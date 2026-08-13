"""
The clothing-agnostic person representation.

This is the single most important piece of data engineering in a small-data
try-on system. The network must never be able to see the garment it is being
asked to predict — if any original garment pixel survives into the input, the
model learns the identity function, training loss collapses, and inference with
a *new* garment falls apart. Equally, we must keep everything the network should
not have to invent (face, hair, body proportions), because a few dozen images
are nowhere near enough to learn to redraw a face.

So we decompose the person into three disjoint parts:

    keep      : identity pixels, copied verbatim (face, hair, hat)
    erase     : the garment region + the skin whose visibility depends on the
                garment (arms for a top, legs for trousers), replaced with grey
    describe  : a deliberately *lossy* body silhouette + coarse parse groups,
                which tell the network the pose and build without leaking the
                garment's texture or exact outline
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .labels import LabelScheme, mask_from_labels, parse_to_groups

# Value used to fill erased regions, in [-1, 1] space. 0.0 == mid grey.
ERASE_VALUE = 0.0


def body_shape(parse: torch.Tensor, blur_size: tuple[int, int] = (16, 12)) -> torch.Tensor:
    """A low-frequency silhouette of the whole person.

    We downsample the binary silhouette to ~16x12 and push it back up. The
    result carries pose and body proportion but has had the garment's precise
    contour destroyed — exactly the trade we want. Keeping a high-resolution
    silhouette here is the classic leak that makes VTON models look great on
    train pairs and fail on unpaired inference.
    """
    silhouette = (parse > 0).float()[None]
    height, width = silhouette.shape[-2:]
    small = F.interpolate(silhouette, size=blur_size, mode="area")
    return F.interpolate(small, size=(height, width), mode="bilinear",
                         align_corners=False)[0]


def select_garment_labels(
    scheme: LabelScheme,
    parse: torch.Tensor,
    person: torch.Tensor,
    garment: torch.Tensor,
    garment_mask: torch.Tensor,
    hint: str | None = None,
    min_area: float = 0.01,
) -> tuple[tuple[int, ...], bool]:
    """Work out which parse region *this* garment corresponds to.

    A fixed `garment_type` assumes every sample in the dataset swaps the same
    kind of clothing. A dataset containing dresses, jeans, skirts and tops
    breaks that assumption immediately: asking for "upper" on a jeans sample
    targets the wrong region, so the model is trained to warp trousers onto a
    T-shirt's silhouette.

    Two signals decide it per sample:

    1. **The filename.** `0007_garment_dress.jpg` says what it is, and this is
       by far the more reliable signal when present.
    2. **Colour.** Compare the garment's mean colour against the mean colour of
       each clothing region in the person image. The region the garment was cut
       from will match closely; the others will not.

    Colour matching alone is fooled by a black top over black trousers, so the
    filename hint wins whenever it names a role the scheme knows and that region
    is actually present.

    Returns `(labels, confident)`. `confident` is False when neither signal
    resolved anything and the caller is getting a blanket "all upper-body
    labels" guess — which trains the warper against a region that has nothing
    to do with the garment, so those samples are worth excluding rather than
    silently accepting.
    """
    present = set(parse.unique().tolist())

    if hint and hint in scheme.roles:
        hinted = tuple(i for i in scheme.roles[hint] if i in present)
        if hinted:
            return hinted, True

    garment_area = garment_mask.sum()
    if garment_area < 1:
        return scheme.garment_labels("upper"), False
    garment_colour = (garment * garment_mask).sum(dim=[1, 2]) / garment_area

    scored: list[tuple[float, int]] = []
    total = parse.numel()
    for label in scheme.all_garment:
        if label not in present:
            continue
        region = mask_from_labels(parse, (label,))
        area = region.sum()
        if area / total < min_area:
            continue
        colour = (person * region).sum(dim=[1, 2]) / area
        scored.append((float((colour - garment_colour).pow(2).sum().sqrt()), label))

    if not scored:
        return scheme.garment_labels("upper"), False

    scored.sort()
    best_distance, best_label = scored[0]

    # Keep any other region that matches almost as well — a coat over a top is
    # two labels describing one visible garment.
    selected = [best_label] + [
        label for distance, label in scored[1:]
        if distance <= best_distance * 1.3 + 0.05
    ]
    return tuple(sorted(selected)), True


def build_agnostic(
    person: torch.Tensor,
    parse: torch.Tensor,
    scheme: LabelScheme,
    garment_labels: tuple[int, ...],
    dilate: int = 5,
) -> dict[str, torch.Tensor]:
    """Split a dressed person into keep / erase / describe components.

    Args:
        person: (3, H, W) in [-1, 1].
        parse:  (1, H, W) int64 label map.
        scheme: the parsing convention `parse` follows.
        garment_labels: ids of the garment being swapped in this sample.
        dilate: kernel size for expanding the erase region. A few pixels of
            slack hides parsing errors along the garment boundary; without it a
            rim of the original garment survives and the model latches onto it.
    """
    garment_mask = mask_from_labels(parse, garment_labels)
    erase_labels = tuple(garment_labels) + scheme.erase_extra(garment_labels)
    erase_mask = mask_from_labels(parse, erase_labels)
    identity_mask = mask_from_labels(parse, scheme.identity)

    if dilate > 1:
        erase_mask = _dilate(erase_mask, dilate)
        # Identity always wins over erasure, otherwise dilation eats the chin.
        erase_mask = (erase_mask * (1.0 - identity_mask)).clamp(0.0, 1.0)

    agnostic = person * (1.0 - erase_mask) + ERASE_VALUE * erase_mask
    head = person * identity_mask + ERASE_VALUE * (1.0 - identity_mask)

    return {
        "agnostic": agnostic,
        "head": head,
        "shape": body_shape(parse),
        "parse_groups": parse_to_groups(parse, scheme),
        # Ground truth for the warper: where the garment must land, and what it
        # must look like once it gets there.
        "garment_mask": garment_mask,
        "erase_mask": erase_mask,
        "target_garment": person * garment_mask + ERASE_VALUE * (1.0 - garment_mask),
    }


def _dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Binary dilation via max-pooling."""
    padding = kernel_size // 2
    pooled = F.max_pool2d(mask[None], kernel_size=kernel_size, stride=1, padding=padding)
    return pooled[0]


def stack_condition(sample: dict) -> torch.Tensor:
    """Concatenate the agnostic components into the tensor the networks consume.

    Channel budget:
        3  agnostic RGB
        3  preserved identity RGB
        1  blurred body shape
        7  coarse parse groups
        = 14 channels
    """
    return torch.cat(
        [sample["agnostic"], sample["head"], sample["shape"], sample["parse_groups"]],
        dim=0,
    )


CONDITION_CHANNELS = 3 + 3 + 1 + 7
