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
                garment (arms for an upper garment), replaced with flat grey
    describe  : a deliberately *lossy* body silhouette + coarse parse groups,
                which tell the network the pose and build without leaking the
                garment's texture or exact outline
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .labels import (
    ERASE_EXTRA,
    GARMENT_SETS,
    IDENTITY,
    mask_from_labels,
    parse_to_groups,
)

# Value used to fill erased regions, in [-1, 1] space. 0.0 == mid grey.
ERASE_VALUE = 0.0


def body_shape(parse: torch.Tensor, blur_size: tuple[int, int] = (16, 12)) -> torch.Tensor:
    """A low-frequency silhouette of the whole person.

    We downsample the binary silhouette to ~16x12 and push it back up. The
    result carries pose and body proportion but has had the garment's precise
    contour destroyed — exactly the trade we want. Keeping a high-resolution
    silhouette here is the classic leak that makes VTON models look great on
    train pairs and fail on unpaired inference.

    Args:
        parse: (1, H, W) int64 label map.
    Returns:
        (1, H, W) float tensor in [0, 1].
    """
    silhouette = (parse > 0).float()[None]
    height, width = silhouette.shape[-2:]
    small = F.interpolate(silhouette, size=blur_size, mode="area")
    return F.interpolate(small, size=(height, width), mode="bilinear",
                         align_corners=False)[0]


def build_agnostic(
    person: torch.Tensor,
    parse: torch.Tensor,
    garment_type: str = "upper",
    dilate: int = 5,
) -> dict[str, torch.Tensor]:
    """Split a dressed person into keep / erase / describe components.

    Args:
        person: (3, H, W) in [-1, 1].
        parse:  (1, H, W) int64 CIHP label map.
        garment_type: one of "upper", "lower", "full".
        dilate: kernel size for expanding the erase region. A few pixels of
            slack hides parsing errors along the garment boundary; without it
            a rim of the original garment survives and the model latches on
            to it.

    Returns:
        dict with keys: agnostic, head, shape, parse_groups, garment_mask,
        target_garment.
    """
    garment_labels = GARMENT_SETS[garment_type]

    garment_mask = mask_from_labels(parse, garment_labels)
    erase_mask = mask_from_labels(parse, garment_labels + ERASE_EXTRA[garment_type])
    identity_mask = mask_from_labels(parse, IDENTITY)

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
        "parse_groups": parse_to_groups(parse),
        # Ground truth for the warper: where the garment must land, and what it
        # must look like once it gets there.
        "garment_mask": garment_mask,
        "target_garment": person * garment_mask + ERASE_VALUE * (1.0 - garment_mask),
    }


def _dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Binary dilation via max-pooling."""
    padding = kernel_size // 2
    pooled = F.max_pool2d(mask[None], kernel_size=kernel_size, stride=1, padding=padding)
    return pooled[0]


def stack_condition(sample: dict) -> torch.Tensor:
    """Concatenate the agnostic components into the tensor the networks consume.

    Channel budget (upper-body, 7 parse groups):
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
