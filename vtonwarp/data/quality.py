"""
Per-sample quality measurement, shared by the checker and by training.

A contact sheet shows six samples. A 47-sample dataset can hide a dozen broken
ones behind them, and on a dataset that size each broken sample is 2% of the
training signal — enough to visibly degrade the warp. So every sample is
measured, the bad ones are named, and training drops them by default.

This lives in one module because the checker and the dataset have already
drifted apart twice: once on which folder supplies the garment mask, and once on
whether the segmentation folder is the parse map. Both were silent, and both
made the checker report something different from what training actually did.
Anything that decides "is this sample usable" belongs here, used by both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .agnostic import select_garment_labels
from .io import (
    bbox_from_mask,
    canonicalise_garment,
    crop,
    garment_mask_from_rgb,
    load_mask,
    read_labels,
    read_rgb,
    resize_labels,
    resize_rgb,
)
from .labels import LabelScheme, mask_from_labels, role_from_filename

# Each threshold marks a distinct failure mode; see `flags_for`.
DEFAULT_THRESHOLDS = {
    "noise": 0.25,
    "garment": 0.01,
    "mask_full": 0.90,
    "mask_tiny": 0.02,
    "identity": 0.15,
}


@dataclass
class SampleReport:
    key: str
    flags: list[str]
    measures: dict


def fragmentation(parse: torch.Tensor) -> float:
    """Fraction of neighbouring pixel pairs carrying different labels.

    A real segmentation is made of solid regions, so neighbours disagree only
    along boundaries and this stays at a few percent. Speckle pushes it towards
    0.5 — the signature of argmax over a near-uniform probability stack, or of a
    label map that was saved through lossy JPEG compression.
    """
    right = (parse[:, :, 1:] != parse[:, :, :-1]).float().mean()
    down = (parse[:, 1:, :] != parse[:, :-1, :]).float().mean()
    return float((right + down) / 2)


def parse_field_for(paths: dict, parse_source: str) -> str:
    """Which folder actually supplies the label map for this sample."""
    if parse_source == "segmentation":
        return "segmentation"
    if parse_source == "auto" and paths.get("cihp") is None:
        return "segmentation"
    return "cihp"


def resolve_garment_mask(paths: dict, garment: torch.Tensor, *, height: int,
                         width: int, parse_source: str,
                         segmentation_role: str = "auto") -> torch.Tensor:
    """Mask of the *flat* garment, from the segmentation folder or the image.

    If the segmentation folder is being read as the parse map it describes the
    person, so using it here would mask a product shot with a body silhouette.
    """
    usable = (
        paths.get("segmentation") is not None
        and parse_field_for(paths, parse_source) != "segmentation"
        and segmentation_role in ("auto", "garment_mask")
    )
    if usable:
        mask = load_mask(paths["segmentation"], height, width)
        # A mask covering most of the frame is a person silhouette, not a flat
        # garment; reject it rather than corrupt the warp supervision.
        if 0.02 < mask.mean().item() < 0.75:
            return (mask > 0.5).float()

    return garment_mask_from_rgb(garment)


def read_person(person_path, parse_path, *, num_classes: int,
                max_side: int | None = 1536):
    """Read a photo and its parse map, aligned and bounded in size.

    `max_side` caps the working resolution. It is still six times the training
    frame, so the crop loses nothing that would reach the output, but it keeps
    every intermediate tensor two orders of magnitude smaller than a raw phone
    photo.
    """
    person = read_rgb(person_path, max_side=max_side)
    parse = read_labels(parse_path, num_classes, max_side=max_side)

    # Parse maps are often exported at a different resolution to the photo.
    if parse.shape[-2:] != person.shape[-2:]:
        parse = resize_labels(parse, *person.shape[-2:])
    return person, parse


def crop_box_for(parse: torch.Tensor, scheme: LabelScheme,
                 labels: tuple[int, ...] | None, *, mode: str = "garment",
                 context: float = 0.6, margin: float = 0.05,
                 aspect: float | None = None) -> tuple[int, int, int, int]:
    """Choose the region of the photo to train on.

    `mode="person"` frames the whole body. That sounds right and measures badly:
    a standing person has an aspect around 0.27, so growing the box to the
    frame's 0.75 adds nearly three times the width in pure background. On
    full-body photos it leaves an upper garment covering ~6% of the frame.

    `mode="garment"` frames the garment plus `context` of slack around it, then
    extends the box to include the head so identity is preserved. That puts the
    resolution where the editing happens — the same half-body framing VITON-style
    datasets have by construction — and roughly triples garment coverage on
    full-body shots.

    Falls back to the whole body when the garment region is empty.
    """
    if mode == "garment" and labels:
        garment = mask_from_labels(parse, labels)
        if garment.sum() > 0:
            region = torch.zeros_like(garment)
            top, left, box_h, box_w = bbox_from_mask(garment, margin=context)
            region[:, top:top + box_h, left:left + box_w] = 1.0

            # Keep the face in frame: cropping it out leaves the model rendering
            # a headless torso, and the identity channel with nothing to carry.
            head = mask_from_labels(parse, scheme.identity)
            if head.sum() > 0:
                region = torch.maximum(region, head)

            return bbox_from_mask(region, margin=margin, aspect=aspect)

    return bbox_from_mask((parse > 0).float(), margin=margin, aspect=aspect)


def load_person(person_path, parse_path, *, height: int, width: int,
                num_classes: int, crop_to_person: bool = True,
                margin: float = 0.05, scheme: LabelScheme | None = None,
                labels: tuple[int, ...] | None = None,
                crop_mode: str = "garment", crop_context: float = 0.6):
    """Load a person and their parse map, cropped and resized to the frame.

    The crop is taken at native resolution and resized once, so the detail a
    small part of a large photo actually contains is preserved rather than
    discarded by an early downscale.
    """
    person, parse = read_person(person_path, parse_path, num_classes=num_classes)

    if crop_to_person:
        box = crop_box_for(parse, scheme, labels, mode=crop_mode,
                           context=crop_context, margin=margin,
                           aspect=width / height)
        person, parse = crop(person, box), crop(parse, box)

    return resize_rgb(person, height, width), resize_labels(parse, height, width)


def flags_for(measures: dict, confident: bool, thresholds: dict | None = None) -> list[str]:
    """Name the failure modes a sample exhibits."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    flags = []

    if measures["noise"] > limits["noise"]:
        # Adjacent pixels disagree far too often: this is not a segmentation.
        flags.append("NOISY_PARSE")
    if measures["garment"] < limits["garment"]:
        # Nothing for the warper to aim at.
        flags.append("NO_GARMENT")
    if not confident:
        # Neither the filename nor colour identified the garment's region, so
        # the selection is a blanket guess across every upper-body label.
        flags.append("GUESSED")
    if measures["mask"] > limits["mask_full"]:
        # Background would be warped onto the body as if it were cloth.
        flags.append("MASK_FULL")
    elif measures["mask"] < limits["mask_tiny"]:
        flags.append("MASK_TINY")
    if measures["identity"] > limits["identity"]:
        # Face/hair/hat covering this much means the parse is not tracking the
        # person at all.
        flags.append("IDENTITY_BIG")

    return flags


def load_sample(paths: dict, *, height: int, width: int, scheme: LabelScheme,
                field: str, parse_source: str = "auto",
                segmentation_role: str = "auto", canonicalise: bool = True,
                garment_fill: float = 0.8, crop_to_person: bool = True,
                crop_margin: float = 0.05, crop_mode: str = "garment",
                crop_context: float = 0.6, max_side: int | None = 1536):
    """The one loading path. Dataset, checker and inference all call this.

    Order matters and is not obvious: the crop depends on which garment is being
    swapped, but identifying that needs the garment image. So the garment is
    prepared first, the region is chosen from the *uncropped* person, and only
    then is the photo cropped around it.

    Returns (person, parse, garment, garment_mask, labels, confident,
    raw_mask_fraction), all at the target frame size.
    """
    person_native, parse_native = read_person(
        paths["person"], paths[field], num_classes=scheme.num_classes,
        max_side=max_side)

    # The garment is masked at its own native resolution, so a garment occupying
    # a small part of a large photo keeps the detail that part contains.
    garment_native = read_rgb(paths["garment"], max_side=max_side)
    mask_native = resolve_garment_mask(
        paths, garment_native, height=garment_native.shape[-2],
        width=garment_native.shape[-1], parse_source=parse_source,
        segmentation_role=segmentation_role)

    # Measured before canonicalisation: cropping and rescaling makes every
    # garment fill the frame, so afterwards a thumbnail on a large canvas would
    # measure the same as a full-bleed shot and MASK_TINY could never fire.
    raw_mask_fraction = float(mask_native.mean())

    garment, mask = canonicalise_garment(
        garment_native, mask_native, height, width,
        fill=garment_fill if canonicalise else 1.0)

    # Identify the garment's region against the whole photo, before cropping.
    labels, confident = select_garment_labels(
        scheme,
        resize_labels(parse_native, height, width),
        resize_rgb(person_native, height, width),
        garment, mask,
        hint=role_from_filename(Path(paths["garment"]).stem),
    )

    if crop_to_person:
        box = crop_box_for(parse_native, scheme, labels, mode=crop_mode,
                           context=crop_context, margin=crop_margin,
                           aspect=width / height)
        person_native = crop(person_native, box)
        parse_native = crop(parse_native, box)

    person = resize_rgb(person_native, height, width)
    parse = resize_labels(parse_native, height, width)
    return person, parse, garment, mask, labels, confident, raw_mask_fraction


def load_for_diagnosis(paths: dict, field: str, *, height: int, width: int,
                       parse_source: str = "auto", canonicalise: bool = True,
                       crop_to_person: bool = True, crop_margin: float = 0.05,
                       max_side: int | None = 1536):
    """Load a sample without committing to a label scheme.

    Identifying the scheme is the question being asked, so the loading cannot
    depend on the answer: the parse is read with a permissive class count, and
    the crop uses the whole-body silhouette, which is `parse > 0` under every
    convention. Both schemes then score the same pixels, which is what makes the
    comparison fair.
    """
    person, parse = read_person(paths["person"], paths[field],
                                num_classes=32, max_side=max_side)

    if crop_to_person:
        box = crop_box_for(parse, None, None, mode="person",
                           margin=crop_margin, aspect=width / height)
        person, parse = crop(person, box), crop(parse, box)

    garment = read_rgb(paths["garment"], max_side=max_side)
    mask = resolve_garment_mask(paths, garment, height=garment.shape[-2],
                                width=garment.shape[-1],
                                parse_source=parse_source)
    garment, mask = canonicalise_garment(garment, mask, height, width,
                                         fill=0.8 if canonicalise else 1.0)

    return (resize_rgb(person, height, width),
            resize_labels(parse, height, width), garment, mask)


def inspect_record(record, root: Path, *, height: int, width: int,
                   scheme: LabelScheme, thresholds: dict | None = None,
                   **load_kwargs) -> SampleReport:
    """Measure one sample exactly as training would load it."""
    paths = record.resolve(root)
    field = parse_field_for(paths, load_kwargs.get("parse_source", "auto"))

    if paths.get(field) is None:
        return SampleReport(record.key, ["NO_PARSE_FILE"], {})

    try:
        _, parse, _, _, labels, confident, raw_mask = load_sample(
            paths, height=height, width=width, scheme=scheme, field=field,
            **load_kwargs)
    except Exception as error:
        return SampleReport(record.key, [f"UNREADABLE({type(error).__name__})"], {})

    measures = {
        "noise": fragmentation(parse),
        "garment": float(mask_from_labels(parse, labels).mean()),
        "mask": raw_mask,
        "identity": float(mask_from_labels(parse, scheme.identity).mean()),
    }
    return SampleReport(record.key, flags_for(measures, confident, thresholds),
                        measures)


def audit(records, root: Path, **kwargs) -> list[SampleReport]:
    return [inspect_record(record, root, **kwargs) for record in records]
