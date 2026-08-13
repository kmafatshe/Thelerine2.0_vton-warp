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
    canonicalise_garment,
    garment_mask_from_rgb,
    load_image,
    load_label_map,
    load_mask,
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


def inspect_record(record, root: Path, *, height: int, width: int,
                   scheme: LabelScheme, parse_source: str = "auto",
                   segmentation_role: str = "auto", canonicalise: bool = True,
                   garment_fill: float = 0.8,
                   thresholds: dict | None = None) -> SampleReport:
    """Measure one sample exactly as training would load it."""
    paths = record.resolve(root)
    field = parse_field_for(paths, parse_source)

    if paths.get(field) is None:
        return SampleReport(record.key, ["NO_PARSE_FILE"], {})

    try:
        person = load_image(paths["person"], height, width)
        garment = load_image(paths["garment"], height, width)
        parse = load_label_map(paths[field], height, width, scheme.num_classes)
    except Exception as error:
        return SampleReport(record.key, [f"UNREADABLE({type(error).__name__})"], {})

    mask = resolve_garment_mask(paths, garment, height=height, width=width,
                                parse_source=parse_source,
                                segmentation_role=segmentation_role)

    # Measure the mask *before* canonicalisation. Cropping and rescaling makes
    # every garment fill the frame, so a thumbnail on a large canvas would
    # measure the same as a full-bleed shot — and MASK_TINY would never fire.
    # What matters is how much real resolution the source image devoted to the
    # garment, because canonicalising a 14x18 patch up to 256x192 produces a
    # blurred smear that the warper then treats as fabric detail.
    raw_mask_fraction = float(mask.mean())

    if canonicalise:
        garment, mask = canonicalise_garment(garment, mask, fill=garment_fill)

    labels, confident = select_garment_labels(
        scheme, parse, person, garment, mask,
        hint=role_from_filename(Path(paths["garment"]).stem),
    )

    measures = {
        "noise": fragmentation(parse),
        "garment": float(mask_from_labels(parse, labels).mean()),
        "mask": raw_mask_fraction,
        "identity": float(mask_from_labels(parse, scheme.identity).mean()),
    }
    return SampleReport(record.key, flags_for(measures, confident, thresholds),
                        measures)


def audit(records, root: Path, **kwargs) -> list[SampleReport]:
    return [inspect_record(record, root, **kwargs) for record in records]
