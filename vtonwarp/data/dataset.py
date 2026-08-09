"""
The triplet dataset.

One sample = (flat garment, dressed person, conditional maps). Because the
person in the dataset is already wearing the garment that sits next to them in
`garments/`, every sample is *self-paired*: the dressed person is a free,
pixel-perfect ground truth for "what should this garment look like on this
body". That is what makes supervised training possible at all on ~100 images.

At inference we simply feed a different person's conditionals with this
garment; nothing about the model changes.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .agnostic import build_agnostic, stack_condition
from .augment import PairedAugment
from .io import garment_mask_from_rgb, load_image, load_label_map, load_mask
from .labels import GARMENT_SETS, mask_from_labels
from .manifest import TripletRecord


class TripletVTONDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        records: list[TripletRecord],
        height: int = 256,
        width: int = 192,
        garment_type: str = "upper",
        dilate: int = 5,
        augment: PairedAugment | None = None,
        segmentation_role: str = "auto",
    ):
        """
        Args:
            segmentation_role: how to interpret the `segmentation/` folder.
                "auto"          - infer from content (see `_segmentation_role`)
                "garment_mask"  - mask of the garment *on the person*
                "person_mask"   - full-body silhouette
                "parse"         - a second label map; used if `cihp/` is absent
                "ignore"        - do not read it at all
        """
        self.root = Path(root).resolve()
        self.records = records
        self.height = height
        self.width = width
        self.garment_type = garment_type
        self.dilate = dilate
        self.augment = augment or PairedAugment(enabled=False)
        self.segmentation_role = segmentation_role

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        paths = record.resolve(self.root)

        person = load_image(paths["person"], self.height, self.width)
        garment = load_image(paths["garment"], self.height, self.width)

        parse = self._load_parse(paths)
        garment_mask = self._load_garment_mask(paths, garment)

        person, parse, garment, garment_mask = self.augment(
            person, parse, garment, garment_mask
        )

        sample = build_agnostic(person, parse, self.garment_type, self.dilate)
        sample.update(
            {
                "key": record.key,
                "subject": record.subject,
                "person": person,
                "parse": parse,
                "garment": garment,
                "garment_input_mask": garment_mask,
                "condition": stack_condition(sample),
            }
        )
        return sample

    # ------------------------------------------------------------------

    def _load_parse(self, paths: dict) -> torch.Tensor:
        """Get a CIHP label map, falling back to `segmentation/` if needed."""
        if paths["cihp"] is not None:
            return load_label_map(paths["cihp"], self.height, self.width)

        if paths["segmentation"] is not None and self.segmentation_role in ("auto", "parse"):
            return load_label_map(paths["segmentation"], self.height, self.width)

        raise FileNotFoundError(
            f"No parse map for sample; cihp/ and segmentation/ both missing or "
            f"unusable. Run scripts/check_dataset.py to see what was matched."
        )

    def _load_garment_mask(self, paths: dict, garment: torch.Tensor) -> torch.Tensor:
        """Mask of the flat garment product image.

        Note this is the mask of the garment *lying flat*, not on the body — it
        tells the warper which pixels of the source image are actually garment.
        If the dataset's segmentation folder describes the person instead, we
        fall back to thresholding the product shot's background.
        """
        if paths["segmentation"] is not None and self.segmentation_role in (
            "auto",
            "garment_mask",
        ):
            mask = load_mask(paths["segmentation"], self.height, self.width)
            # A mask that covers most of the frame is a person silhouette, not a
            # flat garment; reject it rather than corrupt the warp supervision.
            if 0.02 < mask.mean().item() < 0.75:
                return (mask > 0.5).float()

        return garment_mask_from_rgb(garment)


def collate(batch: list[dict]) -> dict:
    """Default collate, but leaves string fields as lists."""
    out = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        out[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return out


def garment_region_mask(parse: torch.Tensor, garment_type: str) -> torch.Tensor:
    """Batched helper: (B, 1, H, W) mask of the target garment region."""
    labels = GARMENT_SETS[garment_type]
    return torch.cat([mask_from_labels(p, labels)[None] for p in parse], dim=0)
