"""
The triplet dataset.

One sample = (flat garment, dressed person, conditional maps). Because the
person in the dataset is already wearing the garment that sits next to them in
`garments/`, every sample is *self-paired*: the dressed person is a free,
pixel-perfect ground truth for "what should this garment look like on this
body". That is what makes supervised training possible at all on ~50 images.

At inference we simply feed a different person's conditionals with this
garment; nothing about the model changes.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .agnostic import build_agnostic, select_garment_labels, stack_condition
from .augment import PairedAugment
from .io import (
    canonicalise_garment,
    garment_mask_from_rgb,
    load_image,
    load_label_map,
    load_mask,
)
from .labels import get_scheme, role_from_filename
from .manifest import TripletRecord


class TripletVTONDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        records: list[TripletRecord],
        height: int = 256,
        width: int = 192,
        garment_type: str = "auto",
        label_scheme: str = "cihp",
        dilate: int = 5,
        augment: PairedAugment | None = None,
        segmentation_role: str = "auto",
        canonicalise: bool = True,
        garment_fill: float = 0.8,
    ):
        """
        Args:
            garment_type: "auto" picks the parse region per sample from the
                garment filename and colour (see agnostic.select_garment_labels)
                — required for a dataset mixing dresses, tops and trousers.
                "upper"/"lower"/"full" force one region for every sample.
            label_scheme: which parsing convention the maps follow, "cihp"
                (= LIP) or "atr". Getting this wrong silently targets the wrong
                body parts; `check_dataset.py --diagnose-labels` identifies it.
            segmentation_role: how to interpret the `segmentation/` folder.
                "auto" | "garment_mask" | "person_mask" | "parse" | "ignore"
            canonicalise: crop each garment to its mask and rescale to a
                consistent size before it reaches the network.
        """
        self.root = Path(root).resolve()
        self.records = records
        self.height = height
        self.width = width
        self.garment_type = garment_type
        self.scheme = get_scheme(label_scheme)
        self.dilate = dilate
        self.augment = augment or PairedAugment(enabled=False)
        self.segmentation_role = segmentation_role
        self.canonicalise = canonicalise
        self.garment_fill = garment_fill

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

        if self.canonicalise:
            garment, garment_mask = canonicalise_garment(
                garment, garment_mask, fill=self.garment_fill
            )

        person, parse, garment, garment_mask = self.augment(
            person, parse, garment, garment_mask
        )

        if self.garment_type == "auto":
            labels = select_garment_labels(
                self.scheme, parse, person, garment, garment_mask,
                hint=role_from_filename(Path(paths["garment"]).stem),
            )
        else:
            labels = self.scheme.garment_labels(self.garment_type)

        sample = build_agnostic(person, parse, self.scheme, labels, self.dilate)
        sample.update(
            {
                "key": record.key,
                "subject": record.subject,
                "person": person,
                "parse": parse,
                "garment": garment,
                "garment_input_mask": garment_mask,
                "garment_labels": ",".join(str(i) for i in labels),
                "condition": stack_condition(sample),
            }
        )
        return sample

    # ------------------------------------------------------------------

    def _load_parse(self, paths: dict) -> torch.Tensor:
        """Get a label map, falling back to `segmentation/` if needed."""
        if paths["cihp"] is not None:
            return load_label_map(paths["cihp"], self.height, self.width,
                                  self.scheme.num_classes)

        if paths["segmentation"] is not None and self.segmentation_role in ("auto", "parse"):
            return load_label_map(paths["segmentation"], self.height, self.width,
                                  self.scheme.num_classes)

        raise FileNotFoundError(
            "No parse map for sample; cihp/ and segmentation/ both missing or "
            "unusable. Run scripts/check_dataset.py to see what was matched."
        )

    def _load_garment_mask(self, paths: dict, garment: torch.Tensor) -> torch.Tensor:
        """Mask of the flat garment product image.

        Note this is the mask of the garment *lying flat*, not on the body — it
        tells the warper which pixels of the source image are actually garment.
        If the dataset's segmentation folder describes the person instead, we
        fall back to segmenting the product shot from its background.
        """
        if paths["segmentation"] is not None and self.segmentation_role in (
            "auto",
            "garment_mask",
        ):
            mask = load_mask(paths["segmentation"], self.height, self.width)
            # A mask covering most of the frame is a person silhouette, not a
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
