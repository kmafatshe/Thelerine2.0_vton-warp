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

from .agnostic import build_agnostic, stack_condition
from .augment import PairedAugment
from .labels import get_scheme
from .quality import load_sample, parse_field_for
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
        parse_source: str = "auto",
        canonicalise: bool = True,
        garment_fill: float = 0.8,
        crop_to_person: bool = True,
        crop_margin: float = 0.05,
        crop_mode: str = "garment",
        crop_context: float = 0.6,
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
            parse_source: which folder holds the label map — "cihp",
                "segmentation", or "auto" (prefer cihp, fall back). A dataset
                may carry two conditional folders where only one is a label
                map; `check_dataset.py --diagnose-labels` scores both.
            segmentation_role: how to interpret the `segmentation/` folder.
                "auto" | "garment_mask" | "person_mask" | "parse" | "ignore"
            canonicalise: crop each garment to its mask and rescale to a
                consistent size before it reaches the network.
            crop_to_person: crop each photo to the person's bounding box, at
                native resolution. Essential when the source images are scenes
                rather than studio shots — otherwise try-on happens on a few
                thousand pixels in the middle of a landscape.
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
        self.parse_source = parse_source
        self.canonicalise = canonicalise
        self.garment_fill = garment_fill
        self.crop_to_person = crop_to_person
        self.crop_margin = crop_margin
        self.crop_mode = crop_mode
        self.crop_context = crop_context

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        paths = record.resolve(self.root)

        person, parse, garment, garment_mask, labels, confident, _ = load_sample(
            paths, height=self.height, width=self.width, scheme=self.scheme,
            field=self._parse_field(paths), parse_source=self.parse_source,
            segmentation_role=self.segmentation_role,
            canonicalise=self.canonicalise, garment_fill=self.garment_fill,
            crop_to_person=self.crop_to_person, crop_margin=self.crop_margin,
            crop_mode=self.crop_mode, crop_context=self.crop_context,
        )

        if self.garment_type != "auto":
            labels, confident = self.scheme.garment_labels(self.garment_type), True

        person, parse, garment, garment_mask = self.augment(
            person, parse, garment, garment_mask
        )

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
                "labels_confident": confident,
                "condition": stack_condition(sample),
            }
        )
        return sample

    # ------------------------------------------------------------------

    def _parse_field(self, paths: dict) -> str:
        """Which record field holds this sample's label map."""
        field = parse_field_for(paths, self.parse_source)
        if paths.get(field) is None:
            raise FileNotFoundError(
                f"No parse map for this sample from source {self.parse_source!r}. "
                "Run scripts/check_dataset.py --diagnose-labels to see which "
                "folder holds the label maps."
            )
        return field


def collate(batch: list[dict]) -> dict:
    """Default collate, but leaves string fields as lists."""
    out = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        out[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return out
