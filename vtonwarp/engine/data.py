"""Dataloader construction shared by both training stages."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from ..data.augment import PairedAugment
from ..data.dataset import TripletVTONDataset, collate
from ..data.manifest import build_manifest, read_manifest, split_records, write_manifest


def build_dataloaders(config):
    root = Path(config.data.root)
    # `manifest: null` in YAML means "use the default", so `or` not `get`.
    manifest_path = Path(config.data.get("manifest") or root / "manifest.json")

    if manifest_path.exists() and not config.data.get("rebuild_manifest", False):
        train_records, val_records = read_manifest(manifest_path)
    else:
        records = build_manifest(
            root,
            person_dir=config.data.get("person_dir", "person"),
            garment_dir=config.data.get("garment_dir", "garments"),
            cihp_dir=config.data.get("cihp_dir", "cihp"),
            segmentation_dir=config.data.get("segmentation_dir", "segmentation"),
        )
        if not records:
            raise RuntimeError(
                f"No samples found under {root}. Run scripts/check_dataset.py to "
                "see how filenames were matched."
            )
        train_records, val_records = split_records(
            records, config.data.get("val_fraction", 0.15), config.get("seed", 42)
        )
        write_manifest(manifest_path, train_records, val_records)
        print(f"[data] wrote manifest with {len(train_records)} train / "
              f"{len(val_records)} val samples -> {manifest_path}")

    shared = dict(
        root=root,
        height=config.data.height,
        width=config.data.width,
        garment_type=config.data.get("garment_type", "upper"),
        dilate=config.data.get("erase_dilate", 5),
        segmentation_role=config.data.get("segmentation_role", "auto"),
    )

    augment = PairedAugment(**dict(config.get("augment", {})))

    train_set = TripletVTONDataset(records=train_records, augment=augment, **shared)
    val_set = TripletVTONDataset(
        records=val_records or train_records[: min(4, len(train_records))],
        augment=PairedAugment(enabled=False),
        **shared,
    )

    workers = config.train.get("num_workers", 0)
    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate,
        drop_last=len(train_set) > config.train.batch_size,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(config.train.batch_size, len(val_set)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    return train_loader, val_loader


def infinite(loader):
    """Cycle a loader forever; training is measured in steps, not epochs.

    With ~100 samples an "epoch" is 25 steps, which makes epoch-based schedules
    and logging almost useless. Counting steps keeps every knob comparable
    regardless of how the dataset grows.
    """
    while True:
        for batch in loader:
            yield batch
