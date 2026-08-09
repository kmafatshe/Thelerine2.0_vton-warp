#!/usr/bin/env python3
"""
Inference: put an arbitrary garment on an arbitrary person.

Single pair:

    python infer.py --checkpoint outputs/tryon/tryon.pt \
        --person data/person/personA/0001.jpg \
        --parse  data/cihp/personA/0001.npy \
        --garment data/garments/personB/0007.jpg \
        --out results/0001_wearing_0007.png

Cross-product over the whole dataset (every garment on every person):

    python infer.py --checkpoint outputs/tryon/tryon.pt --root data --grid \
        --out results/grid.png

The grid is the honest evaluation. Training is self-paired, so reconstructing a
person in their own clothes proves nothing — a model can score well there and
still collapse the moment the garment does not match the body it was cut from.
Look at the off-diagonal cells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vtonwarp.data.agnostic import CONDITION_CHANNELS, build_agnostic, stack_condition
from vtonwarp.data.io import garment_mask_from_rgb, load_image, load_label_map, load_mask
from vtonwarp.data.manifest import read_manifest
from vtonwarp.engine.checkpoint import load_checkpoint
from vtonwarp.engine.visualize import contact_sheet, to_uint8
from vtonwarp.models.composer import Composer
from vtonwarp.models.warper import GarmentWarper
from vtonwarp.utils.config import Config, resolve_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--person")
    parser.add_argument("--parse", help="CIHP label map for the person")
    parser.add_argument("--garment")
    parser.add_argument("--garment-mask", default=None)
    parser.add_argument("--out", default="results/tryon.png")
    parser.add_argument("--root", help="dataset root, required with --grid")
    parser.add_argument("--grid", action="store_true",
                        help="render every garment on every validation person")
    parser.add_argument("--limit", type=int, default=4, help="grid size cap")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


class TryOnPipeline:
    """Loads both stages from a stage-2 checkpoint and runs them end to end."""

    def __init__(self, checkpoint_path: str, device):
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        config = Config(checkpoint["config"])
        self.config = config
        self.device = device
        self.height = config.data.height
        self.width = config.data.width
        self.garment_type = config.data.get("garment_type", "upper")
        self.dilate = config.data.get("erase_dilate", 5)

        warp_config = Config(load_checkpoint(config.train.warp_checkpoint,
                                             map_location="cpu")["config"])
        self.warper = GarmentWarper(
            height=self.height, width=self.width,
            condition_channels=CONDITION_CHANNELS,
            base=warp_config.model.base,
            feature_channels=warp_config.model.feature_channels,
            grid_size=warp_config.model.grid_size,
            max_displacement=warp_config.model.max_displacement,
            use_residual_flow=warp_config.model.use_residual_flow,
        )
        self.warper.load_state_dict(checkpoint["models"]["warper"])

        self.composer = Composer(
            condition_channels=CONDITION_CHANNELS,
            base=config.model.base, depth=config.model.depth,
            max_channels=config.model.max_channels,
        )
        # EMA weights are what you want at inference; see engine/ema.py.
        self.composer.load_state_dict(
            checkpoint["ema"].get("composer") or checkpoint["models"]["composer"]
        )

        self.warper.to(device).eval()
        self.composer.to(device).eval()

    # ------------------------------------------------------------------

    def condition_from(self, person_path, parse_path) -> tuple[torch.Tensor, torch.Tensor]:
        person = load_image(Path(person_path), self.height, self.width)
        parse = load_label_map(Path(parse_path), self.height, self.width)
        sample = build_agnostic(person, parse, self.garment_type, self.dilate)
        return stack_condition(sample)[None], sample["agnostic"][None]

    def garment_from(self, garment_path, mask_path=None):
        garment = load_image(Path(garment_path), self.height, self.width)
        if mask_path:
            mask = load_mask(Path(mask_path), self.height, self.width)
        else:
            mask = garment_mask_from_rgb(garment)
        return garment[None], mask[None]

    @torch.no_grad()
    def __call__(self, condition, garment, garment_mask) -> dict:
        condition = condition.to(self.device)
        garment = garment.to(self.device)
        garment_mask = garment_mask.to(self.device)

        warp = self.warper(garment, garment_mask, condition)
        result = self.composer(condition, warp["warped"], warp["warped_mask"])
        result["warped"] = warp["warped_masked"]
        return result


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(tensor).numpy()).save(path)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    pipeline = TryOnPipeline(args.checkpoint, device)

    if args.grid:
        run_grid(pipeline, args)
        return

    if not (args.person and args.parse and args.garment):
        raise SystemExit("--person, --parse and --garment are required without --grid")

    condition, agnostic = pipeline.condition_from(args.person, args.parse)
    garment, mask = pipeline.garment_from(args.garment, args.garment_mask)
    result = pipeline(condition, garment, mask)

    save_image(result["output"][0], args.out)
    contact_sheet(
        {
            "garment": garment,
            "agnostic": agnostic,
            "warped": result["warped"].cpu(),
            "alpha": result["alpha"].cpu(),
            "output": result["output"].cpu(),
        },
        Path(args.out).with_name(Path(args.out).stem + "_debug.png"),
    )
    print(f"[infer] wrote {args.out}")


def run_grid(pipeline: TryOnPipeline, args) -> None:
    """Every validation garment on every validation person."""
    root = Path(args.root or pipeline.config.data.root)
    manifest = Path(pipeline.config.data.get("manifest") or root / "manifest.json")
    _, val_records = read_manifest(manifest)
    records = val_records[: args.limit]
    if len(records) < 2:
        print("[infer] fewer than two validation samples; grid will be trivial")

    columns: dict[str, list[torch.Tensor]] = {}
    for garment_record in records:
        paths = garment_record.resolve(root)
        garment, mask = pipeline.garment_from(paths["garment"])
        column = []
        for person_record in records:
            person_paths = person_record.resolve(root)
            parse_path = person_paths["cihp"] or person_paths["segmentation"]
            if parse_path is None:
                print(f"[infer] skipping {person_record.key}: no parse map")
                continue
            condition, _ = pipeline.condition_from(person_paths["person"], parse_path)
            result = pipeline(condition, garment, mask)
            column.append(result["output"][0].cpu())
        columns[f"garment {garment_record.key}"] = torch.stack(column)

    contact_sheet(columns, args.out, max_rows=len(records))
    print(f"[infer] wrote {args.out}  (row = person, column = garment)")


if __name__ == "__main__":
    main()
