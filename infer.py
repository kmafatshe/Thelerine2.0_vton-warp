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

from vtonwarp.data.agnostic import (
    CONDITION_CHANNELS,
    build_agnostic,
    select_garment_labels,
    stack_condition,
)
from vtonwarp.data.io import (
    canonicalise_garment,
    garment_mask_from_rgb,
    load_mask,
    read_rgb,
)
from vtonwarp.data.quality import load_sample
from vtonwarp.data.labels import get_scheme, role_from_filename
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
    parser.add_argument("--pairs", default="same-type",
                        choices=("same-type", "all"),
                        help="which garment/person combinations to render")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


class TryOnPipeline:
    """Loads both stages from a stage-2 checkpoint and runs them end to end."""

    def __init__(self, checkpoint_path: str, device):
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

        # A warp.pt has no composer and no warp_checkpoint key, so every
        # attribute lookup below fails with something unrelated to the mistake.
        if "composer" not in checkpoint.get("models", {}):
            raise SystemExit(
                f"{checkpoint_path} is a stage-1 (warp) checkpoint. Inference "
                "needs the stage-2 checkpoint, normally named tryon.pt — train "
                "stage 2 first with train_tryon.py."
            )

        config = Config(checkpoint["config"])
        self.config = config
        self.device = device
        self.height = config.data.height
        self.width = config.data.width
        self.garment_type = config.data.get("garment_type", "auto")
        self.scheme = get_scheme(config.data.get("label_scheme", "cihp"))
        self.dilate = config.data.get("erase_dilate", 5)
        self.canonicalise = config.data.get("canonicalise_garment", True)
        self.garment_fill = config.data.get("garment_fill", 0.8)
        self.crop_to_person = config.data.get("crop_to_person", True)
        self.crop_margin = config.data.get("crop_margin", 0.05)
        self.crop_mode = config.data.get("crop_mode", "garment")
        self.crop_context = config.data.get("crop_context", 0.6)
        self.preserve_legs = config.data.get("preserve_legs", True)

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

    def condition_from(self, person_path, parse_path, garment_path,
                       mask_path=None):
        """Build everything one try-on needs, through the shared loading path.

        The garment is not optional: with `garment_type: auto` it decides which
        region of the person to erase *and* where to crop. Passing it separately
        from the person is what let the two disagree.
        """
        paths = {"person": Path(person_path), "cihp": Path(parse_path),
                 "garment": Path(garment_path),
                 "segmentation": Path(mask_path) if mask_path else None}

        person, parse, garment, mask, labels, _, _ = load_sample(
            paths, height=self.height, width=self.width, scheme=self.scheme,
            field="cihp", parse_source="cihp",
            segmentation_role="garment_mask" if mask_path else "ignore",
            canonicalise=self.canonicalise, garment_fill=self.garment_fill,
            crop_to_person=self.crop_to_person, crop_margin=self.crop_margin,
            crop_mode=self.crop_mode, crop_context=self.crop_context,
        )
        if self.garment_type != "auto":
            labels = self.scheme.garment_labels(self.garment_type)

        sample = build_agnostic(person, parse, self.scheme, labels, self.dilate,
                                preserve_legs=self.preserve_legs)
        return (stack_condition(sample)[None], sample["agnostic"][None],
                garment[None], mask[None])

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

    condition, agnostic, garment, mask = pipeline.condition_from(
        args.person, args.parse, args.garment, args.garment_mask)
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
    """Every garment on every person, as a matrix.

    By default only *same-type* combinations are rendered — a dress onto someone
    wearing a dress, trousers onto someone wearing trousers. That is not a way
    of hiding failures; it is the model's actual competence, and the reason is
    structural.

    Training is self-paired, so every example the warper saw was a garment
    deformed onto the region it was cut from. The region to fill comes from the
    *source person's* parse, so putting a T-shirt on someone wearing a dress
    asks the warper to stretch a small garment across a dress-shaped hole — a
    deformation well outside anything it was trained on. Cross-category try-on
    needs a model that first predicts a new parse for the incoming garment
    (ACGPN, PF-AFN), which is a separate network and far more data.

    Incompatible cells are left blank rather than filled with a plausible-looking
    failure. Pass --pairs all to render them anyway.
    """
    root = Path(args.root or pipeline.config.data.root)
    manifest = Path(pipeline.config.data.get("manifest") or root / "manifest.json")
    _, val_records = read_manifest(manifest)
    records = val_records[: args.limit]
    if len(records) < 2:
        print("[infer] fewer than two validation samples; grid will be trivial")

    def garment_role(record):
        """The garment category this record's own garment belongs to."""
        return role_from_filename(Path(record.garment).stem)

    roles = {record.key: garment_role(record) for record in records}
    unknown = [key for key, role in roles.items() if role is None]
    if unknown and args.pairs == "same-type":
        print(f"[infer] {len(unknown)} garment(s) have no recognisable type in "
              f"their filename; they will pair with anything")

    columns: dict[str, list[torch.Tensor]] = {}
    skipped = 0
    for garment_record in records:
        garment_paths = garment_record.resolve(root)
        garment_role_name = roles[garment_record.key]
        column = []

        for person_record in records:
            person_role = roles[person_record.key]
            compatible = (
                args.pairs == "all"
                or garment_role_name is None or person_role is None
                or garment_role_name == person_role
            )
            if not compatible:
                # Flat grey, matching the erase colour, so a blank cell reads as
                # "not attempted" rather than "attempted and came out empty".
                column.append(torch.zeros(3, pipeline.height, pipeline.width))
                skipped += 1
                continue

            person_paths = person_record.resolve(root)
            parse_path = person_paths["cihp"] or person_paths["segmentation"]
            condition, _, garment, mask = pipeline.condition_from(
                person_paths["person"], parse_path, garment_paths["garment"])
            result = pipeline(condition, garment, mask)
            column.append(result["output"][0].cpu())

        label = f"{garment_record.key}"
        if garment_role_name:
            label += f" ({garment_role_name})"
        columns[label] = torch.stack(column)

    if skipped:
        print(f"[infer] left {skipped} cross-type cell(s) blank; --pairs all "
              f"renders them")

    contact_sheet(columns, args.out, max_rows=len(records),
                  caption=" | ".join([
                      "rows = people, columns = garments",
                      f"pairs={args.pairs}",
                      "blank = incompatible type",
                  ]))
    print(f"[infer] wrote {args.out}  (row = person, column = garment)")


if __name__ == "__main__":
    main()
