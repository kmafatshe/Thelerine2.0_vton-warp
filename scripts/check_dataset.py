#!/usr/bin/env python3
"""
Dataset sanity check. Run this first, every time, before training.

    python scripts/check_dataset.py --root /path/to/dataset

Small-dataset projects fail far more often from a silently mismatched file than
from a bad architecture — one wrong parse map is 1% of your data. This script
reports:

  * how many person images were matched to a garment, and which were not
  * what the parse maps actually contain (which CIHP classes are present)
  * whether the garment region is a plausible fraction of the image
  * whether segmentation/ looks like a garment mask, a silhouette or a parse map
  * a contact sheet of the derived agnostic representation, so you can *see*
    that the garment really has been erased

If the "garment coverage" number is near zero your parse maps are using a
different label convention; edit UPPER_GARMENT in vtonwarp/data/labels.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter
from pathlib import Path

import torch

from vtonwarp.data.agnostic import build_agnostic
from vtonwarp.data.io import garment_mask_from_rgb, load_image, load_label_map, load_mask
from vtonwarp.data.labels import CIHP_LABELS, GARMENT_SETS
from vtonwarp.data.manifest import build_manifest
from vtonwarp.engine.visualize import contact_sheet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--garment-type", default="upper")
    parser.add_argument("--out", default="outputs/dataset_check.png")
    parser.add_argument("--samples", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    print(f"[check] root: {root}")

    records = build_manifest(root)
    print(f"[check] matched {len(records)} triplets")
    if not records:
        raise SystemExit("nothing matched — check folder names and file stems")

    by_subject = Counter(r.subject or "(flat)" for r in records)
    print(f"[check] subjects: {dict(by_subject)}")
    print(f"[check] with cihp: {sum(r.cihp is not None for r in records)}")
    print(f"[check] with segmentation: "
          f"{sum(r.segmentation is not None for r in records)}")

    label_counter = Counter()
    coverage = []
    seg_means = []
    columns = {k: [] for k in
               ("person", "parse", "agnostic", "head", "shape", "garment",
                "garment mask", "target garment")}

    for record in records[: args.samples]:
        paths = record.resolve(root)
        person = load_image(paths["person"], args.height, args.width)
        garment = load_image(paths["garment"], args.height, args.width)

        source = paths["cihp"] or paths["segmentation"]
        parse = load_label_map(source, args.height, args.width)
        label_counter.update(parse.unique().tolist())

        sample = build_agnostic(person, parse, args.garment_type)
        coverage.append(float(sample["garment_mask"].mean()))

        if paths["segmentation"] is not None:
            seg_means.append(float(load_mask(paths["segmentation"],
                                             args.height, args.width).mean()))

        columns["person"].append(person)
        columns["parse"].append((parse.float() / 19.0).repeat(3, 1, 1) * 2 - 1)
        columns["agnostic"].append(sample["agnostic"])
        columns["head"].append(sample["head"])
        columns["shape"].append(sample["shape"])
        columns["garment"].append(garment)
        columns["garment mask"].append(garment_mask_from_rgb(garment))
        columns["target garment"].append(sample["target_garment"])

    print("\n[check] CIHP classes present in sampled parse maps:")
    for label_id, count in sorted(label_counter.items()):
        name = CIHP_LABELS[label_id] if label_id < len(CIHP_LABELS) else "?"
        marker = " <- garment" if label_id in GARMENT_SETS[args.garment_type] else ""
        print(f"          {label_id:>2} {name:<15} in {count} sample(s){marker}")

    mean_coverage = sum(coverage) / len(coverage)
    print(f"\n[check] garment region covers {mean_coverage * 100:.1f}% of the frame")
    if mean_coverage < 0.02:
        print("        !! almost no garment pixels — your parse maps probably use a "
              "different label convention. Edit vtonwarp/data/labels.py.")
    elif mean_coverage > 0.6:
        print("        !! garment covers most of the frame — the parse map may be a "
              "silhouette rather than a class map.")

    if seg_means:
        mean_seg = sum(seg_means) / len(seg_means)
        guess = ("garment mask" if 0.02 < mean_seg < 0.4
                 else "person silhouette" if mean_seg >= 0.4 else "mostly empty")
        print(f"[check] segmentation/ mean occupancy {mean_seg:.2f} -> looks like a "
              f"{guess}")

    contact_sheet({k: torch.stack(v) for k, v in columns.items()}, args.out,
                  max_rows=args.samples)
    print(f"\n[check] wrote {args.out}")
    print("        Look at the 'agnostic' column: if you can still see the original "
          "garment, raise data.erase_dilate or fix the parse maps.")


if __name__ == "__main__":
    main()
