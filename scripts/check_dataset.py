#!/usr/bin/env python3
"""
Dataset sanity check. Run this first, every time, before training.

    python scripts/check_dataset.py --root /path/to/dataset

Small-dataset projects fail far more often from a silently mismatched file than
from a bad architecture — one wrong parse map is 1% of your data. This script
reports:

  * which folder it picked for each of the four roles, and what is inside them
  * how many person images were matched to a garment, and which were not
  * what the parse maps actually contain (which CIHP classes are present)
  * whether the garment region is a plausible fraction of the image
  * whether segmentation/ looks like a garment mask, a silhouette or a parse map
  * a contact sheet of the derived agnostic representation, so you can *see*
    that the garment really has been erased

Folder names are auto-detected (`cond/` resolves to the cihp role, `seg/` to
segmentation, and so on). Override any of them explicitly with --cihp-dir etc.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter

import torch

from vtonwarp.data.agnostic import build_agnostic
from vtonwarp.data.io import garment_mask_from_rgb, load_image, load_label_map, load_mask
from vtonwarp.data.labels import CIHP_LABELS, GARMENT_SETS
from vtonwarp.data.manifest import build_manifest, describe_layout, resolve_layout
from vtonwarp.engine.visualize import contact_sheet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--person-dir", default=None)
    parser.add_argument("--garment-dir", default=None)
    parser.add_argument("--cihp-dir", default=None)
    parser.add_argument("--segmentation-dir", default=None)
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

    overrides = {
        "person_dir": args.person_dir,
        "garment_dir": args.garment_dir,
        "cihp_dir": args.cihp_dir,
        "segmentation_dir": args.segmentation_dir,
    }
    layout = resolve_layout(root, overrides)
    print("[check] folder roles:")
    for role, name in layout.items():
        print(f"          {role:<18} -> {name or '(not found)'}")

    records = build_manifest(root, **layout)
    print(f"\n[check] matched {len(records)} triplets")
    if not records:
        print("\n" + describe_layout(root))
        raise SystemExit(
            "\nNothing matched. Person and garment files pair up by their "
            "normalised key (shown above) — if the keys differ between folders, "
            "rename the files or extend ROLE_TOKENS in vtonwarp/data/manifest.py."
        )

    by_subject = Counter(r.subject or "(flat)" for r in records)
    n_cihp = sum(r.cihp is not None for r in records)
    n_seg = sum(r.segmentation is not None for r in records)
    print(f"[check] subjects: {dict(by_subject)}")
    print(f"[check] with cihp: {n_cihp}")
    print(f"[check] with segmentation: {n_seg}")

    # A parse map is mandatory: the whole agnostic representation is derived
    # from it. Fail here with something actionable rather than deep inside a
    # loader with a None path.
    if n_cihp == 0 and n_seg == 0:
        print("\n" + describe_layout(root))
        raise SystemExit(
            "\nNo parse maps were matched to any person, so the clothing-agnostic\n"
            "input cannot be built. Two things to check in the report above:\n"
            "  1. Is a parse folder present at all? If it is named something\n"
            "     unusual, pass it explicitly:  --cihp-dir <folder>\n"
            "  2. Do its files reduce to the same key as the person images? The\n"
            "     'key' shown for each sample is what matching compares. If the\n"
            "     keys differ, rename the files so the ids line up.\n"
            "Without parse maps you would need a human-parsing model (e.g. CIHP\n"
            "PGN or SCHP) to generate them before training."
        )

    label_counter = Counter()
    coverage = []
    seg_means = []
    skipped = 0
    columns = {k: [] for k in
               ("person", "parse", "agnostic", "head", "shape", "garment",
                "garment mask", "target garment")}

    for record in records:
        if len(columns["person"]) >= args.samples:
            break
        paths = record.resolve(root)
        source = paths["cihp"] or paths["segmentation"]
        if source is None:
            skipped += 1
            continue

        person = load_image(paths["person"], args.height, args.width)
        garment = load_image(paths["garment"], args.height, args.width)
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

    if skipped:
        print(f"[check] skipped {skipped} sample(s) with no parse map")

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
