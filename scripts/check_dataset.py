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

from vtonwarp.data.agnostic import build_agnostic, select_garment_labels
from vtonwarp.data.io import (
    canonicalise_garment,
    garment_mask_from_rgb,
    load_image,
    load_label_map,
    load_mask,
)
from vtonwarp.data.labels import ATR, CIHP, get_scheme, role_from_filename
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
    parser.add_argument("--garment-type", default="auto")
    parser.add_argument("--label-scheme", default="cihp",
                        help="cihp (= LIP, 20 classes) or atr (18 classes)")
    parser.add_argument("--diagnose-labels", action="store_true",
                        help="score each parsing convention against the data")
    parser.add_argument("--no-canonicalise", action="store_true")
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

    scheme = get_scheme(args.label_scheme)

    if args.diagnose_labels:
        diagnose_labels(records, root, args)
        return

    label_counter = Counter()
    coverage = []
    seg_means = []
    chosen = Counter()
    skipped = 0
    columns = {k: [] for k in
               ("person", "parse", "agnostic", "erased", "head", "shape",
                "garment", "garment mask", "target garment")}

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
        parse = load_label_map(source, args.height, args.width, scheme.num_classes)
        label_counter.update(parse.unique().tolist())

        mask = _garment_mask(paths, garment, args)
        if not args.no_canonicalise:
            garment, mask = canonicalise_garment(garment, mask)

        if args.garment_type == "auto":
            labels = select_garment_labels(
                scheme, parse, person, garment, mask,
                hint=role_from_filename(Path(paths["garment"]).stem),
            )
        else:
            labels = scheme.garment_labels(args.garment_type)
        chosen[tuple(scheme.labels[i] for i in labels)] += 1

        sample = build_agnostic(person, parse, scheme, labels)
        coverage.append(float(sample["garment_mask"].mean()))

        if paths["segmentation"] is not None:
            seg_means.append(float(load_mask(paths["segmentation"],
                                             args.height, args.width).mean()))

        columns["person"].append(person)
        columns["parse"].append(
            (parse.float() / max(1, scheme.num_classes - 1)).repeat(3, 1, 1) * 2 - 1)
        columns["agnostic"].append(sample["agnostic"])
        columns["erased"].append(sample["erase_mask"])
        columns["head"].append(sample["head"])
        columns["shape"].append(sample["shape"])
        columns["garment"].append(garment)
        columns["garment mask"].append(mask)
        columns["target garment"].append(sample["target_garment"])

    if skipped:
        print(f"[check] skipped {skipped} sample(s) with no parse map")

    print(f"\n[check] label scheme '{scheme.name}' — classes present:")
    for label_id, count in sorted(label_counter.items()):
        name = scheme.labels[label_id] if label_id < scheme.num_classes else "?"
        marker = " <- garment" if label_id in scheme.all_garment else ""
        print(f"          {label_id:>2} {name:<15} in {count} sample(s){marker}")

    print("\n[check] garment region selected per sample:")
    for names, count in chosen.most_common():
        print(f"          {'+'.join(names):<28} {count} sample(s)")

    mean_coverage = sum(coverage) / len(coverage)
    print(f"\n[check] garment region covers {mean_coverage * 100:.1f}% of the frame")
    if mean_coverage < 0.02:
        print("        !! almost no garment pixels. Either the label scheme is wrong\n"
              "           (try --diagnose-labels) or the parse maps are empty.")
    elif mean_coverage > 0.6:
        print("        !! garment covers most of the frame — the parse map may be a "
              "silhouette rather than a class map.")

    mean_mask = float(torch.stack(columns["garment mask"]).mean())
    print(f"[check] flat-garment mask covers {mean_mask * 100:.1f}% of the frame")
    if mean_mask > 0.9:
        print("        !! the garment mask is almost the whole frame, so background\n"
              "           will be warped onto the body. Provide explicit garment\n"
              "           masks, or check the product shots have a plain background.")

    if seg_means:
        mean_seg = sum(seg_means) / len(seg_means)
        guess = ("garment mask" if 0.02 < mean_seg < 0.4
                 else "person silhouette" if mean_seg >= 0.4 else "mostly empty")
        print(f"[check] segmentation/ mean occupancy {mean_seg:.2f} -> looks like a "
              f"{guess}")

    contact_sheet({k: torch.stack(v) for k, v in columns.items()}, args.out,
                  max_rows=args.samples)
    print(f"\n[check] wrote {args.out}")
    print("        'agnostic': the original garment must be completely gone.")
    print("        'garment mask': must be the garment only, not the background.")
    print("        'target garment': must show the same garment as the 'garment' column.")


def _garment_mask(paths, garment, args):
    """Same precedence the dataset uses, so the check reflects training."""
    if paths["segmentation"] is not None:
        mask = load_mask(paths["segmentation"], args.height, args.width)
        if 0.02 < mask.mean().item() < 0.75:
            return (mask > 0.5).float()
    return garment_mask_from_rgb(garment)


def diagnose_labels(records, root, args, limit: int = 12):
    """Decide which parsing convention the maps follow.

    Three convention-independent tests, none of which needs the answer in
    advance:

    * **Colour match** (the strongest). The parse region a garment was cut from
      should match that garment's colour. A scheme whose "garment" ids point at
      the trousers when the garment is a shirt scores badly here.
    * **Vertical position.** Whatever ids mean face/hair/hat must sit near the
      top of the image, and shoes near the bottom.
    * **Coverage.** A scheme whose garment ids find no sizeable region in most
      samples is not describing this data.

    A component with no data in a given scheme is *excluded* rather than given a
    neutral value — an earlier version defaulted absent labels to the midpoint,
    which flattered whichever scheme happened to have fewer of its labels
    present and picked the wrong answer on data I knew the ground truth for.
    """
    print("\n[diagnose] scoring parsing conventions against the data...\n")

    samples = []
    max_label = 0
    for record in records[:limit]:
        paths = record.resolve(root)
        source = paths["cihp"] or paths["segmentation"]
        if source is None:
            continue
        person = load_image(paths["person"], args.height, args.width)
        garment = load_image(paths["garment"], args.height, args.width)
        # Load with a generous class count so nothing is clamped away before we
        # know which scheme applies.
        parse = load_label_map(source, args.height, args.width, 32)
        max_label = max(max_label, int(parse.max()))

        mask = _garment_mask(paths, garment, args)
        if not args.no_canonicalise:
            garment, mask = canonicalise_garment(garment, mask)
        samples.append((person, garment, parse, mask))

    if not samples:
        raise SystemExit("no usable samples to diagnose")

    print(f"  parse maps contain class ids 0..{max_label}\n")
    results = []

    for scheme in (CIHP, ATR):
        if max_label >= scheme.num_classes:
            print(f"  {scheme.name:<6} RULED OUT — data contains id {max_label}, "
                  f"but this scheme only defines 0..{scheme.num_classes - 1}")
            continue

        identity_y, feet_y, colour, found = [], [], [], 0

        for person, garment, parse, mask in samples:
            y = _mean_y(parse, scheme.identity)
            if y is not None:
                identity_y.append(y)
            y = _mean_y(parse, scheme.feet)
            if y is not None:
                feet_y.append(y)

            area = mask.sum()
            if area <= 0:
                continue
            garment_colour = (garment * mask).sum(dim=[1, 2]) / area

            best = None
            for label in scheme.all_garment:
                region = (parse == label).float()
                if region.sum() < 0.01 * region.numel():
                    continue
                here = (person * region).sum(dim=[1, 2]) / region.sum()
                distance = float((here - garment_colour).pow(2).sum().sqrt())
                best = distance if best is None else min(best, distance)
            if best is not None:
                colour.append(best)
                found += 1

        # (label, weight, penalty, available)
        terms = [
            ("colour", 2.5, _mean(colour), bool(colour)),
            ("coverage", 1.5, 1.0 - found / len(samples), True),
            ("identity@top", 1.0, abs(_mean(identity_y) - 0.15), bool(identity_y)),
            ("feet@bottom", 1.0, abs(_mean(feet_y) - 0.85), bool(feet_y)),
        ]
        weight = sum(w for _, w, _, ok in terms if ok)
        score = sum(w * v for _, w, v, ok in terms if ok) / max(weight, 1e-6)

        detail = "  ".join(
            f"{name}={value:.3f}" if ok else f"{name}=n/a"
            for name, _, value, ok in terms
        )
        print(f"  {scheme.name:<6} {detail}   -> score {score:.3f}")
        results.append((score, scheme))

    if not results:
        raise SystemExit(
            "\nNo scheme can describe these maps. Check the 'parse' column of the "
            "contact sheet — it should show flat bands of distinct greys, not a "
            "photo or a binary mask."
        )

    results.sort(key=lambda r: r[0])
    best_score, best_scheme = results[0]
    print(f"\n[diagnose] best fit: '{best_scheme.name}'  ({len(samples)} samples)")

    if len(results) > 1 and results[1][0] - best_score < 0.03:
        print("[diagnose] !! the two schemes score almost identically, so this is a\n"
              "              coin flip. Run the normal check under each and compare\n"
              "              the 'target garment' column against the 'garment' one.")

    print(f"[diagnose] set  data.label_scheme: {best_scheme.name}  in both configs,")
    print(f"           then rerun:  --label-scheme {best_scheme.name}  (no "
          f"--diagnose-labels)")
    print( "           and confirm the contact sheet before training.")


def _mean(values, default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _mean_y(parse, labels) -> float | None:
    """Mean vertical position (0 = top, 1 = bottom), or None if absent."""
    mask = torch.zeros_like(parse, dtype=torch.bool)
    for label in labels:
        mask |= parse == label
    if not mask.any():
        return None
    ys = torch.nonzero(mask[0])[:, 0].float()
    return float(ys.mean() / max(1, parse.shape[-2] - 1))


if __name__ == "__main__":
    main()
