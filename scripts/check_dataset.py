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
from vtonwarp.data.io import canonicalise_garment, load_mask, read_rgb
from vtonwarp.data.labels import (
    ATR,
    CIHP,
    get_scheme,
    mask_from_labels,
    role_from_filename,
)
from vtonwarp.data.manifest import build_manifest, describe_layout, resolve_layout
from vtonwarp.data.quality import audit as quality_audit
from vtonwarp.data.quality import (
    load_for_diagnosis,
    load_sample,
    parse_field_for,
)
from vtonwarp.data.quality import resolve_garment_mask
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
    parser.add_argument("--no-crop", action="store_true",
                        help="do not crop photos to the person")
    parser.add_argument("--crop-margin", type=float, default=0.05)
    parser.add_argument("--crop-mode", default="garment",
                        choices=("garment", "person"))
    parser.add_argument("--crop-context", type=float, default=0.6)
    parser.add_argument("--audit", action="store_true",
                        help="per-sample quality report over the whole dataset")
    parser.add_argument("--parse-source", default="auto",
                        choices=("auto", "cihp", "segmentation"))
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

    if args.audit:
        audit(records, root, args, scheme)
        return

    label_counter = Counter()
    coverage = []
    mask_fractions = []
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
        source = (paths.get(args.parse_source) if args.parse_source != "auto"
                  else paths["cihp"] or paths["segmentation"])
        if source is None:
            skipped += 1
            continue

        parse_field = parse_field_for(paths, args.parse_source)
        person, parse, garment, mask, labels, _, raw_mask_fraction = load_sample(
            paths, height=args.height, width=args.width, scheme=scheme,
            field=parse_field, parse_source=args.parse_source,
            canonicalise=not args.no_canonicalise,
            crop_to_person=not args.no_crop, crop_margin=args.crop_margin,
            crop_mode=args.crop_mode, crop_context=args.crop_context,
        )
        label_counter.update(parse.unique().tolist())
        if args.garment_type != "auto":
            labels = scheme.garment_labels(args.garment_type)
        chosen[tuple(scheme.labels[i] for i in labels)] += 1

        sample = build_agnostic(person, parse, scheme, labels)
        coverage.append(float(sample["garment_mask"].mean()))
        mask_fractions.append(raw_mask_fraction)

        if paths["segmentation"] is not None and parse_field != "segmentation":
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

    mean_mask = sum(mask_fractions) / len(mask_fractions)
    print(f"[check] flat-garment mask covers {mean_mask * 100:.1f}% of its "
          f"product shot (before canonicalisation)")
    if mean_mask > 0.9:
        print("        !! the garment mask is almost the whole frame, so background\n"
              "           will be warped onto the body. Provide explicit garment\n"
              "           masks, or check the product shots have a plain background.")

    if parse_field == "segmentation":
        print("[check] segmentation/ is being read as the parse map "
              "(data.parse_source), so it is not also used as the garment mask")
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


def audit(records, root, args, scheme, max_listed: int = 40):
    """Per-sample quality report over the whole dataset.

    The measurement itself lives in vtonwarp/data/quality.py, so what this
    prints is exactly what training will act on. See that module for what each
    flag means.
    """
    print(f"\n[audit] checking all {len(records)} samples "
          f"(scheme '{scheme.name}', parse source '{args.parse_source}')\n")

    reports = quality_audit(
        records, root, height=args.height, width=args.width, scheme=scheme,
        parse_source=args.parse_source, canonicalise=not args.no_canonicalise,
        crop_to_person=not args.no_crop, crop_margin=args.crop_margin,
        crop_mode=args.crop_mode, crop_context=args.crop_context,
    )
    flagged = [r for r in reports if r.flags]

    print(f"  {'sample':<22} {'noise':>7} {'garment':>8} {'mask':>7} "
          f"{'ident':>7}  flags")
    for report in flagged[:max_listed]:
        m = report.measures
        if m:
            print(f"  {report.key:<22} {m['noise']:>7.3f} {m['garment']:>8.3f} "
                  f"{m['mask']:>7.3f} {m['identity']:>7.3f}  "
                  f"{','.join(report.flags)}")
        else:
            print(f"  {report.key:<22} {'-':>7} {'-':>8} {'-':>7} {'-':>7}  "
                  f"{','.join(report.flags)}")
    if len(flagged) > max_listed:
        print(f"  ... and {len(flagged) - max_listed} more")

    print(f"\n[audit] {len(reports) - len(flagged)} clean, {len(flagged)} "
          f"flagged, of {len(reports)}")

    measured = [r.measures for r in reports if r.measures]
    if measured:
        median = lambda k: sorted(m[k] for m in measured)[len(measured) // 2]  # noqa: E731
        print(f"[audit] medians — noise {median('noise'):.3f}  "
              f"garment {median('garment'):.3f}  mask {median('mask'):.3f}  "
              f"identity {median('identity'):.3f}")

    counts = Counter(f for r in flagged for f in r.flags)
    if counts:
        print("[audit] flag counts:", dict(counts.most_common()))
    print("\n[audit] training drops flagged samples automatically; set "
          "data.audit_filter=false to keep them")


def _garment_mask(paths, garment, args, parse_field=None):
    """Delegates to the shared resolver so the check reflects training."""
    return resolve_garment_mask(
        paths, garment, height=garment.shape[-2], width=garment.shape[-1],
        parse_source=("segmentation" if parse_field == "segmentation"
                      else args.parse_source),
    )


def diagnose_labels(records, root, args, limit: int = 12):
    """Decide which folder holds the parse map, and which convention it uses.

    Both questions are silent failures. A dataset can carry two conditional
    folders where only one is a label map — the other being pose heatmaps or
    one-hot probabilities, which argmax turns into a well-formed parse map that
    describes nothing. And a real label map still means nothing until you know
    whether id 5 is "upper clothes" (CIHP) or "skirt" (ATR).

    So every (folder, scheme) combination is scored on three
    convention-independent tests:

    * **Colour match** (the strongest). The parse region a garment was cut from
      should match that garment's colour.
    * **Vertical position.** Ids meaning face/hair/hat must sit near the top of
      the image, and shoes near the bottom. A combination that puts heads at
      mid-height is not describing this person.
    * **Coverage.** Garment ids that find no sizeable region in most samples are
      not describing this data.

    A component with no data is *excluded* rather than given a neutral value —
    defaulting absent labels to the midpoint flattered whichever scheme had
    fewer of its labels present, and picked the wrong answer on data with known
    ground truth.
    """
    print("\n[diagnose] scoring parse sources and conventions...\n")

    sources = [
        (name, field) for name, field in
        (("cihp folder", "cihp"), ("segmentation folder", "segmentation"))
        if any(getattr(r, field) for r in records)
    ]
    if not sources:
        raise SystemExit("no conditional files at all — nothing to diagnose")

    results = []
    for source_name, field in sources:
        usable = [r for r in records if getattr(r, field)][:limit]
        if not usable:
            continue

        samples, max_label = [], 0
        for record in usable:
            paths = record.resolve(root)
            person, parse, garment, mask = load_for_diagnosis(
                paths, field, height=args.height, width=args.width,
                # `field` is the candidate being scored, so pass it as the parse
                # source: when scoring the segmentation folder as the parse map,
                # it describes the person and must not also mask the garment.
                parse_source=field,
                canonicalise=not args.no_canonicalise,
                # Deliberately uncropped. The crop is derived from the parse
                # silhouette, so cropping with a *wrong* parse changes the
                # framing that the scoring then judges — a confound that pushed
                # a known-correct dataset from 0.125 to 0.503. Scheme identity
                # is a question about labels, not framing.
                crop_to_person=False,
            )
            max_label = max(max_label, int(parse.max()))

            samples.append((person, garment, parse, mask))

        print(f"  {source_name}: {len(samples)} samples, class ids 0..{max_label}")

        for scheme in (CIHP, ATR):
            if max_label >= scheme.num_classes:
                print(f"      {scheme.name:<6} ruled out — contains id {max_label}, "
                      f"scheme defines 0..{scheme.num_classes - 1}")
                continue
            score, detail = _score(samples, scheme)
            print(f"      {scheme.name:<6} {detail}   -> score {score:.3f}")
            results.append((score, source_name, field, scheme))
        print()

    if not results:
        raise SystemExit(
            "No scheme can describe these maps. Check the 'parse' column of the "
            "contact sheet — it should show flat bands of distinct greys, not a "
            "photo, a binary mask or noise."
        )

    results.sort(key=lambda r: r[0])
    best_score, best_source, best_field, best_scheme = results[0]

    print(f"[diagnose] best fit: {best_source}, scheme '{best_scheme.name}' "
          f"(score {best_score:.3f})")
    print(f"[diagnose] set in both configs:")
    print(f"             data.label_scheme: {best_scheme.name}")
    print(f"             data.parse_source: "
          f"{'cihp' if best_field == 'cihp' else 'segmentation'}")
    print(f"           then rerun this script with --label-scheme "
          f"{best_scheme.name} --parse-source "
          f"{'cihp' if best_field == 'cihp' else 'segmentation'}")
    print( "           and confirm the contact sheet before training.")

    if best_score > 0.45:
        print("\n[diagnose] !! even the best combination scores poorly. Likely the\n"
              "              parse maps are misaligned with the person images, or\n"
              "              neither folder holds a real label map. The 'contents:'\n"
              "              lines in the layout report above say what each folder\n"
              "              actually contains — a multi-channel stack is not a\n"
              "              label map, whatever argmax makes of it.")


def _score(samples, scheme):
    """Score one (data, scheme) pairing. Lower is better."""
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

    terms = [
        ("colour", 2.5, _mean(colour), bool(colour)),
        ("coverage", 1.5, 1.0 - found / len(samples), True),
        ("identity@top", 1.0, abs(_mean(identity_y) - 0.15), bool(identity_y)),
        ("feet@bottom", 1.0, abs(_mean(feet_y) - 0.85), bool(feet_y)),
    ]
    weight = sum(w for _, w, _, ok in terms if ok)
    score = sum(w * v for _, w, v, ok in terms if ok) / max(weight, 1e-6)
    detail = "  ".join(f"{n}={v:.3f}" if ok else f"{n}=n/a"
                       for n, _, v, ok in terms)
    return score, detail


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
