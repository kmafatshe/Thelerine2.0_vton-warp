#!/usr/bin/env python3
"""
Generate a synthetic dataset with the expected folder layout.

    python scripts/make_dummy_dataset.py --out data/dummy --count 24

This exists so you can smoke-test the whole pipeline — manifest, agnostic
construction, both training stages, inference — in a couple of minutes on CPU
before pointing it at real data, and so that a broken run can be diagnosed as
"my data" versus "the code".

The synthetic people are crude (blocks and ellipses), but they exercise exactly
the same code path: a flat garment on white, a person wearing that garment, a
20-class CIHP label map, and a garment mask.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

H, W = 256, 192

PALETTE = [
    (196, 64, 72), (64, 110, 196), (72, 158, 96), (206, 152, 60),
    (140, 84, 178), (60, 168, 172), (198, 106, 148), (96, 96, 104),
]
SKIN = [(226, 190, 160), (198, 152, 118), (150, 106, 78), (108, 74, 52)]


def stripes(size, colour, rng):
    """A simple patterned fabric so warping errors are visible."""
    w, h = size
    image = Image.new("RGB", (w, h), colour)
    draw = ImageDraw.Draw(image)
    accent = tuple(min(255, c + 55) for c in colour)
    step = rng.integers(8, 18)
    for offset in range(-h, w, int(step) * 2):
        draw.line([(offset, 0), (offset + h, h)], fill=accent, width=int(step))
    return image


def make_sample(index: int, rng):
    colour = PALETTE[index % len(PALETTE)]
    skin = SKIN[rng.integers(0, len(SKIN))]

    torso_w = int(rng.integers(70, 96))
    torso_h = int(rng.integers(90, 118))
    torso_x = W // 2 - torso_w // 2 + int(rng.integers(-8, 9))
    torso_y = int(rng.integers(70, 88))

    fabric = stripes((torso_w, torso_h), colour, rng)

    # ---- person ----------------------------------------------------------
    person = Image.new("RGB", (W, H), (238, 238, 240))
    parse = np.zeros((H, W), dtype=np.uint8)
    draw = ImageDraw.Draw(person)

    # legs / trousers
    pants = (58, 62, 78)
    for side in (-1, 1):
        x0 = W // 2 + side * 8 - 20
        draw.rectangle([x0, torso_y + torso_h - 6, x0 + 34, H - 18], fill=pants)
        parse[torso_y + torso_h - 6: H - 18, max(x0, 0): x0 + 34] = 9

    # arms
    arm_w = 20
    for side, label in ((-1, 14), (1, 15)):
        x0 = torso_x - arm_w if side < 0 else torso_x + torso_w
        draw.rectangle([x0, torso_y + 8, x0 + arm_w, torso_y + torso_h - 6], fill=skin)
        parse[torso_y + 8: torso_y + torso_h - 6, max(x0, 0): x0 + arm_w] = label

    # head
    head_r = 26
    cx, cy = W // 2, torso_y - head_r + 6
    draw.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r], fill=skin)
    yy, xx = np.mgrid[0:H, 0:W]
    head_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= head_r ** 2
    parse[head_mask] = 13
    hair = head_mask & (yy < cy - head_r // 3)
    draw.chord([cx - head_r, cy - head_r, cx + head_r, cy + head_r], 180, 360,
               fill=(48, 38, 34))
    parse[hair] = 2

    # torso garment
    person.paste(fabric, (torso_x, torso_y))
    parse[torso_y: torso_y + torso_h, torso_x: torso_x + torso_w] = 5

    # ---- flat garment ----------------------------------------------------
    garment = Image.new("RGB", (W, H), (255, 255, 255))
    flat_w, flat_h = int(torso_w * 1.25), int(torso_h * 1.25)
    flat_x, flat_y = W // 2 - flat_w // 2, H // 2 - flat_h // 2
    garment.paste(stripes((flat_w, flat_h), colour, rng), (flat_x, flat_y))

    garment_mask = np.zeros((H, W), dtype=np.uint8)
    garment_mask[flat_y: flat_y + flat_h, flat_x: flat_x + flat_w] = 255

    return person, garment, parse, garment_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/dummy")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.out)
    rng = np.random.default_rng(args.seed)

    for index in range(args.count):
        subject = "personA" if index % 2 == 0 else "personB"
        name = f"{index:04d}"
        person, garment, parse, mask = make_sample(index, rng)

        for folder in ("person", "garments", "cihp", "segmentation"):
            (root / folder / subject).mkdir(parents=True, exist_ok=True)

        person.save(root / "person" / subject / f"{name}_person.jpg", quality=95)
        garment.save(root / "garments" / subject / f"{name}_garment.jpg", quality=95)
        np.save(root / "cihp" / subject / f"{name}_cihp.npy", parse)
        Image.fromarray(mask).save(root / "segmentation" / subject / f"{name}_seg.png")

    print(f"[dummy] wrote {args.count} synthetic triplets to {root.resolve()}")


if __name__ == "__main__":
    main()
