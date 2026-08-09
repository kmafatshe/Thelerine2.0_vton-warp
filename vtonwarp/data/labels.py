"""
Human-parsing label semantics, across the conventions people actually ship.

A parse map is just an H x W array of integers, and the integers mean nothing
without knowing which taxonomy produced them. The two common ones disagree badly
in the range we care about:

    id   CIHP / LIP (20 classes)      ATR (18 classes)
    4    sunglasses                   upper clothes
    5    upper clothes                skirt
    6    dress                        pants
    7    coat                         dress
    9    pants                        left shoe
    11   scarf                        face
    12   skirt                        left leg

So a model that assumes CIHP but is fed ATR maps will "erase the garment" by
erasing the skirt, the trousers and the person's face. This module keeps each
convention in one place and makes the choice explicit and checkable, rather than
hardcoding a set of magic numbers and hoping.

`scripts/check_dataset.py --diagnose-labels` scores each scheme against your
data and tells you which one fits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LabelScheme:
    """One parsing convention, and the semantic roles its ids play."""

    name: str
    labels: tuple[str, ...]

    upper: tuple[int, ...]        # tops, coats, anything worn on the torso
    lower: tuple[int, ...]        # trousers, skirts
    dress: tuple[int, ...]        # full-body garments, covering both
    arms: tuple[int, ...]
    torso_skin: tuple[int, ...]
    legs: tuple[int, ...]
    feet: tuple[int, ...]
    identity: tuple[int, ...]     # never erased: face, hair, hat

    # Maps a garment word found in a filename onto the ids it should select.
    roles: dict[str, tuple[int, ...]]

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def all_garment(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.upper + self.lower + self.dress)))

    def garment_labels(self, garment_type: str) -> tuple[int, ...]:
        if garment_type == "upper":
            return tuple(sorted(set(self.upper + self.dress)))
        if garment_type == "lower":
            return self.lower
        if garment_type == "full":
            return self.all_garment
        raise ValueError(f"unknown garment_type {garment_type!r}")

    def erase_extra(self, garment_labels: tuple[int, ...]) -> tuple[int, ...]:
        """Skin whose visibility depends on the garment, so must be erased too.

        Swapping a top changes the sleeves, so the arms become unknown; swapping
        trousers changes the hemline, so the legs do. A dress does both. Getting
        this wrong leaves the old sleeves in the input and the model learns to
        copy them.
        """
        selected = set(garment_labels)
        extra: tuple[int, ...] = ()
        if selected & set(self.upper + self.dress):
            extra += self.arms + self.torso_skin
        if selected & set(self.lower + self.dress):
            extra += self.legs
        return tuple(sorted(set(extra)))

    def groups(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Coarse structural groups fed to the networks as one-hot channels.

        Feeding all 18-20 classes to a network trained on ~50 images wastes
        capacity on distinctions it can never learn (left vs right shoe). Seven
        groups carry the structure that matters.
        """
        return (
            ("background", (0,)),
            ("identity", self.identity),
            ("upper_garment", self.upper + self.dress),
            ("skin", self.arms + self.torso_skin),
            ("lower_garment", self.lower),
            ("legs", self.legs),
            ("feet", self.feet),
        )


# ---------------------------------------------------------------------------
# CIHP / LIP — 20 classes. The convention used by CIHP-PGN and by SCHP's
# `lip` and `pascal`-trained checkpoints.
# ---------------------------------------------------------------------------
CIHP = LabelScheme(
    name="cihp",
    labels=(
        "background", "hat", "hair", "glove", "sunglasses", "upper_clothes",
        "dress", "coat", "socks", "pants", "torso_skin", "scarf", "skirt",
        "face", "left_arm", "right_arm", "left_leg", "right_leg",
        "left_shoe", "right_shoe",
    ),
    upper=(5, 7, 11),
    lower=(9, 12),
    dress=(6,),
    arms=(14, 15),
    torso_skin=(10,),
    legs=(16, 17),
    feet=(8, 18, 19),
    identity=(1, 2, 4, 13),
    roles={
        "upper": (5,), "coat": (7,), "dress": (6,),
        "pants": (9,), "skirt": (12,),
    },
)

# ---------------------------------------------------------------------------
# ATR — 18 classes. What SCHP's `atr` checkpoint emits, and the most common
# alternative in the wild.
# ---------------------------------------------------------------------------
ATR = LabelScheme(
    name="atr",
    labels=(
        "background", "hat", "hair", "sunglasses", "upper_clothes", "skirt",
        "pants", "dress", "belt", "left_shoe", "right_shoe", "face",
        "left_leg", "right_leg", "left_arm", "right_arm", "bag", "scarf",
    ),
    upper=(4, 17),
    lower=(5, 6),
    dress=(7,),
    arms=(14, 15),
    torso_skin=(),
    legs=(12, 13),
    feet=(9, 10),
    identity=(1, 2, 3, 11),
    roles={
        "upper": (4,), "coat": (4,), "dress": (7,),
        "pants": (6,), "skirt": (5,),
    },
)

SCHEMES = {"cihp": CIHP, "lip": CIHP, "atr": ATR}

# Filename words that name a garment type, mapped onto a role. Used as a prior
# when picking which parse region a garment corresponds to.
GARMENT_WORDS = {
    "dress": "dress", "gown": "dress", "jumpsuit": "dress",
    "pants": "pants", "trouser": "pants", "trousers": "pants", "jeans": "pants",
    "shorts": "pants", "denim": "pants",
    "skirt": "skirt",
    "top": "upper", "shirt": "upper", "tshirt": "upper", "tee": "upper",
    "blouse": "upper", "jersey": "upper", "vest": "upper", "crop": "upper",
    "coat": "coat", "jacket": "coat", "hoodie": "coat", "blazer": "coat",
    "cardigan": "coat", "knit": "coat",
}


def get_scheme(name: str) -> LabelScheme:
    key = (name or "cihp").lower()
    if key not in SCHEMES:
        raise ValueError(
            f"unknown label scheme {name!r}; choose from {sorted(SCHEMES)}"
        )
    return SCHEMES[key]


def role_from_filename(stem: str) -> str | None:
    """Guess the garment role from a filename, e.g. `0007_garment_dress`."""
    lowered = stem.lower()
    for word, role in GARMENT_WORDS.items():
        if word in lowered:
            return role
    return None


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def mask_from_labels(parse: torch.Tensor, labels) -> torch.Tensor:
    """Binary mask selecting a set of class ids. parse: (1, H, W) or (H, W)."""
    if parse.dim() == 2:
        parse = parse.unsqueeze(0)
    mask = torch.zeros_like(parse, dtype=torch.float32)
    for label in labels:
        mask = mask + (parse == label).float()
    return mask.clamp(0.0, 1.0)


def parse_to_groups(parse: torch.Tensor, scheme: LabelScheme) -> torch.Tensor:
    """Collapse a label map into the scheme's coarse group encoding."""
    if parse.dim() == 2:
        parse = parse.unsqueeze(0)
    return torch.cat([mask_from_labels(parse, ids) for _, ids in scheme.groups()],
                     dim=0)


NUM_PARSE_GROUPS = 7

# Backwards-compatible aliases so existing imports keep working.
CIHP_LABELS = CIHP.labels
NUM_CIHP_CLASSES = 20
