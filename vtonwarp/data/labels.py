"""
CIHP / LIP human-parsing label semantics.

The `cihp/` folder in the dataset stores a per-pixel *label map*: an H x W array
where every value is an integer class id from the 20-class CIHP taxonomy below.
This file is the single place that knows what those integers mean, so that the
rest of the codebase never hardcodes magic numbers like `5`.

Everything downstream (which pixels to erase, which to preserve, what the
garment target region is) is derived from these groupings.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# The 20 CIHP classes, in label order.
# ---------------------------------------------------------------------------
CIHP_LABELS = (
    "background",   # 0
    "hat",          # 1
    "hair",         # 2
    "glove",        # 3
    "sunglasses",   # 4
    "upper_clothes",  # 5
    "dress",        # 6
    "coat",         # 7
    "socks",        # 8
    "pants",        # 9
    "torso_skin",   # 10
    "scarf",        # 11
    "skirt",        # 12
    "face",         # 13
    "left_arm",     # 14
    "right_arm",    # 15
    "left_leg",     # 16
    "right_leg",    # 17
    "left_shoe",    # 18
    "right_shoe",   # 19
)

NUM_CIHP_CLASSES = len(CIHP_LABELS)

# ---------------------------------------------------------------------------
# Semantic groupings.
# ---------------------------------------------------------------------------

# Pixels belonging to an upper-body garment. This is the region the warped
# garment must land on, and therefore the primary supervision signal for the
# warping stage.
UPPER_GARMENT = (5, 6, 7, 11)

# Lower-body garments, used when `garment_type: lower` or `full`.
LOWER_GARMENT = (9, 12)

# Bare skin that gets revealed / covered depending on sleeve length. These
# pixels must be erased from the input, otherwise the network can cheat by
# copying the original sleeves straight through.
ARMS = (14, 15)
TORSO_SKIN = (10,)

# Identity: never erased, always copied verbatim from the source person. This
# is what stops a small-data model from smearing the face.
IDENTITY = (1, 2, 4, 13)

LEGS = (16, 17)
FEET = (8, 18, 19)

GARMENT_SETS = {
    "upper": UPPER_GARMENT,
    "lower": LOWER_GARMENT,
    "full": UPPER_GARMENT + LOWER_GARMENT,
}

# Which non-garment labels also need erasing, per garment type. Changing an
# upper garment changes the sleeves, so arms and torso skin become unknown.
ERASE_EXTRA = {
    "upper": ARMS + TORSO_SKIN,
    "lower": LEGS,
    "full": ARMS + TORSO_SKIN + LEGS,
}

# ---------------------------------------------------------------------------
# Reduced parse used as a network input.
#
# Feeding all 20 one-hot channels to a network trained on ~100 images wastes
# capacity on distinctions it can never learn (left vs right shoe). We collapse
# them into 7 coarse structural groups instead.
# ---------------------------------------------------------------------------
PARSE_GROUPS = (
    ("background", (0,)),
    ("identity", IDENTITY),
    ("upper_garment", UPPER_GARMENT),
    ("skin", ARMS + TORSO_SKIN + (3,)),
    ("lower_garment", LOWER_GARMENT),
    ("legs", LEGS),
    ("feet", FEET),
)

NUM_PARSE_GROUPS = len(PARSE_GROUPS)


def mask_from_labels(parse: torch.Tensor, labels) -> torch.Tensor:
    """Binary mask selecting a set of CIHP class ids.

    Args:
        parse: (1, H, W) or (H, W) integer label map.
        labels: iterable of class ids to select.

    Returns:
        (1, H, W) float tensor in {0., 1.}.
    """
    if parse.dim() == 2:
        parse = parse.unsqueeze(0)

    mask = torch.zeros_like(parse, dtype=torch.float32)
    for label in labels:
        mask = mask + (parse == label).float()
    return mask.clamp(0.0, 1.0)


def parse_to_groups(parse: torch.Tensor) -> torch.Tensor:
    """Collapse a 20-class label map into a 7-channel one-hot group encoding.

    Args:
        parse: (1, H, W) or (H, W) integer label map.

    Returns:
        (NUM_PARSE_GROUPS, H, W) float tensor.
    """
    if parse.dim() == 2:
        parse = parse.unsqueeze(0)

    channels = [mask_from_labels(parse, ids) for _, ids in PARSE_GROUPS]
    return torch.cat(channels, dim=0)
