"""
Tolerant loaders for the heterogeneous files a hand-built VTON dataset contains.

Conditional maps show up in the wild as any of:
  * `.npy` of shape (H, W) with integer class ids
  * `.npy` of shape (H, W, C) or (C, H, W) one-hot / probability maps
  * indexed PNGs where the palette index *is* the class id
  * plain greyscale PNGs where classes were scaled to 0-255

`load_label_map` normalises all of those to a single (1, H, W) int64 tensor, and
`load_mask` normalises anything binary-ish to a (1, H, W) float tensor in [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .labels import NUM_CIHP_CLASSES


def _read_array(path: Path) -> np.ndarray:
    """Read a .npy/.npz or an image file into a numpy array."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        bundle = np.load(path)
        return bundle[list(bundle.keys())[0]]
    # Keep the raw mode: 'P' preserves palette indices, which are class ids.
    image = Image.open(path)
    if image.mode == "P":
        return np.array(image)
    if image.mode in ("RGB", "RGBA"):
        return np.array(image.convert("RGB"))
    return np.array(image.convert("L"))


def load_image(path: Path, height: int, width: int) -> torch.Tensor:
    """RGB image -> (3, H, W) float tensor scaled to [-1, 1]."""
    image = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    return tensor * 2.0 - 1.0


def load_label_map(path: Path, height: int, width: int) -> torch.Tensor:
    """Any parse encoding -> (1, H, W) int64 tensor of CIHP class ids."""
    array = _read_array(Path(path))

    if array.ndim == 3:
        # One-hot / probability map. Find the channel axis and argmax it.
        channel_axis = int(np.argmin(array.shape))
        if array.shape[channel_axis] == 3 and array.dtype == np.uint8:
            # An RGB visualisation of a parse map. Collapsing colour to a class
            # id is lossy and dataset specific, so we refuse rather than guess.
            raise ValueError(
                f"{path} looks like an RGB colourised parse map. Export the raw "
                "label indices (.npy or indexed PNG) instead."
            )
        array = np.argmax(array, axis=channel_axis)

    labels = torch.from_numpy(np.ascontiguousarray(array)).long()

    # Greyscale PNGs are often written as class_id * (255 // num_classes).
    if labels.max() >= NUM_CIHP_CLASSES:
        scale = 255.0 / (NUM_CIHP_CLASSES - 1)
        labels = torch.round(labels.float() / scale).long()

    labels = labels.clamp(0, NUM_CIHP_CLASSES - 1)

    # Nearest-neighbour resize: label ids must never be interpolated.
    labels = F.interpolate(
        labels[None, None].float(), size=(height, width), mode="nearest"
    )
    return labels[0].long()


def load_mask(path: Path, height: int, width: int) -> torch.Tensor:
    """Any binary-ish map -> (1, H, W) float tensor in [0, 1]."""
    array = _read_array(Path(path)).astype(np.float32)

    if array.ndim == 3:
        channel_axis = int(np.argmin(array.shape))
        array = array.mean(axis=channel_axis) if array.shape[channel_axis] <= 4 \
            else np.argmax(array, axis=channel_axis).astype(np.float32)

    tensor = torch.from_numpy(np.ascontiguousarray(array))[None, None]
    if tensor.max() > 1.0:
        tensor = tensor / tensor.max()

    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear",
                           align_corners=False)
    return tensor[0].clamp(0.0, 1.0)


def garment_mask_from_rgb(garment: torch.Tensor, threshold: float = 0.92) -> torch.Tensor:
    """Fallback garment mask for product shots on a white background.

    Used only when the dataset provides no explicit garment mask. Pixels that
    are near-white *and* connected to the border are treated as background; we
    approximate that cheaply with a brightness threshold, which is reliable for
    the flat-lay catalogue images these datasets are built from.

    Args:
        garment: (3, H, W) in [-1, 1].
    """
    rgb = (garment + 1.0) / 2.0
    brightness = rgb.mean(dim=0, keepdim=True)
    saturation = rgb.max(dim=0, keepdim=True).values - rgb.min(dim=0, keepdim=True).values
    background = (brightness > threshold) & (saturation < 0.08)
    return (~background).float()
