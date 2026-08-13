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


def read_rgb(path: Path) -> torch.Tensor:
    """RGB image at its native resolution -> (3, H, W) in [-1, 1].

    Cropping must happen at native resolution. Resizing a 4000px photo down to
    256px and *then* cropping to the person throws away the detail before it can
    be used: the person ends up reconstructed from the handful of pixels that
    survived, not from the pixels the camera actually captured.
    """
    image = Image.open(path).convert("RGB")
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    return tensor * 2.0 - 1.0


def resize_rgb(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(image[None], size=(height, width), mode="bilinear",
                         align_corners=False, antialias=True)[0]


def resize_labels(labels: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Nearest-neighbour: label ids must never be interpolated."""
    out = F.interpolate(labels[None].float(), size=(height, width), mode="nearest")
    return out[0].long()


def bbox_from_mask(mask: torch.Tensor, margin: float = 0.0,
                   aspect: float | None = None) -> tuple[int, int, int, int]:
    """Bounding box of a mask as (top, left, height, width).

    Args:
        mask: (1, H, W) in [0, 1].
        margin: fraction of the box size to add on each side, so a crop keeps
            some context rather than cutting flush against the shoulders.
        aspect: width / height to force. The box is *grown* on the short axis
            rather than shrunk, so nothing inside it is ever cut off, and the
            later resize introduces no distortion.
    """
    _, height, width = mask.shape
    rows = torch.nonzero(mask[0].amax(dim=1) > 0.5).flatten()
    cols = torch.nonzero(mask[0].amax(dim=0) > 0.5).flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return 0, 0, height, width

    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    box_h, box_w = bottom - top, right - left

    pad_y = int(box_h * margin)
    pad_x = int(box_w * margin)
    top, bottom = top - pad_y, bottom + pad_y
    left, right = left - pad_x, right + pad_x

    if aspect is not None:
        box_h, box_w = bottom - top, right - left
        if box_w / max(box_h, 1) < aspect:
            target = int(box_h * aspect)
            centre = (left + right) // 2
            left, right = centre - target // 2, centre - target // 2 + target
        else:
            target = int(box_w / aspect)
            centre = (top + bottom) // 2
            top, bottom = centre - target // 2, centre - target // 2 + target

    # Shift back inside the image rather than clipping, so the aspect ratio the
    # caller asked for survives; only clamp if the box is larger than the image.
    box_h, box_w = bottom - top, right - left
    top = max(0, min(top, height - box_h))
    left = max(0, min(left, width - box_w))
    return top, left, min(box_h, height - top), min(box_w, width - left)


def crop(tensor: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    top, left, height, width = box
    return tensor[:, top:top + height, left:left + width]


def load_image(path: Path, height: int, width: int) -> torch.Tensor:
    """RGB image -> (3, H, W) float tensor scaled to [-1, 1]."""
    return resize_rgb(read_rgb(path), height, width)


def read_labels(path: Path, num_classes: int = 20) -> torch.Tensor:
    """Any parse encoding -> (1, H, W) int64 class ids at native resolution."""
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
    if labels.max() >= num_classes:
        scale = 255.0 / (num_classes - 1)
        labels = torch.round(labels.float() / scale).long()

    return labels.clamp(0, num_classes - 1)[None]


def load_label_map(path: Path, height: int, width: int,
                   num_classes: int = 20) -> torch.Tensor:
    """Any parse encoding -> (1, H, W) int64 tensor, resized to the frame."""
    return resize_labels(read_labels(path, num_classes), height, width)


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


def garment_mask_from_rgb(garment: torch.Tensor, tolerance: float = 0.12,
                          border: float = 0.04) -> torch.Tensor:
    """Segment a product shot from its background, whatever colour that is.

    Used when the dataset provides no explicit mask for the flat garment. An
    earlier version assumed a *white* background; datasets built by cutting
    garments out onto black then produced an inverted mask, and the warper spent
    its time moving black background pixels around. So the background colour is
    now measured rather than assumed.

    The border ring of the image is background by construction — a product shot
    is never cropped tight to the garment — so its median colour is the
    background colour. Pixels within `tolerance` of it are background.

    Args:
        garment: (3, H, W) in [-1, 1].
        tolerance: RGB distance (in 0-1 units) counted as "same as background".
        border: fraction of the image edge sampled to estimate that colour.
    """
    rgb = (garment + 1.0) / 2.0
    _, height, width = rgb.shape

    ring_h = max(1, int(height * border))
    ring_w = max(1, int(width * border))
    edges = torch.cat(
        [
            rgb[:, :ring_h, :].reshape(3, -1),
            rgb[:, -ring_h:, :].reshape(3, -1),
            rgb[:, :, :ring_w].reshape(3, -1),
            rgb[:, :, -ring_w:].reshape(3, -1),
        ],
        dim=1,
    )
    background_colour = edges.median(dim=1).values.view(3, 1, 1)

    distance = (rgb - background_colour).pow(2).sum(dim=0, keepdim=True).sqrt()
    mask = (distance > tolerance).float()

    # A busy background (a room, a floor) will not be uniform, so the estimate
    # can fail. Falling back to "keep everything" is only right when detection
    # found essentially nothing; an earlier threshold of 1% also swallowed
    # genuinely small garments — a thumbnail on a large canvas came back as a
    # full-frame mask, and the warper was handed the whole black background as
    # if it were fabric.
    if mask.mean() < 0.001 or mask.mean() > 0.97:
        return torch.ones_like(mask)

    return _largest_component(mask)


def _largest_component(mask: torch.Tensor) -> torch.Tensor:
    """Keep only the biggest blob, dropping speckle and stray objects.

    Product shots often include a hanger, a label, or JPEG noise that survives
    the colour threshold. Those fragments get warped along with the garment and
    smear across the body.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return mask

    array = mask[0].numpy()
    labelled, count = ndimage.label(array > 0.5)
    if count <= 1:
        return mask

    sizes = ndimage.sum(array > 0.5, labelled, range(1, count + 1))
    keep = int(sizes.argmax()) + 1
    # Fill interior holes so a light-coloured print inside a dark garment is
    # not punched out of the mask.
    filled = ndimage.binary_fill_holes(labelled == keep)
    return torch.from_numpy(filled.astype("float32"))[None]


def canonicalise_garment(garment: torch.Tensor, mask: torch.Tensor,
                         height: int | None = None, width: int | None = None,
                         fill: float = 0.8, pad_value: float = 0.0):
    """Crop a garment to its mask and place it in a frame at a consistent size.

    Product shots vary enormously in framing — some fill the frame, some are a
    thumbnail on a large canvas. That variation lands entirely on the warper,
    which then has to learn a large scale change *and* the body-shape
    deformation, from a handful of examples.

    Given `height`/`width` this crops at the input's own resolution and resizes
    once into the target frame, so a garment occupying 3% of a large photo keeps
    the detail that 3% actually contains. Without them it operates in place, at
    whatever resolution it was handed.

    Args:
        garment: (3, H, W) in [-1, 1].
        mask: (1, H, W) in [0, 1].
        fill: fraction of the frame the garment's longest side should occupy.
    """
    if height is None or width is None:
        height, width = garment.shape[-2:]

    box = bbox_from_mask(mask)
    if box[2] <= 0 or box[3] <= 0:
        return resize_rgb(garment, height, width), resize_labels(
            mask.long(), height, width).float()

    cropped = crop(garment, box)
    cropped_mask = crop(mask, box)

    crop_h, crop_w = cropped.shape[-2:]
    scale = min(height * fill / crop_h, width * fill / crop_w)
    new_h, new_w = max(1, round(crop_h * scale)), max(1, round(crop_w * scale))

    resized = resize_rgb(cropped, new_h, new_w)
    resized_mask = F.interpolate(cropped_mask[None], size=(new_h, new_w),
                                 mode="nearest")[0]

    out = torch.full((3, height, width), pad_value)
    out_mask = torch.zeros((1, height, width))
    oy, ox = (height - new_h) // 2, (width - new_w) // 2
    out[:, oy:oy + new_h, ox:ox + new_w] = resized
    out_mask[:, oy:oy + new_h, ox:ox + new_w] = resized_mask

    # Blank the background so no residual colour leaks into the warp.
    return out * out_mask + pad_value * (1 - out_mask), out_mask
