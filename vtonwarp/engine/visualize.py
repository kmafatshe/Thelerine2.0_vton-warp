"""
Visual debugging.

Numbers lie about try-on quality — L1 will happily improve while the output gets
blurrier. The contact sheet written here is the real evaluation metric, and with
a small dataset you can afford to look at every validation sample every time.

Read a sheet left to right: if the warped garment column is wrong, the problem
is stage 1 and nothing in stage 2 will fix it.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """(C, H, W) in [-1, 1] or [0, 1] -> (H, W, 3) uint8."""
    tensor = tensor.detach().float().cpu()
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
        low, high = 0.0, 1.0
    else:
        low, high = -1.0, 1.0
    tensor = (tensor - low) / (high - low)
    return (tensor.clamp(0, 1) * 255).round().byte().permute(1, 2, 0)


def contact_sheet(columns: dict[str, torch.Tensor], path: str | Path,
                  max_rows: int = 8, caption: str | None = None) -> None:
    """Write a grid image: one row per sample, one column per named tensor.

    Args:
        columns: name -> (B, C, H, W) tensor. Single-channel entries are shown
            as greyscale.
        caption: stamped along the top. Sheets are written to a fixed path per
            step, so a previous run's output sits in the same folder until the
            new run reaches the same step — and an old sheet is otherwise
            indistinguishable from a current one. Recording the step and the
            settings that produced it makes that obvious at a glance.
    """
    from PIL import Image, ImageDraw

    names = list(columns)
    rows = min(max_rows, next(iter(columns.values())).shape[0])
    height, width = next(iter(columns.values())).shape[-2:]
    caption_height = 16 if caption else 0
    label_height = 16 + caption_height

    sheet = Image.new("RGB", (width * len(names), height * rows + label_height),
                      (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    if caption:
        draw.text((4, 3), caption, fill=(255, 210, 90))

    for col, name in enumerate(names):
        draw.text((col * width + 4, 3 + caption_height), name, fill=(230, 230, 230))
        tensor = columns[name]
        if tensor.shape[-2:] != (height, width):
            tensor = F.interpolate(tensor, size=(height, width), mode="bilinear",
                                   align_corners=False)
        for row in range(rows):
            patch = Image.fromarray(to_uint8(tensor[row]).numpy())
            sheet.paste(patch, (col * width, label_height + row * height))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 2) displacement field -> (B, 3, H, W) visualisation.

    Red encodes horizontal displacement, green vertical, so a healthy warp looks
    like a smooth colour gradient and a folded one shows abrupt patches.
    """
    flow = flow.permute(0, 3, 1, 2)
    scaled = (flow / (flow.abs().amax() + 1e-6)).clamp(-1, 1)
    blue = torch.zeros_like(scaled[:, :1])
    return torch.cat([scaled, blue], dim=1)
