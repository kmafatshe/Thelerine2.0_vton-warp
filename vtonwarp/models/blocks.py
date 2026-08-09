"""
Shared building blocks.

Two deliberate choices, both driven by the dataset size:

* **GroupNorm, not BatchNorm.** Small datasets force small batches (2-4 on CPU).
  BatchNorm's statistics are meaningless at that batch size and its train/eval
  mismatch is the classic cause of "looks fine while training, garbage at
  inference". GroupNorm is batch-size independent.

* **Narrow, deep-ish networks.** Capacity is the enemy here. Every extra channel
  is another few thousand parameters competing for the same hundred images.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU, optionally strided."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2),
            norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Two convs at the same resolution, then a stride-2 downsample."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(ConvBlock(in_ch, out_ch), ConvBlock(out_ch, out_ch))
        self.down = ConvBlock(out_ch, out_ch, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.body(x)
        return self.down(skip), skip


class UpBlock(nn.Module):
    """Bilinear upsample + concat skip + two convs.

    Bilinear upsampling rather than transposed convolution: with few images the
    checkerboard artefacts of ConvTranspose2d never get trained away.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.body = nn.Sequential(
            ConvBlock(in_ch + skip_ch, out_ch), ConvBlock(out_ch, out_ch)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against odd spatial sizes losing a pixel on the way down.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.body(torch.cat([x, skip], dim=1))


def zero_init(module: nn.Module) -> nn.Module:
    """Zero a layer's weights and bias.

    Used on the final layer of both flow predictors so the network *starts* as
    an identity warp. Starting from identity means the first gradients describe
    a small correction to something already sensible, instead of unwarping a
    random garbling. On a small dataset this is often the difference between
    converging and not.
    """
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module
