"""
Stage 2 — the composer.

The warper has already put real garment pixels in roughly the right place. The
composer's job is *not* to draw a person. It is to answer, for every pixel:

    "should this pixel come from the warped garment, or does it need to be
     synthesised (skin newly revealed by a shorter sleeve, a shadow under a
     collar, the boundary between garment and neck)?"

It therefore emits two things:

    render : (B, 3, H, W)  synthesised content, used only where copying fails
    alpha  : (B, 1, H, W)  per-pixel preference for the warped garment

    output = alpha * warped_garment + (1 - alpha) * render

This factorisation is what makes the architecture work on ~100 images. A plain
image-to-image generator must produce every output pixel from its weights, so
garment texture is stored *in the network* and a tiny dataset simply cannot
supply enough examples to learn it. Here the texture is an input. The network
only has to learn a soft mask and some boundary shading — a far lower-entropy
target that a small model can fit from few examples.

The alpha map is explicitly regularised towards the warped mask during training
(see losses). Left unregularised, the network discovers it can set alpha ~ 0 and
blur everything through `render`, which minimises L1 fastest and produces the
washed-out result that plagues naive try-on models.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock, DownBlock, UpBlock


class Composer(nn.Module):
    def __init__(
        self,
        condition_channels: int = 14,
        base: int = 32,
        depth: int = 4,
        max_channels: int = 256,
    ):
        super().__init__()
        # Inputs: body condition + warped garment RGB + warped garment mask.
        in_channels = condition_channels + 3 + 1

        channels = [min(base * (2 ** i), max_channels) for i in range(depth + 1)]

        self.stem = ConvBlock(in_channels, channels[0])
        self.downs = nn.ModuleList(
            [DownBlock(channels[i], channels[i + 1]) for i in range(depth)]
        )
        self.middle = nn.Sequential(
            ConvBlock(channels[depth], channels[depth]),
            ConvBlock(channels[depth], channels[depth]),
        )
        self.ups = nn.ModuleList(
            [
                UpBlock(channels[i + 1], channels[i + 1], channels[i])
                for i in reversed(range(depth))
            ]
        )

        # Two heads on a shared trunk: the decision of *where* to copy and the
        # content to fall back on are the same reasoning problem.
        self.render_head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], 3, 3, padding=1),
            nn.Tanh(),
        )
        self.alpha_head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, condition: torch.Tensor, warped: torch.Tensor,
                warped_mask: torch.Tensor) -> dict:
        x = torch.cat([condition, warped, warped_mask], dim=1)
        x = self.stem(x)

        skips = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        x = self.middle(x)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        render = self.render_head(x)
        alpha = self.alpha_head(x)

        # Copying can only happen where the warped garment actually exists;
        # multiplying by the warped mask stops alpha claiming empty space.
        alpha = alpha * warped_mask
        output = alpha * warped + (1.0 - alpha) * render

        return {"output": output, "render": render, "alpha": alpha}
