"""
A conditional PatchGAN discriminator, plus differentiable augmentation.

The adversarial term is optional and off by default for the first part of
training — it is what finally sharpens fabric texture and the garment/skin
boundary, but on a small dataset an unconstrained discriminator memorises the
training images within a few hundred steps and then feeds the generator pure
noise. Two defences:

* **Spectral normalisation** bounds the discriminator's Lipschitz constant, so
  its gradients stay informative instead of exploding into the generator.
* **DiffAugment** (Zhao et al., 2020) applies the *same differentiable*
  augmentation to real and fake images before the discriminator sees them.
  Because the augmentation is differentiable, gradients still flow to the
  generator, but the discriminator can no longer memorise exact training pixels.
  This is the single technique that makes GAN training viable on ~100 images.

The discriminator is *conditional*: it sees the body condition alongside the
image, so it judges "is this a plausible person wearing this garment in this
pose" rather than just "is this a plausible image".
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int, base: int = 32, layers: int = 3):
        super().__init__()
        sequence = [
            spectral_norm(nn.Conv2d(in_channels, base, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base
        for i in range(1, layers):
            prev, channels = channels, min(base * (2 ** i), 256)
            sequence += [
                spectral_norm(nn.Conv2d(prev, channels, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        sequence += [
            spectral_norm(nn.Conv2d(channels, channels, 4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, 4, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*sequence)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image, condition], dim=1))


# ---------------------------------------------------------------------------
# DiffAugment
# ---------------------------------------------------------------------------

def diff_augment(x: torch.Tensor, policy: str = "color,translation,cutout") -> torch.Tensor:
    for name in policy.split(","):
        name = name.strip()
        if name:
            x = _POLICIES[name](x)
    return x


def _rand_brightness(x: torch.Tensor) -> torch.Tensor:
    factor = torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5
    return x + factor


def _rand_saturation(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2
    return (x - mean) * factor + mean


def _rand_contrast(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5
    return (x - mean) * factor + mean


def _rand_color(x: torch.Tensor) -> torch.Tensor:
    return _rand_contrast(_rand_saturation(_rand_brightness(x)))


def _rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    shift_h = int(x.size(2) * ratio + 0.5)
    shift_w = int(x.size(3) * ratio + 0.5)
    pad = F.pad(x, [shift_w, shift_w, shift_h, shift_h])
    top = random.randint(0, 2 * shift_h)
    left = random.randint(0, 2 * shift_w)
    return pad[:, :, top: top + x.size(2), left: left + x.size(3)]


def _rand_cutout(x: torch.Tensor, ratio: float = 0.3) -> torch.Tensor:
    h, w = x.size(2), x.size(3)
    cut_h, cut_w = int(h * ratio + 0.5), int(w * ratio + 0.5)
    top = random.randint(0, max(h - cut_h, 0))
    left = random.randint(0, max(w - cut_w, 0))
    mask = torch.ones_like(x)
    mask[:, :, top: top + cut_h, left: left + cut_w] = 0.0
    return x * mask


_POLICIES = {
    "color": _rand_color,
    "translation": _rand_translation,
    "cutout": _rand_cutout,
}
