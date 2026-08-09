"""
Augmentation designed for a tiny paired dataset.

Three rules govern everything here:

1.  **Geometry applied to the person must be applied to the parse map too**, or
    the label map stops describing the image and every downstream mask is wrong.
    Images resample bilinearly, label maps must resample nearest.

2.  **Photometry must be shared between the person and the garment.** If we
    brightened only the garment, the ground-truth person would still show the
    original colour and we would be training the warper to reproduce a colour
    shift it cannot see. Shared jitter instead simulates a change of lighting,
    which is a genuine invariance we want.

3.  **The garment gets its own extra geometric jitter.** The flat garment photo
    and the worn garment are independent observations, so perturbing one of them
    is free supervision: it stops the warper from settling on a near-identity
    transform that happens to work for the training set's fixed framing. This is
    the single highest-value augmentation for the warping stage.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


class PairedAugment:
    def __init__(
        self,
        hflip: float = 0.5,
        person_shift: float = 0.05,
        person_scale: float = 0.08,
        person_rotate: float = 5.0,
        garment_shift: float = 0.10,
        garment_scale: float = 0.15,
        garment_rotate: float = 12.0,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.15,
        enabled: bool = True,
    ):
        self.hflip = hflip
        self.person_affine = (person_shift, person_scale, person_rotate)
        self.garment_affine = (garment_shift, garment_scale, garment_rotate)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.enabled = enabled

    # -- public ------------------------------------------------------------

    def __call__(
        self, person: torch.Tensor, parse: torch.Tensor, garment: torch.Tensor,
        garment_mask: torch.Tensor,
    ):
        """Returns augmented (person, parse, garment, garment_mask)."""
        if not self.enabled:
            return person, parse, garment, garment_mask

        if random.random() < self.hflip:
            person = torch.flip(person, dims=[-1])
            parse = torch.flip(parse, dims=[-1])
            garment = torch.flip(garment, dims=[-1])
            garment_mask = torch.flip(garment_mask, dims=[-1])

        theta = self._random_affine(*self.person_affine)
        person = self._warp(person, theta, mode="bilinear")
        parse = self._warp(parse.float(), theta, mode="nearest").long()

        theta_g = self._random_affine(*self.garment_affine)
        garment = self._warp(garment, theta_g, mode="bilinear", fill=1.0)
        garment_mask = self._warp(garment_mask, theta_g, mode="nearest")

        person, garment = self._photometric(person, garment)
        return person, parse, garment, garment_mask

    # -- internals ---------------------------------------------------------

    def _random_affine(self, shift: float, scale: float, rotate: float) -> torch.Tensor:
        """Build a 2x3 affine matrix in normalised [-1, 1] coordinates."""
        angle = math.radians(random.uniform(-rotate, rotate))
        zoom = 1.0 + random.uniform(-scale, scale)
        tx = random.uniform(-shift, shift)
        ty = random.uniform(-shift, shift)

        cos, sin = math.cos(angle) / zoom, math.sin(angle) / zoom
        return torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32)

    def _warp(self, tensor: torch.Tensor, theta: torch.Tensor, mode: str,
              fill: float = 0.0) -> torch.Tensor:
        """Apply an affine matrix to a (C, H, W) tensor.

        `fill` is added and subtracted around the sample so that the padding
        colour is meaningful: garment product shots pad with white (1.0 in
        [-1, 1] space), masks and people pad with zero.
        """
        batched = tensor[None]
        grid = F.affine_grid(theta[None], list(batched.shape), align_corners=False)
        shifted = batched - fill
        out = F.grid_sample(shifted, grid, mode=mode, padding_mode="zeros",
                            align_corners=False)
        return (out + fill)[0]

    def _photometric(self, person: torch.Tensor, garment: torch.Tensor):
        """Identical brightness/contrast/saturation jitter for both images."""
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        s = 1.0 + random.uniform(-self.saturation, self.saturation)

        def apply(x: torch.Tensor) -> torch.Tensor:
            rgb = (x + 1.0) / 2.0
            rgb = rgb * b
            rgb = (rgb - rgb.mean()) * c + rgb.mean()
            grey = rgb.mean(dim=0, keepdim=True)
            rgb = grey + (rgb - grey) * s
            return (rgb.clamp(0.0, 1.0) * 2.0) - 1.0

        return apply(person), apply(garment)
