"""
Stage 1 — the garment warper.

This is the part that replaces "generate a dressed person" with "move the real
garment pixels to where they belong". It answers a purely geometric question:
given a flat garment photo and a description of a body, what deformation maps
one onto the other?

Pipeline:

    garment (RGB + mask) ──> encoder ──┐
                                       ├──> correlation ──> regressor ──> TPS
    body condition (14ch) ──> encoder ─┘                                   │
                                                                           v
                                              coarse warp ──> refiner ──> residual flow
                                                                           │
                                                                           v
                                                                     final warped garment

The two-step coarse-then-residual design is the key small-data decision. The TPS
stage has 50 parameters and captures the global "stretch this rectangle over
that torso" transform. The residual stage adds a *bounded* dense field that can
express folds, shoulder seams and sleeve creases which no 5x5 lattice can. If
you predicted a dense field directly it would overfit; if you stopped at TPS the
result would look like a decal.

A correlation layer is used rather than plain concatenation because matching is
fundamentally a comparison operation: correlation gives the regressor an
explicit "how well does garment location i match body location j" score map,
which is a far easier signal to learn from than two stacks of features it must
first learn to compare.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, norm, zero_init
from .tps import TPSGridGen


class FeatureExtractor(nn.Module):
    """Downsample by 16 into a compact descriptor map."""

    def __init__(self, in_channels: int, base: int = 32, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, base, stride=2),      # /2
            ConvBlock(base, base),
            ConvBlock(base, base * 2, stride=2),         # /4
            ConvBlock(base * 2, base * 2),
            ConvBlock(base * 2, base * 4, stride=2),     # /8
            ConvBlock(base * 4, base * 4),
            ConvBlock(base * 4, out_channels, stride=2),  # /16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def l2_normalise(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)


def correlate(garment_feat: torch.Tensor, body_feat: torch.Tensor) -> torch.Tensor:
    """Dense cosine similarity between every garment and body feature location.

    Returns (B, Hg*Wg, Hb, Wb): channel k of the output is a heat map over the
    body of how strongly it matches garment location k.
    """
    garment_feat = l2_normalise(garment_feat)
    body_feat = l2_normalise(body_feat)

    b, c, hg, wg = garment_feat.shape
    _, _, hb, wb = body_feat.shape

    g_flat = garment_feat.view(b, c, hg * wg)                 # (B, C, Ng)
    p_flat = body_feat.view(b, c, hb * wb).transpose(1, 2)    # (B, Nb, C)

    corr = torch.bmm(p_flat, g_flat)                          # (B, Nb, Ng)
    corr = F.relu(corr)
    corr = corr / (corr.pow(2).sum(dim=2, keepdim=True).sqrt() + 1e-6)
    return corr.transpose(1, 2).reshape(b, hg * wg, hb, wb)


class TPSRegressor(nn.Module):
    """Correlation volume -> control-point offsets."""

    def __init__(self, in_channels: int, num_points: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(in_channels, hidden, stride=2),
            ConvBlock(hidden, hidden // 2, stride=2),
            nn.AdaptiveAvgPool2d((4, 3)),
        )
        self.head = nn.Linear((hidden // 2) * 12, num_points * 2)
        # Identity warp at initialisation.
        zero_init(self.head)

    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        x = self.body(corr).flatten(1)
        return self.head(x)


class ResidualFlowNet(nn.Module):
    """Predicts a small bounded dense correction on top of the TPS grid."""

    def __init__(self, in_channels: int, base: int = 32, max_displacement: float = 0.12):
        super().__init__()
        self.max_displacement = max_displacement
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base, stride=2),   # /2
            ConvBlock(base, base * 2, stride=2),      # /4
            ConvBlock(base * 2, base * 2),
            ConvBlock(base * 2, base * 2),
        )
        self.head = zero_init(nn.Conv2d(base * 2, 2, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Returns (B, H, W, 2) displacement, bounded to +-max_displacement."""
        features = self.encoder(x)
        flow = torch.tanh(self.head(features)) * self.max_displacement
        flow = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
        return flow.permute(0, 2, 3, 1)


class GarmentWarper(nn.Module):
    def __init__(
        self,
        height: int = 256,
        width: int = 192,
        condition_channels: int = 14,
        base: int = 32,
        feature_channels: int = 128,
        grid_size: int = 5,
        max_displacement: float = 0.12,
        use_residual_flow: bool = True,
    ):
        super().__init__()
        self.height, self.width = height, width
        self.use_residual_flow = use_residual_flow

        self.garment_encoder = FeatureExtractor(4, base, feature_channels)
        self.body_encoder = FeatureExtractor(condition_channels, base, feature_channels)

        feat_h, feat_w = height // 16, width // 16
        self.tps = TPSGridGen(height, width, grid_size)
        self.regressor = TPSRegressor(feat_h * feat_w, self.tps.num_points)

        # Refiner sees the coarse result plus the body description, so it can
        # reason about *where the coarse warp went wrong*.
        self.refiner = ResidualFlowNet(
            4 + condition_channels, base, max_displacement
        ) if use_residual_flow else None

    def forward(self, garment: torch.Tensor, garment_mask: torch.Tensor,
                condition: torch.Tensor) -> dict:
        """
        Args:
            garment: (B, 3, H, W) flat garment in [-1, 1].
            garment_mask: (B, 1, H, W) mask of the flat garment.
            condition: (B, C, H, W) clothing-agnostic body description.
        """
        garment_in = torch.cat([garment, garment_mask], dim=1)

        g_feat = self.garment_encoder(garment_in)
        b_feat = self.body_encoder(condition)
        corr = correlate(g_feat, b_feat)

        offsets = self.regressor(corr)
        coarse_grid = self.tps(offsets)

        coarse = self._sample(garment, coarse_grid)
        coarse_mask = self._sample(garment_mask, coarse_grid, mode="nearest")

        if self.refiner is None:
            grid, residual = coarse_grid, None
        else:
            refine_in = torch.cat([coarse, coarse_mask, condition], dim=1)
            residual = self.refiner(refine_in, (self.height, self.width))
            grid = coarse_grid + residual

        warped = self._sample(garment, grid)
        warped_mask = self._sample(garment_mask, grid)

        return {
            "warped": warped,
            "warped_mask": warped_mask,
            # Background of the product shot removed, so losses and the
            # composition stage never see warped white space as "garment".
            "warped_masked": warped * warped_mask,
            "coarse": coarse,
            "coarse_mask": coarse_mask,
            "grid": grid,
            "offsets": offsets,
            "residual": residual,
        }

    @staticmethod
    def _sample(x: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        # padding_mode="border" would smear the garment's edge pixels across the
        # background; "zeros" leaves genuinely empty space empty, which the
        # composition stage can then detect via the warped mask.
        return F.grid_sample(x, grid, mode=mode, padding_mode="zeros",
                             align_corners=False)
