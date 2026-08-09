"""
VGG19 perceptual and style losses.

This is where the missing training data comes from. An ImageNet-pretrained VGG
already encodes edges, textures, fabric weave and shading — knowledge distilled
from a million images. Comparing outputs in VGG feature space instead of pixel
space means our ~100 samples only need to teach *geometry and composition*; the
notion of "what a plausible textured surface looks like" is borrowed.

Pixel L1 alone has a well-known failure mode: when the model is uncertain about
where an edge is, the L1-optimal answer is to blur it. Perceptual loss penalises
that blur because blurred features are far from sharp features even when pixel
means match. Style (Gram-matrix) loss goes further and compares *feature
correlations*, which is what makes a knit look like a knit — it is deliberately
position-insensitive, so it survives small warping errors that would otherwise
be punished.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 in torchvision's vgg19.features
SLICE_POINTS = (2, 7, 12, 21, 30)
# Deeper layers are weighted down: they carry semantics we already get for free
# from the copied pixels, while the shallow layers carry the texture detail we
# actually need.
LAYER_WEIGHTS = (1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGFeatures(nn.Module):
    """Frozen VGG19 truncated into five slices."""

    def __init__(self, weights_path: str | None = None):
        super().__init__()
        from torchvision.models import vgg19

        try:
            backbone = vgg19(weights="IMAGENET1K_V1").features
        except Exception as error:  # offline, or no cached weights
            if weights_path is None:
                raise RuntimeError(
                    "Could not obtain pretrained VGG19 weights. Either connect to "
                    "the internet once so torchvision can cache them, or set "
                    "loss.perceptual_weight: 0.0 in the config to train without "
                    "perceptual loss (quality will suffer noticeably)."
                ) from error
            backbone = vgg19().features
            backbone.load_state_dict(torch.load(weights_path, map_location="cpu"))

        self.slices = nn.ModuleList()
        start = 0
        for end in SLICE_POINTS:
            self.slices.append(nn.Sequential(*[backbone[i] for i in range(start, end)]))
            start = end

        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # Never leave eval mode: the frozen backbone must stay deterministic.
        return super().train(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = ((x + 1.0) / 2.0 - self.mean) / self.std
        features = []
        for slice_ in self.slices:
            x = slice_(x)
            features.append(x)
        return features


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    b, c, h, w = features.shape
    flat = features.view(b, c, h * w)
    return torch.bmm(flat, flat.transpose(1, 2)) / (c * h * w)


class PerceptualLoss(nn.Module):
    def __init__(self, weights_path: str | None = None, style_weight: float = 0.0):
        super().__init__()
        self.vgg = VGGFeatures(weights_path)
        self.style_weight = style_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            prediction, target: (B, 3, H, W) in [-1, 1].
            mask: optional (B, 1, H, W) restricting the loss to a region. The
                mask is resized to each feature map so that, e.g., the warping
                stage is only judged on the garment area and not on the
                background it was never asked to produce.
        """
        pred_features = self.vgg(prediction)
        with torch.no_grad():
            target_features = self.vgg(target)

        loss = prediction.new_zeros(())
        for weight, pred, ref in zip(LAYER_WEIGHTS, pred_features, target_features):
            if mask is None:
                loss = loss + weight * F.l1_loss(pred, ref)
            else:
                scaled = F.interpolate(mask, size=pred.shape[-2:], mode="bilinear",
                                       align_corners=False)
                diff = (pred - ref).abs() * scaled
                loss = loss + weight * diff.sum() / (scaled.sum() * pred.shape[1] + 1e-6)

            if self.style_weight > 0:
                loss = loss + self.style_weight * weight * F.l1_loss(
                    gram_matrix(pred), gram_matrix(ref)
                )

        return loss
