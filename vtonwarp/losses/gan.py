"""
Hinge GAN objective.

The hinge loss is used rather than the original non-saturating BCE because it
stops rewarding the discriminator once a sample is confidently classified. On a
small dataset the discriminator wins quickly and unbounded losses translate that
win into enormous generator gradients; the hinge caps them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def r1_penalty(real_logits: torch.Tensor, real_images: torch.Tensor) -> torch.Tensor:
    """Gradient penalty on real samples.

    Keeps the discriminator's decision boundary flat near the data manifold. On
    a tiny dataset the manifold is a handful of points, and without R1 the
    discriminator builds arbitrarily sharp spikes around them.
    """
    grad = torch.autograd.grad(
        outputs=real_logits.sum(), inputs=real_images, create_graph=True
    )[0]
    return grad.pow(2).flatten(1).sum(dim=1).mean()
