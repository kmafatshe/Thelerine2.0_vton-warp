"""
Regularisers for the warping field and the composition mask.

Warping is an ill-posed problem: many deformations map a garment onto a body
with the same pixel loss, and most of them are physically absurd (locally
folded, torn, or self-intersecting). With thousands of training pairs a network
learns to prefer the sensible ones. With a hundred, it does not — so we encode
the preference directly.

Everything here costs no data. That is the point.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def total_variation(flow: torch.Tensor) -> torch.Tensor:
    """First-order smoothness of a (B, H, W, 2) displacement field.

    Penalises neighbouring pixels moving in different directions, which is what
    tearing looks like numerically.
    """
    dx = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    dy = (flow[:, 1:, :, :] - flow[:, :-1, :, :]).abs().mean()
    return dx + dy


def second_order_smoothness(flow: torch.Tensor) -> torch.Tensor:
    """Second-order smoothness: penalises *change* in the deformation rate.

    First-order TV alone biases towards a piecewise-constant field, i.e. a
    translation. Second-order terms allow smooth stretching (which real fabric
    does) while still forbidding kinks. Using both is what lets the residual
    flow model drape without shattering.
    """
    dxx = flow[:, :, 2:, :] - 2 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :]
    dyy = flow[:, 2:, :, :] - 2 * flow[:, 1:-1, :, :] + flow[:, :-2, :, :]
    return dxx.abs().mean() + dyy.abs().mean()


def mask_alignment(warped_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Shape agreement between the warped garment and the region it must fill.

    This is the workhorse supervision of the warping stage and it is almost free:
    the CIHP parse already tells us exactly which pixels of the training image
    are garment. Matching silhouettes is a much better-conditioned objective
    than matching RGB, because it is insensitive to colour, lighting and texture
    and therefore gives a clean gradient even before anything else has been
    learned.

    Combines L1 with a soft Dice term; L1 alone gives weak gradients when the
    masks barely overlap, which is exactly the situation early in training.
    """
    l1 = F.l1_loss(warped_mask, target_mask)

    intersection = (warped_mask * target_mask).sum(dim=[1, 2, 3])
    union = warped_mask.sum(dim=[1, 2, 3]) + target_mask.sum(dim=[1, 2, 3])
    dice = 1.0 - (2 * intersection + 1.0) / (union + 1.0)
    return l1 + dice.mean()


def alpha_regularisation(alpha: torch.Tensor, warped_mask: torch.Tensor) -> torch.Tensor:
    """Push the composition mask towards "copy the warped garment".

    Without this term the composer minimises L1 fastest by ignoring the warped
    garment and hallucinating a blurry average — the model looks like it is
    training well while producing exactly the washed-out results a plain
    image-to-image model gives. Pulling alpha towards the warped garment mask
    forces it to justify every synthesised pixel.
    """
    return F.l1_loss(alpha, warped_mask)


def alpha_sparsity(alpha: torch.Tensor) -> torch.Tensor:
    """Encourage a crisp, near-binary alpha rather than a soft global blend."""
    return (alpha * (1.0 - alpha)).mean()
