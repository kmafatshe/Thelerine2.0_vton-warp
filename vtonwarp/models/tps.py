"""
Thin-Plate-Spline sampling grid.

A TPS transform is defined by a small grid of control points (we use 5x5 = 25).
Move the control points, and the whole plane deforms smoothly around them,
minimising bending energy. That gives us a warp with only 50 free parameters
(25 points x 2 coords) that can still bend a flat T-shirt around a torso.

Why this matters for a small dataset: a dense per-pixel flow field at 256x192
has ~98,000 free parameters and can trivially overfit — it can memorise a
mapping for every training image that means nothing for a new one. A 50-DOF
transform physically *cannot* overfit that way; smoothness is baked into the
parameterisation rather than begged for with a loss term. We predict the coarse
TPS first and only then allow a small dense residual on top.

Maths
-----
Given source control points P_i and their displaced targets Q_i, TPS solves for
coefficients (W, A) satisfying

    f(x) = A0 + A1*x + A2*y + sum_i W_i * U(||(x,y) - P_i||)
    U(r) = r^2 * log(r^2)

subject to the interpolation constraints f(P_i) = Q_i and the natural boundary
conditions sum W_i = sum W_i*P_ix = sum W_i*P_iy = 0. Stacking those gives the
linear system L * [W; A] = [Q; 0] with

        | K  P |                 K_ij = U(||P_i - P_j||)
    L = | Pt 0 |                 P    = [1, P_x, P_y]

Because the *source* control points are a fixed lattice, L is constant and we
invert it once at construction time. At runtime each forward pass is two small
batched matrix multiplies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TPSGridGen(nn.Module):
    def __init__(self, out_h: int, out_w: int, grid_size: int = 5):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.grid_size = grid_size
        self.num_points = grid_size * grid_size

        # Fixed lattice of source control points in normalised [-1, 1] space.
        axis = torch.linspace(-1.0, 1.0, grid_size)
        py, px = torch.meshgrid(axis, axis, indexing="ij")
        control_x = px.reshape(-1, 1)   # (N, 1)
        control_y = py.reshape(-1, 1)

        self.register_buffer("control_x", control_x)
        self.register_buffer("control_y", control_y)
        self.register_buffer("L_inv", self._build_L_inverse(control_x, control_y))

        # Output sampling lattice: every pixel of the target image.
        ys = torch.linspace(-1.0, 1.0, out_h)
        xs = torch.linspace(-1.0, 1.0, out_w)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # .contiguous() matters: torch.meshgrid returns stride-0 views, and a
        # stride-0 buffer cannot be written to by copy_ (which the EMA does).
        self.register_buffer("grid_x", grid_x[None, ..., None].contiguous())
        self.register_buffer("grid_y", grid_y[None, ..., None].contiguous())

        # U(||pixel - control_i||) for every pixel: (1, H, W, N). Constant, so
        # we precompute it — this is the expensive part of TPS and it never
        # changes because the source lattice is fixed.
        self.register_buffer(
            "kernel", self._radial_basis(self.grid_x, self.grid_y).contiguous()
        )

    # -- construction ------------------------------------------------------

    @staticmethod
    def _build_L_inverse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        dx = x - x.t()
        dy = y - y.t()
        dist_sq = dx.pow(2) + dy.pow(2)
        dist_sq[dist_sq == 0] = 1.0            # U(0) := 0, and log(1) = 0
        K = dist_sq * torch.log(dist_sq)

        P = torch.cat([torch.ones(n, 1), x, y], dim=1)          # (N, 3)
        top = torch.cat([K, P], dim=1)                          # (N, N+3)
        bottom = torch.cat([P.t(), torch.zeros(3, 3)], dim=1)   # (3, N+3)
        L = torch.cat([top, bottom], dim=0)                     # (N+3, N+3)
        return torch.inverse(L)

    def _radial_basis(self, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
        """U(r) between every sample point and every control point."""
        cx = self.control_x.view(1, 1, 1, -1)
        cy = self.control_y.view(1, 1, 1, -1)
        dist_sq = (px - cx).pow(2) + (py - cy).pow(2)
        dist_sq = torch.where(dist_sq == 0, torch.ones_like(dist_sq), dist_sq)
        return dist_sq * torch.log(dist_sq)

    # -- runtime -----------------------------------------------------------

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """Turn predicted control-point offsets into a sampling grid.

        Args:
            offsets: (B, 2*N) displacement of each control point in normalised
                coordinates. All zeros == identity transform.

        Returns:
            (B, out_h, out_w, 2) grid consumable by F.grid_sample.
        """
        batch = offsets.shape[0]
        n = self.num_points
        offsets = offsets.view(batch, n, 2)

        target_x = self.control_x.view(1, n, 1) + offsets[:, :, 0:1]
        target_y = self.control_y.view(1, n, 1) + offsets[:, :, 1:2]

        map_x = self._solve(target_x)
        map_y = self._solve(target_y)
        return torch.cat([map_x, map_y], dim=-1)

    def _solve(self, target: torch.Tensor) -> torch.Tensor:
        """Evaluate the spline for one coordinate axis. target: (B, N, 1)."""
        batch, n = target.shape[0], self.num_points
        rhs = torch.cat([target, target.new_zeros(batch, 3, 1)], dim=1)  # (B, N+3, 1)
        coeffs = torch.matmul(self.L_inv.unsqueeze(0), rhs)              # (B, N+3, 1)

        weights = coeffs[:, :n].view(batch, 1, 1, n)     # non-linear part
        affine = coeffs[:, n:].view(batch, 3, 1)         # global affine part

        bending = (self.kernel * weights).sum(dim=-1, keepdim=True)       # (B,H,W,1)
        linear = (
            affine[:, 0].view(batch, 1, 1, 1)
            + affine[:, 1].view(batch, 1, 1, 1) * self.grid_x
            + affine[:, 2].view(batch, 1, 1, 1) * self.grid_y
        )
        return linear + bending

    # -- regularisation ----------------------------------------------------

    def grid_regularisation(self, offsets: torch.Tensor) -> torch.Tensor:
        """Penalise non-uniform spacing between neighbouring control points.

        Without this the regressor is free to fold the lattice over itself,
        producing a warp that mirrors part of the garment onto the body. The
        loss compares horizontal and vertical gaps between adjacent control
        points against their neighbours, so uniform stretching is free but
        local folding is expensive.
        """
        batch = offsets.shape[0]
        g = self.grid_size
        points = offsets.view(batch, g, g, 2)
        points = points + torch.stack(
            [self.control_x.view(g, g), self.control_y.view(g, g)], dim=-1
        )[None]

        dx = points[:, :, 1:, 0] - points[:, :, :-1, 0]     # horizontal gaps
        dy = points[:, 1:, :, 1] - points[:, :-1, :, 1]     # vertical gaps

        loss = (dx[:, :, 1:] - dx[:, :, :-1]).abs().mean()
        loss = loss + (dy[:, 1:, :] - dy[:, :-1, :]).abs().mean()
        # Also forbid outright inversion (a gap that flips sign).
        loss = loss + torch.relu(-dx).mean() + torch.relu(-dy).mean()
        return loss
