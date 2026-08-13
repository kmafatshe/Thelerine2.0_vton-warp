"""
Learning-rate schedule.

One-cycle is the right default here: a short warmup keeps the zero-initialised
flow heads from being kicked off identity by the first few large gradients, and
the long cosine decay is what lets the residual field settle instead of
oscillating.

The wrapper exists because `OneCycleLR` divides by its warmup length and so
fails outright on very short runs — the two- and ten-step runs used to smoke-test
the pipeline. Crashing there is pure noise, and it has bitten twice.
"""

from __future__ import annotations

import torch


def build_onecycle(optimiser, max_lr: float, total_steps: int,
                   warmup_fraction: float = 0.1):
    """One-cycle schedule, degrading to a constant LR when it cannot apply.

    `OneCycleLR` needs at least one step in each of its two phases; with a
    handful of total steps no valid `pct_start` exists. A constant rate is the
    honest answer there — such runs only ever check that the code executes.
    """
    # The warmup must be at least 2 steps for the phase boundary to be valid,
    # and must leave at least one step for the decay phase.
    minimum = 2.0 / total_steps
    maximum = 1.0 - 1.0 / total_steps
    if minimum > maximum:
        return torch.optim.lr_scheduler.LambdaLR(optimiser, lambda _: 1.0)

    pct_start = min(max(warmup_fraction, minimum), maximum)
    return torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=max_lr, total_steps=total_steps,
        pct_start=pct_start, div_factor=10.0, final_div_factor=100.0,
    )
