"""
Exponential moving average of model weights.

On a small dataset the loss surface is noisy — each batch is a large fraction of
the whole dataset, so consecutive gradient steps disagree strongly and the
weights oscillate. Averaging them recovers a point near the centre of that
oscillation, which reliably outperforms any single iterate. It costs one extra
copy of the model and typically buys the largest single quality improvement of
anything in this file tree.

Always evaluate and export from the EMA weights, not the live ones.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 200):
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.step_count = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step_count += 1
        # Ramp the decay in: early on the EMA should track the model closely,
        # otherwise it lags behind random initialisation for thousands of steps.
        decay = min(self.decay, (1 + self.step_count) / (self.warmup + self.step_count))

        # Parameters are averaged; buffers (running stats, precomputed grids)
        # are copied verbatim, because averaging a constant lookup table is
        # meaningless and averaging an integer counter is invalid.
        ema_params = dict(self.module.named_parameters())
        for name, param in model.named_parameters():
            ema_params[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

        ema_buffers = dict(self.module.named_buffers())
        for name, buffer in model.named_buffers():
            ema_buffers[name].copy_(buffer)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)
