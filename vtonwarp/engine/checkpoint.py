"""Checkpoint save/load, including the config that produced it."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, *, step: int, config: dict,
                    models: dict, ema: dict | None = None,
                    optimisers: dict | None = None,
                    schedulers: dict | None = None) -> None:
    """Write a checkpoint atomically.

    Optimiser and scheduler state are included so a run can resume exactly —
    which matters on Colab, where a disconnect can otherwise cost hours. The
    write goes to a temporary file first: a checkpoint truncated by a session
    dying mid-save is worse than no checkpoint at all.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": dict(config),
        "models": {name: model.state_dict() for name, model in models.items()},
        "ema": ema or {},
        "optimisers": {name: opt.state_dict()
                       for name, opt in (optimisers or {}).items()},
        "schedulers": {name: sched.state_dict()
                       for name, sched in (schedulers or {}).items()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)


def load_into(model, state: dict, strict: bool = True):
    model.load_state_dict(state, strict=strict)
    return model


def maybe_resume(path: str | Path, name: str, *, model, ema=None,
                 optimiser=None, scheduler=None) -> int:
    """Restore a run in place and return the step to continue from.

    Returns 0 when there is nothing to resume, so the caller can always write
    `for step in range(start + 1, total + 1)`.
    """
    path = Path(path)
    if not path.exists():
        return 0

    checkpoint = load_checkpoint(path, map_location="cpu")
    model.load_state_dict(checkpoint["models"][name])

    if ema is not None and checkpoint["ema"].get(name):
        ema.load_state_dict(checkpoint["ema"][name])
        # Keep the EMA decay ramp where it was, or the average would briefly
        # snap back towards the live weights after every restart.
        ema.step_count = checkpoint["step"]
    if optimiser is not None and checkpoint["optimisers"].get(name):
        optimiser.load_state_dict(checkpoint["optimisers"][name])

    step = int(checkpoint["step"])

    # The scheduler is fast-forwarded rather than restored from its state dict.
    # OneCycleLR bakes `total_steps` into its state, so restoring it would crash
    # the moment you resume with a different train.steps — which is exactly what
    # you do when a Colab session dies and you extend the run. Replaying the
    # schedule against the *current* config is always consistent.
    if scheduler is not None:
        for _ in range(step):
            scheduler.step()

    return step
