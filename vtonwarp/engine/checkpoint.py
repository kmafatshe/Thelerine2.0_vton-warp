"""Checkpoint save/load, including the config that produced it."""

from __future__ import annotations

from pathlib import Path

import torch


def _plain(value):
    """Deep-convert to plain Python types.

    `dict(config)` is shallow, so nested sections stayed `Config` instances and
    got pickled as such. PyTorch 2.6 made `weights_only=True` the default for
    `torch.load`, which refuses any custom class — so checkpoints written by
    earlier versions of this file fail to load on current PyTorch with an error
    about an unsupported global. Storing only primitives keeps checkpoints
    loadable under the strict default, which is also the safer one.
    """
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_checkpoint(path: str | Path, *, step: int, config: dict,
                    models: dict, ema: dict | None = None,
                    optimisers: dict | None = None) -> None:
    """Write a checkpoint atomically.

    Optimiser state is included so a run can resume exactly, which matters on
    Colab where a disconnect can otherwise cost hours. Scheduler state is not:
    `maybe_resume` replays the schedule against the current config rather than
    restoring it, and OneCycleLR's state contains a bound method, which pickles
    via `getattr` and is rejected by the strict `weights_only` loader. Saving
    state nothing reads, at the cost of making the file unloadable, is a bad
    trade.

    The write goes to a temporary file first: a checkpoint truncated by a
    session dying mid-save is worse than no checkpoint at all.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": _plain(config),
        "models": {name: model.state_dict() for name, model in models.items()},
        "ema": ema or {},
        "optimisers": {name: opt.state_dict()
                       for name, opt in (optimisers or {}).items()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    """Load a checkpoint, preferring PyTorch's strict weights-only mode.

    Checkpoints written before `_plain` existed contain a custom class and can
    only be read with the permissive loader. That is an arbitrary-code-execution
    path in general, so it is used only as a fallback and only for files this
    project wrote itself.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        print(f"[checkpoint] {path} predates the plain-config format; loaded with "
              "weights_only=False. Re-saving it will use the strict format.")
        return checkpoint


def load_into(model, state: dict, strict: bool = True):
    model.load_state_dict(state, strict=strict)
    return model


SEMANTIC_DATA_KEYS = (
    "height", "width", "label_scheme", "parse_source", "garment_type",
    "crop_to_person", "crop_mode", "crop_context", "crop_margin",
    "canonicalise_garment", "garment_fill", "erase_dilate",
)


def check_resume_compatible(checkpoint: dict, config) -> list[str]:
    """Report data settings that changed since the checkpoint was written.

    Resuming carries the optimiser state and step count forward, which only
    makes sense if the inputs still mean the same thing. After changing the
    label scheme, the parse source or the crop, the old weights were fitted to
    different images entirely — continuing from them is worse than starting
    over, and silently does the wrong thing.
    """
    saved = checkpoint.get("config", {}).get("data", {})
    current = config.get("data", {})
    return [
        f"{key}: {saved.get(key)!r} -> {current.get(key)!r}"
        for key in SEMANTIC_DATA_KEYS
        if key in saved and saved.get(key) != current.get(key)
    ]


def maybe_resume(path: str | Path, name: str, *, model, ema=None,
                 optimiser=None, scheduler=None, config=None,
                 extra_models: dict | None = None,
                 extra_optimisers: dict | None = None) -> int:
    """Restore a run in place and return the step to continue from.

    Returns 0 when there is nothing to resume, so the caller can always write
    `for step in range(start + 1, total + 1)`.
    """
    path = Path(path)
    if not path.exists():
        return 0

    checkpoint = load_checkpoint(path, map_location="cpu")

    if config is not None:
        changed = check_resume_compatible(checkpoint, config)
        if changed:
            raise SystemExit(
                f"Refusing to resume {path}: the data settings changed since it "
                "was written, so its weights were fitted to different inputs.\n  "
                + "\n  ".join(changed)
                + "\n\nStart a fresh run instead — set FRESH_START = True in the "
                "notebook, or pass train.resume=false. To keep the old run for "
                "comparison, point output_dir somewhere new."
            )

    model.load_state_dict(checkpoint["models"][name])

    if ema is not None and checkpoint["ema"].get(name):
        ema.load_state_dict(checkpoint["ema"][name])
        # Keep the EMA decay ramp where it was, or the average would briefly
        # snap back towards the live weights after every restart.
        ema.step_count = checkpoint["step"]
    if optimiser is not None and checkpoint["optimisers"].get(name):
        optimiser.load_state_dict(checkpoint["optimisers"][name])

    # Anything else the run needs to continue rather than restart — notably the
    # discriminator, which is half of an adversarial game and cannot be
    # reinitialised mid-run without destabilising the generator it faces.
    for key, module in (extra_models or {}).items():
        if checkpoint["models"].get(key):
            module.load_state_dict(checkpoint["models"][key])
    for key, opt in (extra_optimisers or {}).items():
        if checkpoint["optimisers"].get(key):
            opt.load_state_dict(checkpoint["optimisers"][key])

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
