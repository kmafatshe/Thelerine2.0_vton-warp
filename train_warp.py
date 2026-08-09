#!/usr/bin/env python3
"""
Stage 1 training: teach the warper where the garment goes.

    python train_warp.py --config configs/warp.yaml

What is being supervised
------------------------
For every training person we already know, from the CIHP parse, exactly which
pixels are garment. So we have two free ground truths:

    target_mask : the silhouette the warped garment must match
    target_rgb  : the actual worn garment pixels it must match

The mask term dominates early (shape is easier and better conditioned than
colour), the RGB and perceptual terms take over once the garment lands in
roughly the right place, and the regularisers keep the field physically sane
throughout.

Train this stage to convergence *before* touching stage 2. A composer trained on
top of a bad warp learns to ignore the warp, and that habit never goes away.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vtonwarp.data.agnostic import CONDITION_CHANNELS
from vtonwarp.engine.checkpoint import maybe_resume, save_checkpoint
from vtonwarp.engine.data import build_dataloaders, infinite
from vtonwarp.engine.ema import ModelEMA
from vtonwarp.engine.visualize import contact_sheet, flow_to_rgb
from vtonwarp.losses.perceptual import PerceptualLoss
from vtonwarp.losses.regularisers import (
    mask_alignment,
    second_order_smoothness,
    total_variation,
)
from vtonwarp.models.warper import GarmentWarper
from vtonwarp.utils.config import load_config, resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/warp.yaml")
    parser.add_argument("overrides", nargs="*", help="e.g. train.steps=5000")
    return parser.parse_args()


def warp_losses(outputs: dict, batch: dict, perceptual, weights) -> tuple[torch.Tensor, dict]:
    target_mask = batch["garment_mask"]
    target_rgb = batch["target_garment"]

    # Restrict RGB comparison to the union of predicted and target garment so
    # that empty background never dilutes the signal.
    region = torch.clamp(outputs["warped_mask"] + target_mask, 0.0, 1.0)

    shape = mask_alignment(outputs["warped_mask"], target_mask)
    coarse_shape = mask_alignment(outputs["coarse_mask"], target_mask)

    pixel = (outputs["warped"] - target_rgb).abs()
    pixel = (pixel * region).sum() / (region.sum() * 3 + 1e-6)

    terms = {
        "warp/shape": shape,
        "warp/coarse_shape": coarse_shape,
        "warp/pixel": pixel,
        "warp/grid_reg": outputs["_grid_reg"],
    }

    if perceptual is not None and weights["perceptual"] > 0:
        terms["warp/perceptual"] = perceptual(
            outputs["warped"] * region, target_rgb * region, mask=region
        )

    if outputs["residual"] is not None:
        terms["warp/tv"] = total_variation(outputs["residual"])
        terms["warp/smooth"] = second_order_smoothness(outputs["residual"])

    key_to_weight = {
        "warp/shape": weights["shape"],
        "warp/coarse_shape": weights["coarse_shape"],
        "warp/pixel": weights["pixel"],
        "warp/perceptual": weights["perceptual"],
        "warp/grid_reg": weights["grid"],
        "warp/tv": weights["tv"],
        "warp/smooth": weights["smooth"],
    }
    total = sum(key_to_weight[name] * value for name, value in terms.items())
    return total, {name: float(value) for name, value in terms.items()}


def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "auto"))
    output_dir = Path(config.output_dir)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(config)
    stream = infinite(train_loader)

    model = GarmentWarper(
        height=config.data.height,
        width=config.data.width,
        condition_channels=CONDITION_CHANNELS,
        base=config.model.base,
        feature_channels=config.model.feature_channels,
        grid_size=config.model.grid_size,
        max_displacement=config.model.max_displacement,
        use_residual_flow=config.model.use_residual_flow,
    ).to(device)

    print(f"[warp] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters "
          f"on {device}")

    weights = dict(config.loss)
    perceptual = None
    if weights.get("perceptual", 0) > 0:
        perceptual = PerceptualLoss(
            weights_path=weights.get("vgg_weights"),
            style_weight=weights.get("style", 0.0),
        ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, betas=(0.5, 0.999),
        weight_decay=config.train.get("weight_decay", 1e-4),
    )
    # 10% warmup, but never fewer than 2 steps — OneCycleLR divides by the
    # warmup length and a very short smoke run would otherwise crash.
    warmup_pct = min(0.5, max(0.1, 2.0 / config.train.steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=config.train.lr, total_steps=config.train.steps,
        pct_start=warmup_pct, div_factor=10.0, final_div_factor=100.0,
    )
    ema = ModelEMA(model, decay=config.train.get("ema_decay", 0.999))

    checkpoint_path = output_dir / "warp.pt"
    start_step = 0
    if config.train.get("resume", True):
        start_step = maybe_resume(checkpoint_path, "warper", model=model, ema=ema,
                                  optimiser=optimiser, scheduler=scheduler)
        if start_step:
            print(f"[warp] resumed from step {start_step}")
        if start_step >= config.train.steps:
            print("[warp] checkpoint is already at the requested step count; "
                  "raise train.steps to continue")
            return

    accumulate = config.train.get("accumulate", 1)
    start = time.time()

    for step in range(start_step + 1, config.train.steps + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        logs: dict[str, float] = {}

        for _ in range(accumulate):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in next(stream).items()}

            outputs = model(batch["garment"], batch["garment_input_mask"],
                            batch["condition"])
            outputs["_grid_reg"] = model.tps.grid_regularisation(outputs["offsets"])

            loss, terms = warp_losses(outputs, batch, perceptual, weights)
            (loss / accumulate).backward()
            logs = {k: logs.get(k, 0.0) + v / accumulate for k, v in terms.items()}

        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       config.train.get("grad_clip", 1.0))
        optimiser.step()
        scheduler.step()
        ema.update(model)

        if step % config.train.log_every == 0:
            elapsed = time.time() - start
            summary = "  ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in logs.items())
            print(f"[warp] step {step:>6}/{config.train.steps}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  {summary}  "
                  f"({elapsed / (step - start_step):.2f}s/step)")

        if step % config.train.sample_every == 0 or step == config.train.steps:
            evaluate(ema.module, val_loader, device,
                     output_dir / "samples" / f"warp_{step:06d}.png")

        if step % config.train.save_every == 0 or step == config.train.steps:
            save_checkpoint(
                checkpoint_path, step=step, config=config,
                models={"warper": model}, ema={"warper": ema.state_dict()},
                optimisers={"warper": optimiser}, schedulers={"warper": scheduler},
            )

    print(f"[warp] done in {(time.time() - start) / 60:.1f} min -> "
          f"{checkpoint_path}")


@torch.no_grad()
def evaluate(model, loader, device, path: Path) -> None:
    model.eval()
    batch = next(iter(loader))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    outputs = model(batch["garment"], batch["garment_input_mask"], batch["condition"])

    # Overlay the warped garment on the agnostic person: the fastest way to see
    # whether the geometry is right.
    overlay = (batch["agnostic"] * (1 - outputs["warped_mask"])
               + outputs["warped"] * outputs["warped_mask"])

    columns = {
        "garment": batch["garment"],
        "agnostic": batch["agnostic"],
        "coarse": outputs["coarse"] * outputs["coarse_mask"],
        "warped": outputs["warped_masked"],
        "overlay": overlay,
        "target": batch["target_garment"],
        "gt": batch["person"],
    }
    if outputs["residual"] is not None:
        columns["flow"] = flow_to_rgb(outputs["residual"])

    contact_sheet(columns, path)
    print(f"[warp] wrote {path}")


if __name__ == "__main__":
    main()
