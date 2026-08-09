#!/usr/bin/env python3
"""
Stage 2 training: learn the composition.

    python train_tryon.py --config configs/tryon.yaml

The warper is loaded from stage 1 and kept **frozen by default**. That is
deliberate: the composer's fastest route to a low loss is to blur away the
warp's mistakes, and if gradients can reach the warper it will happily *make*
the warp worse in exchange for an easier composition problem. Freeze it, get a
good composer, then optionally unfreeze for a short low-learning-rate joint
finetune (`train.finetune_warper_after`).

Loss structure:

    L1              anchors global colour and position
    perceptual      restores the detail L1 blurs away
    alpha vs mask   forces the model to copy real garment pixels
    alpha sparsity  keeps the copy/synthesise decision crisp
    GAN (optional)  sharpens fabric texture; enabled after `gan_start_step`

The adversarial term stays off for the first few thousand steps. Starting a
discriminator against a randomly initialised generator on 100 images produces a
discriminator that wins instantly and a generator that never recovers.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vtonwarp.data.agnostic import CONDITION_CHANNELS
from vtonwarp.engine.checkpoint import load_checkpoint, maybe_resume, save_checkpoint
from vtonwarp.engine.data import build_dataloaders, infinite
from vtonwarp.engine.ema import ModelEMA
from vtonwarp.engine.visualize import contact_sheet
from vtonwarp.losses.gan import discriminator_loss, generator_loss
from vtonwarp.losses.perceptual import PerceptualLoss
from vtonwarp.losses.regularisers import alpha_regularisation, alpha_sparsity
from vtonwarp.models.composer import Composer
from vtonwarp.models.discriminator import PatchDiscriminator, diff_augment
from vtonwarp.models.warper import GarmentWarper
from vtonwarp.utils.config import load_config, resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tryon.yaml")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def load_warper(config, device):
    """Rebuild the stage-1 warper from its checkpoint, preferring EMA weights."""
    checkpoint = load_checkpoint(config.train.warp_checkpoint, map_location="cpu")
    saved = checkpoint["config"]["model"]

    # The TPS kernel is precomputed per resolution, so the two stages must agree.
    trained_at = (checkpoint["config"]["data"]["height"],
                  checkpoint["config"]["data"]["width"])
    if trained_at != (config.data.height, config.data.width):
        raise SystemExit(
            f"resolution mismatch: {config.train.warp_checkpoint} was trained at "
            f"{trained_at[0]}x{trained_at[1]} but this config uses "
            f"{config.data.height}x{config.data.width}. Retrain stage 1 at the new "
            f"resolution, or match it here."
        )

    warper = GarmentWarper(
        height=config.data.height,
        width=config.data.width,
        condition_channels=CONDITION_CHANNELS,
        base=saved["base"],
        feature_channels=saved["feature_channels"],
        grid_size=saved["grid_size"],
        max_displacement=saved["max_displacement"],
        use_residual_flow=saved["use_residual_flow"],
    )
    state = checkpoint["ema"].get("warper") or checkpoint["models"]["warper"]
    warper.load_state_dict(state)
    return warper.to(device)


def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "auto"))
    output_dir = Path(config.output_dir)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(config)
    stream = infinite(train_loader)

    warper = load_warper(config, device)
    warper.eval()
    warper.requires_grad_(False)

    composer = Composer(
        condition_channels=CONDITION_CHANNELS,
        base=config.model.base,
        depth=config.model.depth,
        max_channels=config.model.max_channels,
    ).to(device)
    print(f"[tryon] composer {sum(p.numel() for p in composer.parameters()) / 1e6:.2f}M "
          f"parameters on {device}")

    weights = dict(config.loss)
    perceptual = None
    if weights.get("perceptual", 0) > 0:
        perceptual = PerceptualLoss(
            weights_path=weights.get("vgg_weights"),
            style_weight=weights.get("style", 0.0),
        ).to(device)

    optimiser = torch.optim.AdamW(
        composer.parameters(), lr=config.train.lr, betas=(0.5, 0.999),
        weight_decay=config.train.get("weight_decay", 1e-4),
    )
    # 10% warmup, but never fewer than 2 steps — OneCycleLR divides by the
    # warmup length and a very short smoke run would otherwise crash.
    warmup_pct = min(0.5, max(0.1, 2.0 / config.train.steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=config.train.lr, total_steps=config.train.steps,
        pct_start=warmup_pct, div_factor=10.0, final_div_factor=100.0,
    )
    ema = ModelEMA(composer, decay=config.train.get("ema_decay", 0.999))

    use_gan = weights.get("gan", 0) > 0
    discriminator = optimiser_d = None
    if use_gan:
        discriminator = PatchDiscriminator(
            in_channels=3 + CONDITION_CHANNELS, base=config.model.get("d_base", 32)
        ).to(device)
        optimiser_d = torch.optim.AdamW(
            discriminator.parameters(), lr=config.train.get("lr_d", config.train.lr),
            betas=(0.5, 0.999),
        )

    checkpoint_path = output_dir / "tryon.pt"
    start_step = 0
    if config.train.get("resume", True):
        start_step = maybe_resume(checkpoint_path, "composer", model=composer,
                                  ema=ema, optimiser=optimiser, scheduler=scheduler)
        if start_step:
            print(f"[tryon] resumed from step {start_step}")
        if start_step >= config.train.steps:
            print("[tryon] checkpoint is already at the requested step count; "
                  "raise train.steps to continue")
            return

    finetune_after = config.train.get("finetune_warper_after", 0)
    optimiser_w = None
    accumulate = config.train.get("accumulate", 1)
    start = time.time()

    for step in range(start_step + 1, config.train.steps + 1):
        composer.train()

        # Optional joint finetune, at a deliberately tiny learning rate.
        if finetune_after and step == finetune_after:
            warper.requires_grad_(True)
            warper.train()
            optimiser_w = torch.optim.AdamW(
                warper.parameters(), lr=config.train.lr * 0.05, betas=(0.5, 0.999)
            )
            print(f"[tryon] unfroze warper at step {step}")

        gan_active = use_gan and step >= config.train.get("gan_start_step", 0)

        optimiser.zero_grad(set_to_none=True)
        if optimiser_w:
            optimiser_w.zero_grad(set_to_none=True)
        logs: dict[str, float] = {}

        for _ in range(accumulate):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in next(stream).items()}

            with torch.set_grad_enabled(optimiser_w is not None):
                warp = warper(batch["garment"], batch["garment_input_mask"],
                              batch["condition"])

            result = composer(batch["condition"], warp["warped"], warp["warped_mask"])

            loss, terms = composition_losses(
                result, warp, batch, perceptual, weights
            )

            if gan_active:
                fake_logits = discriminator(
                    diff_augment(result["output"], config.train.get(
                        "diffaug", "color,translation,cutout")),
                    batch["condition"],
                )
                adversarial = generator_loss(fake_logits)
                loss = loss + weights["gan"] * adversarial
                terms["tryon/gan_g"] = adversarial

            (loss / accumulate).backward()
            logs = {k: logs.get(k, 0.0) + float(v) / accumulate for k, v in terms.items()}

            if gan_active:
                logs["tryon/gan_d"] = logs.get("tryon/gan_d", 0.0) + _train_discriminator(
                    discriminator, optimiser_d, result["output"].detach(),
                    batch["person"], batch["condition"], config
                ) / accumulate

        torch.nn.utils.clip_grad_norm_(composer.parameters(),
                                       config.train.get("grad_clip", 1.0))
        optimiser.step()
        if optimiser_w:
            optimiser_w.step()
        scheduler.step()
        ema.update(composer)

        if step % config.train.log_every == 0:
            summary = "  ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in logs.items())
            print(f"[tryon] step {step:>6}/{config.train.steps}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  {summary}  "
                  f"({(time.time() - start) / (step - start_step):.2f}s/step)")

        if step % config.train.sample_every == 0 or step == config.train.steps:
            evaluate(warper, ema.module, val_loader, device,
                     output_dir / "samples" / f"tryon_{step:06d}.png")

        if step % config.train.save_every == 0 or step == config.train.steps:
            save_checkpoint(
                checkpoint_path, step=step, config=config,
                models={"composer": composer, "warper": warper},
                ema={"composer": ema.state_dict()},
                optimisers={"composer": optimiser},
                schedulers={"composer": scheduler},
            )

    print(f"[tryon] done in {(time.time() - start) / 60:.1f} min -> "
          f"{checkpoint_path}")


def composition_losses(result, warp, batch, perceptual, weights):
    person = batch["person"]
    output = result["output"]

    terms = {
        "tryon/l1": F.l1_loss(output, person),
        # The garment region is a small fraction of the image, so an unweighted
        # L1 is dominated by the background the model copies for free. This term
        # puts the pressure back where the work is.
        "tryon/l1_garment": _masked_l1(output, person, batch["garment_mask"]),
        "tryon/alpha": alpha_regularisation(result["alpha"], warp["warped_mask"]),
        "tryon/alpha_sparse": alpha_sparsity(result["alpha"]),
    }

    if perceptual is not None and weights.get("perceptual", 0) > 0:
        terms["tryon/perceptual"] = perceptual(output, person)

    key_to_weight = {
        "tryon/l1": weights["l1"],
        "tryon/l1_garment": weights["l1_garment"],
        "tryon/alpha": weights["alpha"],
        "tryon/alpha_sparse": weights["alpha_sparse"],
        "tryon/perceptual": weights.get("perceptual", 0.0),
    }
    total = sum(key_to_weight[name] * value for name, value in terms.items())
    return total, terms


def _masked_l1(prediction, target, mask):
    diff = (prediction - target).abs() * mask
    return diff.sum() / (mask.sum() * prediction.shape[1] + 1e-6)


def _train_discriminator(discriminator, optimiser, fake, real, condition, config):
    policy = config.train.get("diffaug", "color,translation,cutout")
    optimiser.zero_grad(set_to_none=True)
    real_logits = discriminator(diff_augment(real, policy), condition)
    fake_logits = discriminator(diff_augment(fake, policy), condition)
    loss = discriminator_loss(real_logits, fake_logits)
    loss.backward()
    optimiser.step()
    return float(loss)


@torch.no_grad()
def evaluate(warper, composer, loader, device, path: Path) -> None:
    warper.eval()
    composer.eval()
    batch = next(iter(loader))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    warp = warper(batch["garment"], batch["garment_input_mask"], batch["condition"])
    result = composer(batch["condition"], warp["warped"], warp["warped_mask"])

    contact_sheet(
        {
            "garment": batch["garment"],
            "agnostic": batch["agnostic"],
            "warped": warp["warped_masked"],
            "render": result["render"],
            "alpha": result["alpha"],
            "output": result["output"],
            "ground truth": batch["person"],
        },
        path,
    )
    print(f"[tryon] wrote {path}")


if __name__ == "__main__":
    main()
