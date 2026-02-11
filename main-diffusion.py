"""
Training script for Latent Diffusion restoration of low-dose mammograms.

Uses:
  - Pre-trained VQ-VAE-2 (frozen) wrapped via VQVAE2Wrapper
  - UNet + LatentDiffusion from diffusion.py  (adapted from CompVis/latent-diffusion)
  - Paired CBIS-DDSM dataset  (degraded, clean, meta)
  - Concat conditioning:  UNet input = cat([z_noisy_clean, z_degraded], dim=1)
  - DDIM sampler for fast evaluation

Usage::

    python main-diffusion.py --task cbis-ddsm \\
        --vqvae-path runs/.../checkpoints/cbis-ddsm-state-dict-0096.pt
"""

import argparse
import datetime
import os
import time
import traceback
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae import VQVAE
from vqvae2_wrapper import VQVAE2Wrapper
from diffusion import UNet, LatentDiffusion
from datasets import get_dataset
from hps import HPS_DIFFUSION as HPS
from helper import get_device, get_parameter_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--task", type=str, default="cbis-ddsm")
    parser.add_argument("--vqvae-path", type=str, required=True,
                        help="Path to pretrained VQ-VAE-2 checkpoint")
    parser.add_argument("--load-path", type=str, default=None,
                        help="Path to diffusion checkpoint to resume")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--save-jpg", action="store_true")
    args = parser.parse_args()

    cfg = HPS[args.task]
    device = get_device(args.cpu)
    save_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ======================================================================
    # 1.  Load & wrap VQ-VAE-2  (frozen first-stage model)
    # ======================================================================
    print(f"> Loading VQ-VAE-2 from {args.vqvae_path}")
    vqvae = VQVAE(
        in_channels=cfg.in_channels,
        hidden_channels=cfg.hidden_channels,
        res_channels=cfg.res_channels,
        nb_res_layers=cfg.nb_res_layers,
        nb_levels=cfg.nb_levels,
        embed_dim=cfg.embed_dim,
        nb_entries=cfg.nb_entries,
        scaling_rates=cfg.scaling_rates,
    ).to(device)

    if os.path.exists(args.vqvae_path):
        vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=device))
        print("  VQ-VAE weights loaded ✓")
    else:
        print(f"  ⚠ Checkpoint not found – using random weights (debug mode)")

    wrapper = VQVAE2Wrapper(vqvae).to(device)

    # ======================================================================
    # 2.  Build UNet  +  LatentDiffusion
    # ======================================================================
    # Latent: (B, hidden_channels, H/8, W/8)  →  128 ch at 128×96
    # Concat conditioning doubles the input channels
    z_ch = cfg.hidden_channels                           # 128
    unet = UNet(
        in_channels=z_ch * 2,                            # concat(z_noisy, z_cond)
        out_channels=z_ch,                               # predict noise (same shape)
        model_channels=cfg.model_channels,               # base width
        channel_mult=tuple(cfg.channel_mult),
        num_res_blocks=cfg.num_res_blocks,
        attention_levels=tuple(cfg.attention_levels),
        dropout=cfg.dropout,
        num_head_channels=cfg.num_head_channels,
        use_checkpoint=cfg.use_checkpoint,
    ).to(device)

    ldm = LatentDiffusion(
        first_stage_model=wrapper,
        unet=unet,
        timesteps=cfg.timesteps,
        beta_schedule=cfg.beta_schedule,
        linear_start=cfg.linear_start,
        linear_end=cfg.linear_end,
        scale_factor=cfg.scale_factor,
        use_ema=cfg.use_ema,
        ema_decay=cfg.ema_decay,
        l_simple_weight=cfg.l_simple_weight,
        original_elbo_weight=cfg.original_elbo_weight,
        learn_logvar=cfg.learn_logvar,
        logvar_init=cfg.logvar_init,
    ).to(device)

    print(f"> UNet  parameters: {get_parameter_count(unet):,}")
    print(f"> Total parameters: {get_parameter_count(ldm):,}")

    # Optionally resume
    if args.load_path:
        print(f"> Resuming from {args.load_path}")
        state = torch.load(args.load_path, map_location=device)
        ldm.load_state_dict(state, strict=False)

    # ======================================================================
    # 3.  Optimiser  +  AMP scaler
    # ======================================================================
    trainable_params = list(unet.parameters())
    if cfg.learn_logvar:
        print("> Including logvar as trainable parameter")
        trainable_params.append(ldm.logvar)
    opt = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda")

    # ======================================================================
    # 4.  Dataset  (paired: degraded, clean, meta)
    # ======================================================================
    if args.batch_size:
        cfg.mini_batch_size = args.batch_size
    print(f"> Loading {cfg.display_name} dataset")
    train_loader, test_loader = get_dataset(args.task, cfg, shuffle_test=True)

    # ======================================================================
    # 5.  Directories
    # ======================================================================
    if not args.no_save:
        root_dir = Path("runs_diffusion") / f"{args.task}-{save_id}"
        chk_dir = root_dir / "checkpoints"
        img_dir = root_dir / "images"
        for d in (root_dir, chk_dir, img_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # 6.  Calibrate latent scale factor  (1/std of encoder outputs)
    # ======================================================================
    if not ldm._scale_calibrated.item():
        ldm.calibrate_scale_factor(train_loader, max_batches=20)
    else:
        print(f"> Using pre-set scale_factor = {ldm.scale_factor.item():.4f}")

    # ======================================================================
    # 7.  Training  loop
    # ======================================================================
    for epoch in range(cfg.max_epochs):
        print(f"\n> Epoch {epoch + 1}/{cfg.max_epochs}")
        t0 = time.time()
        ldm.train()
        epoch_loss = 0.0

        pb = tqdm(train_loader, disable=args.no_tqdm)
        for batch in pb:
            # Expected: (degraded, clean, meta)
            if len(batch) == 3:
                x_deg, x_clean, _ = batch
            else:
                print("Error: expected paired dataset (return_pair=True)")
                break

            x_deg = x_deg.to(device)
            x_clean = x_clean.to(device)

            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss, loss_dict = ldm.training_loss(x_clean, x_deg)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(),
                                           max_norm=cfg.grad_clip_norm)
            scaler.step(opt)
            scaler.update()

            # EMA update (after each optimiser step, matches ddpm.py)
            ldm.update_ema()

            epoch_loss += loss.item()
            pb.set_description(f"loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        loss_info = " | ".join(
            f"{k}: {v.item():.6f}" for k, v in loss_dict.items()
        ) if loss_dict else ""
        print(f"  avg train loss: {avg_loss:.6f}  ({loss_info})")

        # ------------------------------------------------------------------
        # Evaluation sampling
        # ------------------------------------------------------------------
        if epoch % cfg.image_frequency == 0:
            print("  sampling…")
            ldm.eval()
            try:
                batch = next(iter(test_loader))
                x_deg, x_clean, _ = batch if len(batch) == 3 else (batch[0], batch[0], None)
                n_vis = min(4, x_deg.shape[0])
                x_deg = x_deg[:n_vis].to(device)
                x_clean = x_clean[:n_vis].to(device)

                with ldm.ema_scope("Evaluation"):
                    x_recon = ldm.sample(
                        x_deg,
                        use_ddim=True,
                        ddim_steps=cfg.ddim_steps,
                        ddim_eta=cfg.ddim_eta,
                        verbose=not args.no_tqdm,
                        unconditional_guidance_scale=cfg.unconditional_guidance_scale,
                    )

                # Visualise: Degraded | Clean | Restored  (one row per image)
                vis = torch.stack(
                    [x_deg.cpu(), x_clean.cpu(), x_recon.cpu()], dim=1
                ).flatten(0, 1)

                if not args.no_save:
                    ext = "jpg" if args.save_jpg else "png"
                    save_image(
                        vis,
                        img_dir / f"recon-{str(epoch).zfill(4)}.{ext}",
                        nrow=3, normalize=True, value_range=(0, 1),
                    )
            except Exception:
                traceback.print_exc()

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        if not args.no_save and epoch % cfg.checkpoint_frequency == 0:
            torch.save(ldm.state_dict(),
                       chk_dir / f"diffusion-{str(epoch).zfill(4)}.pt")

        print(f"  elapsed: {time.time() - t0:.1f}s")
