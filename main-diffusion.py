import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import datetime
import time
from pathlib import Path
from math import sqrt
import os

from vqvae import VQVAE
from diffusion import UNet, LatentDiffusion
from datasets import get_dataset
from hps import HPS_DIFFUSION as HPS
from helper import get_device, get_parameter_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--task', type=str, default='cbis-ddsm')
    parser.add_argument('--vqvae-path', type=str, required=True, help='Path to pretrained VQVAE checkpoint')
    parser.add_argument('--load-path', type=str, default=None, help='Path to diffusion checkpoint to resume')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--save-jpg', action='store_true')
    args = parser.parse_args()
    
    cfg = HPS[args.task]
    device = get_device(args.cpu)
    
    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # --------------------------------------------------------------------------
    # Load VQ-VAE (Frozen)
    # --------------------------------------------------------------------------
    print(f"> Loading VQ-VAE model from {args.vqvae_path}")
    vqvae = VQVAE(
        in_channels=cfg.in_channels,
        hidden_channels=cfg.hidden_channels, 
        res_channels=cfg.res_channels, 
        nb_res_layers=cfg.nb_res_layers, 
        nb_levels=cfg.nb_levels, 
        embed_dim=cfg.embed_dim, 
        nb_entries=cfg.nb_entries, 
        scaling_rates=cfg.scaling_rates
    ).to(device)
    
    if os.path.exists(args.vqvae_path):
        vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=device))
    else:
        print(f"Warning: VQVAE checkpoint not found at {args.vqvae_path}. Initializing random weights (DEBUG MODE).")
    
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
        
    # --------------------------------------------------------------------------
    # Initialize Diffusion Model
    # --------------------------------------------------------------------------
    print(f"> Initialising Diffusion model")
    
    # Calculate latent channels (hidden_channels * nb_levels) due to stacking
    # As per our diffusion.py implementation which stacks all levels upsampled
    latent_channels = cfg.hidden_channels * cfg.nb_levels # 128 * 3 = 384
    
    unet = UNet(
        dim=cfg.unet_dim,
        channels=latent_channels,
        cond_channels=latent_channels, # Conditioning on same shaped latent
        dim_mults=cfg.unet_dim_mults
    ).to(device)
    
    diffusion = LatentDiffusion(
        vqvae=vqvae,
        unet=unet,
        timesteps=cfg.timesteps
    ).to(device)
    
    print(f"> Number of parameters (UNet): {get_parameter_count(unet)}")
    
    if args.load_path:
        print(f"> Loading diffusion parameters from {args.load_path}")
        diffusion.load_state_dict(torch.load(args.load_path, map_location=device))

    opt = torch.optim.Adam(unet.parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    
    # --------------------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------------------
    if args.batch_size:
        cfg.mini_batch_size = args.batch_size

    print(f"> Loading {cfg.display_name} dataset")
    # shuffle_test=True to see different samples in eval
    train_loader, test_loader = get_dataset(args.task, cfg, shuffle_test=True)

    # --------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------
    if not args.no_save:
        runs_dir = Path("runs_diffusion")
        root_dir = runs_dir / f"{args.task}-{save_id}"
        chk_dir = root_dir / "checkpoints"
        img_dir = root_dir / "images"
        
        runs_dir.mkdir(exist_ok=True)
        root_dir.mkdir(exist_ok=True)
        chk_dir.mkdir(exist_ok=True)
        img_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    for eid in range(cfg.max_epochs):
        print(f"> Epoch {eid+1}/{cfg.max_epochs}:")
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        diffusion.train()
        pb = tqdm(train_loader, disable=args.no_tqdm)
        
        for i, batch in enumerate(pb):
            # Parse batch
            if len(batch) == 3:
                x_deg, x_clean, _ = batch 
                if isinstance(x_clean, dict): # Handle dataset quirk if any
                     x_deg, _ = batch
                     x_clean = x_deg # Fallback if not paired (should not happen)
            else:
                print("Error: Dataset did not return pairs. Ensure return_pair=True in dataset config.")
                break
                
            x_deg = x_deg.to(device)
            x_clean = x_clean.to(device)
            
            opt.zero_grad()
            
            with torch.amp.autocast("cuda"):
                loss = diffusion.get_loss(x_clean, x_deg)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            epoch_loss += loss.item()
            pb.set_description(f"loss: {loss.item():.4f}")
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"> Training loss: {avg_loss:.4f}")
        
        # ----------------------------------------------------------------------
        # Evaluation / Sampling
        # ----------------------------------------------------------------------
        if eid % cfg.image_frequency == 0:
            print("> Generating evaluation samples...")
            diffusion.eval()
            
            # Get one batch from test loader
            try:
                batch = next(iter(test_loader))
                if len(batch) == 3:
                    x_deg, x_clean, _ = batch
                else:
                    x_deg, _ = batch
                    x_clean = x_deg
                    
                x_deg = x_deg[:4].to(device) # Limit to 4 images
                x_clean = x_clean[:4].to(device)
                
                # Sample
                x_recon = diffusion.sample(x_deg)
                
                # Viz: Input | Target | Restored
                x_deg, x_clean, x_recon = x_deg.cpu(), x_clean.cpu(), x_recon.cpu()
                
                vis = torch.stack([x_deg, x_clean, x_recon], dim=1).flatten(0, 1)
                
                if not args.no_save:
                    save_image(vis, img_dir / f"recon-{str(eid).zfill(4)}.{'jpg' if args.save_jpg else 'png'}", nrow=3, normalize=True, value_range=(-1,1))
            except Exception as e:
                print(f"Sampling failed: {e}")

        if not args.no_save and eid % cfg.checkpoint_frequency == 0:
            torch.save(diffusion.state_dict(), chk_dir / f"diffusion-state-dict-{str(eid).zfill(4)}.pt")

        print(f"> Epoch time taken: {time.time() - epoch_start_time:.2f} seconds.")
        print()
