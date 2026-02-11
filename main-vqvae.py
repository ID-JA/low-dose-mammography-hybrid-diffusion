import torch
import torchvision
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import datetime
import time
from pathlib import Path
from math import sqrt

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from trainer import VQVAETrainer
from datasets import get_dataset
from hps import HPS_VQVAE as HPS
from helper import get_device, get_parameter_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--task', type=str, default='cbis-ddsm')
    parser.add_argument('--load-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--save-jpg', action='store_true')
    args = parser.parse_args()
    cfg = HPS[args.task]

    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    print(f"> Initialising VQ-VAE-2 model")
    trainer = VQVAETrainer(cfg, args)
    print(f"> Number of parameters: {get_parameter_count(trainer.net)}")

    if args.load_path:
        print(f"> Loading model parameters from checkpoint")
        trainer.load_checkpoint(args.load_path)

    if args.batch_size:
        cfg.batch_size = args.batch_size

    if args.evaluate:
        print(f"> Loading {cfg.display_name} dataset")
        _, test_loader = get_dataset(args.task, cfg, shuffle_test=True)
        print(f"> Generating evaluation batch of reconstructions")
        file_name = f"./recon-{save_id}-eval.{'jpg' if args.save_jpg else 'png'}"
        nb_generated = 0
        imgs = []
        pb = tqdm(total=cfg.batch_size)
        for batch in test_loader:
            if len(batch) == 3 and not isinstance(batch[1], dict):
                _, x, _ = batch           # use clean image
            elif len(batch) == 2:
                x, _ = batch
            else:
                x = batch[0]

            loss, r_loss, l_loss, y = trainer.eval(x, target=None)
            x_cpu, y_cpu = x.cpu().clamp(0, 1), y.cpu().clamp(0, 1)
            psnr = peak_signal_noise_ratio(y_cpu, x_cpu, data_range=1.0)
            ssim = structural_similarity_index_measure(y_cpu, x_cpu, data_range=1.0)
            print(f"  PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
            imgs.append(y_cpu)
            nb_generated += y.shape[0]
            pb.update(y.shape[0])
            if nb_generated >= cfg.batch_size:
                break
        print(f"> Assembling Image")
        save_image(torch.cat(imgs, dim=0), file_name, nrow=int(sqrt(cfg.batch_size)), normalize=True, value_range=(0, 1))
        print(f"> Saved to {file_name}")
        exit()
        
    if not args.no_save:
        runs_dir = Path(f"runs")
        root_dir = runs_dir / f"{args.task}-{save_id}"
        chk_dir = root_dir / "checkpoints"
        img_dir = root_dir / "images"
        log_dir = root_dir / "logs"

        runs_dir.mkdir(exist_ok=True)
        root_dir.mkdir(exist_ok=True)
        chk_dir.mkdir(exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)

    print(f"> Loading {cfg.display_name} dataset")
    train_loader, test_loader = get_dataset(args.task, cfg)

    for eid in range(cfg.max_epochs):
        print(f"> Epoch {eid+1}/{cfg.max_epochs}:")
        epoch_loss, epoch_r_loss, epoch_l_loss = 0.0, 0.0, 0.0
        epoch_start_time = time.time()
        pb = tqdm(train_loader, disable=args.no_tqdm)
        for i, batch in enumerate(pb):
            # VQ-VAE trains as autoencoder on CLEAN images only.
            # Dataset with return_pair=True returns (degraded, clean, meta).
            if len(batch) == 3 and not isinstance(batch[1], dict):
                _, x, meta = batch        # use clean image as input
            elif len(batch) == 2:
                x, meta = batch
            else:
                x = batch[0]

            loss, r_loss, l_loss, _ = trainer.train(x, target=None)
            epoch_loss += loss
            epoch_r_loss += r_loss
            epoch_l_loss += l_loss
            pb.set_description(f"training_loss: {epoch_loss / (i+1)} [r_loss: {epoch_r_loss/ (i+1)}, l_loss: {epoch_l_loss / (i+1)}]")
        print(f"> Training loss: {epoch_loss / len(train_loader)} [r_loss: {epoch_r_loss / len(train_loader)}, l_loss: {epoch_l_loss / len(train_loader)}]")
        
        epoch_loss, epoch_r_loss, epoch_l_loss = 0.0, 0.0, 0.0
        epoch_psnr, epoch_ssim, eval_count = 0.0, 0.0, 0
        pb = tqdm(test_loader, disable=args.no_tqdm)
        for i, batch in enumerate(pb):
            # VQ-VAE evaluates as autoencoder on CLEAN images only.
            if len(batch) == 3 and not isinstance(batch[1], dict):
                _, x, _ = batch           # use clean image
            elif len(batch) == 2:
                x, _ = batch
            else:
                x = batch[0]

            loss, r_loss, l_loss, y = trainer.eval(x, target=None)
            epoch_loss += loss
            epoch_r_loss += r_loss
            epoch_l_loss += l_loss

            x_cpu, y_cpu = x.cpu().clamp(0, 1), y.cpu().clamp(0, 1)
            epoch_psnr += peak_signal_noise_ratio(y_cpu, x_cpu, data_range=1.0).item()
            epoch_ssim += structural_similarity_index_measure(y_cpu, x_cpu, data_range=1.0).item()
            eval_count += 1

            pb.set_description(f"evaluation: {epoch_loss / (i+1)} [r_loss: {epoch_r_loss/ (i+1)}, l_loss: {epoch_l_loss / (i+1)}]")
            if i == 0 and not args.no_save and eid % cfg.image_frequency == 0:
                # Grid: Input | Reconstruction  (autoencoder, no separate target)
                x_vis, y_vis = x.cpu(), y.cpu()
                vis = torch.stack([x_vis, y_vis], dim=1).flatten(0, 1)
                save_image(vis, img_dir / f"recon-{str(eid).zfill(4)}.{'jpg' if args.save_jpg else 'png'}", nrow=2, normalize=True, value_range=(0, 1))

        if eid % cfg.checkpoint_frequency == 0 and not args.no_save:
            trainer.save_checkpoint(chk_dir / f"{args.task}-state-dict-{str(eid).zfill(4)}.pt")

        print(f"> Evaluation loss: {epoch_loss / len(test_loader)} [r_loss: {epoch_r_loss / len(test_loader)}, l_loss: {epoch_l_loss / len(test_loader)}]")
        print(f"> PSNR: {epoch_psnr / max(eval_count, 1):.2f} dB | SSIM: {epoch_ssim / max(eval_count, 1):.4f}")
        print(f"> Epoch time taken: {time.time() - epoch_start_time:.2f} seconds.")
        print()
