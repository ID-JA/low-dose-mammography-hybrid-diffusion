# Low-Dose Mammography Restoration via Hybrid Diffusion

A PyTorch implementation of a two-stage hybrid generative pipeline for restoring low-dose mammograms to full-dose quality. The approach combines a **VQ-VAE-2** (Vector Quantized Variational Autoencoder) as a learned first-stage compressor with a **Latent Diffusion Model (LDM)** that performs image-to-image restoration entirely in the compressed latent space.

## Overview

Low-dose mammography reduces radiation exposure but introduces noise that degrades diagnostic image quality. This project addresses the problem by:

1. **Stage 1 — VQ-VAE-2**: A hierarchical 5-level VQ-VAE-2 is trained on clean (full-dose) mammograms to learn a compact discrete latent representation. Once trained, the encoder/decoder are frozen and used as the first-stage model.
2. **Stage 2 — Latent Diffusion Model**: A UNet-based denoising diffusion model operates in the latent space of the frozen VQ-VAE-2. It is conditioned on the degraded (low-dose) latent via concatenation and learns to denoise toward the clean latent. At inference time, DDIM sampling produces a restored latent that is decoded back to pixel space.

The dataset used is **CBIS-DDSM**, with synthetic low-dose degradation applied at a configurable noise level.

## Project Structure

```
├── hps.py                 # All hyperparameters for VQ-VAE-2 and LDM
├── main-vqvae.py          # Training / evaluation script for VQ-VAE-2
├── main-diffusion.py      # Training script for Latent Diffusion Model
├── main-latents.py        # Latent dataset extraction from trained VQ-VAE-2
├── vqvae.py               # VQ-VAE-2 architecture (encoder, decoder, codebook)
├── vqvae2_wrapper.py      # Wrapper exposing VQ-VAE-2 as an LDM first-stage model
├── diffusion.py           # UNet, LatentDiffusion, DDIMSampler, LitEma
├── trainer.py             # VQ-VAE-2 training loop helper
├── datasets.py            # Dataset loaders (CBIS-DDSM)
├── datasets_cbis.py       # CBIS-DDSM dataset class with paired degradation
├── helper.py              # Utility functions
├── logger.py              # Logging utilities
└── preprocessing/
    └── degrade.py         # Low-dose noise simulation
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA support recommended)
- torchvision, torchmetrics, tqdm, pandas, Pillow

## Usage

### 1. Train VQ-VAE-2 (Stage 1)

Train the VQ-VAE-2 autoencoder on clean mammograms:

```bash
python main-vqvae.py --task cbis-ddsm
```

Evaluate a trained VQ-VAE-2 checkpoint:

```bash
python main-vqvae.py --task cbis-ddsm --load-path <checkpoint_path> --evaluate
```

**Available flags:**

| Flag           | Description                               |
| -------------- | ----------------------------------------- |
| `--cpu`        | Disable GPU, run on CPU                   |
| `--batch-size` | Override batch size from `hps.py`         |
| `--load-path`  | Resume training from a saved checkpoint   |
| `--no-tqdm`    | Disable progress bars                     |
| `--no-save`    | Disable saving checkpoints and images     |
| `--no-amp`     | Disable Automatic Mixed Precision         |
| `--save-jpg`   | Save output images as JPEG instead of PNG |
| `--evaluate`   | Run evaluation only (no training)         |

### 2. Train Latent Diffusion Model (Stage 2)

Train the LDM conditioned on degraded latents, using the frozen VQ-VAE-2:

```bash
python main-diffusion.py --task cbis-ddsm --vqvae-path <path_to_vqvae_checkpoint>
```

Resume LDM training from a checkpoint:

```bash
python main-diffusion.py --task cbis-ddsm --vqvae-path <vqvae_checkpoint> --load-path <diffusion_checkpoint>
```

**Available flags:**

| Flag           | Description                                            |
| -------------- | ------------------------------------------------------ |
| `--cpu`        | Disable GPU, run on CPU                                |
| `--vqvae-path` | **(Required)** Path to pre-trained VQ-VAE-2 checkpoint |
| `--load-path`  | Resume diffusion training from a saved checkpoint      |
| `--batch-size` | Override batch size from `hps.py`                      |
| `--no-tqdm`    | Disable progress bars                                  |
| `--no-save`    | Disable saving checkpoints and images                  |
| `--save-jpg`   | Save output images as JPEG instead of PNG              |

### 3. Extract Latent Dataset (Optional)

Generate a latent dataset from a trained VQ-VAE-2 for downstream use:

```bash
python main-latents.py <path_to_vqvae_checkpoint> --task cbis-ddsm
```

## Hyperparameters

All hyperparameters are defined in `hps.py`.

### VQ-VAE-2

| Hyperparameter             | Value           |
| -------------------------- | --------------- |
| Input image shape          | 1 × 1024 × 768  |
| Hidden channels            | 128             |
| Residual channels          | 32              |
| Residual layers per block  | 2               |
| Hierarchical levels        | 5               |
| Scaling rates (per level)  | [4, 2, 2, 2, 2] |
| Embedding dimension        | 64              |
| Codebook entries           | 512             |
| Codebook EMA decay         | 0.99            |
| Residual block type        | ReZero          |
| Optimizer                  | Adam            |
| Learning rate              | 1e-4            |
| Batch size                 | 4               |
| Commitment loss weight (β) | 0.25            |
| Reconstruction loss        | MSE             |
| Mixed precision (AMP)      | Yes             |
| Max epochs                 | 100             |

### Latent Diffusion Model

| Hyperparameter               | Value                             |
| ---------------------------- | --------------------------------- |
| Diffusion timesteps (T)      | 1000                              |
| Noise schedule               | Linear                            |
| β_start                      | 0.0015                            |
| β_end                        | 0.0195                            |
| UNet input channels          | 256 (concat conditioning)         |
| UNet output channels         | 128                               |
| Base model channels          | 192                               |
| Channel multipliers          | [1, 2, 4]                         |
| Residual blocks per level    | 2                                 |
| Self-attention at levels     | [1, 2]                            |
| Attention head channels      | 64                                |
| Dropout                      | 0.0                               |
| Gradient checkpointing       | Yes                               |
| Conditioning strategy        | Concatenation                     |
| Optimizer                    | AdamW                             |
| Learning rate                | 2e-4                              |
| Batch size                   | 4                                 |
| Max epochs                   | 200                               |
| Gradient clipping (max norm) | 1.0                               |
| Loss                         | L_simple (MSE on predicted noise) |
| VLB (ELBO) weight            | 0.0 (disabled)                    |
| EMA decay                    | 0.9999                            |
| DDIM sampling steps          | 50                                |
| DDIM η                       | 0.0 (deterministic)               |

### Dataset

| Parameter                 | Value                    |
| ------------------------- | ------------------------ |
| Dataset                   | CBIS-DDSM                |
| Image mode                | Grayscale (C=1)          |
| Resolution                | 1024 × 768               |
| Noise level (degradation) | 0.2                      |
| Train/test split          | Official CBIS-DDSM split |
| Data loader workers       | 4                        |

## Acknowledgements

This project builds upon and adapts code from the following open-source repositories:

- **[vvvm23/vqvae-2](https://github.com/vvvm23/vqvae-2)** — The VQ-VAE-2 architecture, hierarchical encoder/decoder design, codebook with EMA updates, and ReZero residual blocks are based on this implementation of _"Generating Diverse High-Fidelity Images with VQ-VAE-2"_ (Razavi et al., 2019).

- **[CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)** — The Latent Diffusion Model, UNet backbone, noise schedule, DDIM sampler, LitEma, and training/loss formulation are adapted from the official implementation of _"High-Resolution Image Synthesis with Latent Diffusion Models"_ (Rombach et al., 2022).

## Citations

```bibtex
@misc{razavi2019generating,
      title={Generating Diverse High-Fidelity Images with VQ-VAE-2},
      author={Ali Razavi and Aaron van den Oord and Oriol Vinyals},
      year={2019},
      eprint={1906.00446},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@inproceedings{rombach2022high,
      title={High-Resolution Image Synthesis with Latent Diffusion Models},
      author={Robin Rombach and Andreas Blattmann and Dominik Lorber and Patrick Esser and Bj{\"o}rn Ommer},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022},
      pages={10684--10695}
}

@misc{bachlechner2020rezero,
      title={ReZero is All You Need: Fast Convergence at Large Depth},
      author={Thomas Bachlechner and Bodhisattwa Prasad Majumder and Huanru Henry Mao and Garrison W. Cottrell and Julian McAuley},
      year={2020},
      eprint={2003.04887},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
