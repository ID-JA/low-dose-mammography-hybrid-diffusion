"""
VQ-VAE-2 wrapper that presents a VQModelInterface-compatible API for Latent Diffusion.

The wrapper operates on the **bottom-level** encoder output (the finest-scale latent)
which is (B, hidden_channels=128, H/8, W/8).  For decoding, it re-derives the upper
encoder levels by running the frozen encoders[1:] on the denoised bottom latent, then
calls the full VQ-VAE-2 top-down quantisation + decoder path.

References
----------
CompVis/latent-diffusion  –  ldm.models.autoencoder.VQModelInterface
"""

import torch
import torch.nn as nn


class VQVAE2Wrapper(nn.Module):
    """Wrap a pre-trained VQVAE (from vqvae.py) for use as ``first_stage_model``
    in a Latent Diffusion pipeline.

    Public API consumed by the diffusion model:
        encode(x)   -> z   : (B, C_latent, H_lat, W_lat) tensor
        decode(z)   -> x̂   : (B, C_img, H, W) reconstruction
    """

    def __init__(self, vqvae: nn.Module):
        super().__init__()
        self.vqvae = vqvae

        # Freeze all VQ-VAE weights
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Override train() so that the VQ-VAE always stays in eval mode
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        self.vqvae.eval()
        return self

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the bottom-level encoder feature map (before quantisation).

        Parameters
        ----------
        x : (B, C_img, H, W)

        Returns
        -------
        z : (B, hidden_channels, H/8, W/8)  – continuous latent tensor.
        """
        encoder_outputs = self.vqvae.encode(x)  # list of [enc0, enc1, enc2, …]
        return encoder_outputs[0]                # bottom-level (finest scale)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, z: torch.Tensor, force_not_quantize: bool = False) -> torch.Tensor:
        """Decode a bottom-level latent back to pixel space.

        The upper encoder levels are re-derived by running the frozen
        ``encoders[1:]`` on *z*, then the full VQ-VAE-2 top-down path
        (quantisation + decoders) is executed.

        Parameters
        ----------
        z : (B, hidden_channels, H_lat, W_lat)
        force_not_quantize : bool
            If ``True``, bypass VQ codebook nearest-neighbour lookup and
            pass continuous latents through ``conv_in`` projection only.
            Used at sampling time when the diffusion model has already
            denoised the latent (matches ``VQModelInterface`` behaviour).

        Returns
        -------
        x_hat : (B, C_img, H, W)
        """
        # Rebuild hierarchical encoder outputs from the bottom latent
        encoder_outputs = [z]
        for enc in self.vqvae.encoders[1:]:
            encoder_outputs.append(enc(encoder_outputs[-1]))

        if force_not_quantize:
            # Continuous decode (skip codebook quantisation)
            return self.vqvae.decode_continuous(encoder_outputs)
        else:
            # Standard quantised decode
            return self.vqvae.decode_latents(encoder_outputs)
