"""
Latent Diffusion Model for low-dose mammography restoration.

Faithful adaptation of CompVis/latent-diffusion (Rombach et al., 2022).
Self-contained – no PyTorch-Lightning or taming-transformers needed.

Aligned 1:1 with the official LDM codebase:
  - UNet uses zero_module on ResBlock out-conv, Attention proj_out, and final output conv
  - LitEma for exponential-moving-average weights (used at evaluation)
  - Loss with per-timestep logvar weighting + optional VLB term
  - clip_denoised = False in latent space
  - Gradient checkpointing support
  - DDIM sampler with classifier-free guidance support
  - Concat conditioning for image restoration

Public classes
--------------
UNet            – OpenAI-style denoising backbone (matches openaimodel.py)
LatentDiffusion – VQ-VAE wrapper, noise schedule, loss, sampling
DDIMSampler     – Fast deterministic sampler (eta = 0 -> DDIM)
LitEma          – Exponential moving average of model parameters
"""

import math
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ======================================================================
# Utilities  (matching ldm/modules/diffusionmodules/util.py)
# ======================================================================

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """Index *a* with timestep indices *t* and reshape for broadcasting."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    if repeat:
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1)))
    return torch.randn(shape, device=device)


def zero_module(module):
    """Zero out the parameters of a module and return it.

    Matches ldm/modules/diffusionmodules/util.py::zero_module exactly.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# ======================================================================
# Gradient checkpointing  (matching ldm/modules/diffusionmodules/util.py)
# ======================================================================

class _CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True)
                             for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """Evaluate *func* with optional gradient checkpointing.

    Matches ldm/modules/diffusionmodules/util.py::checkpoint.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return _CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


# ======================================================================
# Noise schedule  (from ldm/modules/diffusionmodules/util.py)
# ======================================================================

def make_beta_schedule(schedule, n_timestep,
                       linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                             n_timestep, dtype=np.float64) ** 2)
    elif schedule == "cosine":
        steps = (np.arange(n_timestep + 1, dtype=np.float64) / n_timestep
                 + cosine_s)
        alphas = np.cos(steps / (1 + cosine_s) * np.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, 0, 0.999)
    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep,
                            dtype=np.float64)
    elif schedule == "sqrt":
        betas = np.linspace(linear_start, linear_end, n_timestep,
                            dtype=np.float64) ** 0.5
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    return betas


# ======================================================================
# Sinusoidal timestep embedding  (matches util.py::timestep_embedding)
# ======================================================================

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32,
                       device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ======================================================================
# EMA  (from ldm/modules/ema.py -- LitEma, verbatim)
# ======================================================================

class LitEma(nn.Module):
    """Exponential-moving-average shadow of model parameters.

    Matches ``ldm.modules.ema.LitEma`` exactly.
    """

    def __init__(self, model, decay=0.9999, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.m_name2s_name = {}
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int) if use_num_updates
            else torch.tensor(-1, dtype=torch.int),
        )

        for name, p in model.named_parameters():
            if p.requires_grad:
                s_name = name.replace(".", "")
                self.m_name2s_name[name] = s_name
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,
                        (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(
                        m_param[key])
                    shadow_params[sname].sub_(
                        one_minus_decay
                        * (shadow_params[sname] - m_param[key]))

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(
                    shadow_params[self.m_name2s_name[key]].data)

    def store(self, parameters):
        """Save current parameters for later restoration."""
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """Restore the parameters saved with :meth:`store`."""
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


# ======================================================================
# UNet building blocks  (matching openaimodel.py)
# ======================================================================

class GroupNorm32(nn.GroupNorm):
    """GroupNorm that casts to float32 (mixed-precision safe)."""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def _norm(channels, num_groups=32):
    return GroupNorm32(num_groups, channels)


class ResBlock(nn.Module):
    """Residual block with timestep-embedding injection.

    Matches ``openaimodel.py::ResBlock`` with:
      - ``use_scale_shift_norm=False`` (additive embedding, official default)
      - ``zero_module`` on output conv  (critical for identity init)
      - Gradient checkpointing support
    """

    def __init__(self, channels, emb_channels, out_channels=None,
                 dropout=0.0, use_checkpoint=False):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            _norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )

        self.out_layers = nn.Sequential(
            _norm(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        self.skip_connection = (
            nn.Conv2d(channels, self.out_channels, 1)
            if channels != self.out_channels else nn.Identity()
        )

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class SelfAttention(nn.Module):
    """Multi-head self-attention with zero-initialized output projection.

    Uses ``F.scaled_dot_product_attention`` (flash / memory-efficient kernel)
    which is mathematically equivalent to the official manual-einsum path
    but more efficient on modern hardware.

    Matches ``openaimodel.py::AttentionBlock`` structure with:
      - ``zero_module`` on ``proj_out``
      - Gradient checkpointing support
    """

    def __init__(self, channels, num_head_channels=64, use_checkpoint=False):
        super().__init__()
        self.num_heads = max(1, channels // num_head_channels)
        self.use_checkpoint = use_checkpoint
        self.norm = _norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x_flat = self.norm(x).view(b, c, h * w)          # (B, C, N)
        qkv = self.qkv(x_flat)                             # (B, 3C, N)
        q, k, v = qkv.chunk(3, dim=1)                      # each (B, C, N)

        hd = c // self.num_heads
        q = q.view(b, self.num_heads, hd, -1).transpose(2, 3)  # (B,H,N,hd)
        k = k.view(b, self.num_heads, hd, -1).transpose(2, 3)
        v = v.view(b, self.num_heads, hd, -1).transpose(2, 3)

        out = F.scaled_dot_product_attention(q, k, v)       # (B,H,N,hd)
        out = out.transpose(2, 3).reshape(b, c, h * w)
        out = self.proj_out(out)
        return x_in + out.view(b, c, h, w)


class Downsample(nn.Module):
    """Strided convolution downsample (matches openaimodel.py, conv_resample=True)."""
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    """Nearest-neighbour + conv upsample (matches openaimodel.py)."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ======================================================================
# UNet  (OpenAI-style, adapted from openaimodel.py::UNetModel)
# ======================================================================

class UNet(nn.Module):
    """Denoising UNet with skip connections, time conditioning, and
    optional self-attention at selected resolution levels.

    Matches ``openaimodel.py::UNetModel`` structure with:
      - ``zero_module`` on output conv
      - Gradient checkpointing passed to all sub-blocks
      - Self-attention at specified levels

    Parameters
    ----------
    in_channels       : total input channels (z_noisy + z_cond for concat)
    out_channels      : predicted noise channels (= z_channels)
    model_channels    : base channel width
    channel_mult      : per-level channel multipliers
    num_res_blocks    : residual blocks per resolution level
    attention_levels  : level indices where self-attention is inserted
    dropout           : dropout probability
    num_head_channels : channels per attention head
    use_checkpoint    : gradient checkpointing to reduce memory
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 128,
        model_channels: int = 192,
        channel_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_levels: tuple = (2,),
        dropout: float = 0.0,
        num_head_channels: int = 64,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4
        attn_set = set(attention_levels)

        # --- time embedding  (matches openaimodel.py) ---
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # --- input projection ---
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Track channel counts at every skip-producing point
        ch = model_channels
        skip_channels = [ch]

        # --- down path ---
        self.down_blocks = nn.ModuleList()
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, out_ch, dropout,
                                   use_checkpoint=use_checkpoint)]
                ch = out_ch
                if level in attn_set:
                    layers.append(SelfAttention(
                        ch, num_head_channels,
                        use_checkpoint=use_checkpoint))
                self.down_blocks.append(nn.ModuleList(layers))
                skip_channels.append(ch)
            if level < len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)

        # --- middle ---
        self.mid_res1 = ResBlock(ch, time_embed_dim, ch, dropout,
                                 use_checkpoint=use_checkpoint)
        self.mid_attn = SelfAttention(ch, num_head_channels,
                                      use_checkpoint=use_checkpoint)
        self.mid_res2 = ResBlock(ch, time_embed_dim, ch, dropout,
                                 use_checkpoint=use_checkpoint)

        # --- up path ---
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                s_ch = skip_channels.pop()
                layers = [ResBlock(ch + s_ch, time_embed_dim, out_ch, dropout,
                                   use_checkpoint=use_checkpoint)]
                ch = out_ch
                if level in attn_set:
                    layers.append(SelfAttention(
                        ch, num_head_channels,
                        use_checkpoint=use_checkpoint))
                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                self.up_blocks.append(nn.ModuleList(layers))

        # --- output  (zero_module on final conv -- critical) ---
        self.out_norm = _norm(ch)
        self.out_conv = zero_module(
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        """
        x : (B, in_channels, H, W)  including any concatenated conditioning
        t : (B,) long timestep indices
        """
        emb = self.time_embed(timestep_embedding(t, self.model_channels))

        h = self.input_conv(x)
        hs = [h]

        # --- down ---
        for block_layers in self.down_blocks:
            for layer in block_layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
            hs.append(h)

        # --- middle ---
        h = self.mid_res1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, emb)

        # --- up ---
        for block_layers in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in block_layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)

        return self.out_conv(F.silu(self.out_norm(h)))


# ======================================================================
# Latent Diffusion Model  (matching ddpm.py::LatentDiffusion)
# ======================================================================

class LatentDiffusion(nn.Module):
    """Latent Diffusion with concat conditioning for image restoration.

    Faithfully matches the official ``ddpm.py::LatentDiffusion``:
      - Noise schedule registration (betas, alphas, posterior)
      - scale_factor as registered buffer with auto-calibration
      - Per-timestep logvar weighting + VLB loss term
      - clip_denoised = False for latent space
      - EMA via LitEma with ema_scope context manager
      - DDPM and DDIM sampling
    """

    def __init__(
        self,
        first_stage_model: nn.Module,
        unet: UNet,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        linear_start: float = 0.0015,
        linear_end: float = 0.0195,
        parameterization: str = "eps",
        scale_factor: float = 1.0,
        loss_type: str = "l2",
        # --- EMA ---
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        # --- loss weighting (official defaults) ---
        l_simple_weight: float = 1.0,
        original_elbo_weight: float = 0.0,
        learn_logvar: bool = False,
        logvar_init: float = 0.0,
        # --- latent space ---
        clip_denoised: bool = False,
    ):
        super().__init__()
        assert parameterization in ("eps", "x0")
        self.parameterization = parameterization
        self.loss_type = loss_type
        self.clip_denoised = clip_denoised

        # scale_factor as buffer (persisted in checkpoints, auto-calibrated)
        self.register_buffer("scale_factor", torch.tensor(scale_factor))
        # self._scale_calibrated = not (scale_factor == 1.0)
        self.register_buffer(
            "_scale_calibrated",
            torch.tensor(scale_factor != 1.0, dtype=torch.bool),
        )

        # ---- Loss weighting (matches ddpm.py) ----
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight
        self.learn_logvar = learn_logvar

        # ---- First stage (frozen) ----
        self.first_stage_model = first_stage_model
        self.first_stage_model.eval()
        for p in self.first_stage_model.parameters():
            p.requires_grad = False

        # ---- Denoising model ----
        self.model = unet

        # ---- EMA (matches ddpm.py L85-88) ----
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay)
            print(f"  Keeping EMAs of "
                  f"{len(list(self.model_ema.buffers()))} parameters.")

        # ---- Noise schedule ----
        self._register_schedule(beta_schedule, timesteps,
                                linear_start, linear_end)

        # ---- Per-timestep logvar (matches ddpm.py L108-111) ----
        logvar = torch.full(fill_value=logvar_init,
                            size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(logvar, requires_grad=True)
        else:
            self.register_buffer("logvar", logvar)

    def train(self, mode: bool = True):
        super().train(mode)
        self.first_stage_model.eval()  # always frozen
        return self

    # ------------------------------------------------------------------
    # EMA scope  (matches ddpm.py L170-186)
    # ------------------------------------------------------------------
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def update_ema(self):
        """Call after each optimiser step (matches ddpm.py on_train_batch_end)."""
        if self.use_ema:
            self.model_ema(self.model)

    # ------------------------------------------------------------------
    # Schedule registration  (matches ddpm.py::register_schedule)
    # ------------------------------------------------------------------
    def _register_schedule(self, schedule, timesteps, linear_start, linear_end):
        betas = make_beta_schedule(schedule, timesteps,
                                   linear_start, linear_end)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.num_timesteps = int(timesteps)

        _t = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", _t(betas))
        self.register_buffer("alphas_cumprod", _t(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", _t(alphas_cumprod_prev))

        self.register_buffer("sqrt_alphas_cumprod",
                             _t(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             _t(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod",
                             _t(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod",
                             _t(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod",
                             _t(np.sqrt(1.0 / alphas_cumprod - 1)))

        # posterior  q(x_{t-1} | x_t, x_0)  (v_posterior=0 like official default)
        posterior_var = (betas * (1.0 - alphas_cumprod_prev)
                         / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance", _t(posterior_var))
        self.register_buffer("posterior_log_variance_clipped",
                             _t(np.log(np.maximum(posterior_var, 1e-20))))
        self.register_buffer("posterior_mean_coef1",
                             _t(betas * np.sqrt(alphas_cumprod_prev)
                                / (1.0 - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2",
                             _t((1.0 - alphas_cumprod_prev)
                                * np.sqrt(alphas)
                                / (1.0 - alphas_cumprod)))

        # VLB weights  (matches ddpm.py L139-147)
        if self.parameterization == "eps":
            lvlb_weights = (self.betas ** 2
                            / (2 * self.posterior_variance
                               * _t(alphas)
                               * (1 - self.alphas_cumprod)))
        elif self.parameterization == "x0":
            lvlb_weights = (0.5 * torch.sqrt(_t(alphas_cumprod))
                            / (2.0 * (1 - _t(alphas_cumprod))))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    # ------------------------------------------------------------------
    # Scale calibration  (matches ddpm.py on_train_batch_start)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate_scale_factor(self, dataloader, max_batches=20):
        """Compute ``1/std`` of latent space and set *scale_factor*.

        Mirrors the official ``on_train_batch_start`` auto-calibration in
        ``ddpm.py::LatentDiffusion`` lines 486-498.
        """
        print("### USING STD-RESCALING ###")
        self.first_stage_model.eval()
        all_z = []
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            x = batch[1] if len(batch) == 3 else batch[0]  # clean images
            x = x.to(next(self.parameters()).device)
            z = self.first_stage_model.encode(x)
            all_z.append(z.flatten())
        all_z = torch.cat(all_z)
        sf = 1.0 / all_z.std()
        self.scale_factor.fill_(sf)
        self._scale_calibrated.fill_(True)
        print(f"setting self.scale_factor to {sf.item():.6f}")
        print("### USING STD-RESCALING ###")

    # ------------------------------------------------------------------
    # First-stage helpers  (matches ddpm.py encode/decode/get_first_stage_encoding)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, z):
        """Scale encoder output (matches ddpm.py::get_first_stage_encoding)."""
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z, force_not_quantize=False):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(
            z, force_not_quantize=force_not_quantize)

    # ------------------------------------------------------------------
    # Forward diffusion  q(z_t | z_0)
    # ------------------------------------------------------------------
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t,
                          x_start.shape) * noise)

    # ------------------------------------------------------------------
    # UNet forward  (concat conditioning -- matches ddpm.py::apply_model
    # with DiffusionWrapper conditioning_key='concat')
    # ------------------------------------------------------------------
    def apply_model(self, z_noisy, t, cond):
        """Concatenate condition and run UNet."""
        return self.model(torch.cat([z_noisy, cond], dim=1), t)

    # ------------------------------------------------------------------
    # Loss  (matches ddpm.py::LatentDiffusion::p_losses exactly)
    # ------------------------------------------------------------------
    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = F.mse_loss(target, pred)
            else:
                loss = F.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError(f"Unknown loss type '{self.loss_type}'")
        return loss

    def p_losses(self, z_clean, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(z_clean))
        z_noisy = self.q_sample(z_clean, t, noise)
        model_output = self.apply_model(z_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        target = noise if self.parameterization == "eps" else z_clean

        # --- simple loss with per-timestep logvar weighting ---
        loss_simple = self.get_loss(model_output, target, mean=False).mean(
            dim=[1, 2, 3])
        loss_dict[f"{prefix}/loss_simple"] = loss_simple.mean()

        logvar_t = self.logvar[t].to(z_clean.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict[f"{prefix}/loss_gamma"] = loss.mean()
            loss_dict["logvar"] = self.logvar.data.mean()

        loss = self.l_simple_weight * loss.mean()

        # --- VLB loss term ---
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(
            dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict[f"{prefix}/loss_vlb"] = loss_vlb

        loss += self.original_elbo_weight * loss_vlb
        loss_dict[f"{prefix}/loss"] = loss

        return loss, loss_dict

    def training_loss(self, x_clean, x_degraded):
        """End-to-end: encode both -> sample *t* -> return (loss, loss_dict)."""
        z_clean = self.get_first_stage_encoding(
            self.encode_first_stage(x_clean))
        z_cond = self.get_first_stage_encoding(
            self.encode_first_stage(x_degraded))
        t = torch.randint(0, self.num_timesteps, (z_clean.shape[0],),
                          device=z_clean.device, dtype=torch.long)
        return self.p_losses(z_clean, z_cond, t)

    # ------------------------------------------------------------------
    # Reverse-process helpers
    # ------------------------------------------------------------------
    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t,
                          x_t.shape) * noise)

    def q_posterior(self, x_start, x_t, t):
        mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def p_mean_variance(self, x, t, cond, clip_denoised=None):
        if clip_denoised is None:
            clip_denoised = self.clip_denoised
        model_out = self.apply_model(x, t, cond)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t, model_out)
        else:
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        mean, var, log_var = self.q_posterior(x_recon, x, t)
        return mean, var, log_var

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=None,
                 temperature=1.0, noise_dropout=0.0):
        b, *_, device = *x.shape, x.device
        mean, _, log_var = self.p_mean_variance(
            x, t, cond, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device) * temperature
        if noise_dropout > 0.0:
            noise = F.dropout(noise, p=noise_dropout)
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(x.shape) - 1)))
        return mean + nonzero_mask * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, verbose=True):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(self.num_timesteps)),
                      desc="DDPM Sampling", total=self.num_timesteps,
                      disable=not verbose):
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, ts, cond)
        return img

    # ------------------------------------------------------------------
    # High-level restoration API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, x_degraded,
               use_ddim=True, ddim_steps=50, ddim_eta=0.0,
               verbose=True,
               unconditional_guidance_scale=1.0,
               unconditional_conditioning=None):
        """Restore a degraded image.

        Parameters
        ----------
        x_degraded : (B, C_img, H, W)
        use_ddim   : use DDIM (fast); otherwise full DDPM (slow)
        ddim_steps : number of DDIM steps
        ddim_eta   : DDIM stochasticity (0 = deterministic)
        unconditional_guidance_scale : >1.0 enables classifier-free guidance
        unconditional_conditioning : tensor of unconditional condition (e.g. zeros)

        Returns
        -------
        x_restored : (B, C_img, H, W)
        """
        z_cond = self.get_first_stage_encoding(
            self.encode_first_stage(x_degraded))
        shape = z_cond.shape  # (B, C, H_lat, W_lat)

        if use_ddim:
            sampler = DDIMSampler(self)
            z_out, _ = sampler.sample(
                S=ddim_steps, batch_size=shape[0], shape=shape[1:],
                conditioning=z_cond, eta=ddim_eta, verbose=verbose,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        else:
            z_out = self.p_sample_loop(z_cond, shape, verbose=verbose)

        return self.decode_first_stage(z_out)


# ======================================================================
# DDIM Sampler  (matching ldm/models/diffusion/ddim.py)
# ======================================================================

def _make_ddim_timesteps(num_ddim, num_ddpm, method="uniform"):
    if method == "uniform":
        c = num_ddpm // num_ddim
        return np.asarray(list(range(0, num_ddpm, c))) + 1
    elif method == "quad":
        return (np.linspace(0, np.sqrt(num_ddpm * 0.8),
                            num_ddim) ** 2).astype(int) + 1
    raise NotImplementedError(method)


def _make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta):
    """Matches util.py::make_ddim_sampling_parameters."""
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray(
        [alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    return sigmas, alphas, alphas_prev


class DDIMSampler:
    """DDIM sampler for :class:`LatentDiffusion`.

    Faithful adaptation of ``ldm.models.diffusion.ddim.DDIMSampler``
    with classifier-free guidance support.
    """

    def __init__(self, model: LatentDiffusion):
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps

    def _register_buffer(self, name, attr):
        """Matches the official DDIMSampler.register_buffer."""
        if isinstance(attr, torch.Tensor):
            if attr.device != self.model.betas.device:
                attr = attr.to(self.model.betas.device)
        setattr(self, name, attr)

    def _make_schedule(self, ddim_steps, eta=0.0):
        self.ddim_timesteps = _make_ddim_timesteps(
            ddim_steps, self.ddpm_num_timesteps)

        ac = self.model.alphas_cumprod.cpu().numpy()
        sigmas, alphas, alphas_prev = _make_ddim_sampling_parameters(
            ac, self.ddim_timesteps, eta)

        device = self.model.betas.device
        _t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)

        self._register_buffer("ddim_alphas", _t(alphas))
        self._register_buffer("ddim_alphas_prev", _t(alphas_prev))
        self._register_buffer("ddim_sqrt_one_minus_alphas",
                              _t(np.sqrt(1.0 - alphas)))
        self._register_buffer("ddim_sigmas", _t(sigmas))

    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning,
               eta=0.0, x_T=None, verbose=True,
               unconditional_guidance_scale=1.0,
               unconditional_conditioning=None,
               **kwargs):
        self._make_schedule(S, eta=eta)
        C, H, W = shape
        device = self.model.betas.device
        img = default(x_T, lambda: torch.randn(
            (batch_size, C, H, W), device=device))

        timesteps = np.flip(self.ddim_timesteps)
        total = len(timesteps)
        iterator = tqdm(timesteps, desc="DDIM Sampler", total=total,
                        disable=not verbose)

        intermediates = {"x_inter": [img], "pred_x0": [img]}

        for i, step in enumerate(iterator):
            idx = total - i - 1
            ts = torch.full((batch_size,), step, device=device,
                            dtype=torch.long)
            img, pred_x0 = self._p_sample_ddim(
                img, conditioning, ts, idx,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )

            if i % max(total // 5, 1) == 0 or i == total - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def _p_sample_ddim(self, x, cond, t, index,
                       unconditional_guidance_scale=1.0,
                       unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        # --- Classifier-free guidance  (matches ddim.py L173-183) ---
        if (unconditional_conditioning is None
                or unconditional_guidance_scale == 1.0):
            eps = self.model.apply_model(x, t, cond)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, cond])
            eps_uncond, eps_cond = self.model.apply_model(
                x_in, t_in, c_in).chunk(2)
            eps = eps_uncond + unconditional_guidance_scale * (
                eps_cond - eps_uncond)

        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index],
                         device=device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index],
                            device=device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index],
                             device=device)
        sqrt_1m_at = torch.full((b, 1, 1, 1),
                                self.ddim_sqrt_one_minus_alphas[index],
                                device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_1m_at * eps) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * eps
        noise = sigma_t * torch.randn_like(x) if sigma_t.sum().item() > 0 else 0.0
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
