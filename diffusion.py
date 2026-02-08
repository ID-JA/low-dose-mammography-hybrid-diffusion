import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# Helpers
# ==============================================================================

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# ==============================================================================
# U-Net Blocks
# ==============================================================================

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w).permute(0, 1, 3, 2), qkv)
        q = q * self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, -1, h, w)
        return self.to_out(out) + x

# ==============================================================================
# U-Net
# ==============================================================================

class UNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=3,
        cond_channels=3,
        resnet_block_groups=8,
    ):
        super().__init__()
        self.channels = channels
        self.cond_channels = cond_channels # Channels for conditioning (low-dose latent)

        input_channels = channels + cond_channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = ResnetBlock
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                Attention(dim_in) if ind == 1 else nn.Identity(), # Add attn at lower res
                nn.Conv2d(dim_in, dim_out, 4, 2, 1) if not is_last else nn.Conv2d(dim_in, dim_out, 3, 1, 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                block_klass(dim_out + dim_out, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                Attention(dim_out) if ind == num_resolutions - 2 else nn.Identity(),
                nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1) if not is_last else nn.Conv2d(dim_out, dim_in, 3, 1, 1)
            ]))

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * dim_mults[0], dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, cond):
        x = torch.cat((x, cond), dim=1) # Concatenate noisy input + condition
        
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x) # Skip connection
            x = block2(x, t)
            x = attn(x)
            h.append(x) # Skip connection
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

# ==============================================================================
# Latent Diffusion Manager
# ==============================================================================

class LatentDiffusion(nn.Module):
    def __init__(self, vqvae, unet, timesteps=1000, sampling_timesteps=None):
        super().__init__()
        self.vqvae = vqvae
        self.unet = unet
        self.channels = unet.channels

        # Diffusion parameters
        self.timesteps = timesteps
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # DDIM/Fast sampling support (TODO)

        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def prepare_latents(self, x):
        """
        Extract latents from VQVAE and stack them for U-Net.
        This adapts the hierarchical VQ-VAE outputs to a single tensor.
        """
        with torch.no_grad():
            encoder_outputs = self.vqvae.encode(x)
        
        # encoder_outputs contains [L0, L1, L2] where L0 is largest, L2 smallest.
        # We upsample everything to L0 size and concat.
        
        base_h, base_w = encoder_outputs[0].shape[-2:]
        stacked = [encoder_outputs[0]]
        
        for i in range(1, len(encoder_outputs)):
            feat = encoder_outputs[i]
            feat_up = F.interpolate(feat, size=(base_h, base_w), mode='nearest')
            stacked.append(feat_up)
            
        return torch.cat(stacked, dim=1)

    def restore_latents_from_stack(self, stacked_latent, batch_size):
        """
        Unstack the single tensor back to hierarchical lists for VQVAE decoding.
        NB: We need to know original shapes.
        """
        # We assume 3 levels with 64 channels each = 192 channels total.
        # Ideally this should be dynamic, but for now hardcoded to match VQVAE config.
        # VQVAE config: hidden_channels=64.
        
        chunk_size = 64
        chunks = torch.split(stacked_latent, chunk_size, dim=1)
        
        level_latents = []
        # Level 0 is native resolution
        level_latents.append(chunks[0])
        
        # We need to know the downscaling factors to downsample back.
        # VQVAE scaling_rates=[4, 2, 2].
        # L0 shape (B, 64, H, W)
        # L1 shape (B, 64, H/2, W/2) (Scaling rate 2 relative to L0? No, L1 is downscaled by 4 relative to input, L0 by 4. Wait.
        # In VQVAE.build: 
        #   Enc0: scaling_rates[0] = 4.  -> L0 = Input / 4.
        #   Enc1: scaling_rates[1] = 2.  -> L1 = L0 / 2.
        #   Enc2: scaling_rates[2] = 2.  -> L2 = L1 / 2.
        
        l0_h, l0_w = chunks[0].shape[-2:]
        
        # L1
        l1_h, l1_w = l0_h // 2, l0_w // 2
        l1 = F.interpolate(chunks[1], size=(l1_h, l1_w), mode='area') # Area for downsampling?
        level_latents.append(l1)
        
        # L2
        l2_h, l2_w = l1_h // 2, l1_w // 2
        l2 = F.interpolate(chunks[2], size=(l2_h, l2_w), mode='area')
        level_latents.append(l2)
        
        return level_latents

    def get_loss(self, x_clean, x_deg):
        """
        Training step.
        """
        b = x_clean.shape[0]
        device = x_clean.device
        
        z_clean = self.prepare_latents(x_clean) # (B, 192, H, W)
        z_deg = self.prepare_latents(x_deg)     # (B, 192, H, W)
        
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        noise = torch.randn_like(z_clean)
        
        # Forward diffusion
        # x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1-alpha_cumprod) * noise
        sqrt_alphas = self.sqrt_alphas_cumprod[t].reshape(b, 1, 1, 1)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t].reshape(b, 1, 1, 1)
        
        x_noisy = sqrt_alphas * z_clean + sqrt_one_minus_alphas * noise
        
        # Predict noise
        noise_pred = self.unet(x_noisy, t, cond=z_deg)
        
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, x_deg):
        """
        Inference: Generate restored image from low-dose input.
        """
        b = x_deg.shape[0]
        device = x_deg.device
        
        z_deg = self.prepare_latents(x_deg)
        shape = z_deg.shape
        
        # Start from random noise
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            
            noise_pred = self.unet(img, t, cond=z_deg)
            
            # Update img using DDPM formula
            # x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_bar) * eps) + sigma * z
            
            beta = 1.0 - (self.sqrt_alphas_cumprod[i] ** 2 / (self.sqrt_alphas_cumprod[i-1] ** 2 if i > 0 else torch.tensor(1.0).to(device)))
            # Actually simpler to use the betas array I defined in init if accessible
            # Reconstructing betas from alphas_cumprod for safety
            
            alpha_cumprod = self.sqrt_alphas_cumprod[i] ** 2
            alpha_cumprod_prev = self.sqrt_alphas_cumprod[i-1] ** 2 if i > 0 else torch.tensor(1.0).to(device)
            alpha = alpha_cumprod / alpha_cumprod_prev
            beta = 1.0 - alpha
            
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[i]
            
            # Mean
            pred_mean = (1 / torch.sqrt(alpha)) * (img - (beta / sqrt_one_minus_alpha_cumprod) * noise_pred)
            
            if i > 0:
                noise = torch.randn_like(img)
                sigma = torch.sqrt(beta) # Simple sigma choice
                img = pred_mean + sigma * noise
            else:
                img = pred_mean
                
        # Decode
        latents_list = self.restore_latents_from_stack(img, b)
        return self.vqvae.decode_latents(latents_list)

from tqdm import tqdm
