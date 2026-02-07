import torch
import torch.nn.functional as F
import random


def gaussian_blur_2d(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    x: [1, H, W] float in [0,1]
    Simple Gaussian blur using separable kernels.
    """
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, device=x.device) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    # Separable conv: first horizontal then vertical
    g1 = g.view(1, 1, 1, -1)  # [out,in,H,W] for conv2d
    g2 = g.view(1, 1, -1, 1)

    x4 = x.unsqueeze(0)  # [B=1, C=1, H, W]
    x4 = F.conv2d(x4, g1, padding=(0, kernel_size // 2))
    x4 = F.conv2d(x4, g2, padding=(kernel_size // 2, 0))
    return x4.squeeze(0)


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(0.0, 1.0)


def down_up_sample(x: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """
    Simulate low-dose / low-quality acquisition by downsampling then upsampling.
    scale < 1.0 makes it lower resolution then back to original size.
    """
    x4 = x.unsqueeze(0)
    H, W = x.shape[-2], x.shape[-1]
    h2, w2 = int(H * scale), int(W * scale)

    x_low = F.interpolate(x4, size=(h2, w2), mode="bilinear", align_corners=False)
    x_back = F.interpolate(x_low, size=(H, W), mode="bilinear", align_corners=False)
    return x_back.squeeze(0)


def degrade_mammogram(x: torch.Tensor) -> torch.Tensor:
    """
    x: [1,H,W] float in [0,1] (clean)
    returns: [1,H,W] float in [0,1] (simulated low-dose)
    """
    y = x.clone()

    if random.random() < 0.9:
        k = random.choice([3, 5, 7])
        sigma = random.uniform(0.6, 1.6)
        y = gaussian_blur_2d(y, kernel_size=k, sigma=sigma)

    if random.random() < 0.8:
        scale = random.uniform(0.4, 0.8)
        y = down_up_sample(y, scale=scale)

    noise_sigma = random.uniform(0.01, 0.06)
    y = add_gaussian_noise(y, sigma=noise_sigma)

    return y.clamp(0.0, 1.0)
