import torch

def simulate_low_dose(element, noise_level=0.1):
    """
    Simulates low-dose mammography by adding Poisson-Gaussian noise.
    
    Args:
        element (torch.Tensor): Input image tensor (C, H, W) in range [0, 1].
        noise_level (float): Factor to control noise intensity. Higher = more noise.
    
    Returns:
        torch.Tensor: Degraded image tensor.
    """
    if not isinstance(element, torch.Tensor):
        raise TypeError("Input element must be a torch.Tensor")
        
    # 1. Simulate Poisson (Shot) Noise
    img = torch.clamp(element, min=0.0)
    photon_scale = 1000.0 / (noise_level + 1e-6) 
    noisy = torch.poisson(img * photon_scale) / photon_scale
    
    # 2. Add Gaussian (Electronic) Noise
    gaussian_std = noise_level * 0.05 # Smaller component
    gaussian_noise = torch.randn_like(element) * gaussian_std
    
    noisy = noisy + gaussian_noise
    
    # Clamp back to valid range
    noisy = torch.clamp(noisy, 0.0, 1.0)
    
    return noisy
