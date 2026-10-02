import numpy as np
import torch
import torch.fft

def project_divergence_free(u, v):
    # u, v shape: (B, H, W) where H=32, W=64
    B, H, W = u.shape
    
    # Wave numbers
    kx = torch.fft.fftfreq(W, d=1.0).view(1, 1, W)
    ky = torch.fft.fftfreq(H, d=1.0).view(1, H, 1)
    
    u_f = torch.fft.fft2(u)
    v_f = torch.fft.fft2(v)
    
    # k dot v
    k_dot_v = kx * u_f + ky * v_f
    k_sq = kx**2 + ky**2
    k_sq = torch.where(k_sq == 0, torch.ones_like(k_sq), k_sq) # avoid div by zero
    
    # Project
    u_f_proj = u_f - (k_dot_v * kx) / k_sq
    v_f_proj = v_f - (k_dot_v * ky) / k_sq
    
    # Zero out the mean (k=0) projection modification since k_dot_v=0 there anyway
    
    u_proj = torch.fft.ifft2(u_f_proj).real
    v_proj = torch.fft.ifft2(v_f_proj).real
    return u_proj, v_proj
