# ultralytics/nn/modules/hysto_attention.py
# Refined replacement for C2PSA in YOLOv11 for small medical datasets

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HystoAttention"]

class HystoAttention(nn.Module):
    """
    HystoAttention: A Lightweight Attention Block for Hysteroscopy Image Analysis.
    Designed to replace C2PSA in YOLOv11's backbone, optimized for small datasets.

    It combines efficient spatial attention (ESA) with efficient channel attention (ECA)
    in a single path to capture both local and global context without overfitting.

    Args:
        dim (int): Input and output channel dimension.
        reduction (int): Reduction ratio for the ECA module.
    """
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        
        # Efficient Spatial Attention (ESA): Lightweight spatial gating
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 3, padding=1, groups=dim // reduction),
            nn.Sigmoid()
        )
        
        # Efficient Channel Attention (ECA): Fast 1D convolution for channel interaction
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        
        # --- Efficient Spatial Attention (ESA) ---
        spatial_attn = self.spatial_gate(x)
        out = x * spatial_attn
        
        # --- Efficient Channel Attention (ECA) ---
        # ECA expects (B, C, 1) -> we squeeze the last two dims
        channel_attn = self.channel_gate(out.mean(dim=(-2, -1), keepdim=True).squeeze(-1).transpose(-1, -2))
        channel_attn = channel_attn.transpose(-1, -2).unsqueeze(-1)
        out = out * channel_attn
        
        # Residual connection
        return identity + out
