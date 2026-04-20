# ultralytics/nn/modules/dmca.py
# Replace C2PSA in YOLOv11 backbone with this block for HS-CMU dataset.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DMCA", "MALAttention", "SeaAttention"]

class MALAttention(nn.Module):
    """Magnitude-Aware Linear Attention (MALA) from ICCV 2025.
    Corrects magnitude neglect in linear attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Apply magnitude correction
        q_mag = torch.norm(q, dim=-1, keepdim=True)
        q = q * torch.sigmoid(q_mag)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class SeaAttention(nn.Module):
    """Squeeze Axial Attention (Sea_Attention) from ICLR 2023.
    Efficiently extracts global semantics via axial attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Horizontal axial attention
        x_h = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x_h = self._axial_attn(x_h)
        x_h = x_h.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # Vertical axial attention
        x_v = x.permute(0, 3, 2, 1).reshape(B, W*H, C)
        x_v = self._axial_attn(x_v)
        x_v = x_v.reshape(B, W, H, C).permute(0, 3, 2, 1)
        return x_h + x_v
        
    def _axial_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class DMCA(nn.Module):
    """Dynamic Multi-Scale Context Attention (DMCA) Block.
    Replaces C2PSA in YOLOv11 backbone. Designed for HS-CMU hysteroscopy dataset.
    
    Args:
        dim (int): Input and output channel dimension.
        expansion_factor (int): Factor to expand the hidden dimension.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, dim, expansion_factor=2, num_heads=8):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.dim = dim
        
        # Path 1: Multi-Scale Convolution
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.dwconv5 = nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2, groups=hidden_dim)
        self.dwconv7 = nn.Conv2d(hidden_dim, hidden_dim, 7, padding=3, groups=hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1)
        
        # Path 2: Frequency-Aware Modulation
        self.conv_freq1 = nn.Conv2d(dim, hidden_dim, 1)
        self.conv_freq_ll = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 16, hidden_dim, 1),
            nn.Sigmoid()
        )
        self.conv_freq2 = nn.Conv2d(hidden_dim, dim, 1)
        
        # Path 3: Efficient Global Attention (MALA + Sea)
        self.conv_attn1 = nn.Conv2d(dim, dim, 1)
        self.mala = MALAttention(dim, num_heads=num_heads)
        self.sea = SeaAttention(dim, num_heads=num_heads)
        self.conv_attn2 = nn.Conv2d(dim, dim, 1)
        
        # Fusion
        self.fusion = nn.Conv2d(dim * 3, dim, 1)
        
    def forward(self, x):
        identity = x
        
        # Path 1: Multi-Scale Convolution
        p1 = self.conv1(x)
        p1 = self.dwconv3(p1) + self.dwconv5(p1) + self.dwconv7(p1)
        p1 = self.conv2(p1)
        
        # Path 2: Frequency-Aware Modulation
        p2 = self.conv_freq1(x)
        # Simplified wavelet: down-up sampling
        p2_ll = F.avg_pool2d(p2, 2)
        p2_ll = self.conv_freq_ll(p2_ll)
        p2_up = F.interpolate(p2_ll, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = p2 * self.se(p2_up)
        p2 = self.conv_freq2(p2)
        
        # Path 3: Efficient Global Attention
        p3 = self.conv_attn1(x)
        B, C, H, W = p3.shape
        p3_flat = p3.flatten(2).transpose(1, 2)
        p3_flat = self.mala(p3_flat)
        p3 = p3_flat.transpose(1, 2).reshape(B, C, H, W)
        p3 = self.sea(p3)
        p3 = self.conv_attn2(p3)
        
        # Fusion
        out = torch.cat([p1, p2, p3], dim=1)
        out = self.fusion(out)
        
        return identity + out
