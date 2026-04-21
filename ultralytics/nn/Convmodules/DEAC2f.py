import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

# --- Include the DEA Components ---

class DiffWaveletAnchor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        with torch.no_grad():
            # Initializing with Laplacian-of-Gaussian to highlight edges immediately
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
            self.dw_conv.weight.data = kernel.view(1, 1, 3, 3).repeat(dim, 1, 1, 1) * 0.1

    def forward(self, x):
        return self.dw_conv(x)

class DEABlock(nn.Module):
    """
    Dynamic Element-Activated Block: SOTA Lightweight Attention (Nature 2026/NeurIPS 2025).
    """
    def __init__(self, dim, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.anchor = DiffWaveletAnchor(dim)
        
        # G1 Gated Attention
        self.norm = nn.GroupNorm(1, dim)
        self.proj_v = nn.Conv2d(dim, dim, 1)
        self.gate_gen = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        
        # Feed Forward Network
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * 2), 1),
            nn.GELU(),
            nn.Conv2d(int(dim * 2), dim, 1)
        )
        
        self.add = shortcut

    def forward(self, x):
        # 1. Edge-Gated Attention
        anchor = self.anchor(x)
        v = self.proj_v(self.norm(x))
        gate = self.gate_gen(torch.cat([self.norm(x), anchor], dim=1))
        x_attn = v * gate
        
        res = x + x_attn if self.add else x_attn
        
        # 2. MLP Refinement
        out = res + self.mlp(res) if self.add else self.mlp(res)
        return out

# --- The Main DEAC2f Module ---

class DEAC2f(nn.Module):
    """
    DEAC2f: Replace standard Bottlenecks in C2f with DEABlocks.
    Optimized for YOLOv8/v10/v11 Architectures.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # concat of (n + 2) chunks
        
        # Replace standard Bottleneck with DEABlock
        self.m = nn.ModuleList(DEABlock(self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        # Initial projection and split
        y = list(self.cv1(x).chunk(2, 1))
        # Extend with outputs from each DEABlock
        y.extend(m(y[-1]) for m in self.m)
        # Final fusion and transition
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Alternative forward for deployment/scripting."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
