import torch
import torch.nn as nn
import torch.nn.functional as F

class SeeLarge(nn.Module):
    """
    Peripheral Perception: Large-Kernel Static Convolution.
    Decomposed into Horizontal (1xK) and Vertical (Kx1) depth-wise convolutions
    to capture long-range dependencies with linear complexity.
    """
    def __init__(self, dim, kernel_size=11):
        super().__init__()
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        
        # Horizontal Depth-wise
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, pad), 
                              groups=dim, bias=False)
        # Vertical Depth-wise
        self.dw_v = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(pad, 0), 
                              groups=dim, bias=False)
        
    def forward(self, x):
        # Parallel processing mimics peripheral vision gathering
        return self.dw_h(x) + self.dw_v(x)

class FocusSmall(nn.Module):
    """
    Foveal Aggregation: Small-Kernel Dynamic Fusion.
    Uses a lightweight dynamic mechanism to weight features based on 
    local importance, mimicking the fovea's sharp focus.
    """
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.pad = kernel_size // 2
        
        # Feature generation for dynamic weights
        # We effectively generate a spatial attention mask modulated by a small kernel
        self.generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        
        # Small depth-wise aggregation
        self.dw_small = nn.Conv2d(dim, dim, kernel_size, padding=self.pad, 
                                  groups=dim, bias=False)

    def forward(self, x):
        # 1. Generate dynamic content-aware weights (Global context)
        attn = self.generator(x)
        
        # 2. Apply small kernel aggregation (Local detail)
        out = self.dw_small(x)
        
        # 3. Dynamic Fusion
        return out * attn

class LSBlock(nn.Module):
    """
    The Official LS-Block (CVPR 2025).
    Pipeline: Input -> 1x1 Conv -> SeeLarge -> FocusSmall -> 1x1 Conv -> Output
    """
    def __init__(self, dim, large_kernel=11, small_kernel=3, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        
        # 1. Input Projection (Channel Mixing)
        self.proj_in = nn.Conv2d(dim, dim, 1)
        self.norm1 = nn.GroupNorm(1, dim) # LayerNorm equivalent
        
        # 2. The Core LS-Module
        self.see_large = SeeLarge(dim, kernel_size=large_kernel)
        self.focus_small = FocusSmall(dim, kernel_size=small_kernel)
        
        # 3. Output Projection (Channel Mixing)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        
        # 4. Inverted Bottleneck / MLP (optional, often integrated)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        
        self.drop_path = nn.Identity() # Placeholder for DropPath

    def forward(self, x):
        shortcut = x
        
        # --- LS Convolution Stage ---
        x = self.norm1(x)
        x = self.proj_in(x)
        
        # Large Kernel Perception (Static)
        x_large = self.see_large(x)
        
        # Small Kernel Aggregation (Dynamic)
        x_mixed = self.focus_small(x_large)
        
        x = self.proj_out(x_mixed)
        x = shortcut + self.drop_path(x)
        
        # --- MLP Stage ---
        x = x + self.drop_path(self.mlp(x))
        
        return x

# --- YOLO Integration Module (LSC2f) ---
class LSC2f(nn.Module):
    """
    LSC2f: Replaces YOLO C2f with LS-Blocks for SOTA Detection.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e) 
        self.cv1 = nn.Conv2d(c1, c2, 1, 1) # Simplified Conv wrapper
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1) 
        
        # The innovation: Replacing Bottleneck with LSBlock
        self.m = nn.ModuleList(LSBlock(self.c) for _ in range(n))

    def forward(self, x):
        # Split, Pass through LSBlocks, Concat
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

if __name__ == "__main__":
    # Test the block
    x = torch.randn(2, 64, 64, 64)
    model = LSBlock(64)
    print(f"Output Shape: {model(x).shape}")
    
    # Parameter check
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params}")
