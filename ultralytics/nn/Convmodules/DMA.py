import torch
import torch.nn as nn
import math

# --------------------- 1. Basic Convolution Block ---------------------
class Conv(nn.Module):
    """Standard convolution block with Conv2d + BatchNorm + SiLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# --------------------- 2. Heterogeneous Kernel Fusion (HKF) ---------------------
class HeteroKernelFusion(nn.Module):
    """
    Parallel depthwise convolutions with different kernel sizes (3x3, 5x5, 7x7).
    Outputs are fused and weighted by a channel attention mechanism.
    """
    def __init__(self, channels, kernels=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, k, padding=k//2, groups=channels, bias=False) 
            for k in kernels
        ])
        # Channel attention to weight the contribution of each branch
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * len(kernels), channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * len(kernels), 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply multi-scale depthwise convolutions
        feats = [conv(x) for conv in self.convs]
        # Concatenate along channel dimension for attention
        concat_feats = torch.cat(feats, dim=1)
        # Generate channel weights
        weights = self.se(concat_feats)
        # Split weights to match each branch
        split_weights = torch.chunk(weights, len(self.convs), dim=1)
        # Fuse weighted features
        out = sum(f * w for f, w in zip(feats, split_weights))
        return out

# --------------------- 3. Contract-and-Broadcast Self-Attention (CBSA) ---------------------
class CBSA(nn.Module):
    """
    Contract-and-Broadcast Self-Attention.
    Linear complexity attention that works on a small set of representative tokens.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, sr_ratio=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Spatial reduction for K, V (the "contract" part)
        self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C), N = H*W

        # Generate Query from full feature map
        q = self.q(x_flat).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Generate Key & Value from contracted (spatially reduced) feature map
        x_sr = self.sr(x)  # (B, C, H/sr, W/sr)
        x_sr = x_sr.flatten(2).transpose(1, 2)  # (B, N_sr, C), N_sr = (H*W) // (sr**2)
        x_sr = self.norm(x_sr)

        kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, num_heads, N_sr, head_dim)

        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Weighted sum (the "broadcast" part)
        x_weighted = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        # Project back to original shape
        out = self.proj(x_weighted).transpose(1, 2).reshape(B, C, H, W)
        return out

# --------------------- 4. Partial Attention-based Convolution (PATConv) ---------------------
class PATConv(nn.Module):
    """
    Partial Attention-based Convolution.
    Applies convolution and a lightweight attention only to a subset of channels.
    """
    def __init__(self, dim, expansion_factor=2, partial_ratio=0.25):
        super().__init__()
        self.partial_channels = int(dim * partial_ratio)
        self.identity_channels = dim - self.partial_channels
        hidden_dim = int(self.partial_channels * expansion_factor)

        # Convolution path for partial channels
        self.conv1 = Conv(self.partial_channels, hidden_dim, 3, stride=1, groups=self.partial_channels)
        self.conv2 = Conv(hidden_dim, self.partial_channels, 1)

        # Lightweight attention path (Squeeze-and-Excitation)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.partial_channels, max(1, self.partial_channels // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, self.partial_channels // 4), self.partial_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Split input into active (to be processed) and passive (identity) parts
        x_active, x_passive = torch.split(x, [self.partial_channels, self.identity_channels], dim=1)

        # Apply convolutions
        x_active_conv = self.conv1(x_active)
        x_active_conv = self.conv2(x_active_conv)

        # Apply lightweight attention
        x_active_attn = x_active_conv * self.se(x_active_conv)

        # Fuse and output
        out = torch.cat([x_active_attn, x_passive], dim=1)
        return out

# --------------------- 5. Dynamic Multi-Scale Attention (DMA) Block ---------------------
class DMA(nn.Module):
    """
    Dynamic Multi-Scale Attention Block.
    Replaces C3k2 module in YOLO11n.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, use_cbsa=True):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of repeated sub-blocks (Bottlenecks).
            shortcut (bool): Whether to use residual connections.
            g (int): Number of groups for group convolution.
            e (float): Expansion ratio for hidden channels.
            use_cbsa (bool): Whether to enable the CBSA attention block.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        
        # Create n bottleneck blocks, each containing HKF and PATConv
        self.m = nn.ModuleList([
            DMA_Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) 
            for _ in range(n)
        ])
        
        # Optionally add CBSA for global context
        self.cbsa = CBSA(self.c * n) if use_cbsa else nn.Identity()

    def forward(self, x):
        # Split input and process through bottlenecks
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        
        # Apply CBSA to the concatenated feature maps
        if isinstance(self.cbsa, CBSA):
            bottleneck_feats = torch.cat(y[2:], dim=1)
            attn_feats = self.cbsa(bottleneck_feats)
            y[2:] = torch.chunk(attn_feats, len(self.m), dim=1)
        
        return self.cv2(torch.cat(y, 1))

# --------------------- 6. DMA Bottleneck (Internal Component) ---------------------
class DMA_Bottleneck(nn.Module):
    """Standard bottleneck block used inside the DMA module."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        
        # Use Heterogeneous Kernel Fusion (HKF) in the bottleneck
        self.hkf = HeteroKernelFusion(c_)
        
        # Use Partial Attention-based Convolution (PATConv) for the second conv
        self.patconv = PATConv(c_)
        
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.patconv(self.hkf(self.cv1(x))) if self.add else self.patconv(self.hkf(self.cv1(x)))
