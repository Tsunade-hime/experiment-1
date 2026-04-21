import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyModulation(nn.Module):
    """
    Frequency-Dynamic Modulation Branch.
    Uses FFT in float32 for cuFFT compatibility, while keeping trainable
    layers in the model's native dtype (e.g., float16).
    """
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.dim = dim
        
        # Learnable frequency filter: MLP on global spectral energy
        self.filter_mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )
        
        # Channel attention (SE-like)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        original_dtype = x.dtype
        
        # --- Frequency domain (float32 for cuFFT) ---
        x_f32 = x.float()
        x_fft = torch.fft.rfft2(x_f32, norm='ortho')
        magnitude = torch.abs(x_fft)
        
        # Global average over spatial frequencies
        pooled = magnitude.mean(dim=(-2, -1))
        
        # MLP expects original dtype (float16)
        pooled_orig = pooled.to(original_dtype)
        filter_weights = self.filter_mlp(pooled_orig).view(B, C, 1, 1)
        
        # Apply filter to magnitude (float32)
        filter_weights_f32 = filter_weights.to(torch.float32)
        magnitude_filtered = magnitude * filter_weights_f32
        
        # Inverse FFT
        phase = torch.angle(x_fft)
        x_fft_filtered = magnitude_filtered * torch.exp(1j * phase)
        x_spatial_f32 = torch.fft.irfft2(x_fft_filtered, s=(H, W), norm='ortho')
        x_spatial = x_spatial_f32.to(original_dtype)
        
        # Channel attention
        ca = self.channel_att(x_spatial)
        return x_spatial * ca


class TokenStatisticsSelfAttention(nn.Module):
    """
    Token Statistics Self-Attention (TSSA).
    O(N) complexity via variance‑based similarity.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        var_v = v.var(dim=-1, keepdim=True)   # (B, num_heads, N, 1)
        var_k = k.var(dim=-1, keepdim=True)
        
        attn = (var_v @ var_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class SpatialQuantizedRouter(nn.Module):
    """
    Spatial branch with learnable router for binary attention.
    """
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5

    def binary_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        
        q = self.q_conv(x).view(B, C, N)
        k = self.k_conv(x).view(B, C, N)
        v = self.v_conv(x).view(B, C, N)
        
        q_bin = torch.sign(q)
        k_bin = torch.sign(k)
        
        attn = torch.bmm(q_bin.transpose(1, 2), k_bin) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)
        return self.proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_local = self.dwconv(x)
        p = self.router(x)  # (B, 1)
        
        if self.training:
            out_bin = self.binary_attention(x)
            alpha = p.view(-1, 1, 1, 1)
            out = (1 - alpha) * x_local + alpha * out_bin
        else:
            use_binary = (p > 0.5).float().mean() > 0.5
            out = self.binary_attention(x) if use_binary else x_local
        return out


class DFSHA(nn.Module):
    """
    Dynamic Frequency-Statistical Hybrid Attention (DFSHA)
    Drop‑in replacement for C2PSA in YOLOv11.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        
        self.m = nn.ModuleList()
        for _ in range(n):
            self.m.append(nn.ModuleList([
                FrequencyModulation(self.c),
                TokenStatisticsSelfAttention(self.c),
                SpatialQuantizedRouter(self.c)
            ]))
        
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        attn_input = y[-1]
        attn_outputs = []
        for block in self.m:
            freq_out = block[0](attn_input)
            stat_out = block[1](attn_input)
            spat_out = block[2](attn_input)
            attn_outputs.append(freq_out + stat_out + spat_out)
        
        y.extend(attn_outputs)
        y = torch.cat(y, dim=1)
        out = self.cv2(y)
        return x + out if self.add else out
