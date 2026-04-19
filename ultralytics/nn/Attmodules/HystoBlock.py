# ultralytics/nn/modules/hysto_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MKDConv", "WTConv", "HystoBlock"]

class MKDConv(nn.Module):
    """Multi-Kernel Depthwise Convolution"""
    def __init__(self, dim, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim) for k in kernel_sizes
        ])

    def forward(self, x):
        return sum(conv(x) for conv in self.convs)


class WTConv(nn.Module):
    """
    Wavelet Transform Convolution with boundary handling for odd dimensions.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        _, _, h, w = x.shape
        # Pad to even dimensions if needed
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Downsample (wavelet LL approximation)
        x_ll = F.avg_pool2d(x, 2)
        x_ll_conv = self.conv(x_ll)
        # Upsample back
        out = F.interpolate(x_ll_conv, scale_factor=2, mode='bilinear', align_corners=False)

        # Crop to original size
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


class HystoBlock(nn.Module):
    """HystoBlock: Multi-Scale Context-Aware Convolution Block"""
    def __init__(self, dim, expansion_factor=2):
        super().__init__()
        hidden_dim = dim * expansion_factor

        # Multi-Kernel Path
        self.mk_conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.mk_dwconv = MKDConv(hidden_dim)
        self.mk_conv2 = nn.Conv2d(hidden_dim, dim, 1)

        # Wavelet Context Path
        self.wt_conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.wt_dwconv = WTConv(hidden_dim)
        self.wt_conv2 = nn.Conv2d(hidden_dim, dim, 1)

        # Feature Enhancement Block (FEB)
        self.feb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mk_out = self.mk_conv1(x)
        mk_out = self.mk_dwconv(mk_out)
        mk_out = self.mk_conv2(mk_out)

        wt_out = self.wt_conv1(x)
        wt_out = self.wt_dwconv(wt_out)
        wt_out = self.wt_conv2(wt_out)

        combined = torch.cat([mk_out, wt_out], dim=1)
        attention = self.feb(combined)
        att_mk, att_wt = attention.chunk(2, dim=1)
        fused = mk_out * att_mk + wt_out * att_wt

        return x + fused
