"""
FuseConv: Multi-branch re-parameterizable convolution block.
Fully DDP-compatible – all tensors are forced contiguous.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = [
    # ... existing modules ...
    'FuseConv',
    'FuseConvWrapper',
]

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) – DDP‑Safe Version.
    Gradient stride warnings are eliminated by forcing contiguous memory layout.
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs(math.log(channels, 2) + b) / gamma)
        kernel_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # 🔧 Force weight tensor to be contiguous (eliminates DDP warning)
        self.conv.weight.data = self.conv.weight.data.contiguous()
        
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)                          # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)           # [B, 1, C]
        y = self.conv(y)                              # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)         # [B, C, 1, 1]
        y = y.contiguous()                            # 🔧 Ensure contiguity
        y = self.sigmoid(y)
        return (x * y).contiguous()                   # 🔧 Final output contiguous


class ConvBN(nn.Module):
    """Conv2d + BatchNorm2d with fusion support."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.bn(self.conv(x))
    
    def get_fused_weights(self):
        conv_weight = self.conv.weight.data
        bn_weight = self.bn.weight.data
        bn_bias = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var = self.bn.running_var
        bn_eps = self.bn.eps
        
        if bn_weight is not None:
            std = torch.sqrt(bn_running_var + bn_eps)
            fused_weight = conv_weight * (bn_weight / std).reshape(-1, 1, 1, 1)
        else:
            fused_weight = conv_weight
            
        if self.conv.bias is not None:
            conv_bias = self.conv.bias.data
        else:
            conv_bias = torch.zeros(self.conv.out_channels, device=conv_weight.device)
        if bn_weight is not None:
            fused_bias = bn_bias + (conv_bias - bn_running_mean) * (bn_weight / std)
        else:
            fused_bias = conv_bias
            
        return fused_weight, fused_bias


class IdentityConv(nn.Module):
    """Identity branch as a 1x1 conv with fixed weights."""
    def __init__(self, channels):
        super(IdentityConv, self).__init__()
        self.channels = channels
        self.register_buffer('weight', torch.eye(channels).view(channels, channels, 1, 1))
        self.register_buffer('bias', torch.zeros(channels))
        
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=1, padding=0)


class DropPath(nn.Module):
    """Stochastic Depth per sample."""
    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output.contiguous()


class FuseConv(nn.Module):
    """
    FuseConv: Multi-branch re-parameterizable convolution.
    DDP warnings eliminated; includes DropPath regularization.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, act=nn.SiLU(),
                 use_eca=True, use_identity=True, drop_path_rate=0.1, deploy=False):
        super(FuseConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.act = act
        self.use_eca = use_eca
        self.use_identity = use_identity
        self.deploy = deploy
        
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, groups, bias=True)
            self.bn = nn.Identity()
        else:
            # Branch 1: Multi-Scale Perception
            self.ms_dw5 = ConvBN(in_channels, in_channels, 5, stride, 2,
                                 groups=in_channels, bias=False)
            self.ms_dw3 = ConvBN(in_channels, in_channels, 3, 1, 1,
                                 groups=in_channels, bias=False)
            self.ms_pw = ConvBN(in_channels * 2, out_channels, 1, bias=False)
            self.ms_drop = DropPath(drop_path_rate)
            
            # Branch 2: Pathway Boosting
            self.pb_conv1x3 = ConvBN(in_channels, out_channels, (1, 3), stride,
                                     (0, 1), bias=False)
            self.pb_conv3x1 = ConvBN(out_channels, out_channels, (3, 1), 1,
                                     (1, 0), bias=False)
            self.pb_drop = DropPath(drop_path_rate)
            
            # Branch 3: Identity
            if use_identity:
                if in_channels == out_channels and stride == 1:
                    self.identity = IdentityConv(in_channels)
                else:
                    self.identity = ConvBN(in_channels, out_channels, 1, stride, bias=False)
            else:
                self.identity = None
            
            # Attention
            self.eca = ECA(out_channels) if use_eca else nn.Identity()
            
            self.act = act if act is not None else nn.Identity()
            self.fused_conv = None
    
    def forward(self, x):
        if self.deploy:
            return self.act(self.fused_conv(x))
        
        identity = x
        
        # Branch 1: Multi-Scale
        ms_out5 = self.ms_dw5(x)
        ms_out3 = self.ms_dw3(ms_out5)
        ms_out = torch.cat([ms_out5, ms_out3], dim=1)
        ms_out = self.ms_pw(ms_out)
        ms_out = self.ms_drop(ms_out)
        
        # Branch 2: Pathway Boosting
        pb_out = self.pb_conv1x3(x)
        pb_out = self.pb_conv3x1(pb_out)
        pb_out = self.pb_drop(pb_out)
        
        # Combine
        out = ms_out + pb_out
        
        if self.identity is not None:
            out = out + self.identity(identity)
        
        out = self.eca(out)
        out = self.act(out)
        
        return out.contiguous()
    
    def fuse(self):
        """Fuse branches into a single 3x3 convolution."""
        if self.deploy:
            return
        
        ms_dw5_w, ms_dw5_b = self.ms_dw5.get_fused_weights()
        ms_dw3_w, ms_dw3_b = self.ms_dw3.get_fused_weights()
        ms_pw_w, ms_pw_b = self.ms_pw.get_fused_weights()
        
        pb_conv1x3_w, pb_conv1x3_b = self.pb_conv1x3.get_fused_weights()
        pb_conv3x1_w, pb_conv3x1_b = self.pb_conv3x1.get_fused_weights()
        
        if self.identity is not None:
            if isinstance(self.identity, ConvBN):
                id_w, id_b = self.identity.get_fused_weights()
            else:
                id_w = self.identity.weight
                id_b = self.identity.bias
        else:
            id_w = torch.zeros(self.out_channels, self.in_channels, 1, 1,
                              device=ms_dw5_w.device)
            id_b = torch.zeros(self.out_channels, device=ms_dw5_w.device)
        
        fused_w = torch.zeros(self.out_channels, self.in_channels, 3, 3,
                             device=ms_dw5_w.device)
        fused_b = torch.zeros(self.out_channels, device=ms_dw5_w.device)
        
        def pad_to_3x3(kernel):
            kh, kw = kernel.shape[-2:]
            if kh == 3 and kw == 3:
                return kernel
            pad_h = (3 - kh) // 2
            pad_w = (3 - kw) // 2
            return F.pad(kernel, (pad_w, pad_w, pad_h, pad_h))
        
        # Multi-scale branch fusion
        ms_dw5_expanded = torch.zeros(self.out_channels, self.in_channels, 5, 5,
                                      device=ms_dw5_w.device)
        for i in range(self.in_channels):
            ms_dw5_expanded[i, i] = ms_dw5_w[i, 0]
        ms_dw5_3x3 = ms_dw5_expanded[:, :, 1:4, 1:4]
        
        ms_dw3_expanded = torch.zeros(self.out_channels, self.in_channels, 3, 3,
                                      device=ms_dw3_w.device)
        for i in range(self.in_channels):
            ms_dw3_expanded[i, i] = ms_dw3_w[i, 0]
        
        ms_combined = F.conv2d(ms_dw5_3x3 + ms_dw3_expanded, ms_pw_w.permute(1, 0, 2, 3))
        fused_w += ms_combined
        fused_b += ms_pw_b
        
        # Pathway boosting fusion
        pb_1x3_padded = pad_to_3x3(pb_conv1x3_w)
        pb_3x1_padded = pad_to_3x3(pb_conv3x1_w)
        pb_combined = F.conv2d(pb_1x3_padded, pb_3x1_padded.permute(1, 0, 2, 3))
        fused_w += pb_combined
        fused_b += pb_conv3x1_b
        
        # Identity fusion
        id_padded = pad_to_3x3(id_w)
        fused_w += id_padded
        fused_b += id_b
        
        self.fused_conv = nn.Conv2d(self.in_channels, self.out_channels, 3,
                                    self.stride, self.padding, self.dilation,
                                    self.groups, bias=True)
        self.fused_conv.weight.data = fused_w
        self.fused_conv.bias.data = fused_b
        
        # Clean up
        del self.ms_dw5, self.ms_dw3, self.ms_pw, self.ms_drop
        del self.pb_conv1x3, self.pb_conv3x1, self.pb_drop
        if hasattr(self, 'identity'):
            del self.identity
        
        self.deploy = True
        return self


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class FuseConvWrapper(nn.Module):
    """YOLO-compatible wrapper."""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, drop_path_rate=0.1):
        super().__init__()
        self.conv = FuseConv(c1, c2, k, s, autopad(k, p, d), d, g,
                             act=nn.SiLU() if act else nn.Identity(),
                             drop_path_rate=drop_path_rate)
        self.bn = nn.Identity()
        
    def forward(self, x):
        return self.conv(x)
    
    def fuse(self):
        if hasattr(self.conv, 'fuse'):
            self.conv.fuse()
        return self
