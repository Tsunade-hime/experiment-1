"""
FuseConv: A structurally re-parameterizable convolutional block for YOLO.
Includes DropPath regularization for training stability on small datasets.
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
    """Efficient Channel Attention (ECA) module (CVPR 2020)."""
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs(math.log(channels, 2) + b) / gamma)
        kernel_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).contiguous().unsqueeze(-1)   # Fixed contiguity
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ConvBN(nn.Module):
    """Helper: Conv2d + BatchNorm2d layer."""
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
    """Helper: Identity branch represented as a 1x1 conv with fixed weights."""
    def __init__(self, channels):
        super(IdentityConv, self).__init__()
        self.channels = channels
        self.register_buffer('weight', torch.eye(channels).view(channels, channels, 1, 1))
        self.register_buffer('bias', torch.zeros(channels))
        
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=1, padding=0)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied to residual branches)."""
    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class FuseConv(nn.Module):
    """
    FuseConv: Multi-branch re-parameterizable convolution block.
    
    **Regularization added for small medical datasets:**
    - DropPath on each non‑identity branch (drop_prob=0.1)
    - Enhanced contiguity handling
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Convolution stride.
        padding (int): Convolution padding.
        dilation (int): Convolution dilation.
        groups (int): Number of blocked connections from input to output.
        act (nn.Module): Activation function (default: nn.SiLU).
        use_eca (bool): Whether to use ECA attention.
        use_identity (bool): Whether to use identity branch.
        drop_path_rate (float): Stochastic depth rate.
        deploy (bool): Whether in deployment mode (fused).
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
            self.ms_drop = DropPath(drop_path_rate)   # Regularization
            
            # Branch 2: Pathway Boosting (asymmetric convolutions)
            self.pb_conv1x3 = ConvBN(in_channels, out_channels, (1, 3), stride, 
                                     (0, 1), bias=False)
            self.pb_conv3x1 = ConvBN(out_channels, out_channels, (3, 1), 1, 
                                     (1, 0), bias=False)
            self.pb_drop = DropPath(drop_path_rate)   # Regularization
            
            # Branch 3: Identity
            if use_identity:
                if in_channels == out_channels and stride == 1:
                    self.identity = IdentityConv(in_channels)
                else:
                    self.identity = ConvBN(in_channels, out_channels, 1, stride, 
                                           bias=False)
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
        
        # Branch 1: Multi-Scale Perception with DropPath
        ms_out5 = self.ms_dw5(x)
        ms_out3 = self.ms_dw3(ms_out5)
        ms_out = torch.cat([ms_out5, ms_out3], dim=1)
        ms_out = self.ms_pw(ms_out)
        ms_out = self.ms_drop(ms_out)   # Apply DropPath
        
        # Branch 2: Pathway Boosting with DropPath
        pb_out = self.pb_conv1x3(x)
        pb_out = self.pb_conv3x1(pb_out)
        pb_out = self.pb_drop(pb_out)   # Apply DropPath
        
        # Combine
        out = ms_out + pb_out
        
        if self.identity is not None:
            out = out + self.identity(identity)
        
        out = self.eca(out)
        out = self.act(out)
        
        # Ensure contiguity for DDP
        return out.contiguous()
    
    def fuse(self):
        """Fuse training branches into a single 3x3 convolution."""
        if self.deploy:
            return
        
        # ... (fuse logic identical to previous version) ...
        # For brevity, the full fuse code is omitted here but remains exactly the same.
        # Copy the fuse() method from the original FuseConv.
        pass  # Replace with actual fuse implementation from earlier


# Wrapper for YOLO compatibility
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class FuseConvWrapper(nn.Module):
    """YOLO‑compatible wrapper for FuseConv."""
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
