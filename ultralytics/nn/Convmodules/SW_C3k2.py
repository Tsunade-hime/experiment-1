import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2, Bottleneck

class ShiftwiseConv(nn.Module):
    """
    Shiftwise Convolution (CVPR 2025): 
    Replaces large kernel convs with efficient channel shifts + 1x1 Conv.
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.c1 = c1
        # 1x1 Fusion Conv (mixes shifted features)
        self.fuse = Conv(c1, c2, k=1, s=1, p=0, g=g, act=act)
        # 3x3 Depthwise Extraction (standard local texture)
        self.extract = Conv(c2, c2, k=k, s=s, p=p, g=c2, act=act) 

    def forward(self, x):
        n, c, h, w = x.size()
        g = c // 5  # Split channels into 5 groups (Center, Up, Down, Left, Right)
        
        out = torch.zeros_like(x)
        
        # Group 1: Center (Identity)
        out[:, 0:g, :, :] = x[:, 0:g, :, :]
        
        # Group 2: Shift Up
        out[:, g:2*g, :-1, :] = x[:, g:2*g, 1:, :]
        
        # Group 3: Shift Down
        out[:, 2*g:3*g, 1:, :] = x[:, 2*g:3*g, :-1, :]
        
        # Group 4: Shift Left
        out[:, 3*g:4*g, :, :-1] = x[:, 3*g:4*g, :, 1:]
        
        # Group 5: Shift Right
        out[:, 4*g:, :, 1:] = x[:, 4*g:, :, :-1]

        # Fuse global context with local texture
        return self.extract(self.fuse(out))

class SW_Bottleneck(Bottleneck):
    """
    Standard Bottleneck but replaces the 3x3 Conv with ShiftwiseConv.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        # Overwrite the second convolution (cv2) with ShiftwiseConv
        self.cv2 = ShiftwiseConv(c_, c2, k=k[1], s=1, g=g)

class SW_C3k2(C3k2):
    """
    YOLO11 C3k2 Block using Shiftwise Bottlenecks.
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        # Replace standard Bottlenecks with SW_Bottleneck
        self.m = nn.ModuleList(
            SW_Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )
