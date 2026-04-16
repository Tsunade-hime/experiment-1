"""
Hybrid Adaptive Upsampler (HAU) for YOLO.
Integrates dynamic kernel generation, offset-based feature enhancement,
and semantic gating for superior upsampling quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HAU(nn.Module):
    """
    Hybrid Adaptive Upsampler (HAU) Module.

    Args:
        in_channels (int): Number of input channels.
        scale_factor (int): Upsampling scale factor (usually 2).
        kernel_size (int): Size of the dynamic reassembly kernel (default: 3).
        deform_groups (int): Number of groups for the deformable convolution (default: 4).
    """
    def __init__(self, in_channels, scale_factor=2, kernel_size=3, deform_groups=4):
        super(HAU, self).__init__()
        self.scale = scale_factor
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.deform_groups = deform_groups

        # --- Path 1: Kernel Prediction for Dynamic Reassembly (Inspired by CARAFE) ---
        # Compresses channel dimension to make kernel prediction efficient
        self.kernel_compressor = nn.Conv2d(in_channels, in_channels // 2, 1)
        # Predicts the weights for the reassembly kernel for each output pixel
        self.kernel_predictor = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 4, scale_factor**2 * kernel_size**2, 1)
        )

        # --- Path 2: Offset Prediction for Deformable Feature Enhancement (Inspired by DySample) ---
        # Predicts 2D offsets for each pixel in the upsampled space
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2 * scale_factor**2 * deform_groups, 1, bias=False)
        )
        # Deformable convolution to enhance features before upsampling
        self.deform_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=1, groups=deform_groups, bias=False)
        # A simple bilinear upsampler for the offset-enhanced features
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        # --- Path 3: Semantic Gating Mechanism (Inspired by SAPA) ---
        # Analyzes the content of the fused features to generate a gating mask
        self.semantic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        # Zero-initialize the last layer of predictors for stable training start
        nn.init.zeros_(self.kernel_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].weight)
        
    def forward(self, x):
        b, c, h, w = x.shape
        target_h, target_w = h * self.scale, w * self.scale

        # --- Path 1: Dynamic Kernel Reassembly (Inspired by CARAFE) ---
        compressed_features = self.kernel_compressor(x)
        # Predict kernel weights for each output pixel
        kernel_weights = self.kernel_predictor(compressed_features)  # [B, (scale^2 * k^2), H, W]
        kernel_weights = kernel_weights.permute(0, 2, 3, 1).reshape(b, h, w, self.scale**2, self.kernel_size**2)
        kernel_weights = F.softmax(kernel_weights, dim=-1)  # Normalize weights for each output pixel
        
        # Unfold the input feature map into sliding local blocks
        unfolded_x = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size // 2) # [B, C*k^2, H*W]
        unfolded_x = unfolded_x.view(b, c, self.kernel_size**2, h, w).permute(0, 3, 4, 1, 2).contiguous() # [B, H, W, C, k^2]
        
        # Apply predicted kernels to reassemble features
        kernel_out = torch.einsum('bhwck,bhwsk->bhwsc', unfolded_x, kernel_weights) # [B, H, W, S*S, C]
        kernel_out = kernel_out.permute(0, 4, 1, 3, 2).contiguous().view(b, c, target_h, target_w)

        # --- Path 2: Offset-Driven Feature Enhancement (Inspired by DySample) ---
        # Predict offsets for the deformable convolution
        offsets = self.offset_predictor(x) # [B, 2*S^2*G, H, W]
        # Apply deformable convolution to enhance features
        enhanced_x = self.deform_conv(x) 
        # Upsample the enhanced features
        offset_out = self.upsample(enhanced_x)

        # --- Fusion with Semantic Gating ---
        # Fuse the two upsampled feature maps
        fused_features = kernel_out + offset_out
        # Generate a semantic attention gate
        gate = self.semantic_gate(fused_features)
        # Apply gating to achieve a balanced output
        final_output = fused_features * gate

        return final_output
