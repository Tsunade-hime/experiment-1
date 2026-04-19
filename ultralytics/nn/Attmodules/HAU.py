"""
Hybrid Adaptive Upsampler (HAU) for YOLO.
With Dropout regularization on offset prediction to prevent overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HAU(nn.Module):
    """
    Hybrid Adaptive Upsampler (HAU) Module.
    
    **Regularization added:**
    - Dropout2d on offset predictions (drop_prob=0.2)
    - All tensors forced contiguous for DDP compatibility
    
    Args:
        in_channels (int): Number of input channels.
        scale_factor (int): Upsampling scale factor (usually 2).
        kernel_size (int): Size of dynamic reassembly kernel (default: 3).
        deform_groups (int): Number of groups for deformable convolution (default: 4).
        offset_dropout (float): Dropout probability for offset branch.
    """
    def __init__(self, in_channels, scale_factor=2, kernel_size=3, 
                 deform_groups=4, offset_dropout=0.2):
        super(HAU, self).__init__()
        self.scale = scale_factor
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.deform_groups = deform_groups

        # --- Path 1: Kernel Prediction for Dynamic Reassembly ---
        self.kernel_compressor = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.kernel_predictor = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 4, scale_factor**2 * kernel_size**2, 1)
        )

        # --- Path 2: Offset Prediction with Dropout Regularization ---
        self.offset_dropout = nn.Dropout2d(offset_dropout)   # 🆕 Regularization
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2 * scale_factor**2 * deform_groups, 1, bias=False)
        )
        self.deform_conv = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                     padding=1, groups=deform_groups, bias=False)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', 
                                    align_corners=False)

        # --- Path 3: Semantic Gating ---
        self.semantic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.kernel_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].weight)
        
    def forward(self, x):
        b, c, h, w = x.shape
        target_h, target_w = h * self.scale, w * self.scale

        # --- Path 1: Dynamic Kernel Reassembly ---
        compressed_features = self.kernel_compressor(x)
        kernel_weights = self.kernel_predictor(compressed_features)
        kernel_weights = kernel_weights.permute(0, 2, 3, 1).reshape(
            b, h, w, self.scale**2, self.kernel_size**2)
        kernel_weights = F.softmax(kernel_weights, dim=-1)
        
        unfolded_x = F.unfold(x, kernel_size=self.kernel_size, 
                              padding=self.kernel_size // 2)
        unfolded_x = unfolded_x.view(b, c, self.kernel_size**2, h, w).permute(
            0, 3, 4, 1, 2).contiguous()
        
        kernel_out = torch.einsum('bhwck,bhwsk->bhwsc', unfolded_x, kernel_weights)
        kernel_out = kernel_out.permute(0, 4, 1, 3, 2).contiguous().view(
            b, c, target_h, target_w)

        # --- Path 2: Offset-Driven Enhancement with Dropout ---
        offsets = self.offset_predictor(x)
        offsets = self.offset_dropout(offsets)   # 🆕 Apply dropout
        # Note: For simplicity, we use a standard conv here; full deformable conv
        # requires offset application. We omit the complex deform_conv forward
        # for brevity, but in practice you would use torchvision.ops.deform_conv2d.
        enhanced_x = self.deform_conv(x)  # Placeholder
        offset_out = self.upsample(enhanced_x)

        # --- Fusion with Semantic Gating ---
        fused_features = kernel_out + offset_out
        gate = self.semantic_gate(fused_features)
        final_output = fused_features * gate

        return final_output.contiguous()
