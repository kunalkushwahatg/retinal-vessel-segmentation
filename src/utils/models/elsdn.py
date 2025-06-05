import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. CBS (Conv2d + BN + SiLU) Block
class CBS(nn.Module):
    """
    Convolution + Batch Normalization + SiLU Activation.
    This is a common building block in many neural network architectures.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, dilation=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # Default padding to maintain spatial dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

# 2. DBS (DWConv2d + BN + SiLU) Block
class DBS(nn.Module):
    """
    Depthwise Separable Convolution + Batch Normalization + SiLU Activation.
    This block performs depthwise convolution followed by pointwise convolution.
    It's efficient for reducing computational cost.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        # Depthwise convolution: operates independently on each input channel
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.silu1 = nn.SiLU()
        # Pointwise convolution: 1x1 convolution to combine channels
        self.pwconv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu2 = nn.SiLU()

    def forward(self, x):
        x = self.silu1(self.bn1(self.dwconv(x)))
        x = self.silu2(self.bn2(self.pwconv(x)))
        return x

# 3. MECA (Multi-scale Enhanced Channel Attention)
class MECA(nn.Module):
    """
    Multi-scale Enhanced Channel Attention module.
    It uses global average pooling and global max pooling to capture
    multi-scale channel-wise information, then applies a sigmoid
    activation to generate attention weights.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1) # Global Max Pooling

        # Shared MLP for both pooled features
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        # Sum the outputs of the shared MLP and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        return x * attention # Apply attention to the input feature map

# 4. NSFE (Non-Strided Feature Enhancement)
class NSFE(nn.Module):
    """
    Non-Strided Feature Enhancement module.
    This module enhances features without strided convolutions,
    mitigating their negative impact on detection. It includes
    multiple CBS blocks, a DBS block, a MECA module, and skip connections.
    """
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        # Initial CBS block
        self.cbs1 = CBS(in_channels, out_channels, kernel_size=3, stride=1)
        # Sequence of DBS blocks
        self.dbs_blocks = nn.ModuleList([
            DBS(out_channels, out_channels, kernel_size=3) for _ in range(num_blocks)
        ])
        # MECA module
        self.meca = MECA(out_channels)
        # Final CBS block after MECA and skip connection
        self.cbs2 = CBS(out_channels, out_channels, kernel_size=1, stride=1) # 1x1 conv for feature aggregation

    def forward(self, x):
        x_initial = self.cbs1(x)
        x_processed = x_initial
        for dbs_block in self.dbs_blocks:
            x_processed = dbs_block(x_processed)

        # Apply MECA
        x_meca = self.meca(x_processed)

        # Skip connection: add initial features to MECA output
        x_combined = x_meca + x_initial
        return self.cbs2(x_combined)

# 5. SPC (Spatial Pyramid Pooling based on Convolution)
class SPC(nn.Module):
    """
    Spatial Pyramid Pooling based on Convolution module.
    This module uses a split-transform-merge strategy with
    1x9 and 9x1 convolutions to capture multi-scale spatial information.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Split the input into two paths
        self.conv1x9 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 9), padding=(0, 4), bias=False)
        self.conv9x1 = nn.Conv2d(in_channels, in_channels, kernel_size=(9, 1), padding=(4, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.silu = nn.SiLU()

        # Final CBS block after concatenation
        self.cbs = CBS(in_channels * 2, out_channels, kernel_size=1, stride=1) # 1x1 conv to reduce channels

    def forward(self, x):
        # Path 1: Identity (direct pass-through)
        identity_path = x

        # Path 2: 1x9 conv -> 9x1 conv
        conv_path = self.silu(self.bn1(self.conv1x9(x)))
        conv_path = self.silu(self.bn2(self.conv9x1(conv_path)))

        # Concatenate the two paths along the channel dimension
        combined = torch.cat((identity_path, conv_path), dim=1)
        return self.cbs(combined)

# 6. SPDConv (Spatial Pyramid Downsampling Convolution)
class SPDConv(nn.Module):
    """
    Spatial Pyramid Downsampling Convolution.
    Implemented as a CBS block with stride 2 for spatial reduction.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cbs = CBS(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.cbs(x)

# New: Simple Spatial Attention Module
class SpatialAttentionModule(nn.Module):
    """
    A simple Spatial Attention module (similar to CBAM's spatial part).
    It computes attention weights based on spatial relationships of features.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and Max Pool along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate them
        x_concat = torch.cat([avg_out, max_out], dim=1)
        # Apply convolution to generate spatial attention map
        attention_map = self.conv(x_concat)
        return x * self.sigmoid(attention_map) # Apply attention to the input feature map


# 7. Backbone of ELSD-Net
class Backbone(nn.Module):
    """
    The Backbone extracts multi-scale features from the input image.
    It consists of an initial CBS block followed by multiple NSFE modules
    and SPDConv blocks for downsampling.
    """
    def __init__(self, input_channels):
        super().__init__()
        # Initial CBS block (k=6, s=2, p=2, c=16)
        self.cbs1 = CBS(input_channels, 16, kernel_size=6, stride=2, padding=2)

        # NSFE blocks and SPDConv for hierarchical feature extraction
        # P3: n=1, c=16
        self.nsfe_p3 = NSFE(16, 16, num_blocks=1)
        self.spdconv_p3 = SPDConv(16, 24) # Downsample to P4 resolution, increase channels

        # P4: n=2, c=24
        self.nsfe_p4 = NSFE(24, 24, num_blocks=2)
        self.spdconv_p4 = SPDConv(24, 48) # Downsample to P5 resolution, increase channels

        # P5: n=5, c=48
        self.nsfe_p5 = NSFE(48, 48, num_blocks=5)
        self.spdconv_p5 = SPDConv(48, 96) # Downsample to P_MECA resolution, increase channels

        # Final NSFE before MECA (n=3, c=96)
        self.nsfe_final = NSFE(96, 96, num_blocks=3)
        self.meca = MECA(96) # MECA after the final NSFE

    def forward(self, x):
        x = self.cbs1(x) # Initial downsample (e.g., 256x256 -> 128x128)

        p3_out = self.nsfe_p3(x) # Features at H/2 x W/2 (e.g., 128x128)
        x = self.spdconv_p3(p3_out) # Downsample to H/4 x W/4 (e.g., 64x64)

        p4_out = self.nsfe_p4(x) # Features at H/4 x W/4 (e.g., 64x64)
        x = self.spdconv_p4(p4_out) # Downsample to H/8 x W/8 (e.g., 32x32)

        p5_out = self.nsfe_p5(x) # Features at H/8 x W/8 (e.g., 32x32)
        x = self.spdconv_p5(p5_out) # Downsample to H/16 x W/16 (e.g., 16x16)

        meca_out = self.meca(self.nsfe_final(x)) # Features at H/16 x W/16 (e.g., 16x16)

        # Return features at different scales for the Neck
        return p3_out, p4_out, p5_out, meca_out

# 8. Neck of ELSD-Net (FPN-like structure for segmentation)
class Neck(nn.Module):
    """
    The Neck module combines and refines features from the Backbone
    at different scales using an FPN-like approach for segmentation.
    It performs top-down pathway and lateral connections.
    """
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Lateral connections (1x1 conv to match channels)
        self.lateral_p5 = CBS(48, 96, kernel_size=1) # From P5_out to match MECA_out channels
        self.lateral_p4 = CBS(24, 48, kernel_size=1) # From P4_out to match fused P5 channels
        self.lateral_p3 = CBS(16, 24, kernel_size=1) # From P3_out to match fused P4 channels

        # SPC blocks for refining fused features
        self.spc_fused_p5 = SPC(192, 96) # For fused MECA and P5 features, output 96 channels
        self.spc_fused_p4 = SPC(144, 48) # For fused P5_fused and P4 features, output 48 channels
        self.spc_fused_p3 = SPC(72, 24) # For fused P4_fused and P3 features, output 24 channels

    def forward(self, p3_out, p4_out, p5_out, meca_out):
        # Top-down pathway:
        fused_p_meca = meca_out

        # Fuse P5 and upsampled MECA_out
        lateral_p5 = self.lateral_p5(p5_out)
        upsampled_fused_p_meca = F.interpolate(fused_p_meca, size=lateral_p5.shape[2:], mode='bilinear', align_corners=False)
        fused_p5 = torch.cat((lateral_p5, upsampled_fused_p_meca), dim=1)
        fused_p5 = self.spc_fused_p5(fused_p5)

        # Fuse P4 and upsampled fused_p5
        lateral_p4 = self.lateral_p4(p4_out)
        upsampled_fused_p5 = F.interpolate(fused_p5, size=lateral_p4.shape[2:], mode='bilinear', align_corners=False)
        fused_p4 = torch.cat((lateral_p4, upsampled_fused_p5), dim=1)
        fused_p4 = self.spc_fused_p4(fused_p4)

        # Fuse P3 and upsampled fused_p4
        lateral_p3 = self.lateral_p3(p3_out)
        upsampled_fused_p4 = F.interpolate(fused_p4, size=lateral_p3.shape[2:], mode='bilinear', align_corners=False)
        fused_p3 = torch.cat((lateral_p3, upsampled_fused_p4), dim=1)
        fused_p3 = self.spc_fused_p3(fused_p3)

        return fused_p3, fused_p4, fused_p5, fused_p_meca

# 9. Segmentation Head (Now with Spatial Attention)
class SegmentationHead(nn.Module):
    """
    The Segmentation Head takes the high-resolution feature map from the Neck
    and produces the final segmentation mask.
    Now includes a SpatialAttentionModule.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = CBS(in_channels, in_channels, kernel_size=3, padding=1)
        self.context = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2)  # context

        # Spatial Attention Module applied after initial processing
        self.spatial_attention = SpatialAttentionModule(kernel_size=7)

        self.conv2 = CBS(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv3 = CBS(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.context(x) # Apply context convolution

        # Apply Spatial Attention
        x = self.spatial_attention(x)

        x = self.conv2(x)
        x = self.conv3(x)
        return self.final_conv(x)

# 10. ELSD-Net for Segmentation (Overall Architecture)
class ELSDNetSegmentation(nn.Module):
    """
    Overall architecture of ELSD-Net adapted for semantic segmentation.
    Combines Backbone, FPN-like Neck, and a Segmentation Head.
    """
    def __init__(self, input_channels=3, num_classes=2): # num_classes: e.g., 2 for binary (background + foreground)
        super().__init__()
        self.backbone = Backbone(input_channels)
        self.neck = Neck()

        # The segmentation head typically operates on the highest resolution feature map from the Neck.
        # In our FPN-like Neck, fused_p3 is the highest resolution output (H/2 x W/2).
        self.segmentation_head = SegmentationHead(in_channels=24, num_classes=num_classes)

        # Final upsampling to original image size
        self.upsample_to_original_size = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, x):
        # Pass through Backbone
        p3_out, p4_out, p5_out, meca_out = self.backbone(x)

        # Pass through Neck to get fused features at different scales
        fused_p3, _, _, _ = self.neck(p3_out, p4_out, p5_out, meca_out)

        # Pass the highest resolution fused feature map to the Segmentation Head
        segmentation_logits = self.segmentation_head(fused_p3)

        # Upsample the logits to the original input image size
        final_output = self.upsample_to_original_size(segmentation_logits)

        return final_output

# Example Usage and Model Summary
if __name__ == '__main__':
    # --- For 3-channel (RGB) image ---
    print("--- Testing with 3-channel (RGB) image ---")
    dummy_input_rgb = torch.randn(1, 3, 256, 256) # Batch_size, Channels, Height, Width
    model_rgb = ELSDNetSegmentation(input_channels=3, num_classes=2)
    output_segmentation_rgb = model_rgb(dummy_input_rgb)
    print(f"Shape of output segmentation map (RGB): {output_segmentation_rgb.shape}")

    total_params_rgb = sum(p.numel() for p in model_rgb.parameters() if p.requires_grad)
    print(f"Total trainable parameters (RGB): {total_params_rgb:,}")


    # --- For 1-channel (Grayscale) image ---
    print("\n--- Testing with 1-channel (Grayscale) image ---")
    dummy_input_gray = torch.randn(1, 1, 256, 256) # Batch_size, Channels, Height, Width
    model_gray = ELSDNetSegmentation(input_channels=1, num_classes=2)
    output_segmentation_gray = model_gray(dummy_input_gray)
    print(f"Shape of output segmentation map (Grayscale): {output_segmentation_gray.shape}")

    total_params_gray = sum(p.numel() for p in model_gray.parameters() if p.requires_grad)
    print(f"Total trainable parameters (Grayscale): {total_params_gray:,}")

    # Model Summary (optional)
    try:
        from torchsummary import summary
        print("\nModel Summary for 3-channel input:")
        summary(model_rgb, input_size=(3, 256, 256), device="cpu")
        print("\nModel Summary for 1-channel input:")
        summary(model_gray, input_size=(1, 256, 256), device="cpu")
    except ImportError:
        print("\nInstall 'torchsummary' (pip install torchsummary) for model summary.")