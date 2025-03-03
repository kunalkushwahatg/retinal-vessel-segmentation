import torch
import torch.nn as nn
from monai.networks.nets import ResUNet

class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) using AvgPool & MaxPool."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Avg pooling across channels
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling across channels
        attention = torch.cat([avg_pool, max_pool], dim=1)  # Concatenate along channel dimension
        attention = self.sigmoid(self.conv(attention))  # Convolution + Sigmoid
        return x * attention  # Apply attention map to input

class ResidualAttentionUNet(nn.Module):
    """Residual Attention U-Net (RA-UNet) using MONAI's ResUNet with Spatial Attention."""
    def __init__(self, in_channels=1, out_channels=1):
        super(ResidualAttentionUNet, self).__init__()

        # Use MONAI's ResUNet as the backbone
        self.resunet = ResUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),  # Feature maps per level
            strides=(2, 2, 2, 2),  # Downsampling factors
            norm="batch",
            act="RELU",
        )

        # Attention modules applied to skip connections
        self.att1 = SpatialAttention()
        self.att2 = SpatialAttention()
        self.att3 = SpatialAttention()
        self.att4 = SpatialAttention()

    def forward(self, x):
        # Encoder (ResUNet feature extraction)
        enc1 = self.resunet.encoder[0](x)  # First conv layer
        enc2 = self.resunet.encoder[1](enc1)  # Downsample + ResBlock
        enc3 = self.resunet.encoder[2](enc2)  # Downsample + ResBlock
        enc4 = self.resunet.encoder[3](enc3)  # Downsample + ResBlock
        bottleneck = self.resunet.encoder[4](enc4)  # Last ResBlock

        # Apply spatial attention to skip connections
        enc1 = self.att1(enc1)
        enc2 = self.att2(enc2)
        enc3 = self.att3(enc3)
        enc4 = self.att4(enc4)

        # Decoder (upsampling with attention-modified skip connections)
        dec1 = self.resunet.decoder[0](bottleneck, enc4)
        dec2 = self.resunet.decoder[1](dec1, enc3)
        dec3 = self.resunet.decoder[2](dec2, enc2)
        dec4 = self.resunet.decoder[3](dec3, enc1)

        # Output segmentation mask
        return self.resunet.final_conv(dec4)

# Example usage
if __name__ == "__main__":
    model = ResidualAttentionUNet(in_channels=1, out_channels=1)
    input_tensor = torch.randn(1, 1, 128, 128)  # Example input (B, C, H, W)
    output = model(input_tensor)
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output.shape)
