import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if its spatial dimensions don't match x2 (due to pooling/upsampling operations)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1) # Concatenate along the channel dimension
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """
    Attention Gate module.
    g: gating signal from a coarser level (decoder path)
    x: feature map from the skip connection (encoder path)
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi # Element-wise multiplication to apply attention weights


class AttentionUNet(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder path with Attention Gates
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.att1 = AttentionGate(F_g=512 // factor, F_l=512, F_int=256) # F_g is output of Up1, F_l is skip from down3

        self.up2 = Up(512, 256 // factor, bilinear)
        self.att2 = AttentionGate(F_g=256 // factor, F_l=256, F_int=128) # F_g is output of Up2, F_l is skip from down2

        self.up3 = Up(256, 128 // factor, bilinear)
        self.att3 = AttentionGate(F_g=128 // factor, F_l=128, F_int=64) # F_g is output of Up3, F_l is skip from down1

        self.up4 = Up(128, 64, bilinear)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32) # F_g is output of Up4, F_l is skip from inc

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x) # 512 -> 512
        x2 = self.down1(x1) # 512 -> 256
        x3 = self.down2(x2) # 256 -> 128
        x4 = self.down3(x3) # 128 -> 64
        x5 = self.down4(x4) # 64 -> 32 (bottleneck)

        # Decoder with Attention Gates
        # Up1 (from x5 to x4)
        g1 = self.up1(x5, x4) # x5 is gating signal, x4 is skip connection
        x4_attn = self.att1(g=g1, x=x4) # Apply attention to x4 using g1
        x = self.up1(x5, x4_attn) # Perform upsampling with attention-weighted skip connection

        # Up2 (from x to x3)
        g2 = self.up2(x, x3)
        x3_attn = self.att2(g=g2, x=x3)
        x = self.up2(x, x3_attn)

        # Up3 (from x to x2)
        g3 = self.up3(x, x2)
        x2_attn = self.att3(g=g3, x=x2)
        x = self.up3(x, x2_attn)

        # Up4 (from x to x1)
        g4 = self.up4(x, x1)
        x1_attn = self.att4(g=g4, x=x1)
        x = self.up4(x, x1_attn)

        # Output
        logits = self.outc(x)
        return logits

# Example Usage:
if __name__ == "__main__":
    # Assuming input images are grayscale (1 channel) and output is binary segmentation (1 class)
    in_channels = 3
    n_classes = 1 # For binary segmentation (vessel/non-vessel)

    model = AttentionUNet(in_channels=in_channels, n_classes=n_classes)

    # Create a dummy input tensor (batch_size, channels, height, width)
    # For a 512x512 image, and batch size of 2
    dummy_input = torch.randn(2, in_channels, 512, 512)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dummy_input = dummy_input.to(device)

    # Pass the input through the model
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Expected output shape: torch.Size([2, 1, 512, 512]) for binary segmentation
    assert output.shape == torch.Size([2, n_classes, 512, 512])
    print("Model created and tested successfully!")

    # For training, you would use a loss function (e.g., BCEWithLogitsLoss for binary, CrossEntropyLoss for multi-class)
    # and an optimizer.
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Example of a single training step (simplified)
    # labels = torch.randint(0, 2, (2, n_classes, 512, 512)).float().to(device)
    # loss = criterion(output, labels)
    # loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()