import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary # Make sure this is installed: pip install torchsummary

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder (Downsampling path)
        # Block 1
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (or bridge)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder (Upsampling path)
        # Block 1
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # 256 = 128 (from upconv) + 128 (from encoder2 skip)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Block 2
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # 128 = 64 (from upconv) + 64 (from encoder1 skip)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(pool2)

        # Decoder
        up2 = self.upconv2(bottleneck)
        # Crop or pad enc2 if sizes don't perfectly match due to padding/stride
        # For UNet, often skip connections need to be cropped/padded to match
        # Here we assume sizes align after transpose conv and max pooling, which is typical for simple U-Net.
        # If sizes differ, you'd do: crop_enc2 = TF.center_crop(enc2, size_of_up2)
        # But `torch.cat` handles broadcasting if dimensions allow. For U-Net, usually F.pad or cropping is used to align.
        # Simple concat assumes spatial dimensions match:
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))

        # Output
        out = self.final_conv(dec1)
        return out


if __name__ == '__main__':
    # Determine the device to use (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model
    model = UNet(in_channels=3, out_channels=1) # 3 input channels for RGB, 1 output for binary segmentation

    # Move the model to the chosen device (GPU or CPU)
    model.to(device)

    # Summarize the model, ensuring the dummy input is also on the correct device
    # Create a dummy input tensor on the same device
    dummy_input = torch.randn(1, 3, 512, 512).to(device) # Batch size 1, 3 channels, 512x512

    # Use torchsummary with the model and input size
    # Note: torchsummary's summary function expects the input_size tuple, not the tensor itself
    summary(model, (3, 512, 512))

    print("\nModel summary complete.")
    print(f"Model is on: {next(model.parameters()).device}") # Verify model's device

    # Example of running a forward pass with dummy data
    # output = model(dummy_input)
    # print(f"Output shape: {output.shape}")