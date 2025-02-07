import torch
from torchvision import models
import torch.nn as nn
import torch.nn as nn
from monai.networks.nets import UNet,AttentionUnet
from segmentation_models_pytorch import UnetPlusPlus
from torchsummary import summary

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetCustom(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Encoder (Contracting Path)
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (Expansive Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        
        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final Convolution
        return self.final_conv(dec1)

class PupilSegmentationUNet(nn.Module):


    def __init__(self, pretrained=True):
        super().__init__()
        # Encoder (ResNet18 backbone)
        self.encoder = models.resnet18(pretrained=pretrained)
        
        # Decoder with 5 upsampling steps
        self.decoder = nn.Sequential(
            # Upsample 8x8 → 16x16
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Upsample 16x16 → 32x32
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Upsample 32x32 → 64x64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Upsample 64x64 → 128x128
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Upsample 128x128 → 256x256
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Final layer
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)  # Output shape: (1, 512, 8, 8)
        
        # Decoder forward pass
        x = self.decoder(x)  # Output shape: (1, 1, 256, 256)
        return x
    
class Unetwrapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unet = UNet(spatial_dims=2,
                    in_channels=3,
                    out_channels=1,
                    channels=(64, 128, 256, 512, 1024),
                    strides=(2, 2, 2, 2),
                    num_res_units=2)
        
    def forward(self, x):
        return self.unet(x)

class Unetpluspluswrapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unetplusplus = UnetPlusPlus(
      encoder_name="resnet34",  # Pretrained encoder
      encoder_weights="imagenet",
      in_channels=3,
      classes=1,
  )
        
    def forward(self, x):
        return self.unetplusplus(x)

class AttentionUnetwrapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attentionunet = AttentionUnet(
      spatial_dims=2,
      in_channels=3,
      out_channels=1,
      channels=(64, 128, 256, 512),
      strides=(2, 2, 2),
  )
    def forward(self, x):
        return self.attentionunet(x)




if __name__ == "__main__":
    # Test UNet
    # model = UNet(in_channels=3, out_channels=1)
    # x = torch.randn((1, 3, 256, 256))
    # print(model(x).shape)  # Expected: (1, 1, 256, 256)
    
    # Test PupilSegmentationUNet
    model = PupilSegmentationUNet(pretrained=True)
    x = torch.randn((1, 3, 256, 256))
    #print(model(x).shape)  # Expected: (1, 1, 256, 256)
    summary(model, (3, 256, 256))


    model = Unetpluspluswrapper(in_channels=3, out_channels=1)
    x = torch.randn((1, 3, 256, 256))
    #print(model(x).shape)  # Expected: (1, 1, 256, 256)
    #print size of model in mb
    #print(sum(p.numel() for p in model.parameters())/1e6)
    summary(model, (3, 256, 256))


    model = AttentionUnetwrapper(in_channels=3, out_channels=1)
    x = torch.randn((1, 3, 256, 256))
    #print(model(x).shape)  # Expected: (1, 1, 256, 256)
    #print size of model in mb
    #print(sum(p.numel() for p in model.parameters())/1e6)
    summary(model, (3, 256, 256))
