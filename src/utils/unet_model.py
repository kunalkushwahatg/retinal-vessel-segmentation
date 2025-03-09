""" Full assembly of the parts to form the complete network """

from unet_parts import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from haunet_s import CEMBlock

class LocalSelfAttention(nn.Module):
    def __init__(self, in_channels, patch_size=8):
        super(LocalSelfAttention, self).__init__()
        self.patch_size = patch_size
        self.softmax = nn.Softmax(dim=-1)
        self.avg_pool = nn.AvgPool2d(patch_size, patch_size)

        # Linear transformations for query, key, value
        self.query = nn.Linear(in_channels, in_channels)  
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

        self.upsample = nn.Upsample(scale_factor=patch_size, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # Extract non-overlapping patches
        x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H//p, W//p, p, p)
        H_p, W_p = x_patches.shape[2], x_patches.shape[3]  # Number of patches along H and W
        
        # Flatten spatial patches into a sequence
        x_patches = x_patches.contiguous().view(B, C, H_p * W_p, p * p)  # (B, C, Num_Patches, Patch_Size*Patch_Size)

        #compute avg pool on the patch
        x_avg = self.avg_pool(x).view(B, C, H_p * W_p).transpose(1, 2)  # (B, Num_Patches, C)
        

        # Compute Query, Key, and Value
        q = self.query(x_avg).permute(0, 2, 1)  # (B, Num_Patches, C)
        k = self.key(x_avg).permute(0, 2, 1)  # (B, Num_Patches, C)
        v = self.value(x_avg).permute(0, 2, 1)  # (B, Num_Patches, C)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5)  # (B, Num_Patches, Num_Patches)
        attn_weights = self.softmax(attn_scores)  # (B, Num_Patches, Num_Patches)

        attn_output = torch.bmm(attn_weights, v)  # (B, C, Num_Patches)

        # Reshape the output
        attn_output = attn_output.view(B, C , H_p, W_p)
        attn_output = self.upsample(attn_output)
        return attn_output
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=padding, bias=False)
        
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.shape  # Input shape (B, C, H, W)
        avg_pooled = self.avg_pool(x).view(batch, channels)  # Shape: (B, C, 1, 1) → (B, C)
        attention_weights = self.fc(avg_pooled).view(batch, channels, 1, 1)  # Shape: (B, C) → (B, C, 1, 1)
        return x * attention_weights  # Element-wise multiplication with input


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64)) #shape = (B, 64, 256, 256)
        self.spatial1 = SpatialAttention()
        self.down1 = (Down(64, 128))  #shape = (B, 128, 128, 128)
        self.spatial2 = SpatialAttention()
        self.down2 = (Down(128, 256)) #shape = (B, 256, 64, 64)
        self.chann_att1 = ChannelAttention(256)
        self.down3 = (Down(256, 512)) #shape = (B, 512, 32, 32)
        self.chann_att2 = ChannelAttention(512)
        self.down4 = (Down(512, 1024)) #shape = (B, 1024, 16, 16)
        self.chann_att3 = ChannelAttention(1024)
        factor = 2 if bilinear else 1

        self.up1 = (Up(1024, 512 // factor, bilinear)) #shape = (B, 512, 32, 32)
        self.chann_att4 = ChannelAttention(512//factor)
        self.up2 = (Up(512, 256 // factor, bilinear)) #shape = (B, 256, 64, 64)
        self.up3 = (Up(256, 128 // factor, bilinear)) #shape = (B, 128, 128, 128)
        self.spatial3 = SpatialAttention()
        self.up4 = (Up(128, 64, bilinear))
        self.spatial4 = SpatialAttention()
        self.outc = (OutConv(64, n_classes))


    def forward(self, x):
        x1 = self.inc(x) #shape = (B, 64, 256, 256)
        x1 = self.spatial1(x1)
        x2 = self.down1(x1) #shape = (B, 128, 128, 128)
        x2 = self.spatial2(x2)
        x3 = self.down2(x2) #shape = (B, 256, 64, 64)
        x3 = self.chann_att1(x3)
        
        x4 = self.down3(x3) #shape = (B, 512, 32, 32)
        x4 = self.chann_att2(x4)
        x5 = self.down4(x4) #shape = (B, 1024, 16, 16)
        x5 = self.chann_att3(x5)

        x = self.up1(x5, x4) #shape = (B, 512, 32, 32)  
        x = self.chann_att4(x)

        x = self.up2(x, x3) #shape = (B, 256, 64, 64)
        x = self.spatial3(x)
        x = self.up3(x, x2) #shape = (B, 128, 128, 128)
        x = self.spatial4(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

if __name__ == "__main__":
    x = torch.randn((1, 3, 512, 512))
    model = UNet(n_channels=3, n_classes=1)
    print(model(x).shape)  # Expected: (1, 1, 256, 256)
    #print size of model in mb  
    print(sum(p.numel() for p in model.parameters())/1e6)
