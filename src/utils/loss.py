import torch

import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return bce_loss + dice_loss
    

class CompoundLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CompoundLoss, self).__init__()
        self.alpha = alpha
        self.bce_dice_loss = BCEDiceLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        bce_dice_loss = self.bce_dice_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        return self.alpha * bce_dice_loss + (1 - self.alpha) * dice_loss
    
# Example usage
if __name__ == "__main__":
    criterion = DiceLoss()
    inputs = torch.randn(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()
    loss = criterion(inputs, targets)
    print(loss)
    