import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.networks.nets import UNet as MONAIUNet  # Alias to avoid name conflict
from models.attention_unet import AttentionUNet  # Assuming you have this model defined
from monai.networks.layers import Norm  # Needed for model initialization
from evaluate import Evaluation
from dataset import SegmentationDataset
from loss import CompoundLoss
from models.unetbasic import UNet
from trainingconfig import TrainingConfig


# Check CUDA
print("Cuda available:", torch.cuda.is_available())

# Load config
config = TrainingConfig()

# Make output directories
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(f'{config.output_dir}/checkpoints', exist_ok=True)
os.makedirs(f'{config.output_dir}/predictions', exist_ok=True)


# Load data
dataset_path = 'C:/Users/kunal/retinal-vessel-segmentation/data/datasets/CHASE_DB1/'
train_dataset = SegmentationDataset(dataset_path, is_valid=False)
val_dataset = SegmentationDataset(dataset_path, is_valid=True)



if config.debug:
    train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(0, len(train_dataset) // 10))
    val_dataset = torch.utils.data.Subset(val_dataset, torch.arange(0, len(val_dataset) // 10))

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True,
                          persistent_workers=0)

val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True,
                        persistent_workers=0)

evaluation = Evaluation(dataloader=val_loader, device=config.device)

# Initialize model, optimizer, loss


model = MONAIUNet(
    spatial_dims=2,  # 2 for 2D images
    in_channels=config.in_channels,
    out_channels=config.classes,
    channels=(16, 32, 64, 128, 256),  # Example channel progression
    strides=(2, 2, 2, 2),  # Example strides for downsampling
    num_res_units=2,  # Number of residual units in each block
    norm=Norm.BATCH,  # Or Norm.INSTANCE, Norm.BATCH is common
).to(config.device)
#model = UNet(in_channels=config.in_channels, out_channels=config.classes).to(config.device)
model = AttentionUNet(in_channels=config.in_channels, n_classes=config.classes).to(config.device)

optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
criterion = CompoundLoss()

# Train + Validate
best_metric = -np.inf
best_epoch = 0

for epoch in tqdm(range(config.num_epochs)):
    # Training
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images = images.to(config.device, non_blocking=True)
        masks = masks.to(config.device, non_blocking=True)

        #print("Input Image Shape:", images.shape)
        #print("Ground Truth Mask Shape:", masks.shape)
        optimizer.zero_grad(set_to_none=True)
        device_type = config.device.type
        #print("images min:", images.min().item(), "max:", images.max().item())
        #print("masks min:", masks.min().item(), "max:", masks.max().item())

        outputs = model(images)
        #print("outputs min:", outputs.min().item(), "max:", outputs.max().item())

        loss = criterion(outputs, masks.float())
        #print("Loss:", loss.item())
        train_loss += loss.item() * images.size(0)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}")

    if epoch !=0 and epoch % config.evaluate_every == 0:
        print(evaluation.evaluate(model, epoch))

    if  epoch!=0 and epoch % 100 == 0:
        evaluation.plot_results()
        
