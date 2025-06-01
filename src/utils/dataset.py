import os
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from trainingconfig import TrainingConfig
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
import torch
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, dataset_path, is_valid=False):
        self.image_dir = dataset_path + "train/input/" if not is_valid else dataset_path + "val/input/"
        self.label_dir = dataset_path + "train/label/" if not is_valid else dataset_path + "val/label/"
        self.image_list = sorted(os.listdir(self.image_dir))
        self.label_list = sorted(os.listdir(self.label_dir))
        self.config = TrainingConfig()
        self.is_valid = is_valid

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        label_path = os.path.join(self.label_dir, self.label_list[idx])

        # Load image and mask
        image = Image.open(img_path).convert("RGB" if self.config.in_channels != 1 else "L")
        label = Image.open(label_path).convert("L")

        # Resize to expected input size
        image = image.resize((self.config.input_size, self.config.input_size))
        label = label.resize((self.config.input_size, self.config.input_size))

        # Apply data augmentation (only for training)
        if not self.is_valid:
            if torch.rand(1) < 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

            if torch.rand(1) < 0.5:
                angle = torch.empty(1).uniform_(-15, 15).item()
                image = TF.rotate(image, angle)
                label = TF.rotate(label, angle)

            # if torch.rand(1) < 0.5:
            #     color_jitter = tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            #     image = color_jitter(image)

        # Convert to tensors
        input_tensor = TF.to_tensor(image)  # shape: C x H x W, scaled to [0, 1]
        target_tensor = torch.tensor(np.array(label), dtype=torch.float32).unsqueeze(0) / 255.0
 # 1 x H x W

        # Normalize image tensor (skip if grayscale)
        if self.config.in_channels == 3:
            normalize_transform = tf.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
            input_tensor = normalize_transform(input_tensor)
        


        return input_tensor, target_tensor

# Testing and visualization
if __name__ == "__main__":
    dataset = SegmentationDataset(dataset_path='C:/Users/kunal/retinal-vessel-segmentation/data/datasets/CHASE_DB1/', is_valid=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for i in range(5):
        image, label = dataset[i]
        #denormalize_image that is mean and std
        if dataset.dc.input_channels == 3:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            image = image * std[:, None, None] + mean[:, None, None]
        else:
            image = image.squeeze(0)
        image = image.clamp(0, 1)


        print(f"Image shape: {image.shape}, Label shape: {label.shape}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray' if image.shape[0] == 1 else None)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(label[0], cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        plt.show()
