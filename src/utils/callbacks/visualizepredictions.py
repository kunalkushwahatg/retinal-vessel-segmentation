import torch
import matplotlib.pyplot as plt
import random


class VisualizePredictions:
    def __init__(self, vis_dir, dataset, device):
        self.vis_dir = vis_dir
        self.dataset = dataset
        self.device = device

    def denormalize(self, image):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if image.dim() != 3:
            raise ValueError("Input image tensor must have shape (C, H, W).")
        
        # Clone the tensor to avoid modifying the original
        denormalized_image = image.clone()
        
        # Denormalize each channel
        for t, m, s in zip(denormalized_image, mean, std):
            t.mul_(s).add_(m)  # Multiply by std and add mean
        
        # Clip values to ensure they are within the valid range [0, 1]
        denormalized_image = torch.clamp(denormalized_image, 0, 1)
        
        return denormalized_image

    def plot_segmentation(self,image, mask, pred_mask, save_path):
        '''
        It plots the image, ground truth mask, and predicted mask as a figure and saves it to the disk.
        The order of plot is image, ground truth mask, and predicted mask in a row.
        '''
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image.permute(1, 2, 0))
        ax[0].set_title('Image')
        ax[0].axis('off')
        ax[1].imshow(mask.squeeze(), cmap='gray')
        ax[1].set_title('Ground Truth')
        ax[1].axis('off')
        ax[2].imshow(pred_mask, cmap='gray')
        ax[2].set_title('Prediction')
        ax[2].axis('off')
        plt.savefig(save_path)
        plt.close()

    def __call__(self, epoch, model,random_save=False):
        model.eval()
        with torch.no_grad():
            if random_save:
                idx = random.randint(0, len(self.dataset)-1)
                sample = self.dataset[idx]
            else:
                sample = self.dataset[0]  # Get a sample that is 0th element of the dataset
            pred = model(sample[0].unsqueeze(0).to(self.device)) #shape = (1, 1, 256, 256)
            pred_mask = torch.sigmoid(pred).cpu().squeeze().numpy() #shape = (256, 256)

            #make the range of the mask to be 0-255
            pred_mask = pred_mask * 255

            #convert to uint8
            pred_mask = pred_mask.astype('uint8')
             
            # Save image, ground truth, and prediction as a figure
            save_path = f'{self.vis_dir}/epoch_{epoch}.png'
            img = self.denormalize(sample[0])
            self.plot_segmentation(img, sample[1], pred_mask, save_path)

# Usage
