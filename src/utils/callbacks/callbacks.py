import torch
import numpy as np

import matplotlib.pyplot as plt

class PlotLossCallback:
    def __init__(self):
        self.epoch_losses = []

    def on_epoch_end(self, epoch, loss):
        self.epoch_losses.append(loss)
        plt.plot(self.epoch_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

class SaveModelCallback:
    def __init__(self, save_path):
        self.save_path = save_path

    def on_epoch_end(self, epoch, model):
        torch.save(model.state_dict(), f"{self.save_path}/model_epoch_{epoch}.pth")

class EarlyStoppingCallback:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def on_epoch_end(self, epoch, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping")
                return True
        return False

class ImageSegmentationCallback:
    def __init__(self, val_loader, model, device):
        self.val_loader = val_loader
        self.model = model
        self.device = device

    def on_epoch_end(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                outputs = torch.sigmoid(outputs)
                outputs = outputs.cpu().numpy()
                masks = masks.cpu().numpy()

                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(np.transpose(images[0].cpu().numpy(), (1, 2, 0)))
                ax[0].set_title('Input Image')
                ax[1].imshow(masks[0, 0], cmap='gray')
                ax[1].set_title('Ground Truth Mask')
                ax[2].imshow(outputs[0, 0], cmap='gray')
                ax[2].set_title('Predicted Mask')
                plt.show()
                break
        self.model.train()