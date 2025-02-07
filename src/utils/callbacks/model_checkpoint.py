import torch 
import numpy as np


class ModelCheckpoint:
    def __init__(self, save_dir, metric='val_dice', mode='max'):
        self.best_metric = { 'dice': -np.inf } if mode == 'max' else { 'dice': np.inf }
        self.save_dir = save_dir
        self.metric = metric
        self.mode = mode

    def __call__(self, model,epoch, val_metric):

        # This will save the model with the best metric
        print(f'Validation {self.metric}: {val_metric}')
        print(f'Best {self.metric}: {self.best_metric}')
        if (self.mode == 'max' and val_metric['dice'] > self.best_metric['dice']) or (self.mode == 'min' and val_metric['dice'] < self.best_metric):
            torch.save(model.state_dict(), f'{self.save_dir}/best_model.pth')
            self.best_metric = val_metric
        torch.save(model.state_dict(), f'{self.save_dir}/latest_model.pth')



# Usage
# model_checkpoint = ModelCheckpoint('checkpoints')
# model_checkpoint(epoch, model, val_metric)