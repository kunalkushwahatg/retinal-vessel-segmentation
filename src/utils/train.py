import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from dataset import SegmentationDataset
from transform import SegmentationTransform  
from model import PupilSegmentationUNet,Unetwrapper,UnetPlusPlus,AttentionUnetwrapper,UNetCustom
from loss import DiceLoss,CompoundLoss
from callbacks.logger import Logger
from callbacks.model_checkpoint import ModelCheckpoint
from callbacks.visualizepredictions import VisualizePredictions
from callbacks.earlystopping import EarlyStopping   
from evaluation import SegmentationEvaluator  
import os
import numpy as np
from trainingconfig import TrainingConfig
from datasetconfig import DatasetConfig
import gc
import matplotlib.pyplot as plt
from unet_model import UNet

#check if the model is training on GPU or CPU
print("Cuda available: ", torch.cuda.is_available())


class SegmentationTrainer:
    def __init__(self, config=TrainingConfig()):
        self.config = config 
        self._setup_environment() #make directories for logs, checkpoints, predictions
        self._load_data() #load the dataset from image and mask folders and split it into train and val dataloaders
        self._initialize_model() #setups the model, optimizer, loss function, scheduler, and evaluator
        self._setup_callbacks() #initialize logger, checkpointer, visualizer, and early stopper
        
    def _setup_environment(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(f'{self.config.output_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{self.config.output_dir}/predictions', exist_ok=True)
        
        if self.config.debug:
            torch.autograd.set_detect_anomaly(True)
            
    def _load_data(self):
        #resize normalizes and converts the image and mask to tensors
        transform = SegmentationTransform(DatasetConfig().image_size) 

        #load the dataset from image and mask folders
        full_dataset = SegmentationDataset(
            DatasetConfig().image_dir,
            DatasetConfig().mask_dir,
            transform
        )

        print(f"Dataset size: {len(full_dataset)}")


        
        # Debug mode takes 1/10th of the dataset
        if self.config.debug:
            full_dataset = torch.utils.data.Subset(
                full_dataset, 
                torch.arange(0, len(full_dataset)//10)
            )
            
        # Stratified split for medical data
        train_size = int(self.config.train_test_split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        

    def _initialize_model(self):
        self.model = UNet(n_channels=1, n_classes=1).to(self.config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = CompoundLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.1, 
            patience=2
        )

        # SegmentationEvaluator is used to calculate the metrics 
        self.evaluator = SegmentationEvaluator(num_classes=1, device=self.config.device)
        
    def _setup_callbacks(self):

        #setup logger that is based on SummaryWriter from PyTorch can be displayed on tensorboard
        self.logger = Logger(f'{self.config.output_dir}/logs')

        # ModelCheckpoint saves the model with the best validation dice score and also saves the latest model
        self.checkpointer = ModelCheckpoint(
            save_dir=f'{self.config.output_dir}/checkpoints',
            metric='val_dice',
            mode='max'
        )


        self.visualizer = VisualizePredictions(
            f'{self.config.output_dir}/predictions',
            self.val_dataset,
            self.config.device
        )

        self.early_stopper = EarlyStopping(
            patience=self.config.patience,
            delta=self.config.early_stop_delta
        )
        
    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for images, masks in progress_bar:
            images = images.to(self.config.device, non_blocking=True)
            masks = masks.to(self.config.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=self.config.device.type == 'cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            loss.backward()
            #gradient clipping is used to prevent the exploding gradient problem
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())
            
            # Memory optimization: Clear intermediate variables
            del images, masks, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()
            
        return running_loss / len(self.train_dataset)
    
    def _validate(self):
        self.model.eval()
        val_metrics = self.evaluator(self.model, self.val_loader, self.criterion)
        
        # Memory optimization: Reset evaluator and clear cache
        self.evaluator.reset()
        torch.cuda.empty_cache()
        gc.collect()
        
        return val_metrics
    
    def train(self):
        best_metric = -np.inf
        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch()
            val_metrics = self._validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['dice'])
            
            # Logging
            self.logger.log(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_metrics.get('loss', 0),
                metrics=val_metrics
            )
            
            # Visualize predictions
            if epoch % self.config.eval_every == 0:
                self.visualizer(epoch, self.model, random_save=True)
                torch.cuda.empty_cache()  # Clear cache after visualization
                
            # Checkpointing
            if val_metrics['dice'] > best_metric + self.config.early_stop_delta:
                best_metric = val_metrics['dice']
                self.checkpointer(self.model, epoch, val_metrics)
                
            # Early stopping
            if self.early_stopper(val_metrics['dice']):
                print(f"Early stopping at epoch {epoch}")
                break
                
            print(f"Epoch {epoch+1}/{self.config.num_workers} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Dice: {val_metrics['dice']:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Periodic full memory cleanup
            if epoch % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
        return self.model

# Usage
if __name__ == "__main__":
    config = TrainingConfig()
    trainer = SegmentationTrainer(config)
    trained_model = trainer.train()
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()