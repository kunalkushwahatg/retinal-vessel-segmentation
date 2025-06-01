from metrics import EvaluationMetrics
import torch
import matplotlib.pyplot as plt
import numpy as np
from trainingconfig import TrainingConfig

class Evaluation:
    def __init__(self,dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.config = TrainingConfig()
        self.metrics = EvaluationMetrics()
        self.results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            'f1_score': [],
            'iou': [],
            'miou': []
        }

    def plot_images(self, images, masks, outputs,epoch):
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(images[0, 0].cpu().numpy())
        ax[0].set_title('Input Image')
        ax[1].imshow(masks[0, 0].cpu().numpy(), cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[2].imshow(outputs[0, 0].cpu().detach().numpy(), cmap='gray')
        ax[2].set_title('Model Output')
        
        #save the plot
        plt.savefig(self.config.output_dir + f'/predictions/epoch_{epoch}_images.png')
        plt.close()

    def evaluate(self, model, epoch):
        model.eval()
        accuracy = precision = recall = specificity = f1_score = iou = miou = 0.0
        with torch.no_grad():
            for images, masks in self.dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = model(images)
                #apply sigmoid to get probabilities
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()

                values = self.metrics(outputs, masks)

                accuracy += values['accuracy']
                precision += values['precision']
                recall += values['recall']
                specificity += values['specificity']
                f1_score += values['f1_score']
                iou += values['iou']
                miou += values['miou']

            self.plot_images(images, masks, outputs, epoch=epoch)

        num_batches = len(self.dataloader)
        self.results['accuracy'].append(accuracy / num_batches)
        self.results['precision'].append(precision / num_batches)
        self.results['recall'].append(recall / num_batches)
        self.results['specificity'].append(specificity / num_batches)
        self.results['f1_score'].append(f1_score / num_batches)
        self.results['iou'].append(iou / num_batches)
        self.results['miou'].append(miou / num_batches)



        model.train()  # Set model back to training mode    
        return {
            'accuracy': self.results['accuracy'][-1],
            'precision': self.results['precision'][-1],
            'recall': self.results['recall'][-1],
            'specificity': self.results['specificity'][-1],
            'f1_score': self.results['f1_score'][-1],
            'iou': self.results['iou'][-1],
            'miou': self.results['miou'][-1]
        }

    
    def plot_results(self):
        epochs = np.arange(1, len(self.results['accuracy']) + 1,self.config.eval_every)
        # Plotting the evaluation metrics
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.results['accuracy'], label='Accuracy')
        plt.plot(epochs, self.results['precision'], label='Precision')
        plt.plot(epochs, self.results['recall'], label='Recall')
        plt.plot(epochs, self.results['specificity'], label='Specificity')
        plt.plot(epochs, self.results['f1_score'], label='F1 Score')
        plt.plot(epochs, self.results['iou'], label='IoU')
        plt.plot(epochs, self.results['miou'], label='mIoU')
        
        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics Over Epochs')
        plt.legend()
        plt.grid()
        
        #save the plot
        plt.savefig(self.config.output_dir + '/evaluation_metrics_plot.png')
        # save the results to a text file
        with open(self.config.output_dir + '/evaluation_results.txt', 'w') as f:
            for key, values in self.results.items():
                f.write(f"{key}: {values}\n")
        plt.close()

