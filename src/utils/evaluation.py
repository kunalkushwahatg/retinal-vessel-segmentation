import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from collections import defaultdict
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.metrics.confusion_matrix import compute_confusion_matrix_metric

class SegmentationEvaluator:
    def __init__(
        self,
        num_classes: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        activation: str = "sigmoid",  # "sigmoid" or "softmax" we use sigmoid for binary segmentation and softmax for multi-class segmentation
        threshold: float = 0.5,
        use_monai: bool = False
    ):
        self.num_classes = num_classes
        self.device = device
        self.activation = activation
        self.threshold = threshold
        self.use_monai = use_monai

        if self.use_monai:
            self.dice_metric = DiceMetric(include_background=False, reduction="mean")
            self.hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
            self.confusion_matrix = []
        else:
            self.metrics = {
                'dice': self._calculate_dice,
                'iou': self._calculate_iou,
                'sensitivity': self._calculate_sensitivity,
                'specificity': self._calculate_specificity
            }

        self.reset()

    def reset(self):
        self.results = defaultdict(list)
        self.all_preds = []
        self.all_targets = []
        if self.use_monai:
            self.dice_metric.reset()
            self.hd_metric.reset()
            self.confusion_matrix = []

    def __call__(self, model, dataloader, loss_func=None):
        model.eval()
        self.reset()
        total_loss = 0.0

        with torch.no_grad():
            for inputs,targets in dataloader:

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = model(inputs)
                preds = self._postprocess(logits)

                if loss_func is not None:
                    loss = loss_func(logits, targets)
                    total_loss += loss.item() * inputs.size(0)

                self._update_metrics(preds, targets)

                if len(self.all_preds) < 10:
                    self.all_preds.append(preds.cpu().numpy())
                    self.all_targets.append(targets.cpu().numpy())

        metrics = self._aggregate_metrics()
        if loss_func is not None:
            metrics['loss'] = total_loss / len(dataloader.dataset)
            
        return metrics

    def _postprocess(self, logits):

        if self.activation == "softmax":
            probs = torch.softmax(logits, dim=1)
        elif self.activation == "sigmoid":
            probs = torch.sigmoid(logits)
        else:
            raise ValueError(f"Invalid activation: {self.activation}")

        if self.num_classes == 1:
            preds = (probs > self.threshold).float()
        else:
            preds = torch.argmax(probs, dim=1, keepdim=True)
        
        return preds

    def _update_metrics(self, preds, targets):
        if self.use_monai:
            # Convert to one-hot format for MONAI metrics
            if self.num_classes > 1:
                targets_onehot = torch.nn.functional.one_hot(
                    targets.squeeze(1).long(), self.num_classes
                ).permute(0, 3, 1, 2).float()
                preds_onehot = torch.nn.functional.one_hot(
                    preds.squeeze(1).long(), self.num_classes
                ).permute(0, 3, 1, 2).float()
            else:
                targets_onehot = targets
                preds_onehot = preds

            self.dice_metric(y_pred=preds_onehot, y=targets_onehot)
            self.hd_metric(y_pred=preds_onehot, y=targets_onehot)
            
            # Store for confusion matrix metrics
            self.confusion_matrix.append((preds_onehot, targets_onehot))
        else:
            for metric_name, metric_fn in self.metrics.items():
                self.results[metric_name].append(metric_fn(preds, targets))

    def _aggregate_metrics(self):
        if self.use_monai:
            dice = self.dice_metric.aggregate().item()
            hd = self.hd_metric.aggregate().item()
            
            # Compute confusion matrix metrics
            if self.confusion_matrix:
                preds = torch.cat([p for p, _ in self.confusion_matrix])
                targets = torch.cat([t for _, t in self.confusion_matrix])
                sensitivity = compute_confusion_matrix_metric(
                    "sensitivity", preds, targets, include_background=False
                ).mean().item()
                specificity = compute_confusion_matrix_metric(
                    "specificity", preds, targets, include_background=False
                ).mean().item()
                iou = compute_confusion_matrix_metric(
                    "iou", preds, targets, include_background=False
                ).mean().item()
            else:
                sensitivity = specificity = iou = 0.0

            return {
                'dice': dice,
                'hausdorff_distance': hd,
                'iou': iou,
                'sensitivity': sensitivity,
                'specificity': specificity
            }
        else:
           #out = {"dice": 0.8, "iou": 0.7, "sensitivity": 0.9, "specificity": 0.6}
           return {k: np.mean(v[0].cpu().numpy()) for k, v in self.results.items()} 
        


    # Keep other methods (visualize_samples, generate_report, custom metrics) the same
    # ...
    def visualize_samples(self, save_path: Optional[str] = None):
        """Visualize predictions vs ground truth"""
        fig, axes = plt.subplots(3, min(3, len(self.all_preds)), figsize=(15, 10))
        
        for i in range(min(3, len(self.all_preds))):
            # Input image
            axes[0, i].imshow(self.all_preds[i][0, 0], cmap='gray')
            axes[0, i].set_title(f"Sample {i+1} - Input")
            
            # Ground truth
            axes[1, i].imshow(self.all_targets[i][0, 0], cmap='jet', alpha=0.5)
            axes[1, i].set_title("Ground Truth")
            
            # Prediction
            axes[2, i].imshow(self.all_preds[i][0, 0], cmap='jet', alpha=0.5)
            axes[2, i].set_title("Prediction")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    # Custom metric calculations (alternative to MONAI)
    def _calculate_dice(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        smooth = 1e-6
        intersection = (preds * targets).sum()
        return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    def _calculate_iou(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        smooth = 1e-6
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    def _calculate_sensitivity(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        tp = (preds * targets).sum()
        fn = ((1 - preds) * targets).sum()
        return tp / (tp + fn + 1e-6)

    def _calculate_specificity(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        tn = ((1 - preds) * (1 - targets)).sum()
        fp = (preds * (1 - targets)).sum()
        return tn / (tn + fp + 1e-6)

    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Generate printable report from metrics"""
        report = [
            "Segmentation Evaluation Report",
            "--------------------------------",
            f"Dice Score: {metrics['dice']:.4f}",
            f"IoU: {metrics['iou']:.4f}",
            f"Hausdorff Distance: {metrics.get('hausdorff_distance', 0):.2f} mm",
            f"Sensitivity: {metrics['sensitivity']:.4f}",
            f"Specificity: {metrics['specificity']:.4f}",
        ]
        if 'loss' in metrics:
            report.insert(2, f"Loss: {metrics['loss']:.4f}")
        return "\n".join(report)
    


# Usage
se =SegmentationEvaluator(num_classes=1)