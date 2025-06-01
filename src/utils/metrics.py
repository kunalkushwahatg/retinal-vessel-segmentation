import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

class EvaluationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def _binarize(self, pred):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        return (pred >= self.threshold).astype(np.uint8)

    def _prepare(self, pred, target):
        pred = self._binarize(pred)
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        target = target.astype(np.uint8)
        return pred, target

    def accuracy(self, pred, target):
        pred, target = self._prepare(pred, target)
        return np.mean(pred == target)

    def precision(self, pred, target):
        pred, target = self._prepare(pred, target)
        tp = np.sum((pred == 1) & (target == 1))
        fp = np.sum((pred == 1) & (target == 0))
        return tp / (tp + fp + 1e-8)

    def recall(self, pred, target):
        pred, target = self._prepare(pred, target)
        tp = np.sum((pred == 1) & (target == 1))
        fn = np.sum((pred == 0) & (target == 1))
        return tp / (tp + fn + 1e-8)

    def specificity(self, pred, target):
        pred, target = self._prepare(pred, target)
        tn = np.sum((pred == 0) & (target == 0))
        fp = np.sum((pred == 1) & (target == 0))
        return tn / (tn + fp + 1e-8)

    def f1_score(self, pred, target):
        p = self.precision(pred, target)
        r = self.recall(pred, target)
        return 2 * p * r / (p + r + 1e-8)

    def iou(self, pred, target):
        pred, target = self._prepare(pred, target)
        intersection = np.sum((pred == 1) & (target == 1))
        union = np.sum((pred == 1) | (target == 1))
        return intersection / (union + 1e-8)

    def miou(self, pred, target):
        # For binary segmentation, mIoU = IoU of class 0 and class 1 averaged
        pred, target = self._prepare(pred, target)
        iou_1 = self.iou(pred, target)
        pred_inv = 1 - pred
        target_inv = 1 - target
        iou_0 = np.sum((pred_inv == 1) & (target_inv == 1)) / (np.sum((pred_inv == 1) | (target_inv == 1)) + 1e-8)
        return (iou_0 + iou_1) / 2

    def auc(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        pred = pred.flatten()
        target = target.flatten()
        try:
            return roc_auc_score(target, pred)
        except:
            return 0.0  # When only one class exists in y_true

    def __call__(self, pred, target):
        return {
            "accuracy": self.accuracy(pred, target),
            "precision": self.precision(pred, target),
            "recall": self.recall(pred, target),
            "specificity": self.specificity(pred, target),
            "f1_score": self.f1_score(pred, target),
            "iou": self.iou(pred, target),
            "miou": self.miou(pred, target),
            "auc": self.auc(pred, target),
        }


# Example usage:
if __name__ == "__main__":
    # Example tensors
    pred = torch.tensor([[0.1, 0.0], [0.8, 0.2]])
    target = torch.tensor([[0, 1], [1, 0]])

    metrics = EvaluationMetrics(threshold=0.5)
    results = metrics(pred, target)

    for metric_name, value in results.items():
        print(f"{metric_name}: {value:.4f}")