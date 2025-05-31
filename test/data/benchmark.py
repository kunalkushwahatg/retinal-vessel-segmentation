import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

class ImageSegmentationBenchmark:
    def __init__(self, true_masks, predicted_masks):
        self.true_masks = true_masks
        self.predicted_masks = predicted_masks

    def calculate_accuracy(self):
        return accuracy_score(self.true_masks.flatten(), self.predicted_masks.flatten())

    def calculate_precision(self):
        return precision_score(self.true_masks.flatten(), self.predicted_masks.flatten(), average='weighted')

    def calculate_recall(self):
        return recall_score(self.true_masks.flatten(), self.predicted_masks.flatten(), average='weighted')

    def calculate_f1_score(self):
        return f1_score(self.true_masks.flatten(), self.predicted_masks.flatten(), average='weighted')

    def calculate_jaccard_index(self):
        return jaccard_score(self.true_masks.flatten(), self.predicted_masks.flatten(), average='weighted')

    def get_all_metrics(self):
        metrics = {
            'Accuracy': self.calculate_accuracy(),
            'Precision': self.calculate_precision(),
            'Recall': self.calculate_recall(),
            'F1 Score': self.calculate_f1_score(),
            'Jaccard Index': self.calculate_jaccard_index()
        }
        return metrics

# Example usage:
# true_masks = np.array([...])  # Replace with actual true masks
# predicted_masks = np.array([...])  # Replace with actual predicted masks
# benchmark = ImageSegmentationBenchmark(true_masks, predicted_masks)
# metrics = benchmark.get_all_metrics()
# print(metrics)