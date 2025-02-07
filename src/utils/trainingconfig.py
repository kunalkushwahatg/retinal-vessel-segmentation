import torch
class TrainingConfig:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = 'unetpp_revised'
        self.debug = False
        self.num_epochs = 25
        self.batch_size = 2 if self.device.type == 'cuda' else 2
        self.learning_rate = 1e-3
        self.eval_every = 1
        self.patience = 10
        self.output_dir = f'results/{self.model_name}'
        self.num_workers = 4 if self.device.type == 'cuda' else 0
        self.gradient_clip = 1.0
        self.early_stop_delta = 0.01
        self.train_test_split_ratio = 0.8
        self.seed = 42
        