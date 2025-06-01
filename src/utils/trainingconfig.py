import torch
class TrainingConfig:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = 'attention'
        self.in_channels = 3
        self.classes = 1
        self.input_size = 512
        self.debug = False
        self.num_epochs = 1000
        self.batch_size = 2 if self.device.type == 'cuda' else 2
        self.learning_rate = 0.0005
        self.eval_every = 1
        self.patience = 10
        self.output_dir = f'results/{self.model_name}'
        self.num_workers = 4 if self.device.type == 'cuda' else 0
        self.gradient_clip = 1.0
        self.early_stop_delta = 0.001
        self.train_test_split_ratio = 0.8
        self.seed = 42
        self.lr_scheduler = 'ReduceLROnPlateau'
        self.scheduler_patience = 1
        self.scheduler_step_size = 5
        self.prediction_threshold = 0.5 # New config parameter
        self.evaluate_every = 20