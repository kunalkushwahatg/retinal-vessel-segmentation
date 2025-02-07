class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_metric = None

    def __call__(self, val_metric):
        if self.best_metric is None:
            self.best_metric = val_metric
        elif val_metric <= self.best_metric + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        else:
            self.best_metric = val_metric
            self.counter = 0
        return False    