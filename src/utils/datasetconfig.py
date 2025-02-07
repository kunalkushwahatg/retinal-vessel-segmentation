class DatasetConfig:
    def __init__(self):
        self.dataset_name = 'full_data'
        self.dataset_dir = 'data/full_data/'
        self.image_dir = self.dataset_dir + 'images/'
        self.mask_dir = self.dataset_dir + 'output/'
        self.image_size = (512, 512)
        self.num_classes = 1