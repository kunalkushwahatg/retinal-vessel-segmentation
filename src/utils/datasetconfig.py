class DatasetConfig:
    def __init__(self):
        self.dataset_name = 'DRHAGIS'
        self.dataset_dir = 'data/DRHAGIS/augmented/'
        self.image_dir = self.dataset_dir + 'images_green_gabor/'
        self.mask_dir = self.dataset_dir + 'mask/'
        self.image_size = (512, 512)
        self.num_classes = 1
        self.input_channels = 1