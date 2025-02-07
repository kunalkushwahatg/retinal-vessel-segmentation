import os
from torch.utils.data import Dataset,dataloader
from PIL import Image
from transform import SegmentationTransform
import matplotlib.pyplot as plt


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)
        self.output_list = os.listdir(mask_dir)

        # Sort the images and masks to ensure they are in the same order
        self.image_list.sort()
        self.output_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.output_list[idx])
        
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform.image_transform(image)
            mask = self.transform.mask_transform(mask)
            

        return image, mask


# Example usage
if __name__ == "__main__":
    transform = SegmentationTransform(resize=(512, 512))
    dataset = SegmentationDataset('data/full_data/images/', 'data/full_data/output/', transform)


    #make dataloader
    dataloader = dataloader.DataLoader(dataset,batch_size=2,shuffle=True)

    for i in range(10):
        image, mask = dataset[i]
        print(image.shape, mask.shape)

        # Show 2x2 plot with original and transformed images and masks
        plt.figure(figsize=(10, 10))

        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.title('Original Image')
        plt.axis('off')

        # Original mask
        plt.subplot(2, 2, 2)
        plt.imshow(mask[0], cmap='gray')
        plt.title('Original Mask')
        plt.axis('off')

        plt.show()

