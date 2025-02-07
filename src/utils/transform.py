
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt

class SegmentationTransform:
    def __init__(self, resize=(256, 256), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.image_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

    def __call__(self, image, mask):
        image,mask = self.image_transform(image,mask)
        return image, mask
    
    def denormalize(self, image):
        #take a tensor image and return it in the form of a numpy array
        return image * (torch.tensor(self.std).view(3, 1, 1)) + torch.tensor(self.mean).view(3, 1, 1)


# Example usage
if __name__ == "__main__":

    image = Image.open('data/DRHAGIS/augmented/image/9_2.jpg')
    mask = Image.open('data/DRHAGIS/augmented/output/_2.png')


    transform = SegmentationTransform(resize=(512, 512))
    
    transformed_image = transform.image_transform(image)
    transformed_mask = transform.mask_transform(mask)

    

    print(transformed_image.shape, transformed_mask.shape)

    # Show 2x2 plot with original and transformed images and masks
    plt.figure(figsize=(10, 10))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Original mask
    plt.subplot(2, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Original Mask')
    plt.axis('off')

    # Transformed image
    plt.subplot(2, 2, 3)
    plt.imshow(transformed_image.permute(1, 2, 0))
    plt.title('Transformed Image')
    plt.axis('off')

    # Transformed mask
    plt.subplot(2, 2, 4)
    plt.imshow(transformed_mask[0], cmap='gray')
    plt.title('Transformed Mask')
    plt.axis('off')

    plt.show()

