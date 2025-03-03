import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def get_augmentation_pipeline():
    return A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ElasticTransform(p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],is_check_shapes=False)

def augment_image(image, mask):
    augmentation_pipeline = get_augmentation_pipeline()
    augmented = augmentation_pipeline(image=image, mask=mask)
    return augmented['image'], augmented['mask']


def augment_folder(image_folder, mask_folder, output_folder, num_augmentations=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    os.makedirs(os.path.join(output_folder, 'image'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'mask'), exist_ok=True)

    image_list = os.listdir(image_folder) #image name is like 1.jpg, 2.jpg, 3.jpg
    mask_list = os.listdir(mask_folder) #mask name is like 1_manual_orig.png, 2_manual_orig.png, 3_manual_orig.png




    print(f'Augmenting {len(image_list)} images and masks...')
    counter = 199
    for i in tqdm(range(len(image_list) - 1)):
        image_path = os.path.join(image_folder,str(i+1)+'.jpg')
        mask_path = os.path.join(mask_folder,str(i+1)+'.png')

        print("image path", image_path)
        print("mask path", mask_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        #confirm that image and mask are the same size if not that skip the image
        if image.shape[:2] != mask.shape:
            print("path", image_path)
            print("mask", mask_path)
            continue

        #mask = mask[:, :, None]
        
        for j in range(num_augmentations):
            augmented_image, augmented_mask = augment_image(image, mask)
            augmented_image = augmented_image.permute(1, 2, 0).numpy()
            augmented_mask = augmented_mask.squeeze(0).numpy()

            augmented_image_path = os.path.join(output_folder, 'image', f'{counter}.jpg')
            augmented_mask_path = os.path.join(output_folder, 'mask', f'{counter}.png')

            cv2.imwrite(augmented_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(augmented_mask_path, augmented_mask)

            counter += 1




    

if __name__ == '__main__':
    # Load image and mask
    image_path = 'data/DRHAGIS/image/1.jpg'
    mask_path = 'data/DRHAGIS/output/1_manual_orig.png'
    output_path = 'data/DRHAGIS/augmented/'
    
    #augment folder
    image_folder = 'data/DRHAGIS/image'
    mask_folder = 'data/DRHAGIS/output'
    output_folder = 'data/DRHAGIS/augmented'
    augment_folder(image_folder, mask_folder, output_folder, num_augmentations=5)

