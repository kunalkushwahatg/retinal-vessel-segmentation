import cv2
import numpy as np

import matplotlib.pyplot as plt

def apply_gabor_filter(image, ksize=31, sigma=3.0, theta=0, lambd=8.0, gamma=0.5, psi=0):
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

    image = image[:,:,1]
    #filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    return image


def plot_images(original, filtered):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Filtered Image')
    plt.imshow(filtered, cmap='gray')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    images_path = 'C:\\Users\\kunal\\retinal-vessel-segmentation\\data\\datasets\\CHASE_DB1\\train\\input\\'  # Replace with your image path
    import os
    from tqdm import tqdm
    file_path = "C:\\Users\\kunal\\retinal-vessel-segmentation\\data\\datasets\\CHASE_DB1\\"
    train  = "train\\input\\"
    save_path = "train\\input_gabor\\"
    if not os.path.exists(os.path.join(file_path, save_path)):
        os.makedirs(os.path.join(file_path, save_path))
    # Loop through all images in the directory
    import cv2
    for filename in tqdm(os.listdir(os.path.join(file_path, train))):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(file_path, train, filename)
            image = cv2.imread(image_path)
            if image is not None:
                filtered_image = apply_gabor_filter(image)
                save_image_path = os.path.join(file_path, save_path, filename)
                cv2.imwrite(save_image_path, filtered_image)
            else:
                print(f"Could not read image {filename}")
    print("Gabor filter applied and images saved successfully.")
    valPath = "val\\input\\"
    save_val_path = "val\\input_gabor\\"
    if not os.path.exists(os.path.join(file_path, save_val_path)):
        os.makedirs(os.path.join(file_path, save_val_path))
    for filename in tqdm(os.listdir(os.path.join(file_path, valPath))):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(file_path, valPath, filename)
            image = cv2.imread(image_path)
            if image is not None:
                filtered_image = apply_gabor_filter(image)
                save_image_path = os.path.join(file_path, save_val_path, filename)
                cv2.imwrite(save_image_path, filtered_image)
            else:
                print(f"Could not read image {filename}")
    print("Gabor filter applied to validation images and saved successfully.")
