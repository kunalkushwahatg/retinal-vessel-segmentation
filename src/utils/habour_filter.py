import cv2
import numpy as np

import matplotlib.pyplot as plt

def apply_gabor_filter(image, ksize=31, sigma=3.0, theta=0, lambd=8.0, gamma=0.5, psi=0):
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    return filtered_image

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
    images_path = 'data/full_data/images_green/'  # Replace with your image path
    import os
    from tqdm import tqdm
    os.mkdir('data/full_data/images_green_gabor')
    for image_name in tqdm(os.listdir(images_path)):
        image = cv2.imread(images_path + image_name, cv2.IMREAD_GRAYSCALE)
        filtered_image = apply_gabor_filter(image, ksize=31, sigma=3.0, theta=135, lambd=7.0, gamma=1.0)
        cv2.imwrite('data/full_data/images_green_gabor/' + image_name, filtered_image)
        #plot_images(image, filtered_image)
        print(image_name)
    #for lamda 7 and sigma 3 gamma 1 theta 135 kernel size 31
    
        
    