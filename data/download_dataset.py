import requests
from tqdm import tqdm
import zipfile
import os
import numpy as np
from PIL import Image

CHASE_DB1_URL = "https://github.com/ZombaSY/FSG-Net-pytorch/releases/download/1.1.0/CHASE_DB1.zip"
DRIVE_URL = "https://github.com/ZombaSY/FSG-Net-pytorch/releases/download/1.1.0/DRIVE.zip"
STARE_URL = "https://github.com/ZombaSY/FSG-Net-pytorch/releases/download/1.1.0/STARE.zip"
HRF_URL = "https://github.com/ZombaSY/FSG-Net-pytorch/releases/download/1.1.0/HRF.zip"



def download_dataset(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download file: {response.status_code}")
        return
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024 * 1024  # 1MB
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path,
        total=total_size,
        unit='MB',
        unit_scale=True,
        unit_divisor=1024*1024
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            f.write(data)
            bar.update(len(data))


def download_all_datasets():
    datasets = [
        (CHASE_DB1_URL, "CHASE_DB1.zip"),
        (DRIVE_URL, "DRIVE.zip"),
        (STARE_URL, "STARE.zip"),
        (HRF_URL, "HRF.zip")
    ]
    
    # Create a data directory if it doesn't exist
    data_dir = "./data/datasets"
    os.makedirs(data_dir, exist_ok=True)
    
    for url, filename in datasets:
        zip_path = os.path.join(data_dir, filename)
        # Download the dataset
        download_dataset(url, zip_path)
        # Extract the dataset
        extract_zip(zip_path, data_dir+"/"+filename.split(".")[0])
        #delete the zip file
        os.remove(zip_path)
        print(f"Dataset {filename} downloaded and extracted to {data_dir}/{filename.split('.')[0]}")
        

def extract_zip(zip_path, extract_to=None):
    """
    Extract a zip file to the specified directory.
    If extract_to is None, extracts to the same directory as the zip file.
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete: {zip_path}")


def image_to_array(image_path):
    """
    Convert an image file to numpy array.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Image as numpy array with shape (height, width, channels)
    """
    try:
        # Open image using PIL
        img = Image.open(image_path)
        # Convert to numpy array
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"Error converting image {image_path}: {str(e)}")
        return None

def load_dataset_images(dataset_dir):
    """
    Load all images from a dataset directory into numpy arrays.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        
    Returns:
        dict: Dictionary containing image arrays with their filenames as keys
    """
    image_arrays = {}
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            image_path = os.path.join(dataset_dir, filename)
            img_array = image_to_array(image_path)
            if img_array is not None:
                image_arrays[filename] = img_array
    return image_arrays


if __name__ == "__main__":
    download_all_datasets()
