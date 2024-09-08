import os
import shutil
import zipfile
import wget
import sys
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

# Constants
URL = "https://cdn.intra.42.fr/document/document/17447/leaves.zip"
ZIP_FILE = "leaves.zip"
EXTRACT_DIR = "dataset"
SINGLE_IMAGE_DIR = "test_images"
TARGET_DIR = "augmented_directory"
VALIDATION_DIR = "validation"

def clean_up(clean_all=False):
    """Remove existing 'dataset' directory and related files if they exist."""
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
        print(f"Removed existing directory: {EXTRACT_DIR}")

    if clean_all:
        if os.path.exists(ZIP_FILE):
            os.remove(ZIP_FILE)
            print(f"Removed existing file: {ZIP_FILE}")

        if os.path.exists(SINGLE_IMAGE_DIR):
            shutil.rmtree(SINGLE_IMAGE_DIR)
            print(f"Removed existing directory: {SINGLE_IMAGE_DIR}")

        if os.path.exists(VALIDATION_DIR):
            shutil.rmtree(VALIDATION_DIR)
            print(f"Removed existing directory: {VALIDATION_DIR}")

        if os.path.exists(TARGET_DIR):
            shutil.rmtree(TARGET_DIR)
            print(f"Removed existing directory: {TARGET_DIR}")

def download_and_extract_zip(url, zip_file, extract_dir):
    """Download and extract the ZIP file."""
    if not os.path.exists(zip_file):
        print(f"Downloading {url}...")
        wget.download(url, zip_file)
        print("\nDownload complete.")

    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")

def pick_random_image(extract_dir):
    """Pick a random image from the dataset/images folder and copy it to the SINGLE_IMAGE_DIR."""
    # Check if the SINGLE_IMAGE_DIR directory exists, if so, remove it
    if os.path.exists(SINGLE_IMAGE_DIR):
        shutil.rmtree(SINGLE_IMAGE_DIR)
        print(f"Removed existing directory: {SINGLE_IMAGE_DIR}")

    images_dir = os.path.join(extract_dir, 'images')

    if not os.path.exists(images_dir):
        print(f"Directory {images_dir} does not exist. Ensure the dataset is extracted.")
        sys.exit(1)

    subfolders = [f.path for f in os.scandir(images_dir) if f.is_dir()]

    if not subfolders:
        print(f"No subdirectories found in {images_dir}.")
        sys.exit(1)

    random_folder = random.choice(subfolders)
    images = [f for f in os.listdir(random_folder) if os.path.isfile(os.path.join(random_folder, f))]

    if not images:
        print(f"No images found in {random_folder}.")
        sys.exit(1)

    random_image = random.choice(images)
    src_path = os.path.join(random_folder, random_image)

    # Save the image in SINGLE_IMAGE_DIR
    os.makedirs(SINGLE_IMAGE_DIR, exist_ok=True)
    dest_path = os.path.join(SINGLE_IMAGE_DIR, random_image)

    shutil.copy(src_path, dest_path)
    print(f"Copied random image {random_image} to {dest_path}")

def extract_number_from_image_name(image_name):
    """Extract the number from the image name in parentheses."""
    match = re.search(r'\((\d+)\)', image_name)
    if match:
        return match.group(1)
    return None

def show_images(image_name, folders):
    """Show all images that share the same digit inside the specified folders."""
    number = extract_number_from_image_name(image_name)
    if not number:
        print(f"No number found in the image name {image_name}.")
        sys.exit(1)

    matching_images = []
    for folder in folders:
        folder_path = folder
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist.")
            continue

        for file in os.listdir(folder_path):
            if (file.endswith('.png') or file.endswith('.jpg')) and f'({number})' in file:
                matching_images.append(os.path.join(folder_path, file))

    if not matching_images:
        print(f"No images found with the number {number} in the specified folders.")
        sys.exit(1)

    # Show the input image alone and centered in the first line
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    img = mpimg.imread(image_name)
    axes.imshow(img)
    axes.axis('off')
    plt.show()

    # Show the matching images with a maximum of 3 images per line
    num_images = len(matching_images)
    rows = (num_images + 2) // 3  # Calculate the number of rows needed
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))

    for i, img_path in enumerate(matching_images):
        row = i // 3
        col = i % 3
        img = mpimg.imread(img_path)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')

    for j in range(i + 1, rows * 3):
        row = j // 3
        col = j % 3
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Process dataset options.")
    parser.add_argument('--clean', action='store_true', help="Clean the dataset directory.")
    parser.add_argument('--clean_all', action='store_true', help="Clean all related files and directories.")
    parser.add_argument('--single_image', action='store_true', help="Pick a random image and copy it to the SINGLE_IMAGE_DIR.")
    parser.add_argument('--show', nargs='+', type=str, help="Show all images that share the same digit inside the specified folders. The first argument is the image name.")

    args = parser.parse_args()

    if args.show:
        image_name = args.show[0]
        folders = args.show[1:]
        show_images(image_name, folders)
    elif args.single_image:
        download_and_extract_zip(URL, ZIP_FILE, EXTRACT_DIR)
        pick_random_image(EXTRACT_DIR)
    elif args.clean_all:
        clean_up(True)
        print("Cleanup complete.")
    elif args.clean:
        clean_up()
        print("Cleanup complete.")
    else:
        clean_up()
        download_and_extract_zip(URL, ZIP_FILE, EXTRACT_DIR)
        print("Dataset is ready for use.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
