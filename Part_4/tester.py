import os
import sys
import wget
import shutil
import zipfile
import argparse
import subprocess

# Constants
URL = "https://cdn.intra.42.fr/document/document/17447/leaves.zip"
ZIP_FILE = "leaves.zip"
EXTRACT_DIR = "extracted_data"
DATASETS = "datasets"
OUTPUT = "output"
SPLITTED = "splitted"
AUGMENTED = "augmented_directory"

def clean_up():
    """Remove existing 'dataset' directory if they exist."""
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
        print(f"Removed existing directory: {EXTRACT_DIR}")
    
    if os.path.exists(DATASETS):
        shutil.rmtree(DATASETS)
        print(f"Removed existing directory: {DATASETS}")
    
    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
        print(f"Removed existing file: {ZIP_FILE}")

def full_clean_up():
    """Remove all existing directories if they exist."""
    
    clean_up()
    
    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)
        print(f"Removed existing directory: {OUTPUT}")
    if os.path.exists(SPLITTED):
        shutil.rmtree(SPLITTED)
        print(f"Removed existing directory: {SPLITTED}")
    if os.path.exists(AUGMENTED):
        shutil.rmtree(AUGMENTED)
        print(f"Removed existing directory: {AUGMENTED}")

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

def no_test():
    """Just download and unzip the file without running any tests."""
    print("Running no test...")
    download_and_extract_zip(URL, ZIP_FILE, EXTRACT_DIR)
    print("Download and extraction completed.")

def move_and_split_dataset(extract_dir):
    subprocess.run(["sh", "split_plants.sh", extract_dir])

def main(test_type):
    clean_up()

    if test_type == "notest":
        no_test()
    elif test_type == "clean":
        full_clean_up()
    elif test_type == "dns":
        print("Running DNS test...")
        download_and_extract_zip(URL, ZIP_FILE, EXTRACT_DIR)
        move_and_split_dataset(EXTRACT_DIR)
        print("Download and extraction completed.")
    else:
        print("Invalid test_type. Use 'notest'.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Tester for the Distribution.py script')
        parser.add_argument('test_type', choices=['notest',  'clean', 'dns'], help="Specify the type of test: 'dns'")
        
        args = parser.parse_args()
        main(args.test_type)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
