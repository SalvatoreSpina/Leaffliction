import os
import subprocess
import shutil
import zipfile
import wget
import argparse
import sys

# Constants
URL = "https://cdn.intra.42.fr/document/document/17447/leaves.zip"
ZIP_FILE = "leaves.zip"
EXTRACT_DIR = "dataset"
APPLE_DIR = "Apple"

def clean_up():
    """Remove existing 'dataset' and 'Apple' directories if they exist."""
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
        print(f"Removed existing directory: {EXTRACT_DIR}")
    
    if os.path.exists(APPLE_DIR):
        shutil.rmtree(APPLE_DIR)
        print(f"Removed existing directory: {APPLE_DIR}")
    
    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
        print(f"Removed existing file: {ZIP_FILE}")

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

def prepare_apple_test(extract_dir, apple_dir):
    """Prepare the Apple test by organizing the necessary directories."""
    if os.path.exists(apple_dir):
        shutil.rmtree(apple_dir)

    os.makedirs(apple_dir)

    # The Apple subdirectories are inside 'images/' in the extracted dataset
    source_dir = os.path.join(extract_dir, "images")
    
    # Move all subdirectories containing "apple" in their name
    for subdir in os.listdir(source_dir):
        if "apple" in subdir.lower():
            src = os.path.join(source_dir, subdir)
            dest = os.path.join(apple_dir, subdir)
            if os.path.exists(src):
                shutil.move(src, dest)

    print(f"Apple test environment set up in {apple_dir}.")

def run_distribution(directory, save=False, save_path=None):
    """Run the Distribution.py script with the specified directory."""
    command = ["python3", "Distribution.py", directory]
    
    if save:
        command.append("--save")
        if save_path:
            command.extend(["--save_path", save_path])
    
    subprocess.run(command)

def apple_test():
    """Run the Apple test."""
    print("Running Apple test...")
    download_and_extract_zip(URL, ZIP_FILE, EXTRACT_DIR)
    prepare_apple_test(EXTRACT_DIR, APPLE_DIR)
    run_distribution(APPLE_DIR, save=True, save_path=APPLE_DIR)
    print("Apple test completed.")

def full_test():
    """Run the Full test."""
    print("Running Full test...")
    download_and_extract_zip(URL, ZIP_FILE, EXTRACT_DIR)
    run_distribution(EXTRACT_DIR, save=True, save_path=EXTRACT_DIR)
    print("Full test completed.")

def no_test():
    """Just download and unzip the file without running any tests."""
    print("Running no test...")
    download_and_extract_zip(URL, ZIP_FILE, EXTRACT_DIR)
    print("Download and extraction completed.")

def main(test_type):
    clean_up()

    if test_type == "apple":
        apple_test()
    elif test_type == "full":
        full_test()
    elif test_type == "notest":
        no_test()
    else:
        print("Invalid test_type. Use 'apple', 'full', or 'notest'.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Tester for the Distribution.py script')
        parser.add_argument('test_type', choices=['apple', 'full', 'notest'], help="Specify the type of test: 'apple', 'full', or 'notest'")
        
        args = parser.parse_args()
        main(args.test_type)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
