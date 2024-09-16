import hashlib
import zipfile
import shutil
import os
import argparse


def calculate_sha1(file_path):
    """Calculate SHA1 hash of a file."""
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)  # Read in chunks of 64KB
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def unzip_and_copy(zip_file, destination):
    """Unzip the file and copy its contents to the destination folder."""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Temporarily extract files to /tmp
        zip_ref.extractall('/tmp/unzipped_DB')

    # Copy the contents to the destination
    if not os.path.exists(destination):
        os.makedirs(destination)

    for item in os.listdir('/tmp/unzipped_DB'):
        s = os.path.join('/tmp/unzipped_DB', item)
        d = os.path.join(destination, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    # Clean up the temporary unzipped folder
    shutil.rmtree('/tmp/unzipped_DB')


def main(zip_file):
    # Step 1: Calculate SHA-1 of the given zip file
    signature = calculate_sha1(zip_file)
    print(f"Calculated SHA-1 of {zip_file}: {signature}")

    # Step 2: Read the signature from signature.txt
    signature_file = 'signature.txt'
    with open(signature_file, 'r') as f:
        stored_signature = f.read().strip()
    print(f"Signature from {signature_file}: {stored_signature}")

    # Step 3: Ask the user if the signatures match
    user_input = input("Are the signatures equal? (y/n): ").strip().lower()

    if user_input == 'y':
        # Step 4: Unzip and copy to ./Part_4
        destination_folder = './Part_4'
        unzip_and_copy(zip_file, destination_folder)
        print(f"Contents of {zip_file} unzipped \
              and copied to {destination_folder}.")
        print("Operation completed successfully!\
              Now you can proceed to predict.py.")
        # python3 predict.py splitted/datasets/Apples/training/Apples
        # splitted/datasets/Apples/validation/Apples -batch

        # python3 predict.py splitted/datasets/Grapes/training/Grapes/
        # splitted/datasets/Grapes/validation/Grapes -batch
    else:
        print("Signatures do not match. Aborting operation.")


if __name__ == "__main__":
    # Using argparse to take DB.zip as an argument
    parser = argparse.ArgumentParser(description="Verify\
                                    and unzip a zip file if signatures match.")
    parser.add_argument("zip_file", help="Path to the zip file (e.g., DB.zip)")
    args = parser.parse_args()

    # Run the main function with the passed zip file argument
    main(args.zip_file)
