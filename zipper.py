import os
import zipfile
import argparse


def zip_folder(folder_path, output_path):
    """Zip the entire folder and its contents."""
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Add the file to the zip archive,
                # maintaining the folder structure
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip a folder\
                                     and its contents.")
    parser.add_argument("folder_path", help="Path to the folder to zip\
                        (e.g., ./DB)")
    parser.add_argument("output_path", help="Output zip file\
                        (e.g., DB.zip)")
    args = parser.parse_args()

    zip_folder(args.folder_path, args.output_path)
    print(f"Folder '{args.folder_path}'\
          has been zipped as '{args.output_path}'")
