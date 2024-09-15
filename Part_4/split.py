import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

def create_dir_structure(root_dir, categories):
    """
    Create the folder structure for each category in training and validation directories.
    """
    os.makedirs(root_dir, exist_ok=True)
    for category in categories:
        os.makedirs(os.path.join(root_dir, category), exist_ok=True)

def copy_files(file_list, destination):
    """
    Copy files from the list to the destination folder.
    """
    os.makedirs(destination, exist_ok=True)  # Ensure the destination directory exists
    for file_path in file_list:
        shutil.copy(file_path, destination)

def split_dataset(dataset_dir, output_dir, split_ratio):
    categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    # Create Apple and Grape folder structure within 'datasets'
    apples_dir = os.path.join(output_dir, 'Apples')
    grapes_dir = os.path.join(output_dir, 'Grapes')
    os.makedirs(apples_dir, exist_ok=True)
    os.makedirs(grapes_dir, exist_ok=True)

    # Split each category's files into training and validation sets
    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        files = [os.path.join(category_path, f) for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        
        # Split the files into training and validation sets
        train_files, val_files = train_test_split(files, train_size=split_ratio)

        if 'Apple' in category:
            train_category_path = os.path.join(apples_dir, 'training', category)
            val_category_path = os.path.join(apples_dir, 'validation', category)
        elif 'Grape' in category:
            train_category_path = os.path.join(grapes_dir, 'training', category)
            val_category_path = os.path.join(grapes_dir, 'validation', category)
        else:
            continue

        # Create directories for each category
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(val_category_path, exist_ok=True)

        # Copy files to respective folders
        copy_files(train_files, train_category_path)
        copy_files(val_files, val_category_path)

def copy_folders(dataset_dir, output_dirs):
    """
    Mimic the behavior of the original bash script by copying specific folders.
    """
    categories = {
        "Grapes": ["Grape_Black_rot", "Grape_Esca", "Grape_healthy", "Grape_spot"],
        "Apples": ["Apple_Black_rot", "Apple_healthy", "Apple_rust", "Apple_scab"]
    }
    
    for key, category_list in categories.items():
        output_path = output_dirs[key]
        os.makedirs(output_path, exist_ok=True)
        
        for category in category_list:
            source_dir = os.path.join(dataset_dir, category)
            dest_dir = os.path.join(output_path, category)
            shutil.copytree(source_dir, dest_dir)

def ensure_output_dir_exists(output_dir):
    """
    Ensure that the output directory exists, creating it if necessary.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Split dataset or copy folders.")
        parser.add_argument("dataset_dir", help="Path to the dataset directory containing category subdirectories.")
        parser.add_argument("output_dir", help="Base path for the output directory. The script will create 'dataset' or 'datasets' inside this base path.")
        parser.add_argument("--split_ratio", type=float, default=0.85, help="Percentage of data to be used for training. Default is 80%.")
        parser.add_argument("-data", action="store_true", help="If used, split the dataset into training and validation sets. Otherwise, copy folders like the bash script.")

        args = parser.parse_args()

        # Set the appropriate directory structure (either 'dataset' or 'datasets')
        if args.data:
            wrapped_output_dir = os.path.join(args.output_dir, "datasets")
        else:
            wrapped_output_dir = os.path.join(args.output_dir, "dataset")

        # Ensure that the output directory exists
        ensure_output_dir_exists(wrapped_output_dir)

        # If -data is used, split dataset into training and validation sets
        if args.data:
            split_dataset(args.dataset_dir, wrapped_output_dir, args.split_ratio)
        else:
            # Otherwise, just split the folders
            output_dirs = {
                "Grapes": os.path.join(wrapped_output_dir, "Grapes"),
                "Apples": os.path.join(wrapped_output_dir, "Apples")
            }
            copy_folders(args.dataset_dir, output_dirs)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
