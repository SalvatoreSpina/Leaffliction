import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copy
from torchvision import transforms as t
from torchvision.io import read_image, write_jpeg
import torchvision.transforms.functional as F

class ImageAugmentor:
    def __init__(self, directory, augmentation_dir="augmented_directory", validation_dir="validation"):
        self.directory = directory
        self.augmentation_dir = augmentation_dir
        self.validation_dir = validation_dir
        self.transformation_functions = [
            self.flip, self.rotate, self.blur, self.adjust_contrast,
            self.adjust_brightness, self.perspective, self.scale
        ]
        self.num_of_aug = len(self.transformation_functions)

    def flip(self, image, path=None):
        transformed_image = t.RandomHorizontalFlip(1)(image)
        if path:
            write_jpeg(transformed_image, f"{path}_Flip.jpg")
        return transformed_image

    def rotate(self, image, path=None):
        transformed_image = t.functional.rotate(image, 30, interpolation=Image.Resampling.BILINEAR)
        if path:
            write_jpeg(transformed_image, f"{path}_Rotation.jpg")
        return transformed_image

    def blur(self, image, path=None):
        transformed_image = t.GaussianBlur(9)(image)
        if path:
            write_jpeg(transformed_image, f"{path}_Blur.jpg")
        return transformed_image

    def adjust_contrast(self, image, path=None):
        transformed_image = t.functional.adjust_contrast(image, 1.5)
        if path:
            write_jpeg(transformed_image, f"{path}_Contrast.jpg")
        return transformed_image

    def adjust_brightness(self, image, path=None):
        transformed_image = t.ColorJitter((1.8, 2))(image)
        if path:
            write_jpeg(transformed_image, f"{path}_Brightness.jpg")
        return transformed_image

    def perspective(self, image, path=None):
        transformed_image = t.RandomPerspective(0.5, p=1)(image)
        if path:
            write_jpeg(transformed_image, f"{path}_Distortion.jpg")
        return transformed_image

    def scale(self, image, path=None):
        transformed_image = t.RandomAffine(degrees=0, scale=(1.3, 1.7))(image)
        if path:
            write_jpeg(transformed_image, f"{path}_Scaling.jpg")
        return transformed_image

    def augment_images(self, max_augmentations, plant, class_name, image_list):
        """
        Augments images in the specified class directory.
        """
        class_path = os.path.join(self.augmentation_dir, plant, class_name)
        os.makedirs(class_path, exist_ok=True)

        for image_path in image_list:
            copy(image_path, class_path)
            for transform in self.transformation_functions:
                if max_augmentations == 0:
                    break
                save_path = os.path.join(class_path, os.path.splitext(os.path.basename(image_path))[0])
                transform(read_image(image_path), save_path)
                max_augmentations -= 1

        if max_augmentations > 0:
            print("Balancing was not fully achieved.")

    def walk_through_dir(self):
        """
        Walks through the directory and gathers all image paths.
        """
        image_list = []
        for plant in os.listdir(self.directory):
            plant_path = os.path.join(self.directory, plant)
            for disease in os.listdir(plant_path):
                disease_path = os.path.join(plant_path, disease)
                for file in os.listdir(disease_path):
                    file_path = os.path.join(disease_path, file)
                    if os.path.isfile(file_path):
                        image_list.append(file_path)
        return image_list

    def create_dataframe(self, image_list):
        """
        Creates a DataFrame to manage images by plant and disease.
        """
        df = pd.DataFrame(image_list, columns=['filename'])
        df['plant'] = df['filename'].apply(lambda x: x.split(os.sep)[-3])
        df['disease'] = df['filename'].apply(lambda x: x.split(os.sep)[-2])
        return df

    def stratified_sample_df(self, df, col, n_samples):
        """
        Creates a stratified sample DataFrame.
        """
        min_samples = min(n_samples, df[col].value_counts().min())
        sampled_df = df.groupby(col).apply(lambda x: x.sample(min_samples))
        sampled_df.index = sampled_df.index.droplevel(0)
        return sampled_df

    def balance_and_split_dataset(self):
        """
        Splits data into training and validation sets and applies augmentations.
        """
        image_list = self.walk_through_dir()
        dataframe = self.create_dataframe(image_list)

        for plant, frame in dataframe.groupby('plant'):
            aug_counter = {}
            df_test = self.stratified_sample_df(frame, 'disease', 25)
            df_train = frame.drop(df_test.index)

            max_aug_per_class = df_train['disease'].value_counts().min() * self.num_of_aug

            for disease_class in df_train['disease'].value_counts().index:
                aug_for_class = max_aug_per_class - df_train['disease'].value_counts().loc[disease_class]
                aug_counter[disease_class] = aug_for_class

            for img_path, disease_class in zip(df_train['filename'], df_train['disease']):
                train_path = os.path.join(self.augmentation_dir, plant, 'train', disease_class)
                os.makedirs(train_path, exist_ok=True)
                for transform in self.transformation_functions:
                    if aug_counter[disease_class] > 0:
                        save_path = os.path.join(train_path, os.path.splitext(os.path.basename(img_path))[0])
                        transform(read_image(img_path), save_path)
                        aug_counter[disease_class] -= 1
                copy(img_path, train_path)

            for img_path, disease_class in zip(df_test['filename'], df_test['disease']):
                val_path = os.path.join(self.validation_dir, plant, disease_class)
                os.makedirs(val_path, exist_ok=True)
                copy(img_path, val_path)

    def create_balanced_dataset(self):
        """
        Creates a balanced dataset by augmenting the images.
        """
        os.makedirs(self.augmentation_dir, exist_ok=True)
        image_list = self.walk_through_dir()
        dataframe = self.create_dataframe(image_list)

        for plant, frame in dataframe.groupby('plant'):
            print(f"Class: {plant}")
            min_class_count = frame['disease'].value_counts().min()
            max_aug_per_class = min_class_count * self.num_of_aug

            for disease_class in frame['disease'].value_counts().index:
                aug_for_class = max_aug_per_class - frame['disease'].value_counts().loc[disease_class]
                print(f"{disease_class}: {aug_for_class} augmentations needed.")
                self.augment_images(
                    aug_for_class, plant, disease_class,
                    dataframe[(dataframe['disease'] == disease_class) & (dataframe['plant'] == plant)]['filename']
                )

    def show(self, imgs, labels):
        """
        Displays the images with labels.
        """
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = F.to_pil_image(img.detach())
            axs[0, i].set_title(labels[i])
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].axis('off')
        plt.show()

    def one_image(self, img_path):
        """
        Applies augmentations to a single image and displays the original and augmented images.
        """
        img = read_image(img_path)
        augmented_images = [transform(img) for transform in self.transformation_functions]
        labels = ['flip', 'rotate', 'blur', 'contrast', 'brightness', 'perspective', 'scaling']
        
        # Show original image along with augmented images
        self.show([img] + augmented_images, labels=['original'] + labels)
        
        # Save the augmented images
        for i, image in enumerate(augmented_images):
            write_jpeg(image, f"{os.path.splitext(img_path)[0]}_{labels[i]}.jpg")


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='Augmentation',
        description='Program that either applies augmentations to a single image or creates a balanced dataset from a directory.'
    )
    parser.add_argument('directory', metavar='directory', type=str, nargs=1,
                        help='Directory containing images to augment.')
    return parser.parse_args()


if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = (10, 10)
    args = parse_arguments()
    path = args.directory[0]

    if not os.path.exists(path):
        sys.exit("The directory does not exist or is not accessible.")
    
    augmentor = ImageAugmentor(directory=path)

    if os.path.isfile(path):
        augmentor.one_image(path)
    elif os.path.isdir(path):
        augmentor.balance_and_split_dataset()
