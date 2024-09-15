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
    def __init__(self, source_directory, augmented_output_directory="augmented_directory", validation_output_directory="validation"):
        self.source_directory = source_directory
        self.augmented_output_directory = augmented_output_directory
        self.validation_output_directory = validation_output_directory
        self.transformation_functions = [
            self.apply_flip, self.apply_rotation, self.apply_blur, 
            self.adjust_contrast, self.adjust_brightness, 
            self.apply_perspective_transform, self.apply_scaling
        ]
        self.number_of_transformations = len(self.transformation_functions)

    def apply_flip(self, image, save_path=None):
        flipped_image = t.RandomHorizontalFlip(1)(image)
        if save_path:
            write_jpeg(flipped_image, f"{save_path}_Flip.jpg")
        return flipped_image

    def apply_rotation(self, image, save_path=None):
        rotated_image = t.functional.rotate(image, 30, interpolation=Image.Resampling.BILINEAR)
        if save_path:
            write_jpeg(rotated_image, f"{save_path}_Rotation.jpg")
        return rotated_image

    def apply_blur(self, image, save_path=None):
        blurred_image = t.GaussianBlur(9)(image)
        if save_path:
            write_jpeg(blurred_image, f"{save_path}_Blur.jpg")
        return blurred_image

    def adjust_contrast(self, image, save_path=None):
        contrast_adjusted_image = t.functional.adjust_contrast(image, 1.5)
        if save_path:
            write_jpeg(contrast_adjusted_image, f"{save_path}_Contrast.jpg")
        return contrast_adjusted_image

    def adjust_brightness(self, image, save_path=None):
        brightness_adjusted_image = t.ColorJitter((1.8, 2))(image)
        if save_path:
            write_jpeg(brightness_adjusted_image, f"{save_path}_Brightness.jpg")
        return brightness_adjusted_image

    def apply_perspective_transform(self, image, save_path=None):
        perspective_transformed_image = t.RandomPerspective(0.5, p=1)(image)
        if save_path:
            write_jpeg(perspective_transformed_image, f"{save_path}_Distortion.jpg")
        return perspective_transformed_image

    def apply_scaling(self, image, save_path=None):
        scaled_image = t.RandomAffine(degrees=0, scale=(1.3, 1.7))(image)
        if save_path:
            write_jpeg(scaled_image, f"{save_path}_Scaling.jpg")
        return scaled_image

    def augment_images_in_class(self, max_augmentations, plant_name, disease_class, image_paths):
        """
        Augments images in the specified class directory.
        """
        class_output_path = os.path.join(self.augmented_output_directory, plant_name, disease_class)
        os.makedirs(class_output_path, exist_ok=True)

        for image_path in image_paths:
            copy(image_path, class_output_path)
            for transformation in self.transformation_functions:
                if max_augmentations == 0:
                    break
                save_path = os.path.join(class_output_path, os.path.splitext(os.path.basename(image_path))[0])
                transformation(read_image(image_path), save_path)
                max_augmentations -= 1

        if max_augmentations > 0:
            print("Balancing was not fully achieved.")

    def gather_image_paths(self):
        """
        Walks through the directory and gathers all image paths.
        """
        image_paths = []
        for plant_name in os.listdir(self.source_directory):
            plant_directory_path = os.path.join(self.source_directory, plant_name)
            for disease_class in os.listdir(plant_directory_path):
                disease_directory_path = os.path.join(plant_directory_path, disease_class)
                for image_file in os.listdir(disease_directory_path):
                    image_file_path = os.path.join(disease_directory_path, image_file)
                    if os.path.isfile(image_file_path):
                        image_paths.append(image_file_path)
        return image_paths

    def create_image_dataframe(self, image_paths):
        """
        Creates a DataFrame to manage images by plant and disease.
        """
        image_dataframe = pd.DataFrame(image_paths, columns=['image_file_path'])
        image_dataframe['plant_name'] = image_dataframe['image_file_path'].apply(lambda x: x.split(os.sep)[-3])
        image_dataframe['disease_class'] = image_dataframe['image_file_path'].apply(lambda x: x.split(os.sep)[-2])
        return image_dataframe

    def stratified_sample_dataframe(self, dataframe, column_name, sample_size):
        """
        Creates a stratified sample DataFrame.
        """
        min_sample_size = min(sample_size, dataframe[column_name].value_counts().min())
        stratified_sample = dataframe.groupby(column_name).apply(lambda x: x.sample(min_sample_size))
        stratified_sample.index = stratified_sample.index.droplevel(0)
        return stratified_sample

    def balance_and_split_dataset(self, no_validation=False):
        """
        Splits data into training and validation sets and applies augmentations.
        If no_validation is set to True, it will only create the training set without splitting into validation.
        """
        image_paths = self.gather_image_paths()
        image_dataframe = self.create_image_dataframe(image_paths)

        for plant_name, group_dataframe in image_dataframe.groupby('plant_name'):
            augmentation_count = {}

            if not no_validation:
                validation_dataframe = self.stratified_sample_dataframe(group_dataframe, 'disease_class', 25)
                training_dataframe = group_dataframe.drop(validation_dataframe.index)
            else:
                training_dataframe = group_dataframe

            max_augmentations_per_class = training_dataframe['disease_class'].value_counts().min() * self.number_of_transformations

            for disease_class in training_dataframe['disease_class'].value_counts().index:
                augmentations_needed = max_augmentations_per_class - training_dataframe['disease_class'].value_counts().loc[disease_class]
                augmentation_count[disease_class] = augmentations_needed

            for image_path, disease_class in zip(training_dataframe['image_file_path'], training_dataframe['disease_class']):
                training_output_path = os.path.join(self.augmented_output_directory, plant_name, 'train', disease_class)
                os.makedirs(training_output_path, exist_ok=True)
                for transformation in self.transformation_functions:
                    if augmentation_count[disease_class] > 0:
                        save_path = os.path.join(training_output_path, os.path.splitext(os.path.basename(image_path))[0])
                        transformation(read_image(image_path), save_path)
                        augmentation_count[disease_class] -= 1
                copy(image_path, training_output_path)

            if not no_validation:
                for image_path, disease_class in zip(validation_dataframe['image_file_path'], validation_dataframe['disease_class']):
                    validation_output_path = os.path.join(self.validation_output_directory, plant_name, disease_class)
                    os.makedirs(validation_output_path, exist_ok=True)
                    copy(image_path, validation_output_path)

    def create_balanced_dataset(self):
        """
        Creates a balanced dataset by augmenting the images.
        """
        os.makedirs(self.augmented_output_directory, exist_ok=True)
        image_paths = self.gather_image_paths()
        image_dataframe = self.create_image_dataframe(image_paths)

        for plant_name, group_dataframe in image_dataframe.groupby('plant_name'):
            print(f"Class: {plant_name}")
            min_class_count = group_dataframe['disease_class'].value_counts().min()
            max_augmentations_per_class = min_class_count * self.number_of_transformations

            for disease_class in group_dataframe['disease_class'].value_counts().index:
                augmentations_needed = max_augmentations_per_class - group_dataframe['disease_class'].value_counts().loc[disease_class]
                print(f"{disease_class}: {augmentations_needed} augmentations needed.")
                self.augment_images_in_class(
                    augmentations_needed, plant_name, disease_class,
                    image_dataframe[(image_dataframe['disease_class'] == disease_class) & (image_dataframe['plant_name'] == plant_name)]['image_file_path']
                )

    def display_images_with_labels(self, images, labels):
        """
        Displays the images with labels.
        """
        fig, axs = plt.subplots(ncols=len(images), squeeze=False)
        for i, image in enumerate(images):
            image = F.to_pil_image(image.detach())
            axs[0, i].set_title(labels[i])
            axs[0, i].imshow(np.asarray(image))
            axs[0, i].axis('off')
        plt.show()

    def augment_single_image(self, image_path):
        """
        Applies augmentations to a single image and displays the original and augmented images.
        """
        image = read_image(image_path)
        augmented_images = [transformation(image) for transformation in self.transformation_functions]
        labels = ['Flip', 'Rotate', 'Blur', 'Contrast', 'Brightness', 'Perspective', 'Scaling']
        
        # Show original image along with augmented images
        self.display_images_with_labels([image] + augmented_images, labels=['Original'] + labels)
        
        # Save the augmented images
        for i, augmented_image in enumerate(augmented_images):
            write_jpeg(augmented_image, f"{os.path.splitext(image_path)[0]}_{labels[i]}.jpg")

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='ImageAugmentation',
        description='Program that applies augmentations to a single image or creates a balanced dataset from a directory of images.'
    )
    parser.add_argument('source_directory', metavar='source_directory', type=str, nargs=1,
                        help='Directory containing images to augment.')
    parser.add_argument('-no_validation', action='store_true',
                        help='If set, the dataset will not be split into training and validation.')
    return parser.parse_args()


if __name__ == '__main__':

    try:
        plt.rcParams["figure.figsize"] = (10, 10)
        args = parse_arguments()
        directory_path = args.source_directory[0]

        if not os.path.exists(directory_path):
            sys.exit("The directory does not exist or is not accessible.")

        augmentor = ImageAugmentor(source_directory=directory_path)

        if os.path.isfile(directory_path):
            augmentor.augment_single_image(directory_path)
        elif os.path.isdir(directory_path):
            if args.no_validation:
                augmentor.create_balanced_dataset()
            else:
                augmentor.balance_and_split_dataset()
    except Exception as e:
        print(e)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
