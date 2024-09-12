#!/bin/bash

# Step 1: Create a balanced dataset
echo "Running tester.py for dataset balancing..."
python3 tester.py notest

# Step 2: Split the dataset
echo "Splitting dataset into training and validation..."
python3 split.py extracted_data/images splitted -data

# Step 3: Run wrapper script
echo "Running wrapper.sh..."
sh wrapper.sh

# Step 4: Augment training data for Apples
echo "Augmenting Apples training data..."
python3 ../Part_2/Augmentation.py splitted/datasets/Apples/training -no_validation

# Replace Apples training images with augmented ones
echo "Replacing Apples training images with augmented images..."
rm -rf splitted/datasets/Apples/training/Apples
mv augmented_directory/Apples splitted/datasets/Apples/training/

# Step 5: Augment training data for Grapes
echo "Augmenting Grapes training data..."
python3 ../Part_2/Augmentation.py splitted/datasets/Grapes/training -no_validation

# Replace Grapes training images with augmented ones
echo "Replacing Grapes training images with augmented images..."
rm -rf splitted/datasets/Grapes/training/Grapes
mv augmented_directory/Grapes splitted/datasets/Grapes/training/

echo "Full dataset creation and augmentation process completed."

# Step 6: Run the full dataset through the model
python3 train.py splitted/datasets/Apples/training/Apples
python3 train.py splitted/datasets/Apples/training/Grapes
