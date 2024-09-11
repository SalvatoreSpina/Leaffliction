import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import argparse

def fetch_class_names(directory):
    """
    Fetch class names from the subdirectory structure in the training directory.
    """
    return sorted([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

def classify_and_display(image_path, model, class_names):
    # Load and preprocess the image
    img = Image.open(image_path)
    img_resized = img.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_name = class_names[predicted_class]

    # Create a subplot with two images (original and a version of the processed image)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display the original image
    ax[0].imshow(img)
    ax[0].axis('off')
    
    # Display the processed image (for simplicity, use the same image but can be enhanced further)
    ax[1].imshow(img)
    ax[1].axis('off')
    
    # Add a title and predicted class
    fig.suptitle("DL Classification", fontsize=16)
    plt.figtext(0.5, 0.01, f"Class predicted : {class_name}", ha="center", fontsize=12, color="green")

    # Show the plot
    plt.show()

def classify_folder(folder_path, model, class_names):
    """
    Classify all images in a folder and print the predicted vs expected labels.
    """
    correct_predictions = 0
    total_predictions = 0

    for class_name in os.listdir(folder_path):
        print(f"Class: {class_name}")
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                if image_path.lower().endswith(('png', 'jpg', 'jpeg', '.JPG')):
                    img = Image.open(image_path)
                    img_resized = img.resize((150, 150))
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Predict the class
                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions[0])
                    predicted_class_name = class_names[predicted_class]

                    # Print the expected and predicted class
                    print(f"Image: {image_name} | Expected: {class_name} | Predicted: {predicted_class_name}")

                    # Track accuracy
                    total_predictions += 1
                    if predicted_class_name == class_name:
                        correct_predictions += 1
    
    # Calculate and print accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Total Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Classify an image or folder of images using a trained model.')
    parser.add_argument('train_directory', type=str, help='Path to the training directory containing class subdirectories.')
    parser.add_argument('image_or_folder', type=str, help='Path to the image or folder to be classified.')
    parser.add_argument('-range', action='store_true', help='Classify all images in the folder and print the accuracy.')
    args = parser.parse_args()

    # Load the model
    model = tf.keras.models.load_model('output/model.h5')

    class_names = fetch_class_names(args.train_directory)

    if args.range:
        classify_folder(args.image_or_folder, model, class_names)
    else:
        classify_and_display(args.image_or_folder, model, class_names)
