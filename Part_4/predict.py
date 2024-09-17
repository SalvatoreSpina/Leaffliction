import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import argparse
from collections import defaultdict


def get_class_labels(training_directory):
    """
    Retrieve the class labels from the
    subdirectory structure of the training directory.
    """
    return sorted([subdir for subdir in os.listdir(training_directory)
                   if os.path.isdir(os.path.join(training_directory, subdir))])


def get_model_name_based_on_folder(training_directory):
    """
    Determine the model name based on whether 'Apple'
    or 'Grapes' is found in the folder name.
    Defaults to 'Model' if neither is found.
    """
    folder_name = os.path.basename(os.path.normpath(training_directory))
    if "apple" in folder_name.lower():
        return "Apple_Model"
    elif "grape" in folder_name.lower():
        return "Grape_Model"
    else:
        return "Model"


def classify_image_and_display_results(image_path,
                                       trained_model, class_labels):
    """
    Classifies a single image and displays
    both the original and processed images
    along with the predicted class.
    """
    original_image = Image.open(image_path)

    # Preprocess the image (resize and normalize):
    # Resize the image for model input
    processed_image = original_image.resize((150, 150))
    # Normalize pixel values
    processed_image_array = np.array(processed_image) / 255.0
    # Add batch dimension for prediction
    processed_image_array = np.expand_dims(processed_image_array, axis=0)

    # Make a prediction using the trained model
    predictions = trained_model.predict(processed_image_array)
    # Get the index of the highest prediction
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]

    # Display the original and processed images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Show the original image
    ax[0].imshow(original_image)
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    # Show the processed image (after resizing and normalization):
    # Show the resized version before normalization
    ax[1].imshow(processed_image)
    ax[1].axis('off')
    ax[1].set_title('Processed Image (Resized)')

    # Add a title with the predicted class
    fig.suptitle(f"Predicted Class: {predicted_class_label}", fontsize=16)

    # Display the images
    plt.show()


def classify_images_in_folder(folder_path, trained_model, class_labels):
    """
    Classifies all images in the specified folder
    and calculates overall accuracy, as well as class-wise accuracy,
    printing the results and showing a bar graph.
    """
    total_correct = 0
    total_images = 0
    class_correct_count = defaultdict(int)
    class_total_count = defaultdict(int)

    for class_name in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder_path):
            for image_filename in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_filename)

                if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
                    # Load and preprocess the image
                    image = Image.open(image_path)
                    resized_image = image.resize((150, 150))
                    image_array = np.array(resized_image) / 255.0
                    image_array = np.expand_dims(image_array, axis=0)

                    # Predict the class
                    predictions = trained_model.predict(image_array)
                    predicted_class_index = np.argmax(predictions[0])
                    predicted_class_label = class_labels[predicted_class_index]

                    # Print the expected vs predicted class
                    print(f"Image: {image_filename} | Expected: {class_name}\
                           | Predicted: {predicted_class_label}")

                    # Track overall and class-specific accuracy
                    total_images += 1
                    class_total_count[class_name] += 1
                    if predicted_class_label == class_name:
                        total_correct += 1
                        class_correct_count[class_name] += 1

    # Calculate and display overall accuracy
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")

    # Calculate and display accuracy by class
    print(f"Class-wise Accuracy ({total_correct} out of {total_images}):")
    class_accuracies = {}
    for class_name in class_labels:
        if class_total_count[class_name] > 0:
            accuracy = (class_correct_count[class_name]
                        / class_total_count[class_name])
        else:
            accuracy = 0
        class_accuracies[class_name] = accuracy
        print(f"{class_name}: {accuracy * 100:.2f}%")

    # Plot the class-wise accuracies
    plot_class_accuracy_bar_chart(class_accuracies)


def plot_class_accuracy_bar_chart(class_accuracies):
    """
    Plot a bar chart for class-wise accuracies.
    """
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, accuracies, color='lightblue')
    plt.xlabel('Disease Classes')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy')
    plt.ylim(0, 1)  # Accuracy range from 0 to 1
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        # Argument parser setup
        parser = argparse.ArgumentParser(
            description='Classify images using a pre-trained model.')
        parser.add_argument('training_directory', type=str,
                            help='Path to the directory \
                            containing training subdirectories.')
        parser.add_argument('input_image_or_folder', type=str,
                            help='Path to the image \
                                or folder for classification.')
        parser.add_argument('-batch', action='store_true',
                            help='If set, classify all images \
                                in the folder and calculate accuracy.')
        args = parser.parse_args()

        # Determine model name based on folder (Apple or Grape)
        model_name = get_model_name_based_on_folder(args.training_directory)

        # Load the trained model
        model_path = f'output/{model_name}.h5'
        trained_model = tf.keras.models.load_model(model_path)

        # Fetch class labels from the training directory structure
        class_labels = get_class_labels(args.training_directory)

        # Classify based on the input (single image or batch of images)
        if args.batch:
            classify_images_in_folder(args.input_image_or_folder,
                                      trained_model, class_labels)
        else:
            classify_image_and_display_results(args.input_image_or_folder,
                                               trained_model, class_labels)

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        exit(0)
