import os
import zipfile
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from PIL import Image

def get_model_name_based_on_folder(training_directory):
    """
    Determine the model name based on whether 'Apple' or 'Grape' is found in the folder name.
    Defaults to 'Model' if neither is found.
    """
    folder_name = os.path.basename(os.path.normpath(training_directory))
    if "apple" in folder_name.lower():
        return "Apple_Model"
    elif "grape" in folder_name.lower():
        return "Grape_Model"
    else:
        return "Model"

def build_cnn_model():
    """
    Build a Convolutional Neural Network (CNN) model for image classification.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(4, activation='softmax')  # Assuming 4 classes for disease classification
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model(training_directory, batch_size=32, epochs=10):
    """
    Train the CNN model using image data from the specified directory.
    """
    # Data augmentation to generate more diverse training examples
    data_generator = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                        horizontal_flip=True, fill_mode='nearest')

    # Generate batches of augmented images from the directory
    train_data_generator = data_generator.flow_from_directory(training_directory, 
                                                              target_size=(150, 150),
                                                              batch_size=batch_size, 
                                                              class_mode='categorical')
    
    # Build and compile the model
    model = build_cnn_model()
    
    # Train the model using the augmented images
    model.fit(train_data_generator, steps_per_epoch=train_data_generator.samples // batch_size, 
              epochs=epochs)
    
    return model, train_data_generator

def save_model_and_sample_images(trained_model, data_generator, model_name, output_directory='output', 
                                 zip_filename='output.zip'):
    """
    Save the trained model and some augmented images to an output directory and zip them.
    The model and images will be named based on the model type (Apple or Grape).
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Save the trained model with the model name (e.g., Apple_Model.h5 or Grape_Model.h5)
    model_path = os.path.join(output_directory, f'{model_name}.h5')
    trained_model.save(model_path)
    
    # Save some augmented images as examples
    images_batch, labels_batch = next(data_generator)  # Get a batch of images and labels
    for i, (image, label) in enumerate(zip(images_batch, labels_batch)):
        image_pil = Image.fromarray((image * 255).astype(np.uint8))  # Convert to PIL Image
        image_path = os.path.join(output_directory, f'{model_name}_augmented_image_{i}.png')
        image_pil.save(image_path)  # Save the image with a model-specific name

    # Zip the model and images
    zip_path = os.path.join(output_directory, zip_filename)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(model_path)
        for i in range(len(images_batch)):
            image_path = os.path.join(output_directory, f'{model_name}_augmented_image_{i}.png')
            zf.write(image_path)
    
    print(f"{model_name} and augmented images saved to {zip_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a model for leaf disease classification.')
    parser.add_argument('training_directory', type=str, help='Directory with training images.')
    args = parser.parse_args()

    model_name = get_model_name_based_on_folder(args.training_directory)

    trained_model, data_generator = train_cnn_model(args.training_directory)
    save_model_and_sample_images(trained_model, data_generator, model_name)
