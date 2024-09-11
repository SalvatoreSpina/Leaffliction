import os
import zipfile
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from PIL import Image


def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(4, activation='softmax')  # Assuming 4 types of diseases
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(directory):
    batch_size = 32
    image_gen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

    train_generator = image_gen.flow_from_directory(directory, target_size=(150, 150),
                                                    batch_size=batch_size, class_mode='categorical')
    
    model = create_model()
    model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=10)
    return model, train_generator


def save_model_and_images(model, train_generator, output_dir='output', zip_name='output.zip'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.h5')
    model.save(model_path)
    
    # Save some augmented images for demonstration
    images, labels = next(train_generator)  # Fetch a batch of images and labels
    for i, (image_array, label) in enumerate(zip(images, labels)):
        image_pil = Image.fromarray((image_array * 255).astype(np.uint8))  # Convert the numpy array to a PIL Image
        image_path = os.path.join(output_dir, f'image_{i}.png')
        image_pil.save(image_path)  # Save the PIL Image as a PNG file

    # Zip everything
    zip_path = os.path.join(output_dir, zip_name)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(model_path)
        for i in range(len(images)):
            image_path = os.path.join(output_dir, f'image_{i}.png')
            zf.write(image_path)
    
    print(f"Saved trained model and images to {zip_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a model to classify leaf diseases.')
    parser.add_argument('directory', type=str, help='Directory containing the training images.')
    args = parser.parse_args()

    model, train_generator = train_model(args.directory)
    save_model_and_images(model, train_generator)
