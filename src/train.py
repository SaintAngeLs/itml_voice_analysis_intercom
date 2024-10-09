import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from model import create_cnn_model
from PIL import Image  # Importing PIL for resizing

def load_data(spectrogram_dir, target_size=(128, 128)):
    """Load the spectrogram images, resize them, and return the data and labels."""
    X = []
    y = []

    for class_name in ['allowed', 'disallowed']:  # Class names
        class_dir = os.path.join(spectrogram_dir, class_name)

        # Check if the directory exists
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} does not exist. Creating it...")
            os.makedirs(class_dir)

        label = 1 if class_name == 'allowed' else 0  # 1 for allowed (Class 1), 0 for disallowed (Class 0)

        for file_name in os.listdir(class_dir):
            if file_name.endswith('.png'):
                image_path = os.path.join(class_dir, file_name)

                # Load image using PIL and resize to target_size
                image = Image.open(image_path).convert('L')  # Convert to grayscale
                image = image.resize(target_size, Image.LANCZOS)  # Resize to target size
                
                # Convert image to numpy array and normalize
                image_array = np.array(image) / 255.0

                # Append to data
                X.append(image_array)
                y.append(label)

    # Convert lists to numpy arrays
    return np.array(X), np.array(y)


def main():
    spectrogram_dir = './data/spectrograms'  # Path to the generated spectrograms
    X, y = load_data(spectrogram_dir)

    # Reshape X to match CNN input shape (e.g., 128x128x1 for grayscale)
    X = X.reshape(-1, 128, 128, 1)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the CNN model
    input_shape = (128, 128, 1)
    model = create_cnn_model(input_shape)

    # Data augmentation (optional)
    datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
    datagen.fit(X_train)

    # Train the model
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                        epochs=20, validation_data=(X_val, y_val))

    # Save the model
    model.save('./models/cnn_model.h5')

if __name__ == "__main__":
    main()
