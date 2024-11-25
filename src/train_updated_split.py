import os
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
from model import create_cnn_model  # Correct import for your model function

class DataLoader:
    def __init__(self, spectrogram_dir, target_size=(128, 128)):
        self.spectrogram_dir = spectrogram_dir
        self.target_size = target_size

    def load_data(self):
        X, y = [], [] # Lists to store image data and labels
        for class_name in ['allowed', 'disallowed']:
            class_dir = os.path.join(self.spectrogram_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('_spectrogram.png'): # Process only valid PNG files
                    image_path = os.path.join(class_dir, file_name)
                    image = tf.keras.preprocessing.image.load_img(
                        image_path, color_mode='grayscale', target_size=self.target_size
                    ) # Load the image in grayscale
                    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
                    X.append(image_array) # Add the image to the dataset
                    y.append(1 if class_name == 'allowed' else 0) # Assign label: 1 for allowed, 0 for disallowed
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) # Convert lists to Numy arrays

def train_and_evaluate(test_size, val_size, output_dir):
    train_dir = '~/spectrograms/train' # Please update the directory to the train data folder
    log_dir = './logs' # Directory to store TensorBoard logs

    # Load data
    loader = DataLoader(spectrogram_dir=train_dir)
    X, y = loader.load_data()

    # Train/Validation/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

    # Initialize model
    model = create_cnn_model(input_shape=(128, 128, 1)) # Build the CNN model
    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_dir, 'best_model_{}_{}.keras'.format(test_size, val_size)),
        monitor='val_loss',
        save_best_only=True
    )
    tensorboard_callback = TensorBoard(log_dir=os.path.join(log_dir, f"test_{test_size}_val_{val_size}"), histogram_freq=1)

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, tensorboard_callback]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=32)
    return test_loss, test_acc

def main():
    test_sizes = [0.3, 0.4, 0.5] # List of test size ratios
    val_sizes = [0.1, 0.15, 0.2] # List of validation size ratios
    output_dir = './train_results'

    os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists.
    results = []

    for test_size in test_sizes:
        for val_size in val_sizes:
            print(f"Running with test_size={test_size} and val_size={val_size}")
            test_loss, test_acc = train_and_evaluate(test_size, val_size, output_dir)
            # Train and evaluate for the combination
            results.append({'test_size': test_size, 'val_size': val_size, 'test_loss': test_loss, 'test_acc': test_acc})

    # Save results to a JSON file
    results_file = os.path.join(output_dir, "split_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
