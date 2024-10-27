"""
Script for Training and Evaluating a CNN Model for Voice-based Access Control Using Spectrograms

This script uses TensorFlow and Keras to load, preprocess, and train a Convolutional Neural Network (CNN) 
model that classifies audio spectrogram images as "allowed" or "disallowed" for access control purposes. 
The key components include data loading, image augmentation, handling class imbalance, and model training 
with several callbacks for optimal performance.

Main Steps:
1. **Data Loading and Preprocessing**: 
   - Load spectrogram images from directory, normalize, and label based on subdirectories ("allowed" or "disallowed").
   - Split data into training, validation, and testing sets.
   - Implement class balancing to handle imbalanced datasets.
   
2. **Model Creation and Compilation**: 
   - Define a CNN model architecture tailored to image classification.
   - Compile the model with the Adam optimizer and binary cross-entropy loss.

3. **Training and Validation**: 
   - Use ImageDataGenerator for data augmentation to improve generalization.
   - Train the model using training and validation datasets, incorporating various callbacks:
     - **EarlyStopping**: Stops training if validation loss does not improve.
     - **ModelCheckpoint**: Saves the best model based on validation loss.
     - **LearningRateScheduler**: Reduces learning rate after a set number of epochs.
     - **TensorBoard**: Logs training metrics for visualization.

4. **Evaluation and Saving**: 
   - Evaluate the model on a separate test dataset and print accuracy and loss.
   - Save the trained model for future inference.

"""

import os #for file and directory operations
import sys #to handle system-specific parameters and functions
import numpy as np #for efficient numerical computations
import tensorflow as tf #a deep learning framework
from tensorflow.keras.preprocessing.image import ImageDataGenerator #for data augmentation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard #for training management
from tensorflow.keras.optimizers import Adam #imports the Adam optimizer
from model import create_cnn_model #Imports the CNN model architecture from another script/module
from sklearn.utils.class_weight import compute_class_weight #to compute class weights for imbalanced datasets
from sklearn.model_selection import train_test_split #to split data into training and testing sets
from datetime import datetime  # For timestamping TensorBoard logs

# # Redirect all output (stdout and stderr) to a log file
# log_file = open(".log", "a")
# sys.stdout = log_file
# sys.stderr = log_file

def load_data(spectrogram_dir, target_size=(128, 128)):
    """Load spectrogram images from the directory and assign labels based on subdirectories."""
    X, y = [], [] # Initialize empty lists to store images and labels
    for class_name in ['allowed', 'disallowed']:  # Loop over classes
        class_dir = os.path.join(spectrogram_dir, class_name)  # Path to each class folder
        for file_name in os.listdir(class_dir): # Loop over files in the class folder
            if file_name.endswith('_spectrogram.png'): # Only process spectrogram images
                image_path = os.path.join(class_dir, file_name) # Full path to the image
                # Load the image as grayscale with target dimensions and normalize it
                image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=target_size)
                image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0 # Convert to array and normalize
                X.append(image_array) # Append image data to X
                y.append(1 if class_name == 'allowed' else 0) # Assign label 1 for "allowed", 0 for "disallowed"
    return np.array(X), np.array(y) # Return images and labels as numpy arrays

def main():
    # Load the data from the train directory (no separate test directory)
    train_dir = './data/spectrograms/train'
    X, y = load_data(train_dir) # Load training data and labels

    # Convert labels to float32 type to avoid data type issues
    y = y.astype(np.float32)

    # Split data into 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Further split training data into training and validation (80% train, 20% validation from the 70%)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Create the CNN model
    model = create_cnn_model(input_shape=(128, 128, 1))
    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Define ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=20, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Ensure the data and labels are in correct format and shape
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Use tf.data.Dataset to control the input pipeline structure explicitly
    def data_generator(X, y, batch_size):
        """Generator to yield batches of data and labels as tf.float32."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(len(y)).batch(batch_size)
        dataset = dataset.map(lambda X, y: (tf.cast(X, tf.float32), tf.cast(y, tf.float32)))
        return dataset

    # Create train and validation datasets
    train_dataset = data_generator(X_train, y_train, batch_size=32)
    val_dataset = data_generator(X_val, y_val, batch_size=32)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Stops training if no improvement
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True) # Saves the best model
    lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.9 if epoch > 10 else lr)# Decreases learning rate after epoch 10

    # TensorBoard logging callback
    log_dir = "./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1) # Logs training metrics to TensorBoard

    # Train the model using the training and validation datasets
    model.fit(
        train_dataset,  # Training dataset
        validation_data=val_dataset,  # Validation dataset
        epochs=50, #number of training epochs
        class_weight=class_weight_dict,  # Class weights to handle imbalance
        callbacks=[early_stopping, checkpoint, lr_scheduler, tensorboard_callback]  # Added TensorBoard callback
    )

    model.save('final_model_2.keras') #save the final model

    # Evaluate the model on the test set
    test_dataset = data_generator(X_test, y_test, batch_size=32)
    test_loss, test_acc = model.evaluate(test_dataset) #Calculates test loss and accuracy
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}") # Prints evaluation results

# Main entry point to run the main function if the script is executed directly
if __name__ == "__main__": 
    main()

# # Close the log file when the program ends
# log_file.close()
