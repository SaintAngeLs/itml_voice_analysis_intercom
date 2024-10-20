import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from model import create_cnn_model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# # Redirect all output (stdout and stderr) to a log file
# log_file = open(".log", "a")
# sys.stdout = log_file
# sys.stderr = log_file

def load_data(spectrogram_dir, target_size=(128, 128)):
    """Load spectrogram images from the directory and assign labels based on subdirectories."""
    X, y = [], []
    for class_name in ['allowed', 'disallowed']:
        class_dir = os.path.join(spectrogram_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('_spectrogram.png'):
                image_path = os.path.join(class_dir, file_name)
                image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=target_size)
                image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
                X.append(image_array)
                y.append(1 if class_name == 'allowed' else 0)
    return np.array(X), np.array(y)

def main():
    # Load the data from the train directory (no separate test directory)
    train_dir = './data/spectrograms/train'
    X, y = load_data(train_dir)

    # Convert labels to float32 type to avoid data type issues
    y = y.astype(np.float32)

    # Split data into 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.9 if epoch > 10 else lr)

    # Train the model
    model.fit(
        train_dataset,  # Training dataset
        validation_data=val_dataset,  # Validation dataset
        epochs=50,
        class_weight=class_weight_dict,  # Class weights to handle imbalance
        callbacks=[early_stopping, checkpoint, lr_scheduler]  # Callbacks for early stopping, checkpointing, and learning rate scheduling
    )

    model.save('final_model.keras')

    # Evaluate the model on the test set
    test_dataset = data_generator(X_test, y_test, batch_size=32)
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

if __name__ == "__main__":
    main()

# # Close the log file when the program ends
# log_file.close()
