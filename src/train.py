import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from model import create_cnn_model
import tensorflow as tf
import logging
import sys

def load_data(spectrogram_dir, target_size=(128, 128)):
    """Load spectrogram images from the provided directory."""
    X = []
    y = []

    for class_name in ['allowed', 'disallowed']:  # Class names
        class_dir = os.path.join(spectrogram_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} does not exist.")
            continue

        # Traverse through user subdirectories
        for file_name in os.listdir(class_dir):
            if file_name.endswith('_spectrogram.png'):
                image_path = os.path.join(class_dir, file_name)

                # Load image using PIL and resize to target_size
                image = Image.open(image_path).convert('L')  # Convert to grayscale
                image = image.resize(target_size, Image.LANCZOS)

                # Convert image to numpy array and normalize (0 to 1)
                image_array = np.array(image, dtype=np.float32) / 255.0  # Convert to float32

                # Append to data arrays
                X.append(image_array)
                y.append(1 if class_name == 'allowed' else 0)  # 1 for 'allowed', 0 for 'disallowed'

    # Convert lists to numpy arrays
    X = np.array(X, dtype=np.float32)  # Ensure X is float32
    y = np.array(y, dtype=np.float32)  # Ensure y is float32 for binary classification

    # Reshape X to match CNN input shape (batch_size, 128, 128, 1)
    X = np.expand_dims(X, axis=-1)  # Add a channel dimension

    return X, y

def focal_loss(gamma=2., alpha=0.25):
    """
    Focal loss for binary classification.
    gamma > 0 reduces the relative loss for well-classified examples.
    alpha balances the importance of different classes.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        cross_entropy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = tf.math.exp(-cross_entropy_loss)  # Probabilities of correct class
        focal_loss = alpha * tf.pow((1 - p_t), gamma) * cross_entropy_loss
        return focal_loss

    return focal_loss_fixed

def scheduler(epoch, lr):
    if epoch > 10:  # After 10 epochs, reduce the learning rate by a factor of 10
        return lr * 0.5
    return lr

def generator_with_weights(datagen, X, y, sample_weights, batch_size=32):
    data_size = len(X)
    indices = np.arange(data_size)
    while True:
        np.random.shuffle(indices)  # Shuffle indices each epoch
        for start_idx in range(0, data_size, batch_size):
            end_idx = min(start_idx + batch_size, data_size)
            batch_indices = indices[start_idx:end_idx]
            x_batch = X[batch_indices]
            y_batch = y[batch_indices]
            sw_batch = sample_weights[batch_indices]

            # Apply data augmentation
            augmented_gen = datagen.flow(x_batch, y_batch, batch_size=batch_size, shuffle=False)
            x_aug_batch, y_aug_batch = next(augmented_gen)

            yield x_aug_batch, y_aug_batch, sw_batch

def main():
    train_spectrogram_dir = './data/spectrograms/train'  # Path to the training spectrograms
    test_spectrogram_dir = './data/spectrograms/test'    # Path to the test spectrograms

    # Load training and test data
    X_train, y_train = load_data(train_spectrogram_dir)
    X_test, y_test = load_data(test_spectrogram_dir)

    if len(X_train) == 0 or len(y_train) == 0:
        print("Error: No training data found. Please ensure spectrograms are available.")
        return
    if len(X_test) == 0 or len(y_test) == 0:
        print("Error: No test data found. Please ensure spectrograms are available.")
        return

    # Ensure labels are float32
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Calculate class weights to address class imbalance
    num_class_0 = np.sum(y_train.astype(int) == 0)
    num_class_1 = np.sum(y_train.astype(int) == 1)

    if num_class_0 == 0:
        class_weights = {0: 1, 1: len(y_train) / num_class_1}
    elif num_class_1 == 0:
        class_weights = {0: len(y_train) / num_class_0, 1: 1}
    else:
        class_weights = {
            0: len(y_train) / num_class_0,
            1: len(y_train) / num_class_1,
        }

    # Compute sample weights
    sample_weights = compute_sample_weight(
        class_weight=class_weights, y=y_train.astype(int)
    )

    class_weights = {
        0: len(y_train) / np.sum(y_train == 0),
        1: len(y_train) / np.sum(y_train == 1),
    }

    # Create the CNN model for binary classification
    input_shape = (128, 128, 1)
    model = create_cnn_model(input_shape)  # Single output for binary classification

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Use Adam optimizer
        loss='binary_crossentropy',  # Binary crossentropy for binary classification
        metrics=['accuracy'],
    )

    lr_scheduler = LearningRateScheduler(scheduler)

    # Data augmentation (optional)
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
    )

    datagen.fit(X_train)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        './models/best_cnn_model.keras', monitor='val_loss', save_best_only=True
    )

    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // 32

    # Train the model with data augmentation and early stopping
    history = model.fit(
        generator_with_weights(datagen, X_train, y_train, sample_weights),
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint, lr_scheduler],
        class_weight=class_weights,
    )

    # Save the final model
    model.save('./models/cnn_model.keras')

    print("Model training completed and saved to './models/cnn_model.keras'.")

# Set up logging
logging.basicConfig(
    filename='error_log.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    filemode='w',
)

# Redirect stdout and stderr to the log file
sys.stdout = open('output_log.log', 'w')
sys.stderr = sys.stdout

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("An error occurred during training.")
