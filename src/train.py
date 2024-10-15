import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from model import create_cnn_model
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def load_data(spectrogram_dir, target_size=(128, 128)):
    """Load spectrogram and MFCC images, resize them, and return the data and labels."""
    X = []
    y = []

    for class_name in ['allowed', 'disallowed']:  # Class names
        class_dir = os.path.join(spectrogram_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} does not exist.")
            continue

        # Traverse through user subdirectories
        for user_dir in os.listdir(class_dir):
            user_path = os.path.join(class_dir, user_dir)
            if os.path.isdir(user_path):  # Ensure it's a directory
                for file_name in os.listdir(user_path):
                    # Process both spectrograms and MFCC images
                    if file_name.endswith('_spectrogram.png') or file_name.endswith('_mfcc.png'):
                        image_path = os.path.join(user_path, file_name)

                        # Load image using PIL and resize to target_size
                        image = Image.open(image_path).convert('L')  # Convert to grayscale
                        image = image.resize(target_size, Image.LANCZOS)

                        # Convert image to numpy array and normalize (0 to 1)
                        image_array = np.array(image) / 255.0

                        # Append to data arrays
                        X.append(image_array)
                        y.append(user_dir)  # Use the user name as the label

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Encode the user labels (convert user names to numeric labels)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder

def focal_loss(gamma=2., alpha=0.25):
    """
    Focal loss for multi-class classification.
    gamma > 0 reduces the relative loss for well-classified examples.
    alpha balances the importance of different classes.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        p_t = tf.math.exp(-cross_entropy_loss)  # Probabilities of correct class
        focal_loss = alpha * tf.pow((1 - p_t), gamma) * cross_entropy_loss
        return focal_loss

    return focal_loss_fixed

def scheduler(epoch, lr):
    if epoch > 10:  # After 10 epochs, reduce the learning rate by a factor of 10
        return lr * 0.1
    return lr

def main():
    spectrogram_dir = './data/spectrograms'  # Path to the generated spectrograms and MFCCs
    X, y, label_encoder = load_data(spectrogram_dir)

    if len(X) == 0 or len(y) == 0:
        print("Error: No data found. Please ensure spectrograms and MFCCs are available.")
        return

    # Reshape X to match CNN input shape (e.g., 128x128x1 for grayscale images)
    X = X.reshape(-1, 128, 128, 1)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calculate class weights to address class imbalance
    class_weights = {i: len(y_train) / np.sum(y_train == i) for i in np.unique(y_train)}

    # Create the CNN model (number of users = number of unique labels)
    input_shape = (128, 128, 1)
    model = create_cnn_model(input_shape, num_classes=len(label_encoder.classes_))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.00001),  # Use Adam optimizer
                  loss=focal_loss(),  # Loss for multi-class classification
                  metrics=['accuracy'])  # Evaluate with accuracy

    lr_scheduler = LearningRateScheduler(scheduler)

    # Data augmentation (optional): Add additional augmentations for more robust training
    datagen = ImageDataGenerator(
        rotation_range=20,  # Rotate up to 20 degrees
        zoom_range=0.2,     # Zoom in/out up to 20%
        horizontal_flip=True,
        vertical_flip=True,  # Include vertical flips if meaningful
        width_shift_range=0.2,  # Shift the spectrogram horizontally
        height_shift_range=0.2,  # Shift the spectrogram vertically
        shear_range=0.2,     # Shear transformations
        brightness_range=[0.8, 1.2],  # Randomly adjust brightness
        fill_mode='nearest'  # How to fill in newly created pixels
    )

    datagen.fit(X_train)

    # Callbacks: Early stopping and model checkpoint to save the best model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('./models/best_cnn_model.keras', monitor='val_loss', save_best_only=True)

    # Train the model with data augmentation and early stopping
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=50,  # Adjust number of epochs as needed
                        validation_data=(X_val, y_val),
                        class_weight=class_weights,  # Class weights for imbalance
                        callbacks=[early_stopping, checkpoint, lr_scheduler])

    # Save the final model and label encoder
    model.save('./models/cnn_model.keras')  # Use .keras extension
    np.save('./models/label_encoder.npy', label_encoder.classes_)

    print("Model training completed and saved to './models/cnn_model.keras'.")

if __name__ == "__main__":
    main()
