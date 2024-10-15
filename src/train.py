import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from model import create_cnn_model
from sklearn.preprocessing import LabelEncoder

def load_data(spectrogram_dir, target_size=(128, 128)):
    """Load spectrogram and MFCC images, resize them, and return the data and labels."""
    X = []
    y = []
    users = []

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
                    if file_name.endswith('_spectrogram.png') or file_name.endswith('_mfcc.png'):  # Match both spectrograms and MFCCs
                        image_path = os.path.join(user_path, file_name)

                        # Load image using PIL and resize to target_size
                        image = Image.open(image_path).convert('L')  # Convert to grayscale
                        image = image.resize(target_size, Image.LANCZOS)

                        # Convert image to numpy array and normalize
                        image_array = np.array(image) / 255.0

                        # Append to data
                        X.append(image_array)
                        y.append(user_dir)  # Use the user name as the label
                        users.append(user_dir)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Encode the user labels (convert user names to numeric labels)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder

def main():
    spectrogram_dir = './data/spectrograms'  # Path to the generated spectrograms and MFCCs
    X, y, label_encoder = load_data(spectrogram_dir)

    if len(X) == 0 or len(y) == 0:
        print("Error: No data found. Please ensure spectrograms and MFCCs are available.")
        return

    # Reshape X to match CNN input shape (e.g., 128x128x1 for grayscale)
    X = X.reshape(-1, 128, 128, 1)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the CNN model (number of users = number of unique labels)
    input_shape = (128, 128, 1)
    model = create_cnn_model(input_shape, num_users=len(label_encoder.classes_))

    # Data augmentation (optional)
    datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
    datagen.fit(X_train)

    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=50,  # Adjust as necessary
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping])

    # Save the model and label encoder
    model.save('./models/cnn_model.h5')
    np.save('./models/label_encoder.npy', label_encoder.classes_)

    print("Model training completed and saved to './models/cnn_model.h5'.")

if __name__ == "__main__":
    main()
