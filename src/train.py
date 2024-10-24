import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam
from model import create_cnn_model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# Allowed speakers (class 1)
class_1_speakers = {'F1', 'F7', 'F8', 'M3', 'M6', 'M8'}  # Allowed speakers

# Directories for training (will be split into training and validation)
train_dirs = [
    'clean', 'cleanraw', 'ipad_balcony1', 'ipad_bedroom1', 'ipad_confroom1',
    'ipad_confroom2', 'ipadflat_confroom1', 'ipadflat_office1', 'ipad_livingroom1',
    'ipad_office1', 'ipad_office2', 'iphone_balcony1', 'iphone_bedroom1', 'produced'
]

def load_data(spectrogram_dir, target_size=(128, 128)):
    """Load spectrogram images and assign labels based on filenames (for allowed/disallowed)."""
    X, y, scripts, effects = [], [], [], []

    for class_name in ['allowed', 'disallowed']:
        class_dir = os.path.join(spectrogram_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} does not exist. Skipping.")
            continue
        for file_name in os.listdir(class_dir):
            if file_name.endswith('_spectrogram.png'):
                # Extract information from the filename
                parts = file_name.split('_')
                
                # Assuming the format is something like: m9_script2_ipad_livingroom1.wav_segment_0_spectrogram.png
                script = parts[1]  # Extract the script part (e.g., "script2")
                effect = parts[2]  # Extract the effect (e.g., "ipad_livingroom1")

                # Full path to the image file
                image_path = os.path.join(class_dir, file_name)

                # Load the image and process it
                try:
                    image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(128, 128))
                    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0

                    # Append the image and its metadata
                    X.append(image_array)
                    y.append(1 if class_name == 'allowed' else 0)
                    scripts.append(script)
                    effects.append(effect)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")

    # Check if data is loaded
    print(f"Loaded {len(X)} images, {len(y)} labels, {len(scripts)} scripts, {len(effects)} effects")

    return np.array(X), np.array(y), np.array(scripts), np.array(effects)




def split_data(X, y, scripts, effects, val_size=0.2):
    """Split data into training and validation sets, ensuring no overlap of scripts and effects."""
    unique_scripts = np.unique(scripts)
    unique_effects = np.unique(effects)

    # First split by effects to ensure no overlap of environmental effects
    train_effects, val_effects = train_test_split(unique_effects, test_size=val_size, random_state=42)
    
    train_indices = np.isin(effects, train_effects)
    val_indices = np.isin(effects, val_effects)

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    scripts_train, scripts_val = scripts[train_indices], scripts[val_indices]

    # Now ensure that no script is in both train and validation sets
    train_scripts, val_scripts = train_test_split(np.unique(scripts_train), test_size=val_size, random_state=42)
    
    train_indices = np.isin(scripts_train, train_scripts)
    val_indices = np.isin(scripts_train, val_scripts)

    X_train, X_val = X_train[train_indices], X_train[val_indices]
    y_train, y_val = y_train[train_indices], y_train[val_indices]

    return X_train, X_val, y_train, y_val


def main(use_augmentation=True):
    # Load data from directories, ensuring no overlap between effects and scripts
    spectrogram_dir = './data/spectrograms/train'

    # Load training and validation data
    X, y, scripts, effects = load_data(spectrogram_dir, train_dirs)

    # Convert labels to float32 type
    y = y.astype(np.float32)

    # Split data into train and validation sets by effects and scripts
    X_train, X_val, y_train, y_val = split_data(X, y, scripts, effects)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Create the CNN model
    model = create_cnn_model(input_shape=(128, 128, 1))
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Define ImageDataGenerator for augmentation
    if use_augmentation:
        datagen = ImageDataGenerator(
            rotation_range=20, 
            zoom_range=0.2, 
            horizontal_flip=True, 
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],  # Adjust brightness
            shear_range=0.2,  # Apply shearing
            width_shift_range=0.2,  # Apply horizontal shifting
            height_shift_range=0.2  # Apply vertical shifting
        )

    else:
        datagen = ImageDataGenerator()  # No augmentation

    # Ensure the data and labels are in correct format and shape
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Use tf.data.Dataset to control the input pipeline structure explicitly
    def data_generator(X, y, batch_size):
        """Generator to yield batches of data and labels as tf.float32."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(len(y)).batch(batch_size)

        def preprocess_image(X, y):
            X = tf.cast(X, tf.float32)
            y = tf.cast(y, tf.float32)
            X = tf.image.adjust_contrast(X, contrast_factor=1.5)  # Adjust contrast
            return X, y

        dataset = dataset.map(preprocess_image)
        return dataset


    # Create train and validation datasets
    train_dataset = data_generator(X_train, y_train, batch_size=32)
    val_dataset = data_generator(X_val, y_val, batch_size=32)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.9 if epoch > 10 else lr)
    csv_logger = CSVLogger('training_log.csv', append=True)  # Save training logs

    # Train the model
    model.fit(
        train_dataset,  # Training dataset
        validation_data=val_dataset,  # Validation dataset
        epochs=50,
        class_weight=class_weight_dict,  # Class weights to handle imbalance
        callbacks=[early_stopping, checkpoint, lr_scheduler, csv_logger]  # Callbacks for early stopping, checkpointing, and learning rate scheduling
    )

    model.save('final_model_1.keras')

    # Evaluate the model on the validation set
    val_loss, val_acc = model.evaluate(val_dataset)
    print(f"Validation accuracy: {val_acc}, Validation loss: {val_loss}")

if __name__ == "__main__":
    main(use_augmentation=True)  # Set to False if you don't want augmentation
