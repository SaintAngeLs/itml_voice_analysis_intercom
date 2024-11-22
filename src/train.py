import os
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from datetime import datetime
from model import CNNModel


class DataLoader:
    """Handles loading and preprocessing of spectrogram image data."""

    def __init__(self, spectrogram_dir, target_size=(128, 128)):
        """
        Initialize the DataLoader.

        Parameters:
        - spectrogram_dir: Directory containing spectrogram images.
        - target_size: Target size to resize the spectrogram images.
        """
        self.spectrogram_dir = spectrogram_dir
        self.target_size = target_size

    def load_data(self):
        """
        Load and preprocess data from the directory.

        Returns:
        - X: Numpy array of spectrogram images.
        - y: Numpy array of labels (1 for "allowed", 0 for "disallowed").
        """
        X, y = [], []
        for class_name in ['allowed', 'disallowed']:
            class_dir = os.path.join(self.spectrogram_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('_spectrogram.png'):
                    image_path = os.path.join(class_dir, file_name)
                    image = tf.keras.preprocessing.image.load_img(
                        image_path, color_mode='grayscale', target_size=self.target_size
                    )
                    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
                    X.append(image_array)
                    y.append(1 if class_name == 'allowed' else 0)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class ModelTrainer:
    """Handles the training and evaluation of a CNN model."""

    def __init__(self, model, output_dir="./outputs", log_dir="./logs"):
        """
        Initialize the ModelTrainer.

        Parameters:
        - model: The Keras model to train.
        - output_dir: Directory to save trained models and results.
        - log_dir: Directory to save TensorBoard logs.
        """
        self.model = model
        self.output_dir = output_dir
        self.log_dir = log_dir

    @staticmethod
    def compute_class_weights(y_train):
        """Compute class weights to handle class imbalance."""
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        return dict(enumerate(class_weights))

    @staticmethod
    def prepare_dataset(X, y, batch_size):
        """
        Prepare a TensorFlow dataset for training or evaluation.

        Parameters:
        - X: Input data.
        - y: Labels.
        - batch_size: Batch size.

        Returns:
        - A tf.data.Dataset object.
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(y)).batch(batch_size)
        dataset = dataset.map(lambda X, y: (tf.cast(X, tf.float32), tf.cast(y, tf.float32)))
        return dataset

    def setup_callbacks(self):
        """Setup callbacks for training."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.output_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True
        )
        lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.9 if epoch > 10 else lr)
        log_dir = os.path.join(self.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        return [early_stopping, checkpoint, lr_scheduler, tensorboard_callback]

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
        """
        Train the model.

        Parameters:
        - X_train, y_train: Training data and labels.
        - X_val, y_val: Validation data and labels.
        - batch_size: Batch size for training.
        - epochs: Number of training epochs.
        """
        # Create datasets
        train_dataset = self.prepare_dataset(X_train, y_train, batch_size)
        val_dataset = self.prepare_dataset(X_val, y_val, batch_size)

        # Compute class weights
        class_weights = self.compute_class_weights(y_train)

        # Setup callbacks
        callbacks = self.setup_callbacks()

        # Train the model
        self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks
        )

    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Evaluate the model on the test set.

        Parameters:
        - X_test, y_test: Test data and labels.
        - batch_size: Batch size for evaluation.

        Returns:
        - Test loss and accuracy.
        """
        test_dataset = self.prepare_dataset(X_test, y_test, batch_size)
        test_loss, test_acc = self.model.evaluate(test_dataset)
        return test_loss, test_acc


def main():
    """Main function to execute the training pipeline."""
    # Directories
    train_dir = './data/spectrograms/train'
    output_dir = './outputs'
    log_dir = './logs'

    # Load data
    loader = DataLoader(spectrogram_dir=train_dir)
    X, y = loader.load_data()

    # Train/Validation/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize model
    model_builder = CNNModel(input_shape=(128, 128, 1), num_classes=1)
    model = model_builder.build()
    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Initialize and run training pipeline
    trainer = ModelTrainer(model=model, output_dir=output_dir, log_dir=log_dir)
    trainer.train(X_train, y_train, X_val, y_val, batch_size=32, epochs=50)

    # Save final model
    model.save(os.path.join(output_dir, 'final_model.keras'))

    # Evaluate the model
    test_loss, test_acc = trainer.evaluate(X_test, y_test, batch_size=32)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")


if __name__ == "__main__":
    log_file = open("output_log.txt", "a")
    sys.stdout = log_file
    sys.stderr = log_file

    main()

    log_file.close()