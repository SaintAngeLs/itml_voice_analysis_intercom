import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from datetime import datetime
from model import CNNModel


class DataLoader:
    """Class to handle loading and preprocessing of spectrogram image data."""

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
        return np.array(X), np.array(y)


class TrainingPipeline:
    """Class to handle the training and evaluation of the CNN model."""

    def __init__(self, model, output_dir="./outputs", log_dir="./logs"):
        """
        Initialize the TrainingPipeline.

        Parameters:
        - model: The Keras model to train.
        - output_dir: Directory to save trained models and results.
        - log_dir: Directory to save TensorBoard logs.
        """
        self.model = model
        self.output_dir = output_dir
        self.log_dir = log_dir

    def setup_callbacks(self):
        """Setup callbacks for training."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            os.path.join(self.output_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True
        )
        lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.9 if epoch > 10 else lr)
        tensorboard_callback = TensorBoard(
            log_dir=os.path.join(self.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1
        )
        return [early_stopping, checkpoint, lr_scheduler, tensorboard_callback]

    def compute_class_weights(self, y_train):
        """Compute class weights to handle class imbalance."""
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        return dict(enumerate(class_weights))

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
        """
        Train the model.

        Parameters:
        - X_train, y_train: Training data and labels.
        - X_val, y_val: Validation data and labels.
        - batch_size: Batch size for training.
        - epochs: Number of training epochs.
        """
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
        )
        train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = datagen.flow(X_val, y_val, batch_size=batch_size)

        class_weights = self.compute_class_weights(y_train)
        callbacks = self.setup_callbacks()

        self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
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
        test_generator = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
        test_loss, test_acc = self.model.evaluate(test_generator)
        return test_loss, test_acc


def main():
    """Main function to execute the training pipeline."""
    # Directories
    train_dir = './data/spectrograms/train'
    test_dir = './data/spectrograms/test'
    output_dir = './outputs'
    log_dir = './logs'

    # Data loading
    data_loader = DataLoader(spectrogram_dir=train_dir)
    X, y = data_loader.load_data()
    y = y.astype(np.float32)

    # Train/Validation/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Model initialization
    model_builder = CNNModel(input_shape=(128, 128, 1), num_classes=1)
    model = model_builder.build()
    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Training pipeline
    pipeline = TrainingPipeline(model=model, output_dir=output_dir, log_dir=log_dir)
    pipeline.train(X_train, y_train, X_val, y_val, batch_size=32, epochs=50)

    # Save the final model
    model.save(os.path.join(output_dir, 'final_model.keras'))

    # Evaluation
    test_loss, test_acc = pipeline.evaluate(X_test, y_test, batch_size=32)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")


if __name__ == "__main__":
    main()
