import os
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from model import CNNModel
import json


class DataLoader:
    """Handles loading and preprocessing of spectrogram image data."""

    def __init__(self, spectrogram_dir, target_size=(128, 128), chunk_size=None):
        """
        Initialize the DataLoader.

        Parameters:
        - spectrogram_dir: Directory containing spectrogram images.
        - target_size: Target size to resize the spectrogram images.
        - chunk_size: Optional size for randomly cropping spectrogram images (default is None).
        """
        self.spectrogram_dir = spectrogram_dir
        self.target_size = target_size
        self.chunk_size = chunk_size

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

                    # Apply random cropping if chunk_size is provided
                    if self.chunk_size:
                        height, width = self.chunk_size
                        image_array = tf.image.random_crop(image_array, size=(height, width, 1))

                    X.append(image_array)
                    y.append(1 if class_name == 'allowed' else 0)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class ROCAUCCallback(Callback):
    """Custom callback to calculate and store ROC AUC after each epoch."""

    def __init__(self, train_data, val_data, output_dir):
        """
        Initialize the ROCAUCCallback.

        Parameters:
        - train_data: Tuple of training features and labels.
        - val_data: Tuple of validation features and labels.
        - output_dir: Directory to save ROC AUC results.
        """
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = output_dir
        self.roc_auc_scores = {"train": [], "val": []}

    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate and store the ROC AUC scores at the end of each epoch.

        Parameters:
        - epoch: Current epoch number.
        - logs: Dictionary of logs for the current epoch (unused here).
        """
        train_preds = self.model.predict(self.train_data[0])
        val_preds = self.model.predict(self.val_data[0])

        # Calculate ROC AUC
        train_roc_auc = roc_auc_score(self.train_data[1], train_preds)
        val_roc_auc = roc_auc_score(self.val_data[1], val_preds)

        # Append scores
        self.roc_auc_scores["train"].append(train_roc_auc)
        self.roc_auc_scores["val"].append(val_roc_auc)

        # Log scores
        print(f"Epoch {epoch + 1} - Train ROC AUC: {train_roc_auc:.4f}, Val ROC AUC: {val_roc_auc:.4f}")

    def on_train_end(self, logs=None):
        """
        Save ROC AUC results to a JSON file at the end of training.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "roc_auc_scores.json"), "w") as f:
            json.dump(self.roc_auc_scores, f)

class LayerChangeTracker(Callback):
    """Tracks changes in the weights of specified layers during training."""
    def __init__(self, reinit_layers=None, reinit_epoch=25, epsilon=0.01):
        super().__init__()
        self.reinit_layers = reinit_layers if reinit_layers else []
        self.reinit_epoch = reinit_epoch
        self.epsilon = epsilon
        self.initial_weights = {}
        self.weight_changes = []
        self.recovery_epochs = {}

    def on_train_begin(self, logs=None):
        self.initial_weights = {layer.name: [np.copy(w) for w in layer.get_weights()] for layer in self.model.layers if len(layer.get_weights()) > 0}

    def on_epoch_end(self, epoch, logs=None):
        epoch_changes = {}
        for layer in self.model.layers:
            if len(layer.get_weights()) > 0:
                initial_weights = self.initial_weights.get(layer.name, [])
                current_weights = layer.get_weights()
                changes = [np.linalg.norm(current - initial) for current, initial in zip(current_weights, initial_weights)]
                epoch_changes[layer.name] = np.sum(changes)
        self.weight_changes.append(epoch_changes)
        if epoch + 1 == self.reinit_epoch:
            self._reinitialize_layers()
        if epoch >= self.reinit_epoch:
            for layer in self.reinit_layers:
                if self.recovery_epochs.get(layer) is None and epoch_changes[layer] < self.epsilon:
                    self.recovery_epochs[layer] = epoch - self.reinit_epoch

    def _reinitialize_layers(self):
        for layer in self.model.layers:
            if layer.name in self.reinit_layers and not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.set_weights([layer.kernel_initializer(shape=w.shape) for w in layer.get_weights()])

    def get_weight_changes(self):
        return self.weight_changes

    def get_recovery_epochs(self):
        return self.recovery_epochs

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

    def setup_callbacks(self, train_data, val_data):
        """
        Setup training callbacks including ROC AUC callback.

        Parameters:
        - train_data: Training data to be passed to ROC AUC callback.
        - val_data: Validation data to be passed to ROC AUC callback.

        Returns:
        - A list of configured callbacks.
        """
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.output_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True
        )
        log_dir = os.path.join(self.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        roc_auc_callback = ROCAUCCallback(train_data=train_data, val_data=val_data, output_dir=self.output_dir)

        layer_tracker = LayerChangeTracker(reinit_layers=['conv2d_27', 'conv2d_34', 'dense_6'], reinit_epoch=25)

        # Early stopping (commented out for running full 50 epochs)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        # Return callbacks
        return [checkpoint, tensorboard_callback, roc_auc_callback]
        # Uncomment the next line for early stopping:
        # return [checkpoint, tensorboard_callback, roc_auc_callback, early_stopping]

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

        # Setup callbacks with training and validation data
        callbacks = self.setup_callbacks(train_data=(X_train, y_train), val_data=(X_val, y_val))

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
    output_dir = './train_results'
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
