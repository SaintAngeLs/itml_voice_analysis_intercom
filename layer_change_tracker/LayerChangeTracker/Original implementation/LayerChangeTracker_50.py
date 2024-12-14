import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split

# Load pre-trained model
# This model is a result of prior training and will be used as the base model for this experiment.
model_path = "/Users/vonguyen/Downloads/final_model_0.4_0.1.keras"
model = load_model(model_path)

# Data Loader Class
class DataLoader:
    def __init__(self, spectrogram_dir, target_size=(128, 128)):
        # Directory containing spectrogram images divided into 'allowed' and 'disallowed' categories
        self.spectrogram_dir = spectrogram_dir
        # Target size to resize all spectrogram images to the same dimensions
        self.target_size = target_size

    def load_data(self):
        X, y = [], []
        for class_name in ['allowed', 'disallowed']:
            class_dir = os.path.join(self.spectrogram_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('_spectrogram.png'):
                    # Load and preprocess the image
                    image_path = os.path.join(class_dir, file_name)
                    image = tf.keras.preprocessing.image.load_img(
                        image_path, color_mode='grayscale', target_size=self.target_size
                    )
                    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
                    X.append(image_array)
                    # Assign label: 1 for 'allowed', 0 for 'disallowed'
                    y.append(1 if class_name == 'allowed' else 0)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Callback to track layer weight changes
class LayerChangeTracker(Callback):
    def __init__(self, reinit_layers=None, reinit_epoch=25):
        super().__init__()
        # Layers to reset during training (conv2d_27, conv2d_34, dense_6 chosen for their critical roles)
        self.reinit_layers = reinit_layers if reinit_layers else []
        # Epoch at which the reset will occur
        self.reinit_epoch = reinit_epoch
        self.initial_weights = {}
        self.weight_changes = []

    def on_train_begin(self, logs=None):
        # Save initial weights for all layers that have trainable parameters
        self.initial_weights = {
            layer.name: [np.copy(w) for w in layer.get_weights()]
            for layer in self.model.layers if len(layer.get_weights()) > 0
        }

    def on_epoch_end(self, epoch, logs=None):
        epoch_changes = {}
        for layer in self.model.layers:
            if len(layer.get_weights()) > 0:
                initial_weights = self.initial_weights.get(layer.name, [])
                current_weights = layer.get_weights()
                # Calculate the norm of weight differences for each layer
                changes = [
                    np.linalg.norm(current - initial)
                    for current, initial in zip(current_weights, initial_weights)
                ]
                epoch_changes[layer.name] = changes

        self.weight_changes.append(epoch_changes)

        # Reinitialize layers if the epoch matches the reinitialization epoch
        if epoch + 1 == self.reinit_epoch:
            self._reinitialize_layers()

    def _reinitialize_layers(self):
        print(f"\nReinitializing layers: {self.reinit_layers}")
        for layer in self.model.layers:
            if layer.name in self.reinit_layers:
                if len(layer.get_weights()) > 0:
                    # Skip BatchNormalization layers as their internal state is critical
                    if not isinstance(layer, tf.keras.layers.BatchNormalization):
                        print(f"Resetting layer: {layer.name}")
                        layer.set_weights([
                            layer.kernel_initializer(layer.get_weights()[0].shape),
                            layer.bias_initializer(layer.get_weights()[1].shape)
                        ])
                    else:
                        print(f"Skipping BatchNormalization layer: {layer.name}")
                else:
                    print(f"Skipping layer: {layer.name} (No weights to reset)")

    def get_weight_changes(self):
        # Return the recorded weight changes for analysis
        return self.weight_changes

# Initialize data loader
train_data_dir = "/Users/vonguyen/Downloads/spectrograms/train"  # Training data directory
test_data_dir = "/Users/vonguyen/Downloads/spectrograms/test"    # Testing data directory

train_loader = DataLoader(spectrogram_dir=train_data_dir)
X_train, y_train = train_loader.load_data()

test_loader = DataLoader(spectrogram_dir=test_data_dir)
X_test, y_test = test_loader.load_data()

# Split train/validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Initialize tracker
tracker = LayerChangeTracker(
    reinit_layers=['conv2d_27', 'conv2d_34', 'dense_6'],  # Critical layers chosen for reset
    reinit_epoch=25  # Reset at midpoint to observe recovery
)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),  # Low learning rate for fine-tuning
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model for 50 epochs
try:
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Fixed to 50 epochs as per the experiment design
        batch_size=32,  # Moderate batch size to balance memory usage and learning stability
        callbacks=[tracker]
    )
except Exception as e:
    print(f"Error during training: {str(e)}")
    raise

# Save weight changes to a JSON file
weight_changes = tracker.get_weight_changes()
output_file = "layer_weight_changes_50_epochs.json"

# Convert numpy arrays to lists for JSON serialization
weight_changes_serializable = [
    {layer_name: np.sum(changes).tolist() for layer_name, changes in epoch.items()}
    for epoch in weight_changes
]

with open(output_file, "w") as f:
    json.dump(weight_changes_serializable, f, indent=4)

print(f"Weight changes saved to {output_file}")
