import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Load pre-trained model
model_path = "/Users/vonguyen/Downloads/final_model_0.4_0.1.keras"
model = load_model(model_path)

# Data Loader Class
class DataLoader:
    def __init__(self, spectrogram_dir, target_size=(128, 128)):
        self.spectrogram_dir = spectrogram_dir
        self.target_size = target_size

    def load_data(self):
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

# Callback to track layer weight changes
class LayerChangeTracker(Callback):
    def __init__(self, reinit_layers=None, reinit_epoch=25, epsilon=0.01):
        super().__init__()
        self.reinit_layers = reinit_layers if reinit_layers else []
        self.reinit_epoch = reinit_epoch
        self.epsilon = epsilon
        self.initial_weights = {}
        self.weight_changes = []
        self.recovery_epochs = {}

    def on_train_begin(self, logs=None):
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
                changes = [
                    np.linalg.norm(current - initial)
                    for current, initial in zip(current_weights, initial_weights)
                ]
                epoch_changes[layer.name] = np.sum(changes)

        self.weight_changes.append(epoch_changes)

        # Measure stabilization post-reset
        if epoch >= self.reinit_epoch:
            for layer in self.reinit_layers:
                if self.recovery_epochs.get(layer) is None and epoch_changes[layer] < self.epsilon:
                    self.recovery_epochs[layer] = epoch - self.reinit_epoch

        # Reinitialize layers at the defined epoch
        if epoch + 1 == self.reinit_epoch:
            self._reinitialize_layers()

    def _reinitialize_layers(self):
        print(f"\nReinitializing layers: {self.reinit_layers}")
        for layer in self.model.layers:
            if layer.name in self.reinit_layers:
                if len(layer.get_weights()) > 0 and not isinstance(layer, tf.keras.layers.BatchNormalization):
                    print(f"Resetting layer: {layer.name}")
                    layer.set_weights([
                        layer.kernel_initializer(layer.get_weights()[0].shape),
                        layer.bias_initializer(layer.get_weights()[1].shape)
                    ])

    def get_weight_changes(self):
        return self.weight_changes

    def get_recovery_epochs(self):
        return self.recovery_epochs

# Initialize data loader
train_data_dir = "/Users/vonguyen/Downloads/spectrograms/train"
test_data_dir = "/Users/vonguyen/Downloads/spectrograms/test"

train_loader = DataLoader(spectrogram_dir=train_data_dir)
X_train, y_train = train_loader.load_data()

test_loader = DataLoader(spectrogram_dir=test_data_dir)
X_test, y_test = test_loader.load_data()

# Split train/validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Compute class weights to balance the dataset
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Initialize tracker
tracker = LayerChangeTracker(
    reinit_layers=['conv2d_27', 'conv2d_34', 'dense_6'],
    reinit_epoch=25,
    epsilon=0.01  # Define stabilization threshold
)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model for 50 epochs
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[tracker]
)

# Save weight changes and recovery epochs to a JSON file
output_file = "layer_weight_changes_with_recovery.json"
output_data = {
    "weight_changes": tracker.get_weight_changes(),
    "recovery_epochs": tracker.get_recovery_epochs()
}

# Convert all numpy.float32 to Python float
def convert_to_serializable(obj):
    if isinstance(obj, np.float32):  # Convert numpy float32 to Python float
        return float(obj)
    elif isinstance(obj, dict):  # Recursively handle dictionaries
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):  # Recursively handle lists
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

serializable_output_data = convert_to_serializable(output_data)

with open(output_file, "w") as f:
    json.dump(serializable_output_data, f, indent=4)

print(f"Results saved to {output_file}")

# Print the results to terminal as a backup
print("Weight Changes:", json.dumps(serializable_output_data["weight_changes"], indent=4))
print("Recovery Epochs:", json.dumps(serializable_output_data["recovery_epochs"], indent=4))
