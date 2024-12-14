import os
import json
import numpy as np
import itertools
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from datetime import datetime
from model import CNNModel, CNNModelSimpler, CNNModelMoreAdvanced
from train import DataLoader, ModelTrainer, LayerChangeTracker

class ExperimentLogger:
    """Logs experimental results and configurations."""

    def __init__(self, output_dir="./experiment_results"):
        """
        Initialize the logger.

        Parameters:
        - output_dir: Directory to save experiment results.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []

    def config_to_filename(self, config):
        """
        Convert a configuration dictionary to a string suitable for a filename.

        Parameters:
        - config: Dictionary of parameters.

        Returns:
        - A string representing the configuration.
        """
        return "_".join([f"{key}={value}" for key, value in config.items()]).replace(".", "_")

    def log_training_history(self, config, history):
        """
        Save the training history (loss/accuracy per epoch) to a JSON file.

        Parameters:
        - config: Dictionary of the experiment configuration.
        - history: Training history object containing metrics over epochs.
        """
        filename = self.config_to_filename(config) + "_history.json"
        history_data = {
            "config": config,
            "history": history
        }
        with open(os.path.join(self.output_dir, filename), "w") as f:
            json.dump(history_data, f, indent=4)

    def log_result(self, config, roc_auc, val_loss, val_acc):
        """
        Log a single experimental result and save it to a configuration-specific file.

        Parameters:
        - config: Dictionary of the experiment configuration.
        - roc_auc: ROC AUC score for the experiment.
        - val_loss: Validation loss.
        - val_acc: Validation accuracy.
        """
        result = {"config": config, "roc_auc": roc_auc, "val_loss": val_loss, "val_acc": val_acc}
        self.results.append(result)

        # Save individual result
        filename = self.config_to_filename(config) + ".json"
        with open(os.path.join(self.output_dir, filename), "w") as f:
            json.dump(result, f, indent=4)

    def save_results(self):
        """Save all logged results to a file."""
        results_file = os.path.join(self.output_dir, "experiment_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=4)

class ExperimentRunner:
    """Handles running experiments with varying parameters and models."""

    def __init__(self, model_builders, dataloader, trainer, logger):
        """
        Initialize the ExperimentRunner.

        Parameters:
        - model_builders: Dictionary of model names and corresponding model builder classes.
        - dataloader: Instance of the DataLoader class for loading data.
        - trainer: Instance of the ModelTrainer class for training and evaluation.
        - logger: Instance of the ExperimentLogger class for logging results.
        """
        self.model_builders = model_builders
        self.dataloader = dataloader
        self.trainer = trainer
        self.logger = logger

    def run_experiments(self, param_grid):
        """
        Run experiments based on parameter combinations and models.

        Parameters:
        - param_grid: Dictionary where keys are parameter names, and values are lists of values to experiment with.
        """
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        # Load data once, to be used for all experiments
        X, y = self.dataloader.load_data()

        for model_name, model_builder in self.model_builders.items():
            for combination in param_combinations:
                config = dict(zip(param_names, combination))
                config["model_name"] = model_name  # Log the model name
                print(f"Running experiment with model={model_name}, configuration={config}")

                # Apply configuration
                model_builder.dropout_rate = config["dropout_rate"]
                train_test_split_ratio = config["train_test_split_ratio"]

                # Train/Validation/Test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_ratio, random_state=42)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

                # Prepare datasets
                train_dataset = self.trainer.prepare_dataset(X_train, y_train, batch_size=config["batch_size"])
                val_dataset = self.trainer.prepare_dataset(X_val, y_val, batch_size=config["batch_size"])
                test_dataset = self.trainer.prepare_dataset(X_test, y_test, batch_size=config["batch_size"])

                # Build and compile model
                model = model_builder.build()
                optimizer = Adam(learning_rate=config["learning_rate"]) if config["optimizer"] == "adam" else SGD(
                    learning_rate=config["learning_rate"])
                model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

                # Configure TensorBoard callback with a unique log directory
                log_dir = os.path.join("./logs", model_name, self.logger.config_to_filename(config))
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                layer_tracker = LayerChangeTracker(reinit_layers=['conv2d_27', 'conv2d_34', 'dense_6'], reinit_epoch=25)

                # Train the model and capture history
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=10,
                    verbose=1,
                    callbacks=[tensorboard_callback, layer_tracker]
                ).history

                # Save training history
                self.logger.log_training_history(config, history)

                # Evaluate the model
                val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
                y_pred = model.predict(test_dataset).ravel()
                roc_auc = roc_auc_score(y_test, y_pred)

                # Log results
                self.logger.log_result(config, roc_auc, val_loss, val_acc)
                print(f"Model: {model_name}, ROC AUC: {roc_auc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

                # Save the model
                model.save(os.path.join("./outputs", model_name, self.logger.config_to_filename(config) + "_model.h5"))

                # Save weight changes and recovery epochs
                weight_changes = layer_tracker.get_weight_changes()
                recovery_epochs = layer_tracker.get_recovery_epochs()
                changes_file = os.path.join(self.logger.output_dir, self.logger.config_to_filename(config) + "_changes.json")
                with open(changes_file, "w") as wf:
                    json.dump({"weight_changes": self.convert_to_serializable(weight_changes), "recovery_epochs": self.convert_to_serializable(recovery_epochs)}, wf, indent=4)


        self.logger.save_results()

    def convert_to_serializable(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Converts numpy datatype to native Python datatype
        elif isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(v) for v in obj]
        else:
            return obj


if __name__ == "__main__":
    param_grid = {
        "dropout_rate": [0.5, 0.7],
        "learning_rate": [0.001, 0.0001],
        "optimizer": ["adam", "sgd"],
        "batch_size": [32, 64],
        "train_test_split_ratio": [0.4, 0.6],
    }
    dataloader = DataLoader(spectrogram_dir="./data/spectrograms/train", target_size=(256, 256), chunk_size=(128, 128))
    model_builders = {
        "BaselineCNN": CNNModel(input_shape=(128, 128, 1), num_classes=1),
        "SimplerCNN": CNNModelSimpler(input_shape=(128, 128, 1), num_classes=1),
        "MoreAdvancedCNN": CNNModelMoreAdvanced(input_shape=(128, 128, 1), num_classes=1),
    }
    trainer = ModelTrainer(model=None, output_dir="./outputs", log_dir="./logs")
    logger = ExperimentLogger(output_dir="./experiment_results")
    experiment_runner = ExperimentRunner(model_builders, dataloader, trainer, logger)
    experiment_runner.run_experiments(param_grid)
