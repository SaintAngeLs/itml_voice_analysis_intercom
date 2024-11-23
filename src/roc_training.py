import os
import json
import matplotlib.pyplot as plt


class ROCDataLoader:
    """Handles loading of ROC AUC scores data from a file."""

    def __init__(self, file_path):
        """
        Initialize the ROCDataLoader.

        Parameters:
        - file_path: Path to the JSON file containing ROC AUC scores.
        """
        self.file_path = file_path

    def load_data(self):
        """
        Load ROC AUC scores from the JSON file.

        Returns:
        - A dictionary with keys 'train' and 'val', containing lists of ROC AUC scores.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        with open(self.file_path, "r") as f:
            return json.load(f)


class ROCPlotter:
    """Handles plotting of ROC AUC curves."""

    def __init__(self, roc_data):
        """
        Initialize the ROCPlotter.

        Parameters:
        - roc_data: Dictionary containing ROC AUC scores for 'train' and 'val'.
        """
        self.roc_data = roc_data

    def plot(self, output_dir=None):
        """
        Plot the ROC AUC curves for training and validation.

        Parameters:
        - output_dir: Directory to save the plot (optional). If None, the plot is shown but not saved.
        """
        train_scores = self.roc_data.get("train", [])
        val_scores = self.roc_data.get("val", [])

        if not train_scores or not val_scores:
            raise ValueError("ROC AUC data is empty or missing required keys ('train', 'val').")

        # Plot the curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_scores, label="Train ROC AUC", marker='o')
        plt.plot(val_scores, label="Validation ROC AUC", marker='o')
        plt.title("ROC AUC Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("ROC AUC")
        plt.legend()
        plt.grid(True)

        # Save or show the plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, "roc_auc_curve.png")
            plt.savefig(plot_path)
            print(f"ROC AUC plot saved to {plot_path}")
        else:
            plt.show()


class ROCVisualizerApp:
    """Main application class for visualizing ROC AUC curves."""

    def __init__(self, data_loader, plotter):
        """
        Initialize the ROCVisualizerApp.

        Parameters:
        - data_loader: An instance of ROCDataLoader to load ROC AUC scores.
        - plotter: An instance of ROCPlotter to plot the ROC AUC curves.
        """
        self.data_loader = data_loader
        self.plotter = plotter

    def run(self, output_dir=None):
        """
        Run the application to load data and display/save the ROC AUC plot.

        Parameters:
        - output_dir: Directory to save the plot (optional). If None, the plot is shown but not saved.
        """
        print("Loading ROC AUC data...")
        roc_data = self.data_loader.load_data()
        print("Data loaded successfully. Plotting ROC AUC curves...")
        self.plotter.roc_data = roc_data
        self.plotter.plot(output_dir=output_dir)


if __name__ == "__main__":
    # Path to the JSON file with ROC AUC scores
    roc_auc_file = "./train_results/roc_auc_scores.json"
    # Directory to save the plot (set to None to just display the plot)
    output_plot_dir = "./training_plots"

    # Create instances of data loader and plotter
    data_loader = ROCDataLoader(file_path=roc_auc_file)
    plotter = ROCPlotter(roc_data={})

    # Create and run the application
    app = ROCVisualizerApp(data_loader=data_loader, plotter=plotter)
    app.run(output_dir=output_plot_dir)
