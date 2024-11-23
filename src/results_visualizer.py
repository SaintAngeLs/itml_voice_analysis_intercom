import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f_oneway


class ExperimentResultsLoader:
    """Handles loading and preprocessing of experiment results."""

    def __init__(self, results_dir="./experiment_results"):
        self.results_dir = results_dir

    def load_results(self):
        results = []
        for file in os.listdir(self.results_dir):
            if file.endswith(".json") and file != "experiment_results.json":
                with open(os.path.join(self.results_dir, file), "r") as f:
                    results.append(json.load(f))
        return results

    @staticmethod
    def results_to_dataframe(results):
        flattened_results = []
        for result in results:
            flat_result = result["config"].copy()
            flat_result.update({
                "roc_auc": result.get("roc_auc", None),
                "val_loss": result.get("val_loss", None),
                "val_acc": result.get("val_acc", None),
            })
            flattened_results.append(flat_result)
        return pd.DataFrame(flattened_results)


class ResultsVisualizer:
    """Visualizes experiment results with plots and statistical tests."""

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def save_metric_by_parameter(self, metric, parameter, output_dir):
        grouped = self.dataframe.groupby(parameter)[metric].mean()
        plt.figure()
        grouped.plot(kind="bar", title=f"{metric.upper()} by {parameter}", ylabel=metric, xlabel=parameter)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{metric}_by_{parameter}.png"))
        plt.close()

    def save_comparison_heatmap(self, parameter1, parameter2, metric, output_dir):
        pivot_table = self.dataframe.pivot_table(
            index=parameter1, columns=parameter2, values=metric, aggfunc="mean"
        )
        plt.figure(figsize=(8, 6))
        plt.imshow(pivot_table, cmap="viridis", aspect="auto")
        plt.colorbar(label=metric)
        plt.title(f"{metric.upper()} by {parameter1} and {parameter2}")
        plt.xlabel(parameter2)
        plt.ylabel(parameter1)
        plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=45)
        plt.yticks(range(len(pivot_table.index)), pivot_table.index)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{metric}_by_{parameter1}_and_{parameter2}.png"))
        plt.close()

    def perform_anova_test(self, metric, group_by, output_dir):
        # Drop missing values for the metric and group_by columns
        self.dataframe = self.dataframe.dropna(subset=[metric, group_by])

        # Filter groups with at least 2 entries and variance
        groups = self.dataframe.groupby(group_by)[metric].apply(list)
        valid_groups = [g for g in groups if len(g) > 1 and len(set(g)) > 1]

        if len(valid_groups) > 1:
            # Perform ANOVA if there are valid groups
            anova_result = f_oneway(*valid_groups)
            result_text = (
                f"ANOVA Test for {metric} grouped by {group_by}\n"
                f"F-statistic: {anova_result.statistic:.4f}, p-value: {anova_result.pvalue:.4e}"
            )
        else:
            # Inform if ANOVA couldn't be performed
            result_text = (
                f"ANOVA Test for {metric} grouped by {group_by}\n"
                "Not enough variance or insufficient groups for a valid ANOVA test."
            )

        # Save the result to a text file
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"anova_{metric}_by_{group_by}.txt"), "w") as f:
            f.write(result_text)
        print(result_text)

    def save_boxplot(self, metric, group_by, output_dir):
        # Drop missing data before plotting
        self.dataframe = self.dataframe.dropna(subset=[metric, group_by])
        plt.figure(figsize=(10, 6))
        self.dataframe.boxplot(column=metric, by=group_by)
        plt.title(f"Boxplot of {metric.upper()} by {group_by}")
        plt.suptitle("")
        plt.xlabel(group_by)
        plt.ylabel(metric)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"boxplot_{metric}_by_{group_by}.png"))
        plt.close()

    def save_all_plots(self, output_dir="./plots"):
        parameters = [col for col in self.dataframe.columns if col not in ["roc_auc", "val_loss", "val_acc"]]
        metrics = ["roc_auc", "val_loss", "val_acc"]

        for parameter in parameters:
            for metric in metrics:
                if metric in self.dataframe.columns and parameter in self.dataframe.columns:
                    self.save_metric_by_parameter(metric, parameter, output_dir)

        if "dropout_rate" in self.dataframe.columns and "batch_size" in self.dataframe.columns:
            self.save_comparison_heatmap("dropout_rate", "batch_size", "roc_auc", output_dir)
        if "optimizer" in self.dataframe.columns and "train_test_split_ratio" in self.dataframe.columns:
            self.save_comparison_heatmap("optimizer", "train_test_split_ratio", "val_loss", output_dir)

        for metric in metrics:
            if metric in self.dataframe.columns:
                self.perform_anova_test(metric, "model_name", output_dir)
                self.save_boxplot(metric, "model_name", output_dir)


class ExperimentVisualizerApp:
    def __init__(self, results_dir="./experiment_results"):
        self.results_loader = ExperimentResultsLoader(results_dir)

    def run(self, output_dir="./plots"):
        results = self.results_loader.load_results()
        dataframe = self.results_loader.results_to_dataframe(results)
        visualizer = ResultsVisualizer(dataframe)
        print(f"Saving plots and performing analysis to {output_dir}...")
        visualizer.save_all_plots(output_dir)
        print("Analysis and plots saved successfully!")


if __name__ == "__main__":
    results_directory = "./experiment_results"
    plots_directory = "./experiment_results_plots"
    app = ExperimentVisualizerApp(results_directory)
    app.run(output_dir=plots_directory)
