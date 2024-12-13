import math
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from utils.regression_utils import *


class RegressionExperiment:
    def __init__(
        self,
        exp_name: str,
        dataset_paths: list[str] | list[Path],
        num_samples: int | None = None,
        min_sentences: int = 5,
        max_sentences: int = 65,
        sentence_bucket_size: int = 10,
    ):
        self.exp_name = exp_name

        datasets = []
        self.dataset_paths = dataset_paths
        for dataset_path in dataset_paths:
            assert Path(dataset_path).exists(), f"{dataset_path} does not exist"
            datasets.append(pd.read_pickle(dataset_path))
        self.df = pd.concat(datasets, axis=0)

        if num_samples is not None:
            self.df = self.df[:num_samples]

        assert min_sentences < max_sentences
        assert sentence_bucket_size > 0
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.sentence_bucket_size = sentence_bucket_size

        self.experiment_results = {}

    def run(self):
        print("Computing R² and Sentence Coefficients for all datapoints...")
        self.df = self.compute_datapoint_regression_statistics()

        print("Computing dataset regression statistics...")
        self.experiment_results = self.compute_dataset_regression_statistics()

        print(f"Completed experiment {self.exp_name}")
        return self.experiment_results

    def plot(self, plot_save_path: str | Path | None = None, show_plots: bool = False):
        sentence_range = self.max_sentences - self.min_sentences
        num_buckets = math.ceil((sentence_range) / self.sentence_bucket_size)
        fig, axes = plt.subplots(1, num_buckets, figsize=(5 * num_buckets, 8))
        axes = axes.flatten()

        for idx, i in enumerate(
            range(self.min_sentences, self.max_sentences, self.sentence_bucket_size)
        ):
            filtered_df = self.df[
                self.df["num_sentences"].apply(lambda x: i <= x < i + self.sentence_bucket_size)
            ]
            if len(filtered_df) == 0:
                continue

            df_regression_coefs = []
            df_sentence_positions = []
            for regression_coefs in filtered_df["regression_coefs"]:
                df_regression_coefs.extend(regression_coefs)
            for sentence_positions in filtered_df["sentence_positions"]:
                df_sentence_positions.extend(sentence_positions)

            ax = axes[idx]
            ax.scatter(df_sentence_positions, df_regression_coefs, s=10)

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_sentence_positions, df_regression_coefs
            )

            # Generate the line of best fit
            line_x = np.array(df_sentence_positions)
            line_y = slope * line_x + intercept

            # Plot the line of best fit
            ax.plot(line_x, line_y, color="red", label="Line of Best Fit")

            # Display p-value and slope on the plot
            ax.text(
                0.05,
                0.95,
                f"Slope: {slope:.4f}\nP-value: {p_value:.4f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
            )

            # Set title and labels
            ax.set_title(
                f"documens with {i} - {i + self.sentence_bucket_size} sentences",
                fontsize=12,
                pad=10,
            )
            ax.legend()

        # Set global x and y labels
        fig.supxlabel("Sentence Starting Index / Document Length", fontsize=16)
        fig.supylabel("Regression Coefficients", fontsize=16)

        # Adjust layout and save the plot
        plt.tight_layout()
        if plot_save_path is not None:
            Path(plot_save_path).mkdir(parents=True, exist_ok=True)
            filename = Path(plot_save_path) / f"regression_{self.exp_name}.png"
            plt.savefig(filename)

        if show_plots:
            plt.show()

    def compute_datapoint_regression_statistics(self):
        df: pd.DataFrame = self.df.copy()
        df_columns = df.columns

        # Compute R² and regression coefficients for the original and shuffled embeddings
        assert "sentence_embeddings" in df_columns and "embeddings_original" in df_columns
        df["r2"] = df.apply(
            lambda x: get_r2(x["sentence_embeddings"], x["embeddings_original"]), axis=1
        )
        df["regression_coefs"] = df.apply(
            lambda x: get_regression_coefs(x["sentence_embeddings"], x["embeddings_original"]),
            axis=1,
        )

        if "sentence_embeddings_shuffled" in df_columns and "embeddings_shuffled" in df_columns:
            df["shuffled_r2"] = df.apply(
                lambda x: get_r2(x["sentence_embeddings_shuffled"], x["embeddings_shuffled"]),
                axis=1,
            )

            df["shuffled_regression_coefs"] = df.apply(
                lambda x: get_regression_coefs(
                    x["sentence_embeddings_shuffled"], x["embeddings_shuffled"]
                ),
                axis=1,
            )

        return df

    def compute_dataset_regression_statistics(self):
        bucket_ranges = []
        r2_values = []
        regression_slopes = []
        p_values = []
        linreg_r_values = []
        num_samples = []

        assert "num_sentences" in self.df.columns
        for i in range(self.min_sentences, self.max_sentences, self.sentence_bucket_size):
            filtered_df = self.df[
                self.df["num_sentences"].apply(lambda x: i <= x < i + self.sentence_bucket_size)
            ]
            if len(filtered_df) == 0:
                continue

            num_samples.append(len(filtered_df))
            r2_values.append(filtered_df["r2"].mean())

            df_regression_coefs = []
            df_sentence_positions = []
            for regression_coefs in filtered_df["regression_coefs"]:
                df_regression_coefs.extend(regression_coefs)
            for sentence_positions in filtered_df["sentence_positions"]:
                df_sentence_positions.extend(sentence_positions)

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_sentence_positions, df_regression_coefs
            )

            bucket_ranges.append((i, i + self.sentence_bucket_size))
            regression_slopes.append(slope)
            p_values.append(p_value)
            linreg_r_values.append(r_value)

        experiment_results = pd.DataFrame(
            {
                "bucket_ranges": bucket_ranges,
                "r2_values": r2_values,
                "regression_slopes": regression_slopes,
                "p_values": p_values,
                "linreg_r_values": linreg_r_values,
                "num_samples": num_samples,
            }
        )

        return experiment_results
