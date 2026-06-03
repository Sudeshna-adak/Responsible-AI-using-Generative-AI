# =========================================================
# BIAS VISUALIZATION MODULE
# =========================================================

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# =========================================================
# CREATE SAVE DIRECTORY
# =========================================================

def create_folder(path):

    os.makedirs(
        path,
        exist_ok=True
    )

# =========================================================
# BIAS BAR GRAPH
# =========================================================

def bias_bar_graph(
    bias_results,
    save_dir
):

    """
    Creates bias probability bar graph
    """

    try:

        create_folder(save_dir)

        plt.figure(figsize=(10, 6))

        plt.bar(
            bias_results["Bias Type"],
            bias_results["Probability"]
        )

        plt.title(
            "Bias Probability Graph"
        )

        plt.xlabel(
            "Bias Type"
        )

        plt.ylabel(
            "Probability"
        )

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        save_path = os.path.join(
            save_dir,
            "bias_bar_graph.png"
        )

        plt.savefig(save_path)

        plt.close()

        print(
            f"\nBias Bar Graph Saved:\n{save_path}"
        )

    except Exception as e:

        print(
            f"\nBias Bar Graph Error: {e}"
        )

# =========================================================
# BIAS HEATMAP
# =========================================================

def bias_heatmap(
    df,
    save_dir
):

    """
    Creates correlation heatmap
    """

    try:

        create_folder(save_dir)

        numeric_df = df.select_dtypes(
            include=np.number
        )

        corr = numeric_df.corr()

        plt.figure(figsize=(12, 8))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title(
            "Bias Correlation Heatmap"
        )

        plt.tight_layout()

        save_path = os.path.join(
            save_dir,
            "bias_heatmap.png"
        )

        plt.savefig(save_path)

        plt.close()

        print(
            f"\nBias Heatmap Saved:\n{save_path}"
        )

    except Exception as e:

        print(
            f"\nBias Heatmap Error: {e}"
        )

# =========================================================
# FAIRNESS COMPARISON GRAPH
# =========================================================

def fairness_comparison_chart(
    before_score,
    after_score,
    save_dir
):

    """
    Compares fairness before vs after fix
    """

    try:

        create_folder(save_dir)

        labels = [
            "Before Fix",
            "After Fix"
        ]

        scores = [
            before_score,
            after_score
        ]

        plt.figure(figsize=(7, 5))

        bars = plt.bar(
            labels,
            scores
        )

        # Add values on top
        for bar in bars:

            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom"
            )

        plt.ylim(0, 1)

        plt.ylabel(
            "Fairness Score"
        )

        plt.title(
            "Fairness Comparison"
        )

        plt.tight_layout()

        save_path = os.path.join(
            save_dir,
            "fairness_comparison.png"
        )

        plt.savefig(save_path)

        plt.close()

        print(
            f"\nFairness Comparison Saved:\n{save_path}"
        )

    except Exception as e:

        print(
            f"\nFairness Comparison Error: {e}"
        )