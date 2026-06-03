# ==========================================================
# RESPONSIBLE AI PIPELINE
# FULL UPDATED MAIN.PY
# ==========================================================

import os
import gc
import warnings
import tkinter as tk

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog
from sklearn.model_selection import train_test_split

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape

# ==========================================================
# DATASET MODULES
# ==========================================================

from modules.dataset.loader import load_dataset
from modules.dataset.validator import validate_dataset
from modules.dataset.summary import dataset_summary

# ==========================================================
# PREPROCESSING
# ==========================================================

from modules.preprocessing.preprocessing_pipeline import preprocess_dataset
from modules.preprocessing.skewness_fixer import fix_skewness

# ==========================================================
# BALANCING
# ==========================================================

from modules.balancing.imbalance_detector import detect_imbalance
from modules.balancing.imbalance_report import imbalance_report
from modules.balancing.smote_module import apply_smote

# ==========================================================
# SYNTHETIC DATA
# ==========================================================

from modules.synthetic.ctgan_generator import generate_ctgan_data

# ==========================================================
# FAIRNESS / BIAS
# ==========================================================

from modules.bias.fairness_metrics import calculate_fairness
from modules.bias.fairness_fix import apply_fairness_fix

# ==========================================================
# METRICS
# ==========================================================

from modules.metrics.edqs import calculate_edqs
from modules.metrics.eri import calculate_eri
from modules.metrics.rai import calculate_rai

# ==========================================================
# MODEL
# ==========================================================

from modules.model.train_model import train_model
from modules.model.evaluate_model import evaluate_model
from modules.model.save_model import save_model

# ==========================================================
# VISUALIZATION
# ==========================================================

from modules.visualization.before_after_graphs import (
    plot_before_after_counts
)

from modules.visualization.edqs_graph import (
    plot_edqs_comparison
)

from modules.visualization.bias_visualizations import (
    bias_bar_graph,
    bias_heatmap,
    fairness_comparison_chart
)

from modules.visualization.model_metric_graphs import (
    plot_model_metrics_before_after
)

# ==========================================================
# UTILITIES
# ==========================================================

from modules.utils.save_metrics import save_metrics
from modules.utils.logger import (
    set_log_file,
    write_log
)

# ==========================================================
# LLM
# ==========================================================

from modules.llm.llm_decision_engine import (
    generate_llm_analysis,
    generate_llm_recommendations
)

# ==========================================================
# CONFIG
# ==========================================================

FAST_MODE = True
TEST_SIZE = 0.20
RANDOM_STATE = 42

os.makedirs("outputs", exist_ok=True)

METRICS_TEXT_PATH = os.path.join(
    "outputs",
    "all_metrics_log.txt"
)

with open(
    METRICS_TEXT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RESPONSIBLE AI METRICS LOG\n\n"
    )

# ==========================================================
# MEMORY OPTIMIZATION
# ==========================================================

def optimize_memory():

    gc.collect()
    plt.close("all")

# ==========================================================
# PRINT SECTION
# ==========================================================

def print_section(title):

    print(
        f"\n{'='*20} {title} {'='*20}"
    )

# ==========================================================
# SAVE METRICS
# ==========================================================

def append_metrics_to_txt(
    title,
    data
):

    with open(
        METRICS_TEXT_PATH,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            f"\n{'='*60}\n"
            f"{title}\n"
            f"{'='*60}\n\n"
        )

        if isinstance(data, dict):

            for k, v in data.items():

                f.write(
                    f"{k} : {v}\n"
                )

        else:

            f.write(str(data))

        f.write("\n")

# ==========================================================
# LLM ITERATION LOGGER
# ==========================================================

def log_llm_iteration(
    iteration,
    metrics_dict
):

    output_path = os.path.join(
        "outputs",
        "llm_iteration_metrics.txt"
    )

    with open(
        output_path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            f"\n{'='*60}\n"
            f"LLM ITERATION {iteration}\n"
            f"{'='*60}\n\n"
        )

        for k, v in metrics_dict.items():

            f.write(
                f"{k} : {v}\n"
            )

        f.write("\n")

# ==========================================================
# FILE CHOOSER
# ==========================================================

def choose_file():

    try:

        root = tk.Tk()
        root.withdraw()

        root.attributes(
            '-topmost',
            True
        )

        file_path = filedialog.askopenfilename(
            title="Select CSV Dataset",
            filetypes=[
                ("CSV Files", "*.csv")
            ]
        )

        root.destroy()

        if file_path:

            print(
                "\nDataset Selected Successfully"
            )

            return file_path

    except Exception as e:

        print(e)

    file_path = input(
        "\nEnter CSV Path: "
    ).strip()

    if os.path.exists(file_path):

        return file_path

    raise FileNotFoundError(file_path)

# ==========================================================
# TARGET DETECTION
# ==========================================================

def detect_target_column(df):

    print("\nDetecting Target Column...")

    target_keywords = [
        "target",
        "label",
        "class",
        "output",
        "result",
        "response",
        "status",
        "prediction",
        "survived",
        "default",
        "churn",
        "risk",
        "score",
        "rating"
    ]

    excluded_keywords = [
        "id",
        "serial",
        "phone",
        "email",
        "timestamp",
        "date",
        "time"
    ]

    scores = {}

    total_rows = len(df)

    for col in df.columns:

        score = 0

        lower_col = col.lower()

        unique_values = df[col].nunique()

        unique_ratio = unique_values / max(total_rows, 1)

        dtype = str(df[col].dtype)

        if unique_values <= 1:
            continue

        if any(
            x in lower_col
            for x in excluded_keywords
        ):
            score -= 100

        if any(
            x in lower_col
            for x in target_keywords
        ):
            score += 60

        if col == df.columns[-1]:
            score += 30

        if dtype == "object":
            score += 20

        if unique_values <= 10:
            score += 25

        if unique_ratio > 0.95:
            score -= 40

        scores[col] = score

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(
        "\n========== TARGET COLUMN SCORES =========="
    )

    for col, score in sorted_scores:

        print(f"{col} --> {score}")

    target_col = sorted_scores[0][0]

    print(
        f"\nDetected Target Column : {target_col}"
    )

    return target_col

# ==========================================================
# GRAPH FOLDER
# ==========================================================

def create_graph_output_folder(dataset_name):

    graph_dir = os.path.join(
        "outputs",
        "graphs",
        dataset_name.replace(".csv", "")
    )

    os.makedirs(
        graph_dir,
        exist_ok=True
    )

    return graph_dir

# ==========================================================
# SAVE GRAPH
# ==========================================================

def save_current_graph(
    graph_dir,
    graph_name
):

    try:

        output_path = os.path.join(
            graph_dir,
            f"{graph_name}.png"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=100,
            bbox_inches="tight"
        )

        print(
            f"\nGraph Saved Successfully:\n{output_path}"
        )

        plt.close()

    except Exception as e:

        print(e)

# ==========================================================
# MAIN
# ==========================================================

def main():

    print_section(
        "RESPONSIBLE AI PIPELINE STARTED"
    )

    file_path = choose_file()

    df = load_dataset(file_path)

    dataset_name = os.path.basename(file_path)

    graph_dir = create_graph_output_folder(
        dataset_name
    )

    set_log_file(dataset_name)

    optimize_memory()

    validate_dataset(df)

    dataset_summary(df)

    target_col = detect_target_column(df)

    print(
        f"\nTarget Column : {target_col}"
    )

    # ======================================================
    # TASK TYPE
    # ======================================================

    unique_count = df[target_col].nunique()

    total_rows = len(df)

    if (
        pd.api.types.is_numeric_dtype(
            df[target_col]
        )
        and (
            unique_count > 15
            or unique_count / total_rows > 0.05
        )
    ):

        task_type = "regression"

    else:

        task_type = "classification"

    print(
        f"\nTask Type : {task_type.upper()}"
    )

    # ======================================================
    # PREPROCESSING
    # ======================================================

    df_before = df.copy()

    target_series = df[target_col].copy()

    features_df = df.drop(
        columns=[target_col]
    )

    features_df, encoders = preprocess_dataset(
        features_df
    )

    df = pd.concat(
        [
            features_df.reset_index(drop=True),
            target_series.reset_index(drop=True)
        ],
        axis=1
    )

    # ======================================================
    # INITIAL METRICS
    # ======================================================

    edqs_before_metrics = calculate_edqs(
        df,
        target_col,
        task_type
    )

    edqs_before = edqs_before_metrics["edqs"]

    if task_type == "classification":

        imbalance_ratio_before, class_counts_before = (
            detect_imbalance(
                df,
                target_col
            )
        )

        bias_results_before, fairness_before = (
            calculate_fairness(
                df,
                target_col
            )
        )

    else:

        imbalance_ratio_before = 0
        fairness_before = 1.0
        bias_results_before = []

    # ======================================================
    # TRAIN TEST SPLIT
    # ======================================================

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify_option = y \
        if task_type == "classification" \
        else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_option
    )

    # ======================================================
    # BEFORE MODEL
    # ======================================================

    baseline_model = train_model(
        X_train,
        y_train,
        task_type
    )

    baseline_pred = baseline_model.predict(
        X_test
    )

    baseline_metrics = evaluate_model(
        y_test,
        baseline_pred,
        task_type
    )

    # ======================================================
    # TRAIN DATAFRAME
    # ======================================================

    train_df = pd.concat(
        [X_train, y_train],
        axis=1
    )

    # ======================================================
    # FAIRNESS FIX
    # ======================================================

    if task_type == "classification":

        train_df = apply_fairness_fix(
            train_df,
            bias_results_before,
            target_col
        )

    # ======================================================
    # FINAL METRICS
    # ======================================================

    bias_results_after, fairness_after = (
        calculate_fairness(
            train_df,
            target_col
        )
    )

    edqs_after_metrics = calculate_edqs(
        train_df,
        target_col,
        task_type
    )

    edqs_after = edqs_after_metrics["edqs"]

    imbalance_ratio_after, class_counts_after = (
        detect_imbalance(
            train_df,
            target_col
        )
    )

    # ======================================================
    # FINAL MODEL
    # ======================================================

    X_train_final = train_df.drop(
        columns=[target_col]
    )

    y_train_final = train_df[target_col]

    model = train_model(
        X_train_final,
        y_train_final,
        task_type
    )

    y_pred = model.predict(X_test)

    metrics = evaluate_model(
        y_test,
        y_pred,
        task_type
    )

    # ======================================================
    # ERI
    # ======================================================

    eri_before = calculate_eri(
        fairness_score=fairness_before,
        accuracy=baseline_metrics.get(
            "accuracy",
            0
        ),
        imbalance_ratio=imbalance_ratio_before,
        explainability_score=0.50
    )

    eri_after = calculate_eri(
        fairness_score=fairness_after,
        accuracy=metrics.get(
            "accuracy",
            0
        ),
        imbalance_ratio=imbalance_ratio_after,
        explainability_score=0.85
    )

    # ======================================================
    # RAI
    # ======================================================

    rai_before = calculate_rai(
        edqs_before,
        fairness_before,
        baseline_metrics.get(
            "accuracy",
            0
        ),
        eri_before
    )

    rai_after = calculate_rai(
        edqs_after,
        fairness_after,
        metrics.get(
            "accuracy",
            0
        ),
        eri_after
    )

    # ======================================================
    # METRICS DICTIONARY
    # ======================================================

    metrics_data = {

        "EDQS Before": edqs_before,
        "EDQS After": edqs_after,

        "Fairness Before": fairness_before,
        "Fairness After": fairness_after,

        "Bias Before": 1 - fairness_before,
        "Bias After": 1 - fairness_after,

        "ERI Before": eri_before,
        "ERI After": eri_after,

        "RAI Before": rai_before,
        "RAI After": rai_after
    }

    if task_type == "classification":

        metrics_data.update({

            "Accuracy Before":
            baseline_metrics.get(
                "accuracy",
                0
            ),

            "Accuracy After":
            metrics.get(
                "accuracy",
                0
            ),

            "Precision Before":
            baseline_metrics.get(
                "precision",
                0
            ),

            "Precision After":
            metrics.get(
                "precision",
                0
            ),

            "Recall Before":
            baseline_metrics.get(
                "recall",
                0
            ),

            "Recall After":
            metrics.get(
                "recall",
                0
            ),

            "F1 Score Before":
            baseline_metrics.get(
                "f1_score",
                0
            ),

            "F1 Score After":
            metrics.get(
                "f1_score",
                0
            )
        })

    # ======================================================
    # SAVE METRICS
    # ======================================================

    append_metrics_to_txt(
        "FINAL METRICS",
        metrics_data
    )

    log_llm_iteration(
        1,
        metrics_data
    )

    # ======================================================
    # VISUALIZATION
    # ======================================================

    try:

        plt.figure(figsize=(8, 5))

        plot_before_after_counts(
            df_before,
            train_df
        )

        save_current_graph(
            graph_dir,
            "before_after_counts"
        )

        plt.figure(figsize=(8, 5))

        plot_edqs_comparison(
            edqs_before,
            edqs_after,
            task_type
        )

        save_current_graph(
            graph_dir,
            "edqs_comparison"
        )

        fairness_comparison_chart(
            fairness_before,
            fairness_after,
            graph_dir
        )

        plot_model_metrics_before_after(
            baseline_metrics,
            metrics,
            graph_dir
        )

        bias_heatmap(
            train_df,
            graph_dir
        )

    except Exception as e:

        print(e)

    # ======================================================
    # LLM
    # ======================================================

    metrics_path = save_metrics(
        metrics_data
    )

    with open(
        metrics_path,
        "r",
        encoding="utf-8"
    ) as f:

        metrics_text = f.read()

    llm_analysis = generate_llm_analysis(
        metrics_text
    )

    recommendations = generate_llm_recommendations(
        metrics_text
    )

    print(
        "\n========== LLM ANALYSIS =========="
    )

    print(llm_analysis)

    print(
        "\n========== LLM RECOMMENDATIONS =========="
    )

    print(recommendations)

    # ======================================================
    # SAVE MODEL
    # ======================================================

    save_model(model)

    # ======================================================
    # SAVE DATASET
    # ======================================================

    output_path = os.path.join(
        "outputs",
        "final_responsible_ai_dataset.csv"
    )

    train_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nFinal Dataset Saved:\n{output_path}"
    )

    optimize_memory()

    print_section(
        "RESPONSIBLE AI PIPELINE COMPLETED"
    )

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    main()