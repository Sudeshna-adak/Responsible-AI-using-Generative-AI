# =========================================================
# EVALUATE MODEL
# =========================================================

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score

)

import pandas as pd

import numpy as np


# =========================================================
# MODEL EVALUATION FUNCTION
# =========================================================

def evaluate_model(
    y_test,
    y_pred,
    task_type
):

    try:

        # =================================================
        # CLASSIFICATION METRICS
        # =================================================

        if task_type == "classification":

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            print("\n========== MODEL METRICS ==========\n")

            print(f"Accuracy  : {accuracy:.4f}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"F1 Score  : {f1:.4f}")

            metrics = {

                "accuracy": accuracy,

                "precision": precision,

                "recall": recall,

                "f1_score": f1
            }

        # =================================================
        # REGRESSION METRICS
        # =================================================

        else:

            mae = mean_absolute_error(
                y_test,
                y_pred
            )

            mse = mean_squared_error(
                y_test,
                y_pred
            )

            rmse = np.sqrt(mse)

            r2 = r2_score(
                y_test,
                y_pred
            )

            print("\n========== MODEL METRICS ==========\n")

            print(f"MAE  : {mae:.4f}")
            print(f"MSE  : {mse:.4f}")
            print(f"RMSE : {rmse:.4f}")
            print(f"R2   : {r2:.4f}")

            metrics = {

                "accuracy": r2,

                "precision": 1 - mae,

                "recall": 1 - rmse,

                "f1_score": r2
            }

        return metrics

    except Exception as e:

        print(f"\nModel Evaluation Error: {e}")

        return {

            "accuracy": 0,

            "precision": 0,

            "recall": 0,

            "f1_score": 0
        }