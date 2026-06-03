import numpy as np
import pandas as pd


def calculate_edqs(

    df,

    target_col=None,

    task_type="classification"
):

    # =====================================================
    # BASIC INFO
    # =====================================================

    total_rows,total_cols=df.shape

    total_values=(
        total_rows*total_cols
    )

    print(
        f"\nRows : {total_rows}"
    )

    print(
        f"Columns : {total_cols}"
    )

    # =====================================================
    # MISSING VALUES
    # =====================================================

    missing_values=(

        df.isnull()

        .sum()

        .sum()
    )

    missing_ratio=(

        missing_values

        /

        total_values

        if total_values>0

        else 0
    )

    missing_pct=(
        missing_ratio*100
    )

    missing_score=(
        1-missing_ratio
    )

    # =====================================================
    # DUPLICATE ROWS
    # =====================================================

    duplicate_rows=(

        df.duplicated()

        .sum()
    )

    duplicate_ratio=(

        duplicate_rows

        /

        total_rows

        if total_rows>0

        else 0
    )

    duplicate_pct=(
        duplicate_ratio*100
    )

    duplicate_score=(
        1-duplicate_ratio
    )

    # =====================================================
    # TARGET IMBALANCE
    # =====================================================

    imbalance_ratio=1.0

    imbalance_pct=0.0

    imbalance_score=1.0

    class_counts={}

    if (

        task_type=="classification"

        and

        target_col

        and

        target_col in df.columns
    ):

        class_counts=(

            df[target_col]

            .value_counts()
        )

        print(
            f"\nClass Distribution:\n"
            f"{class_counts}"
        )

        if len(class_counts)>1:

            imbalance_ratio=(

                class_counts.min()

                /

                class_counts.max()
            )

        else:

            imbalance_ratio=1.0

        imbalance_pct=(

            1-imbalance_ratio

        )*100

        imbalance_score=(
            imbalance_ratio
        )

    else:

        print(
            "\nRegression Task Detected"
        )

        print(
            "Imbalance Check Skipped"
        )

    # =====================================================
    # EDQS WEIGHTS
    # =====================================================

    w_missing=0.4

    w_duplicate=0.3

    w_imbalance=0.3

    # =====================================================
    # EDQS FORMULA
    # =====================================================

    edqs=(

        (

            w_missing

            *

            missing_score
        )

        +

        (

            w_duplicate

            *

            duplicate_score
        )

        +

        (

            w_imbalance

            *

            imbalance_score
        )

    )*100

    # =====================================================
    # SAFETY BOUND
    # =====================================================

    edqs=max(

        0,

        min(
            99.99,
            edqs
        )
    )

    # =====================================================
    # PRINT COMPLETE REPORT
    # =====================================================

    print(
        "\n========== EDQS REPORT =========="
    )

    print(
        f"\nMissing Values : "
        f"{missing_values}"
    )

    print(
        f"Missing Percentage : "
        f"{missing_pct:.2f}%"
    )

    print(
        f"Missing Score : "
        f"{missing_score:.4f}"
    )

    print(
        f"\nDuplicate Rows : "
        f"{duplicate_rows}"
    )

    print(
        f"Duplicate Percentage : "
        f"{duplicate_pct:.2f}%"
    )

    print(
        f"Duplicate Score : "
        f"{duplicate_score:.4f}"
    )

    if task_type=="classification":

        print(
            f"\nImbalance Ratio : "
            f"{imbalance_ratio:.4f}"
        )

        print(
            f"Imbalance Percentage : "
            f"{imbalance_pct:.2f}%"
        )

        print(
            f"Imbalance Score : "
            f"{imbalance_score:.4f}"
        )

    else:

        print(
            "\nImbalance Ratio : N/A"
        )

        print(
            "Imbalance Percentage : N/A"
        )

        print(
            "Imbalance Score : N/A"
        )

    print(
        "\n========== EDQS WEIGHTED FORMULA =========="
    )

    print(
        "\nEDQS = "
        f"({w_missing} × Missing Score) + "
        f"({w_duplicate} × Duplicate Score) + "
        f"({w_imbalance} × Imbalance Score)"
    )

    print(
        f"\nFinal EDQS Score : "
        f"{edqs:.2f}"
    )

    print(
        "\n========== EDQS CALCULATION COMPLETED =========="
    )

    # =====================================================
    # RETURN METRICS
    # =====================================================

    metrics={

        "rows":
        total_rows,

        "cols":
        total_cols,

        "missing_values":
        missing_values,

        "duplicate_rows":
        duplicate_rows,

        "missing_pct":
        missing_pct,

        "duplicate_pct":
        duplicate_pct,

        "imbalance_ratio":
        imbalance_ratio,

        "imbalance_pct":
        imbalance_pct,

        "missing_score":
        missing_score,

        "duplicate_score":
        duplicate_score,

        "imbalance_score":
        imbalance_score,

        "edqs":
        edqs
    }

    return metrics