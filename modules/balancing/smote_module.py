# =========================================================
# SMOTE MODULE
# ADVANCED RESPONSIBLE AI VERSION
# =========================================================

from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os

# =========================================================
# SMOTE COMPARISON GRAPH
# =========================================================

def save_smote_graph(
    before_counts,
    after_counts,
    save_dir="outputs/graphs"
):

    try:

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        labels=list(
            before_counts.index.astype(str)
        )

        before_values=list(
            before_counts.values
        )

        after_values=[]

        for label in labels:

            after_values.append(

                after_counts.get(
                    int(label),
                    0
                )
            )

        x=np.arange(len(labels))

        width=0.35

        plt.figure(figsize=(8,5))

        plt.bar(
            x-width/2,
            before_values,
            width,
            label="Before SMOTE"
        )

        plt.bar(
            x+width/2,
            after_values,
            width,
            label="After SMOTE"
        )

        plt.xlabel(
            "Classes"
        )

        plt.ylabel(
            "Count"
        )

        plt.title(
            "SMOTE Before vs After"
        )

        plt.xticks(
            x,
            labels
        )

        plt.legend()

        plt.tight_layout()

        save_path=os.path.join(
            save_dir,
            "smote_comparison.png"
        )

        plt.savefig(save_path)

        plt.close()

        print(
            f"\nSMOTE Graph Saved:\n{save_path}"
        )

    except Exception as e:

        print(
            f"\nSMOTE Graph Error: {e}"
        )

# =========================================================
# APPLY SMOTE USING SMOTENC
# =========================================================

def apply_smote(
    df,
    target_col,
    categorical_cols
):

    print("\n========== SMOTE STARTED ==========")

    # =====================================================
    # SAFETY COPY
    # =====================================================

    df=df.copy()

    # =====================================================
    # FEATURES & TARGET
    # =====================================================

    X=df.drop(columns=[target_col])

    y=df[target_col]

    # =====================================================
    # REMOVE NULLS
    # =====================================================

    full_df=pd.concat(
        [X,y],
        axis=1
    )

    full_df=full_df.dropna()

    X=full_df.drop(
        columns=[target_col]
    )

    y=full_df[target_col]

    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================

    class_counts=y.value_counts()

    imbalance_ratio=(

        class_counts.min()

        /

        class_counts.max()
    )

    print("\nBefore SMOTE:")

    print(class_counts)

    print(
        f"\nImbalance Ratio : "
        f"{imbalance_ratio:.4f}"
    )

    # =====================================================
    # IMBALANCE SEVERITY
    # =====================================================

    if imbalance_ratio<0.30:

        severity="HIGH IMBALANCE"

    elif imbalance_ratio<0.70:

        severity="MODERATE IMBALANCE"

    else:

        severity="LOW IMBALANCE"

    print(
        f"Imbalance Severity : {severity}"
    )

    # =====================================================
    # DATASET ALREADY BALANCED
    # =====================================================

    if imbalance_ratio>=0.95:

        print("\nDataset Already Balanced")

        print("\n========== SMOTE SKIPPED ==========")

        return None

    # =====================================================
    # NO CATEGORICAL FEATURES
    # =====================================================

    if len(categorical_cols)==0:

        print(
            "\nSMOTENC Skipped "
            "(no categorical columns)"
        )

        return None

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train,X_test,y_train,y_test=train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    # =====================================================
    # CATEGORICAL COLUMN INDICES
    # =====================================================

    categorical_indices=[]

    for col in categorical_cols:

        if col in X.columns:

            categorical_indices.append(

                X.columns.get_loc(col)
            )

    print("\nCategorical Columns:")

    print(categorical_cols)

    print("\nCategorical Indices:")

    print(categorical_indices)

    # =====================================================
    # NO VALID CATEGORICAL INDICES
    # =====================================================

    if len(categorical_indices)==0:

        print(
            "\nSMOTENC Skipped "
            "(categorical indices empty)"
        )

        return None

    # =====================================================
    # SAFE K NEIGHBORS
    # =====================================================

    minority_count=y_train.value_counts().min()

    if minority_count<=1:

        print(
            "\nSMOTE Skipped "
            "(minority class too small)"
        )

        return None

    k_neighbors=min(
        5,
        minority_count-1
    )

    print(
        f"\nk_neighbors Used : "
        f"{k_neighbors}"
    )

    # =====================================================
    # STORE BEFORE COUNTS
    # =====================================================

    before_counts=y_train.value_counts()

    # =====================================================
    # APPLY SMOTENC
    # =====================================================

    try:

        smote=SMOTENC(

            categorical_features=
            categorical_indices,

            random_state=42,

            k_neighbors=k_neighbors
        )

        X_resampled,y_resampled=(

            smote.fit_resample(

                X_train,

                y_train
            )
        )

        print(
            "\nSMOTENC Applied Successfully"
        )

    except Exception as e:

        print("\nSMOTENC FAILED")

        print(e)

        return None

    # =====================================================
    # AFTER COUNTS
    # =====================================================

    after_counts=pd.Series(
        y_resampled
    ).value_counts()

    print("\nBefore SMOTE (TRAIN):")

    print(before_counts)

    print("\nAfter SMOTE (TRAIN):")

    print(after_counts)

    # =====================================================
    # SAVE GRAPH
    # =====================================================

    save_smote_graph(
        before_counts,
        after_counts
    )

    print("\nTest Data Distribution:")

    print(
        pd.Series(y_test).value_counts()
    )

    # =====================================================
    # CONVERT TO DATAFRAME
    # =====================================================

    X_resampled=pd.DataFrame(

        X_resampled,

        columns=X.columns
    )

    y_resampled=pd.Series(

        y_resampled,

        name=target_col
    )

    # =====================================================
    # FIX CATEGORICAL COLUMNS
    # =====================================================

    for col in categorical_cols:

        if col in X_resampled.columns:

            X_resampled[col]=(

                X_resampled[col]

                .round()

                .astype(int)
            )

    # =====================================================
    # FIX TARGET TYPE
    # =====================================================

    y_resampled=y_resampled.astype(
        int
    )

    # =====================================================
    # TRAIN DATAFRAME
    # =====================================================

    train_df=pd.concat(

        [
            X_resampled,
            y_resampled
        ],

        axis=1
    )

    # =====================================================
    # TEST DATAFRAME
    # =====================================================

    test_df=pd.concat(

        [

            pd.DataFrame(
                X_test,
                columns=X.columns
            ).reset_index(drop=True),

            pd.Series(
                y_test,
                name=target_col
            ).reset_index(drop=True)
        ],

        axis=1
    )

    # =====================================================
    # FINAL DATAFRAME
    # =====================================================

    final_df=pd.concat(

        [
            train_df,
            test_df
        ],

        ignore_index=True
    )

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    before_len=len(final_df)

    final_df=final_df.drop_duplicates()

    removed_duplicates=(

        before_len-

        len(final_df)
    )

    print(
        f"\nDuplicates Removed : "
        f"{removed_duplicates}"
    )

    # =====================================================
    # SHUFFLE DATA
    # =====================================================

    final_df=final_df.sample(

        frac=1,

        random_state=42
    ).reset_index(drop=True)

    # =====================================================
    # RESET INDEX
    # =====================================================

    final_df.reset_index(

        drop=True,

        inplace=True
    )

    # =====================================================
    # FINAL DISTRIBUTION
    # =====================================================

    final_counts=(

        final_df[target_col]

        .value_counts()
    )

    final_ratio=(

        final_counts.min()

        /

        final_counts.max()
    )

    print("\nFinal Combined Distribution:")

    print(final_counts)

    print(
        f"\nFinal Imbalance Ratio : "
        f"{final_ratio:.4f}"
    )

    # =====================================================
    # IMPROVEMENT %
    # =====================================================

    improvement=(

        final_ratio-

        imbalance_ratio

    )*100

    print(
        f"\nSMOTE Improvement : "
        f"{improvement:.2f}%"
    )

    print(
        f"\nFinal Shape : "
        f"{final_df.shape}"
    )

    # =====================================================
    # VERIFY CATEGORICALS
    # =====================================================

    print(
        "\nCategorical Column Verification:"
    )

    for col in categorical_cols:

        if col in final_df.columns:

            unique_values=sorted(

                final_df[col]

                .dropna()

                .unique()

                .tolist()
            )

            print(
                f"{col} -> "
                f"{unique_values[:15]}"
            )

    # =====================================================
    # SAVE REPORT
    # =====================================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    report_path=os.path.join(
        "outputs",
        "smote_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "========== SMOTE REPORT ==========\n\n"
        )

        f.write(
            f"Before Ratio : "
            f"{imbalance_ratio:.4f}\n"
        )

        f.write(
            f"After Ratio : "
            f"{final_ratio:.4f}\n"
        )

        f.write(
            f"Improvement : "
            f"{improvement:.2f}%\n"
        )

        f.write(
            f"Severity : "
            f"{severity}\n"
        )

        f.write(
            f"Final Shape : "
            f"{final_df.shape}\n"
        )

    print(
        f"\nSMOTE Report Saved:\n{report_path}"
    )

    print(
        "\n========== SMOTE COMPLETED =========="
    )

    return final_df