from modules.balancing.smote_module import (
    apply_smote
)

from modules.bias.proxy_bias_remover import (
    remove_proxy_bias
)

from modules.preprocessing.skewness_fixer import (
    fix_skewness
)

import numpy as np


# =========================================================
# APPLY LLM RECOMMENDATIONS
# =========================================================

def apply_llm_recommendations(

    df,

    recommendations,

    target_col,

    categorical_cols
):

    print(
        "\nApplying LLM Recommendations"
    )

    changes_applied = False

    # =====================================
    # REMOVE DUPLICATES
    # =====================================

    if recommendations.get(
        "remove_duplicates"
    ):

        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        changes_applied = True

        print(
            f"Duplicates Removed : "
            f"{before - after}"
        )

    # =====================================
    # FILL MISSING VALUES
    # =====================================

    if recommendations.get(
        "fill_missing_values"
    ):

        for col in df.columns:

            if (
                df[col].dtype == "object"
            ):

                df[col] = (
                    df[col]
                    .fillna(
                        df[col]
                        .mode()[0]
                    )
                )

            else:

                df[col] = (
                    df[col]
                    .fillna(
                        df[col]
                        .median()
                    )
                )

        changes_applied = True

        print(
            "Missing Values Filled"
        )

    # =====================================
    # CHECK CLASS IMBALANCE
    # =====================================

    imbalance_ratio = 1.0

    if target_col in df.columns:

        class_distribution = (
            df[target_col]
            .value_counts(normalize=True)
        )

        if len(class_distribution) > 1:

            imbalance_ratio = (
                class_distribution.min()
                / class_distribution.max()
            )

    print(
        f"Imbalance Ratio : "
        f"{imbalance_ratio:.2f}"
    )

    # =====================================
    # APPLY SMOTE
    # =====================================

    if recommendations.get(
        "apply_smote"
    ) and imbalance_ratio < 0.7:

        df = apply_smote(

            df,

            target_col,

            categorical_cols
        )

        changes_applied = True

        print(
            "SMOTE Applied"
        )

    else:

        print(
            "SMOTE Skipped"
        )

    # =====================================
    # REMOVE PROXY BIAS
    # =====================================

    if recommendations.get(
        "remove_proxy_bias"
    ):

        df = remove_proxy_bias(

            df,

            target_col
        )

        changes_applied = True

        print(
            "Proxy Bias Removed"
        )

    # =====================================
    # FIX SKEWNESS
    # =====================================

    if recommendations.get(
        "fix_skewness"
    ):

        df = fix_skewness(

            df,

            target_col
        )

        changes_applied = True

        print(
            "Skewness Fixed"
        )

    # =====================================
    # REMOVE CONSTANT COLUMNS
    # =====================================

    if recommendations.get(
        "remove_constant_columns"
    ):

        constant_cols = [

            col for col in df.columns

            if df[col].nunique() <= 1
        ]

        if constant_cols:

            protected_cols = [
                target_col
            ]

            constant_cols = [

                col for col in constant_cols

                if col not in protected_cols
            ]

            if constant_cols:

                df = df.drop(
                    columns=constant_cols
                )

                changes_applied = True

                print(
                    f"Constant Columns Removed : "
                    f"{constant_cols}"
                )

    # =====================================
    # REMOVE HIGHLY CORRELATED COLUMNS
    # =====================================

    if recommendations.get(
        "remove_correlated_columns"
    ):

        numeric_df = df.select_dtypes(
            include=["number"]
        )

        corr_matrix = (
            numeric_df.corr().abs()
        )

        upper_triangle = corr_matrix.where(

            np.triu(

                np.ones(
                    corr_matrix.shape
                ),

                k=1

            ).astype(bool)
        )

        drop_cols = [

            column

            for column in upper_triangle.columns

            if any(
                upper_triangle[column] > 0.95
            )
        ]

        protected_cols = [
            target_col
        ]

        drop_cols = [

            col for col in drop_cols

            if col not in protected_cols
        ]

        if drop_cols:

            df = df.drop(
                columns=drop_cols
            )

            changes_applied = True

            print(
                f"Correlated Columns Removed : "
                f"{drop_cols}"
            )

    # =====================================
    # RESET INDEX
    # =====================================

    df = df.reset_index(
        drop=True
    )

    # =====================================
    # CHANGE STATUS
    # =====================================

    if changes_applied:

        print(
            "\nLLM Changes Applied Successfully"
        )

    else:

        print(
            "\nNo LLM Changes Applied"
        )

    return df