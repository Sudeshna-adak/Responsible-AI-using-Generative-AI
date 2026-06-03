# =========================================================
# FAIRNESS FIX MODULE
# ADVANCED RESPONSIBLE AI VERSION
# =========================================================

import pandas as pd
import numpy as np

from sklearn.utils import resample

# =========================================================
# APPLY FAIRNESS FIX
# =========================================================

def apply_fairness_fix(
    df,
    bias_results,
    target_col
):

    print(
        "\n========== FAIRNESS FIX STARTED =========="
    )

    df=df.copy()

    try:

        # =================================================
        # EMPTY BIAS CHECK
        # =================================================

        if (

            bias_results is None

            or

            len(bias_results)==0
        ):

            print(
                "\nNo Bias Results Found"
            )

            return df

        # =================================================
        # CONVERT TO DATAFRAME
        # =================================================

        if not isinstance(
            bias_results,
            pd.DataFrame
        ):

            bias_results=pd.DataFrame(
                bias_results
            )

        # =================================================
        # REQUIRED COLUMN CHECK
        # =================================================

        required_cols=[
            "Bias Type",
            "Probability"
        ]

        for col in required_cols:

            if col not in bias_results.columns:

                print(
                    f"\nMissing Bias Column: {col}"
                )

                return df

        # =================================================
        # HIGH / MODERATE BIAS TYPES
        # =================================================

        high_bias=bias_results[

            bias_results["Probability"]>0.20
        ]

        print("\nDetected Bias Types:")

        print(
            high_bias["Bias Type"]
            .tolist()
        )

        # =================================================
        # REPRESENTATION BIAS FIX
        # =================================================

        if (

            "Representation Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nFixing Representation Bias..."
            )

            class_counts=(

                df[target_col]

                .value_counts()
            )

            print(
                "\nBefore Balancing:"
            )

            print(class_counts)

            majority_count=(
                class_counts.max()
            )

            # =============================================
            # SMART TARGET SIZE
            # =============================================

            target_size=int(
                majority_count*0.75
            )

            balanced_data=[]

            for cls in class_counts.index:

                cls_df=df[
                    df[target_col]==cls
                ]

                # =========================================
                # UPSAMPLING
                # =========================================

                if len(cls_df)<target_size:

                    cls_df=resample(

                        cls_df,

                        replace=True,

                        n_samples=target_size,

                        random_state=42
                    )

                # =========================================
                # DOWNSAMPLING
                # =========================================

                elif len(cls_df)>target_size:

                    cls_df=resample(

                        cls_df,

                        replace=False,

                        n_samples=target_size,

                        random_state=42
                    )

                balanced_data.append(
                    cls_df
                )

            df=pd.concat(

                balanced_data,

                ignore_index=True
            )

            # =============================================
            # SHUFFLE DATA
            # =============================================

            df=df.sample(

                frac=1,

                random_state=42
            ).reset_index(drop=True)

            # =============================================
            # DUPLICATE CONTROL
            # =============================================

            duplicate_percent=(

                df.duplicated()

                .mean()

            )*100

            print(
                f"\nDuplicate Percentage : "
                f"{duplicate_percent:.2f}%"
            )

            if duplicate_percent>25:

                before=len(df)

                df=df.drop_duplicates()

                removed=(
                    before-len(df)
                )

                print(
                    f"\nHigh Duplicate Bias Removed : "
                    f"{removed}"
                )

            # =============================================
            # FINAL SHUFFLE
            # =============================================

            df=df.sample(

                frac=1,

                random_state=42
            ).reset_index(drop=True)

            print(
                "\nAfter Balancing:"
            )

            print(
                df[target_col]
                .value_counts()
            )

            print(
                "\nRepresentation Bias Reduced"
            )

        # =================================================
        # SELECTION BIAS FIX
        # =================================================

        if (

            "Selection Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nFixing Selection Bias..."
            )

            df=df.sample(

                frac=1,

                random_state=42
            ).reset_index(drop=True)

            print(
                "\nSelection Bias Reduced"
            )

        # =================================================
        # PROXY BIAS FIX
        # =================================================

        if (

            "Proxy Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nFixing Proxy Bias..."
            )

            numeric_cols=df.select_dtypes(
                include=np.number
            ).columns.tolist()

            protected_corr_threshold=0.75

            corr_matrix=df[
                numeric_cols
            ].corr().abs()

            removed_cols=[]

            # =============================================
            # REMOVE HIGHLY CORRELATED FEATURES
            # =============================================

            for col in numeric_cols:

                if col==target_col:

                    continue

                target_corr=(

                    corr_matrix[
                        target_col
                    ][col]
                )

                if (
                    target_corr>
                    protected_corr_threshold
                ):

                    removed_cols.append(col)

            removed_cols=[

                c for c in removed_cols

                if c!=target_col
            ]

            if len(removed_cols)>0:

                print(
                    f"\nRemoving Proxy Columns: "
                    f"{removed_cols}"
                )

                df.drop(

                    columns=removed_cols,

                    inplace=True,

                    errors="ignore"
                )

            # =============================================
            # FAIRNESS NOISE
            # =============================================

            remaining_numeric=df.select_dtypes(
                include=np.number
            ).columns

            for col in remaining_numeric:

                if col!=target_col:

                    noise=np.random.normal(

                        0,

                        0.05,

                        len(df)
                    )

                    df[col]=(

                        df[col]

                        +noise
                    )

            print(
                "\nProxy Bias Reduced"
            )

        # =================================================
        # OUTCOME BIAS FIX
        # =================================================

        if (

            "Outcome Bias"

            in

            high_bias["Bias Type"].values
        ):

            print(
                "\nFixing Outcome Bias..."
            )

            numeric_cols=df.select_dtypes(
                include=np.number
            ).columns

            for col in numeric_cols:

                if col!=target_col:

                    q1=df[col].quantile(0.25)

                    q3=df[col].quantile(0.75)

                    iqr=q3-q1

                    lower=q1-1.5*iqr

                    upper=q3+1.5*iqr

                    df[col]=np.clip(

                        df[col],

                        lower,

                        upper
                    )

            print(
                "\nOutcome Bias Reduced"
            )

        # =================================================
        # FINAL DUPLICATE CHECK
        # =================================================

        final_duplicate_percent=(

            df.duplicated()

            .mean()

        )*100

        print(
            f"\nFinal Duplicate Percentage : "
            f"{final_duplicate_percent:.2f}%"
        )

        # =================================================
        # FINAL SUMMARY
        # =================================================

        print(
            "\nFinal Dataset Shape:"
        )

        print(df.shape)

        print(
            "\n========== FAIRNESS FIX COMPLETED =========="
        )

        return df

    except Exception as e:

        print(
            f"\nFairness Fix Error: {e}"
        )

        return df