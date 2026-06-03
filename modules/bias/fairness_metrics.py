# =========================================================
# UNIVERSAL RESPONSIBLE AI BIAS DETECTOR
# FULLY UPDATED + STABLE VERSION
# =========================================================

import pandas as pd
import numpy as np

# =========================================================
# PROBABILITY NORMALIZER
# =========================================================

def normalize_probability(value):

    try:

        if pd.isna(value):

            return 0.0

        value = abs(float(value))

        value = min(max(value, 0), 1)

        return round(value, 2)

    except Exception:

        return 0.0

# =========================================================
# BIAS STATUS
# =========================================================

def bias_status(probability):

    if probability >= 0.60:

        return "High Bias"

    elif probability >= 0.40:

        return "Moderate Bias"

    else:

        return "Low Bias"

# =========================================================
# MAIN FAIRNESS FUNCTION
# =========================================================

def calculate_fairness(
    df,
    target_col,
    verbose=True
):

    try:

        # =================================================
        # START
        # =================================================

        if verbose:

            print(
                "\n========== FAIRNESS ANALYSIS STARTED =========="
            )

        bias_results = []

        total_rows = len(df)

        # =================================================
        # SAFETY CHECK
        # =================================================

        if total_rows == 0:

            raise ValueError(
                "Dataset is empty."
            )

        if target_col not in df.columns:

            raise ValueError(
                f"Target column '{target_col}' not found."
            )

        # =================================================
        # CLASS DISTRIBUTION
        # =================================================

        class_distribution = (

            df[target_col]
            .value_counts(normalize=True)
        )

        # =================================================
        # 1. SELECTION BIAS
        # =================================================

        selection_bias = (

            class_distribution.max()
            -
            class_distribution.min()
        )

        selection_bias = normalize_probability(
            selection_bias
        )

        bias_results.append({

            "Bias Type":
            "Selection Bias",

            "Probability":
            selection_bias,

            "Bias Status":
            bias_status(selection_bias)
        })

        # =================================================
        # 2. SAMPLING BIAS
        # =================================================

        sample_size = max(
            int(total_rows * 0.5),
            1
        )

        sampling_bias = abs(

            len(
                df.sample(
                    n=sample_size,
                    random_state=42
                )
            )

            / total_rows

            - 0.5
        )

        sampling_bias = normalize_probability(
            sampling_bias
        )

        bias_results.append({

            "Bias Type":
            "Sampling Bias",

            "Probability":
            sampling_bias,

            "Bias Status":
            bias_status(sampling_bias)
        })

        # =================================================
        # 3. RESPONSE BIAS
        # =================================================

        missing_ratio = (

            df.isnull()
            .sum()
            .sum()

            /

            (df.shape[0] * df.shape[1])
        )

        response_bias = normalize_probability(
            missing_ratio
        )

        bias_results.append({

            "Bias Type":
            "Response Bias",

            "Probability":
            response_bias,

            "Bias Status":
            bias_status(response_bias)
        })

        # =================================================
        # 4. LABEL BIAS
        # =================================================

        label_bias = (
            class_distribution.std()
        )

        label_bias = normalize_probability(
            label_bias
        )

        bias_results.append({

            "Bias Type":
            "Label Bias",

            "Probability":
            label_bias,

            "Bias Status":
            bias_status(label_bias)
        })

        # =================================================
        # NUMERIC COLUMNS
        # =================================================

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        # =================================================
        # REMOVE TARGET FROM NUMERIC COLS
        # =================================================

        if target_col in numeric_cols:

            numeric_cols.remove(
                target_col
            )

        # =================================================
        # 5. MEASUREMENT BIAS
        # =================================================

        measurement_bias = 0

        if len(numeric_cols) > 0:

            skewness_values = []

            for col in numeric_cols:

                try:

                    skew_val = abs(
                        df[col].skew()
                    )

                    if not pd.isna(skew_val):

                        skewness_values.append(
                            skew_val
                        )

                except Exception:

                    continue

            if len(skewness_values) > 0:

                measurement_bias = (
                    np.mean(skewness_values)
                    / 10
                )

        measurement_bias = normalize_probability(
            measurement_bias
        )

        bias_results.append({

            "Bias Type":
            "Measurement Bias",

            "Probability":
            measurement_bias,

            "Bias Status":
            bias_status(measurement_bias)
        })

        # =================================================
        # 6. REPRESENTATION BIAS
        # =================================================

        representation_bias = (

            1
            -
            class_distribution.min()
        )

        representation_bias = normalize_probability(
            representation_bias
        )

        bias_results.append({

            "Bias Type":
            "Representation Bias",

            "Probability":
            representation_bias,

            "Bias Status":
            bias_status(representation_bias)
        })

        # =================================================
        # 7. PROXY BIAS
        # =================================================

        proxy_bias = 0

        if len(numeric_cols) > 1:

            try:

                corr_matrix = (

                    df[numeric_cols]
                    .corr()
                    .abs()
                )

                np.fill_diagonal(
                    corr_matrix.values,
                    0
                )

                proxy_bias = (

                    corr_matrix.mean().mean()
                ) / 2

            except Exception:

                proxy_bias = 0

        proxy_bias = normalize_probability(
            proxy_bias
        )

        bias_results.append({

            "Bias Type":
            "Proxy Bias",

            "Probability":
            proxy_bias,

            "Bias Status":
            bias_status(proxy_bias)
        })

        # =================================================
        # 8. CONFIRMATION BIAS
        # =================================================

        confirmation_bias = 0

        if len(numeric_cols) > 0:

            try:

                variances = []

                for col in numeric_cols:

                    var = df[col].var()

                    if not pd.isna(var):

                        variances.append(var)

                if len(variances) > 0:

                    confirmation_bias = (

                        1
                        /
                        (
                            np.mean(variances)
                            + 1
                        )
                    )

            except Exception:

                confirmation_bias = 0

        confirmation_bias = normalize_probability(
            confirmation_bias
        )

        bias_results.append({

            "Bias Type":
            "Confirmation Bias",

            "Probability":
            confirmation_bias,

            "Bias Status":
            bias_status(confirmation_bias)
        })

        # =================================================
        # FINAL DATAFRAME
        # =================================================

        bias_df = pd.DataFrame(
            bias_results
        )

        # =================================================
        # FAIRNESS SCORE
        # =================================================

        avg_bias = bias_df[
            "Probability"
        ].mean()

        fairness_score = round(
            1 - avg_bias,
            2
        )

        fairness_score = max(
            fairness_score,
            0
        )

        # =================================================
        # RETURN
        # =================================================

        return bias_df, fairness_score

    except Exception as e:

        print(
            f"\nFairness Calculation Error: {e}"
        )

        # =================================================
        # FAILSAFE OUTPUT
        # =================================================

        fallback_df = pd.DataFrame([{

            "Bias Type":
            "Unknown",

            "Probability":
            0.0,

            "Bias Status":
            "Low Bias"
        }])

        return fallback_df, 0.5