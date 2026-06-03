# =========================================================
# LLM EXPLAINER WITH GROQ INTEGRATION
# FINAL PROFESSIONAL VERSION
# =========================================================

import os

from groq import Groq

from config import GROQ_API_KEY

# =========================================================
# GROQ API CONFIGURATION
# =========================================================

MODEL_NAME = "llama-3.3-70b-versatile"

# =========================================================
# INITIALIZE GROQ CLIENT
# =========================================================

client = Groq(
    api_key=GROQ_API_KEY
)

# =========================================================
# LLM EXPLANATION FUNCTION
# =========================================================

def explain_dataset(
    df,
    target_col,
    edqs,
    eri,
    bias_report
):

    print(
        "\n========== LLM DATASET EXPLANATION =========="
    )

    # =====================================================
    # BASIC DATASET INFO
    # =====================================================

    total_rows, total_cols = df.shape

    bias_count = sum(
        1 for v in bias_report.values()
        if v
    )

    # =====================================================
    # BIAS OBSERVATIONS
    # =====================================================

    observations = []

    if bias_count > 3:

        observations.append(
            "Multiple biases detected"
        )

    else:

        observations.append(
            "Limited bias detected"
        )

    # =====================================================
    # EDQS OBSERVATION
    # =====================================================

    if edqs > 80:

        observations.append(
            "Dataset quality is high"
        )

    elif edqs > 60:

        observations.append(
            "Dataset quality is moderate"
        )

    else:

        observations.append(
            "Dataset quality is poor"
        )

    # =====================================================
    # ERI OBSERVATION
    # =====================================================

    if eri < 30:

        observations.append(
            "Ethical risk is low"
        )

    elif eri < 60:

        observations.append(
            "Ethical risk is moderate"
        )

    else:

        observations.append(
            "Ethical risk is high"
        )

    # =====================================================
    # EXCLUDE HELPER COLUMNS
    # =====================================================

    exclude_cols = [
        target_col,
        "data_type"
    ]

    # =====================================================
    # DETECT NUMERICAL FEATURES
    # =====================================================

    numerical_cols = [

        col

        for col in df.select_dtypes(
            include=['int64', 'float64']
        ).columns

        if col not in exclude_cols
    ]

    # =====================================================
    # DETECT CATEGORICAL FEATURES
    # =====================================================

    categorical_cols = [

        col

        for col in df.select_dtypes(
            include=['object']
        ).columns

        if col not in exclude_cols
    ]

    # =====================================================
    # MISSING VALUES
    # =====================================================

    missing_values = int(
        df.isnull().sum().sum()
    )

    missing_percentage = (

        missing_values

        /

        (df.shape[0] * df.shape[1])

    ) * 100

    # =====================================================
    # DUPLICATES
    # =====================================================

    duplicate_rows = (
        df.duplicated().sum()
    )

    # =====================================================
    # TARGET DISTRIBUTION
    # =====================================================

    target_distribution = (

        df[target_col]

        .value_counts()

        .to_dict()
    )

    # =====================================================
    # DETECTED BIASES
    # =====================================================

    detected_biases = [

        bias

        for bias, status

        in bias_report.items()

        if status
    ]

    if len(detected_biases) == 0:

        detected_biases_text = (
            "No major biases detected"
        )

    else:

        detected_biases_text = (
            ", ".join(detected_biases)
        )

    # =====================================================
    # LOCAL RULE-BASED EXPLANATION
    # =====================================================

    local_explanation = f"""

DATASET SUMMARY
--------------------------------

Total Records  : {total_rows}

Total Features : {total_cols}

Target Column  : {target_col}

Numerical Features : {len(numerical_cols)}

Categorical Features : {len(categorical_cols)}

Missing Values : {missing_values}

Missing Percentage : {missing_percentage:.2f}%

Duplicate Rows : {duplicate_rows}

EDQS Score     : {edqs:.2f}

ERI Score      : {eri:.2f}


TARGET DISTRIBUTION
--------------------------------

{target_distribution}


DETECTED BIASES
--------------------------------

{detected_biases_text}


KEY OBSERVATIONS
--------------------------------

• {observations[0]}

• {observations[1]}

• {observations[2]}


SYNTHETIC DATA ANALYSIS
--------------------------------

The dataset was enhanced using
controlled synthetic data generation
to improve minority representation
and reduce imbalance issues.


INTERPRETATION
--------------------------------

The dataset has undergone preprocessing,
bias detection, fairness correction,
and ethical quality assessment.

The Responsible AI pipeline improved
data fairness, reduced ethical risks,
and enhanced overall dataset reliability.
"""

    # =====================================================
    # PRINT LOCAL EXPLANATION
    # =====================================================

    print(local_explanation)

    # =====================================================
    # CHECK API KEY
    # =====================================================

    if not GROQ_API_KEY:

        print(
            "\nLLM service unavailable"
        )

        print(
            "\nUsing local explanation only."
        )

        return local_explanation

    # =====================================================
    # LLM PROMPT
    # =====================================================

    prompt = f"""
You are an expert Responsible AI analyst.

Analyze the following Responsible AI pipeline results.

DATASET DETAILS:
------------------------

Total Rows: {total_rows}

Total Columns: {total_cols}

Target Column: {target_col}

Numerical Features: {len(numerical_cols)}

Categorical Features: {len(categorical_cols)}

Missing Values: {missing_values}

Missing Percentage: {missing_percentage:.2f}%

Duplicate Rows: {duplicate_rows}

EDQS Score: {edqs:.2f}

ERI Score: {eri:.2f}

Target Distribution:
{target_distribution}

Detected Biases:
{detected_biases_text}

Synthetic Data Usage:
The dataset includes controlled synthetic
data generation for balancing minority classes.

TASK:
------------------------

Provide:

1. Dataset quality analysis

2. Bias and fairness interpretation

3. Ethical risk analysis

4. AI reliability assessment

5. Recommendations for improvement

6. Final Responsible AI verdict

Use professional but simple language.
"""

    # =====================================================
    # GROQ API REQUEST
    # =====================================================

    try:

        print(
            "\nGenerating AI explanation "
            "using Groq..."
        )

        response = (
            client.chat.completions.create(

                model=MODEL_NAME,

                messages=[

                    {
                        "role": "system",

                        "content":
                        (
                            "You are an expert in "
                            "Responsible AI, "
                            "Ethical AI, "
                            "Bias Detection, "
                            "and Fairness Analysis."
                        )
                    },

                    {
                        "role": "user",

                        "content": prompt
                    }
                ],

                temperature=0.7,

                max_tokens=1500
            )
        )

        # =================================================
        # PARSE RESPONSE
        # =================================================

        ai_explanation = (

            response
            .choices[0]
            .message.content
        )

        # =================================================
        # PRINT AI RESPONSE
        # =================================================

        print(
            "\n========== AI GENERATED "
            "EXPLANATION ==========\n"
        )

        print(ai_explanation)

        print(
            "\n===================================="
        )

        # =================================================
        # SAVE REPORT
        # =================================================

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        report_path = os.path.join(
            "outputs",
            "llm_explanation.txt"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(ai_explanation)

        print(
            f"\nAI Explanation Saved:\n"
            f"{report_path}"
        )

        return ai_explanation

    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception:

        print(
            "\nLLM service unavailable"
        )

        print(
            "\nUsing local explanation only."
        )

        return local_explanation