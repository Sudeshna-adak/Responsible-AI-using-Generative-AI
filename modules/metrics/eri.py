# =========================================================
# ETHICAL RISK INDEX (ERI)
# ADVANCED RESPONSIBLE AI VERSION
# =========================================================

import numpy as np


# =========================================================
# CALCULATE ERI
# =========================================================

def calculate_eri(
    fairness_score,
    accuracy,
    imbalance_ratio,
    explainability_score=0.80
):

    """
    =====================================================
    ETHICAL RISK INDEX (ERI)

    Lower ERI  = Better Ethical AI
    Higher ERI = Higher Ethical Risk

    Range: 0 → 1

    Formula:

    ERI =
        0.35 × Fairness Risk
      + 0.30 × Accuracy Risk
      + 0.20 × Imbalance Risk
      + 0.15 × Explainability Risk

    =====================================================
    """

    print("\n========== ERI CALCULATION ==========")

    # =================================================
    # PRINT FORMULA
    # =================================================

    print(
        "\nERI Formula:"
    )

    print(
        """
ERI =
    0.35 × Fairness Risk
  + 0.30 × Accuracy Risk
  + 0.20 × Imbalance Risk
  + 0.15 × Explainability Risk
        """
    )

    # =================================================
    # SAFETY CLIPPING
    # =================================================

    fairness_score = np.clip(
        fairness_score,
        0,
        1
    )

    accuracy = np.clip(
        accuracy,
        0,
        1
    )

    imbalance_ratio = np.clip(
        imbalance_ratio,
        0,
        1
    )

    explainability_score = np.clip(
        explainability_score,
        0,
        1
    )

    # =================================================
    # CONVERT TO RISK VALUES
    # =================================================

    fairness_risk = 1 - fairness_score

    accuracy_risk = 1 - accuracy

    imbalance_risk = 1 - imbalance_ratio

    explainability_risk = (
        1 - explainability_score
    )

    # =================================================
    # PRINT COMPONENTS
    # =================================================

    print("\nRisk Components:")

    print(
        f"Fairness Risk       : {fairness_risk:.4f}"
    )

    print(
        f"Accuracy Risk       : {accuracy_risk:.4f}"
    )

    print(
        f"Imbalance Risk      : {imbalance_risk:.4f}"
    )

    print(
        f"Explainability Risk : {explainability_risk:.4f}"
    )

    # =================================================
    # FINAL ERI
    # =================================================

    eri = (

        0.35 * fairness_risk +

        0.30 * accuracy_risk +

        0.20 * imbalance_risk +

        0.15 * explainability_risk
    )

    # =================================================
    # CLIPPING
    # =================================================

    eri = np.clip(
        eri,
        0,
        1
    )

    eri = round(
        float(eri),
        4
    )

    # =================================================
    # RISK CATEGORY
    # =================================================

    if eri < 0.20:

        risk_level = "LOW ETHICAL RISK"

    elif eri < 0.50:

        risk_level = "MODERATE ETHICAL RISK"

    else:

        risk_level = "HIGH ETHICAL RISK"

    # =================================================
    # FINAL OUTPUT
    # =================================================

    print(
        f"\nFinal ERI Score : {eri}"
    )

    print(
        f"Risk Level      : {risk_level}"
    )

    print(
        "\n========== ERI COMPLETED =========="
    )

    return eri