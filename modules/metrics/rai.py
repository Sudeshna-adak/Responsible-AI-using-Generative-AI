# =========================================================
# RESPONSIBLE AI INDEX (RAI)
# =========================================================

import numpy as np


def calculate_rai(
    edqs,
    fairness,
    accuracy,
    eri
):

    """
    RESPONSIBLE AI INDEX (RAI)

    Higher = Better
    Range  = 0 to 100
    """

    # =====================================================
    # NORMALIZATION
    # =====================================================

    fairness_score = fairness * 100

    accuracy_score = accuracy * 100

    ethical_score = (1 - eri) * 100

    # =====================================================
    # RESPONSIBLE AI FORMULA
    # =====================================================

    rai_score = (

        0.30 * edqs +

        0.30 * fairness_score +

        0.25 * accuracy_score +

        0.15 * ethical_score
    )

    # =====================================================
    # CLIPPING
    # =====================================================

    rai_score = np.clip(
        rai_score,
        0,
        100
    )

    # =====================================================
    # ROUNDING
    # =====================================================

    return round(
        float(rai_score),
        2
    )