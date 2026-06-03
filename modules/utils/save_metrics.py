# =========================================================
# SAVE METRICS MODULE
# =========================================================

import os
from datetime import datetime

# =========================================================
# SAVE METRICS
# =========================================================

def save_metrics(metrics_data):

    """
    Saves all Responsible AI metrics
    into outputs/metrics.txt
    """

    try:

        # =================================================
        # CREATE OUTPUT DIRECTORY
        # =================================================

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        # =================================================
        # OUTPUT FILE
        # =================================================

        output_path = os.path.join(

            "outputs",

            "metrics.txt"
        )

        # =================================================
        # WRITE METRICS
        # =================================================

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "=" * 60 + "\n"
            )

            f.write(
                "RESPONSIBLE AI METRICS SUMMARY\n"
            )

            f.write(
                "=" * 60 + "\n\n"
            )

            # =============================================
            # TIMESTAMP
            # =============================================

            f.write(
                f"Generated On : "
                f"{datetime.now()}\n\n"
            )

            # =============================================
            # METRICS
            # =============================================

            for key, value in metrics_data.items():

                f.write(
                    f"{key}: {value}\n"
                )

            f.write("\n")

            f.write(
                "=" * 60 + "\n"
            )

            f.write(
                "END OF METRICS\n"
            )

            f.write(
                "=" * 60 + "\n"
            )

        print(
            f"\nMetrics Saved Successfully:\n{output_path}"
        )

        return output_path

    except Exception as e:

        print(
            f"\nMetrics Save Error: {e}"
        )

        return None