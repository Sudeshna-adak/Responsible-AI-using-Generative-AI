# =========================================================
# LLM DECISION ENGINE
# ADVANCED RESPONSIBLE AI VERSION
# =========================================================

import json
import os

from groq import Groq

from config import GROQ_API_KEY

# =========================================================
# LOAD API KEY
# =========================================================

client = Groq(
    api_key=GROQ_API_KEY
)

# =========================================================
# LOAD METRICS FILE
# =========================================================

def load_metrics_file():

    try:

        metrics_path = os.path.join(
            "outputs",
            "metrics.txt"
        )

        with open(
            metrics_path,
            "r",
            encoding="utf-8"
        ) as f:

            metrics_text = f.read()

        return metrics_text

    except Exception as e:

        print(
            f"\nMetrics Load Error: {e}"
        )

        return ""

# =========================================================
# GENERATE LLM ANALYSIS
# =========================================================

def generate_llm_analysis(metrics_text=None):

    try:

        # =============================================
        # AUTO LOAD METRICS
        # =============================================

        if metrics_text is None:

            metrics_text = load_metrics_file()

        # =============================================
        # PROMPT
        # =============================================

        prompt = f"""

You are an expert Responsible AI Analyst.

Analyze the Responsible AI metrics carefully.

Generate a professional AI governance report.

Your analysis MUST include:

1. Dataset Quality Analysis
2. Fairness Analysis
3. Bias Severity Analysis
4. Ethical Risk Analysis
5. Explainability Review
6. Model Performance Review
7. Responsible AI Recommendations
8. Final Risk Assessment

Metrics:

{metrics_text}

"""

        # =============================================
        # API CALL
        # =============================================

        completion = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[

                {
                    "role": "system",

                    "content":
                    (
                        "You are an expert "
                        "Responsible AI Governance Engine."
                    )
                },

                {
                    "role": "user",

                    "content": prompt
                }
            ],

            temperature=0.2,

            max_tokens=1200
        )

        # =============================================
        # RESPONSE
        # =============================================

        response = (

            completion
            .choices[0]
            .message
            .content
        )

        # =============================================
        # SAVE ANALYSIS
        # =============================================

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        analysis_path = os.path.join(
            "outputs",
            "llm_analysis.txt"
        )

        with open(
            analysis_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(response)

        print(
            f"\nLLM Analysis Saved:\n{analysis_path}"
        )

        return response.strip()

    except Exception as e:

        print(
            f"\nLLM Analysis Error: {e}"
        )

        return (
            "LLM analysis unavailable"
        )

# =========================================================
# GENERATE LLM RECOMMENDATIONS
# =========================================================

def generate_llm_recommendations(metrics_text=None):

    try:

        # =============================================
        # AUTO LOAD METRICS
        # =============================================

        if metrics_text is None:

            metrics_text = load_metrics_file()

        # =============================================
        # PROMPT
        # =============================================

        prompt = f"""

You are an advanced Responsible AI Optimization Engine.

Analyze the Responsible AI metrics carefully.

Return ONLY valid JSON.

DO NOT explain anything.

DO NOT write markdown.

STRICTLY return JSON only.

Required JSON format:

{{
    "remove_duplicates": false,
    "fill_missing_values": false,
    "apply_smote": false,
    "remove_proxy_bias": false,
    "fix_skewness": false,
    "remove_outliers": false,
    "rebalance_classes": false,
    "apply_fairness_fix": false,
    "increase_regularization": false,
    "retrain_model": true
}}

Decision Rules:

1. If imbalance ratio < 0.75:
   apply_smote = true
   rebalance_classes = true

2. If fairness score < 0.80:
   apply_fairness_fix = true

3. If ERI score > 0.30:
   increase_regularization = true

4. If proxy bias exists:
   remove_proxy_bias = true

5. If skewness detected:
   fix_skewness = true

6. If outliers detected:
   remove_outliers = true

7. If missing values exist:
   fill_missing_values = true

8. Always:
   retrain_model = true

Metrics:

{metrics_text}

"""

        # =============================================
        # API CALL
        # =============================================

        completion = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[

                {
                    "role": "system",

                    "content":
                    (
                        "You are a Responsible AI "
                        "optimization engine."
                    )
                },

                {
                    "role": "user",

                    "content": prompt
                }
            ],

            temperature=0,

            max_tokens=500
        )

        # =============================================
        # RESPONSE
        # =============================================

        response = (

            completion
            .choices[0]
            .message
            .content
        )

        response = response.strip()

        # =============================================
        # REMOVE MARKDOWN
        # =============================================

        if response.startswith("```json"):

            response = (
                response
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        elif response.startswith("```"):

            response = (
                response
                .replace("```", "")
                .strip()
            )

        # =============================================
        # JSON PARSE
        # =============================================

        recommendations = json.loads(
            response
        )

        # =============================================
        # DEFAULT SAFETY
        # =============================================

        defaults = {

            "remove_duplicates": False,

            "fill_missing_values": False,

            "apply_smote": False,

            "remove_proxy_bias": False,

            "fix_skewness": False,

            "remove_outliers": False,

            "rebalance_classes": False,

            "apply_fairness_fix": False,

            "increase_regularization": False,

            "retrain_model": True
        }

        defaults.update(
            recommendations
        )

        recommendations = defaults

        # =============================================
        # SMART RECOMMENDATION FIX
        # =============================================

        try:

            accuracy = 0

            fairness = 0

            eri = 1

            for line in metrics_text.split("\n"):

                lower_line = line.lower()

                if "accuracy" in lower_line:

                    accuracy = float(

                        line.split(":")[-1]
                        .strip()
                    )

                elif "fairness after" in lower_line:

                    fairness = float(

                        line.split(":")[-1]
                        .strip()
                    )

                elif "eri after" in lower_line:

                    eri = float(

                        line.split(":")[-1]
                        .strip()
                    )

            print(
                f"\nParsed Accuracy : "
                f"{accuracy}"
            )

            print(
                f"Parsed Fairness : "
                f"{fairness}"
            )

            print(
                f"Parsed ERI : "
                f"{eri}"
            )

            if (

                accuracy >= 0.85

                and

                fairness >= 0.90

                and

                eri <= 0.10
            ):

                recommendations[
                    "increase_regularization"
                ] = False

                recommendations[
                    "retrain_model"
                ] = False

                print(
                    "\nSmart Recommendation Fix Applied"
                )

        except Exception as e:

            print(e)
            
        # =============================================
        # SAVE RECOMMENDATIONS
        # =============================================

        recommendation_path = os.path.join(
            "outputs",
            "llm_recommendations.txt"
        )

        with open(
            recommendation_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(

                json.dumps(
                    recommendations,
                    indent=4
                )
            )

        # =============================================
        # PRINT
        # =============================================

        print(
            "\n========== LLM RECOMMENDATIONS ==========\n"
        )

        print(
            json.dumps(
                recommendations,
                indent=4
            )
        )

        print(
            f"\nRecommendations Saved:\n"
            f"{recommendation_path}"
        )

        return recommendations

    except Exception as e:

        print(
            f"\nLLM Recommendation Error: {e}"
        )

        # =============================================
        # FAILSAFE
        # =============================================

        return {

            "remove_duplicates": False,

            "fill_missing_values": False,

            "apply_smote": False,

            "remove_proxy_bias": False,

            "fix_skewness": False,

            "remove_outliers": False,

            "rebalance_classes": False,

            "apply_fairness_fix": False,

            "increase_regularization": False,

            "retrain_model": True
        }