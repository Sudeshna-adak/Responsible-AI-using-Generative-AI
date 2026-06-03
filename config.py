import os

from dotenv import load_dotenv

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

# =========================================================
# API KEYS
# =========================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

# =========================================================
# PROJECT SETTINGS
# =========================================================

TARGET_COLUMN = None

TEST_SIZE = 0.2

RANDOM_STATE = 42

SMOTE_THRESHOLD = 0.4

CORRELATION_THRESHOLD = 0.95

EDQS_WEIGHTS = {

    "missing": 0.4,

    "duplicate": 0.3,

    "imbalance": 0.3
}