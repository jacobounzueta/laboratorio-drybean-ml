"""Configuración central del proyecto (TDSP).

Concentra rutas, constantes del dataset y parámetros de reproducibilidad para
que código, notebooks y pruebas compartan exactamente lo mismo.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto (estructura TDSP)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_MODELS_DIR = OUTPUTS_DIR / "models"
OUTPUTS_REPORTS_DIR = OUTPUTS_DIR / "reports"
SPRINT1_REPORTS_DIR = OUTPUTS_REPORTS_DIR / "sprint1"
SPRINT1_FIGURES_DIR = SPRINT1_REPORTS_DIR / "figuras"
SPRINT2_REPORTS_DIR = OUTPUTS_REPORTS_DIR / "sprint2"
SPRINT2_FIGURES_DIR = SPRINT2_REPORTS_DIR / "figuras"

for _directory in (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    OUTPUTS_MODELS_DIR,
    OUTPUTS_REPORTS_DIR,
    SPRINT1_REPORTS_DIR,
    SPRINT1_FIGURES_DIR,
    SPRINT2_REPORTS_DIR,
    SPRINT2_FIGURES_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
UCI_DATASET_ID = 602
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset"
TARGET_COLUMN = "Class"

RAW_DATASET_PATH = DATA_RAW_DIR / "dry_bean.csv"
RAW_METADATA_PATH = DATA_RAW_DIR / "dry_bean_metadata.json"

# ---------------------------------------------------------------------------
# Parámetros de experimentación (reproducibilidad TDSP)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Artefactos procesados del Sprint 1
X_TRAIN_PATH = DATA_PROCESSED_DIR / "X_train.csv"
X_TEST_PATH = DATA_PROCESSED_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_PROCESSED_DIR / "y_train.csv"
Y_TEST_PATH = DATA_PROCESSED_DIR / "y_test.csv"

# Artefactos del modelo final (Sprint 2 / Sprint 3)
FINAL_MODEL_PATH = OUTPUTS_MODELS_DIR / "modelo_final_drybean.joblib"
