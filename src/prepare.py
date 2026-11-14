"""Calidad de datos, separación train/test y pipeline de preprocesamiento.

Corresponde a los pasos 5 a 8 del laboratorio (Sprint 1: PB-02 y preparación
del modelado). Todo se apoya en `config` para mantener la reproducibilidad.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


# ---------------------------------------------------------------------------
# PB-02: calidad de datos
# ---------------------------------------------------------------------------
def build_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Resume nulos, duplicados, tipos y distribución de clases."""
    missing_by_column = df.isna().sum()
    report: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(missing_by_column.sum()),
        "missing_by_column": {
            column: int(count)
            for column, count in missing_by_column.items()
            if count > 0
        },
        "duplicated_rows": int(df.duplicated().sum()),
        "numeric_columns": int(df.select_dtypes(include="number").shape[1]),
        "non_numeric_columns": [
            column
            for column in df.columns
            if not pd.api.types.is_numeric_dtype(df[column])
        ],
        "target_column": config.TARGET_COLUMN,
        "n_classes": int(df[config.TARGET_COLUMN].nunique()),
        "class_counts": {
            str(label): int(count)
            for label, count in df[config.TARGET_COLUMN]
            .value_counts()
            .sort_values(ascending=False)
            .items()
        },
    }
    total = report["rows"]
    report["class_balance_ratio"] = (
        round(max(report["class_counts"].values()) / min(report["class_counts"].values()), 2)
        if total
        else None
    )
    return report


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aplica la limpieza mínima documentada: elimina filas duplicadas.

    No se imputan nulos: si aparecieran, el laboratorio exige reportarlos y
    decidir explícitamente (no se hace de forma automática).
    """
    report = build_quality_report(df)
    rows_before = report["rows"]

    cleaned = df.drop_duplicates().reset_index(drop=True)

    report["rows_after_cleaning"] = int(cleaned.shape[0])
    report["rows_removed"] = int(rows_before - cleaned.shape[0])
    report["cleaning_actions"] = [
        "drop_duplicates(): se eliminan filas idénticas completas.",
        "index reindexado con reset_index(drop=True).",
    ]
    if report["missing_total"] > 0:
        report["cleaning_actions"].append(
            "ATENCION: existen valores nulos; se reportan y NO se imputan en el Sprint 1."
        )
    return cleaned, report


# ---------------------------------------------------------------------------
# Pasos 7 y 8: X/y y train/test split
# ---------------------------------------------------------------------------
def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables predictoras (X) y variable objetivo (y)."""
    X = df.drop(columns=config.TARGET_COLUMN).copy()
    y = df[config.TARGET_COLUMN].astype(str).copy()

    non_numeric = [
        column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])
    ]
    if non_numeric:
        raise TypeError(f"Variables no numéricas inesperadas en X: {non_numeric}")
    if X.isna().any().any():
        raise ValueError("X contiene valores nulos; requiere decisión documentada.")
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """División estratificada y reproducible (documentada en el reporte)."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ---------------------------------------------------------------------------
# Pipeline de preprocesamiento
# ---------------------------------------------------------------------------
def build_preprocessing_pipeline() -> Pipeline:
    """Pipeline de preprocesamiento reutilizable (entrada de los modelos).

    Se define aquí para que el baseline (Sprint 2) y los modelos alternativos
    usen exactamente el mismo escalamiento y no haya fuga de información.
    """
    return Pipeline(steps=[("scaler", StandardScaler())])


# ---------------------------------------------------------------------------
# Persistencia de artefactos procesados
# ---------------------------------------------------------------------------
def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, str]:
    """Guarda los conjuntos procesados para reutilizarlos en el Sprint 2."""
    X_train.to_csv(config.X_TRAIN_PATH, index=False)
    X_test.to_csv(config.X_TEST_PATH, index=False)
    y_train.to_csv(config.Y_TRAIN_PATH, index=False)
    y_test.to_csv(config.Y_TEST_PATH, index=False)
    return {
        "X_train": str(config.X_TRAIN_PATH),
        "X_test": str(config.X_TEST_PATH),
        "y_train": str(config.Y_TRAIN_PATH),
        "y_test": str(config.Y_TEST_PATH),
    }


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga los conjuntos procesados del Sprint 1."""
    X_train = pd.read_csv(config.X_TRAIN_PATH)
    X_test = pd.read_csv(config.X_TEST_PATH)
    y_train = pd.read_csv(config.Y_TRAIN_PATH)[config.TARGET_COLUMN].astype(str)
    y_test = pd.read_csv(config.Y_TEST_PATH)[config.TARGET_COLUMN].astype(str)
    return X_train, X_test, y_train, y_test


def describe_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Resumen del split para trazabilidad y verificación de estratificación."""
    total = len(X_train) + len(X_test)
    return {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_share": round(len(X_train) / total, 4),
        "test_share": round(len(X_test) / total, 4),
        "n_features": int(X_train.shape[1]),
        "features": list(X_train.columns),
        "train_class_share": {
            label: round(share, 4)
            for label, share in y_train.value_counts(normalize=True).sort_index().items()
        },
        "test_class_share": {
            label: round(share, 4)
            for label, share in y_test.value_counts(normalize=True).sort_index().items()
        },
        "max_class_share_diff": round(
            float(
                (
                    y_train.value_counts(normalize=True).sort_index()
                    - y_test.value_counts(normalize=True).sort_index()
                )
                .abs()
                .max()
            ),
            6,
        ),
        "preprocessing_pipeline": str(build_preprocessing_pipeline()),
        "scaled_features_check": {
            "mean": float(np.mean(build_preprocessing_pipeline().fit_transform(X_train).mean(axis=0))),
            "std": float(np.mean(build_preprocessing_pipeline().fit_transform(X_train).std(axis=0))),
        },
    }
