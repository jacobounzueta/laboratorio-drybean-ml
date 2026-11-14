"""Construcción, entrenamiento y evaluación de modelos (Sprint 2: PB-03 y PB-04).

El baseline usa `StandardScaler` + `LogisticRegression`; el modelo alternativo
es un `RandomForestClassifier` con `class_weight="balanced"` para compensar el
desbalance de clases detectado en el Sprint 1.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config

BASELINE_NAME = "Logistic Regression"
FOREST_NAME = "Random Forest"


# ---------------------------------------------------------------------------
# Construcción de modelos
# ---------------------------------------------------------------------------
def build_baseline_pipeline() -> Pipeline:
    """Baseline: escalamiento + regresión logística multiclase."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


def build_forest_model() -> RandomForestClassifier:
    """Modelo alternativo: bosque aleatorio balanceado.

    Random Forest no necesita escalamiento, por lo que se entrena con las
    variables originales; así se compara de forma justa contra el baseline.
    """
    return RandomForestClassifier(
        n_estimators=200,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------
def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[BaseEstimator, float]:
    """Ajusta el modelo y devuelve el tiempo de entrenamiento en segundos."""
    start = perf_counter()
    model.fit(X_train, y_train)
    return model, perf_counter() - start


def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    train_seconds: float | None = None,
) -> dict[str, Any]:
    """Calcula métricas globales, por clase y matriz de confusión."""
    start = perf_counter()
    y_pred = model.predict(X_test)
    predict_seconds = perf_counter() - start

    labels = sorted(y_test.unique().tolist())
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "train_seconds": None if train_seconds is None else round(train_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
    }

    per_class = pd.DataFrame(
        classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
    ).T
    per_class.index.name = "clase"

    return {
        "model_name": model_name,
        "metrics": metrics,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
        "y_pred": y_pred,
    }


def build_comparison_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Tabla comparativa con las métricas clave de cada modelo."""
    columns = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "balanced_accuracy": "balanced_accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "train_seconds": "train_seconds",
    }
    rows = {
        name: {column: result["metrics"][key] for column, key in columns.items()}
        for name, result in results.items()
    }
    comparison = pd.DataFrame.from_dict(rows, orient="index")
    comparison.index.name = "modelo"
    return comparison.round(4)


def select_best_model(
    comparison: pd.DataFrame, tolerance: float = 0.005
) -> dict[str, Any]:
    """Decisión CRISP-DM: priorizar F1 macro y preferir el modelo más simple.

    Si la diferencia de F1 macro es menor que `tolerance`, se elige el modelo
    con menor tiempo de entrenamiento (normalmente el baseline).
    """
    ranking = comparison.sort_values("f1_macro", ascending=False)
    best_name = ranking.index[0]
    runner_up = ranking.index[1]
    gap = float(ranking.loc[best_name, "f1_macro"] - ranking.loc[runner_up, "f1_macro"])

    if gap < tolerance:
        chosen = ranking.loc[[best_name, runner_up]].sort_values("train_seconds").index[0]
        rationale = (
            f"La diferencia de F1 macro entre {best_name} y {runner_up} es {gap:.4f} "
            f"(< {tolerance}), por lo que se prefiere el modelo mas simple y rapido: {chosen}."
        )
    else:
        chosen = best_name
        rationale = (
            f"{best_name} supera a {runner_up} por {gap:.4f} de F1 macro, "
            "por lo que pasa a la fase de despliegue."
        )

    return {
        "selected": chosen,
        "f1_macro_gap": round(gap, 4),
        "tolerance": tolerance,
        "rationale": rationale,
        "ranking": ranking["f1_macro"].round(4).to_dict(),
    }


# ---------------------------------------------------------------------------
# Importancia de variables
# ---------------------------------------------------------------------------
def feature_importance(model: BaseEstimator, feature_names: list[str]) -> pd.DataFrame:
    """Importancia de variables del Random Forest (o coeficientes del baseline)."""
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model[-1]

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "impureza (Gini)"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).mean(axis=0)
        kind = "coeficiente medio absoluto"
    else:
        raise TypeError(f"El modelo {type(estimator)!r} no expone importancia de variables.")

    table = pd.DataFrame({"variable": feature_names, "importancia": values})
    table = table.sort_values("importancia", ascending=False).reset_index(drop=True)
    table.attrs["kind"] = kind
    table.attrs["total"] = float(values.sum())
    return table


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def save_model(
    model: BaseEstimator,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Guarda el modelo con joblib y, si se entrega, sus metadatos en JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    metadata_path = path.with_suffix(".metadata.json")
    payload = {
        "artifact": path.name,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_state": config.RANDOM_STATE,
        **(metadata or {}),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"model": str(path), "metadata": str(metadata_path)}


def load_model(path: Path) -> BaseEstimator:
    """Carga un modelo guardado con joblib."""
    return joblib.load(path)
