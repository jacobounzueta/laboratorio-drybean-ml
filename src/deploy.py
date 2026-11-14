"""Despliegue del modelo: carga del artefacto y predicción reutilizable.

Implementa los pasos 16 a 18 del laboratorio: el modelo entrenado en el Sprint 2
se guarda con `joblib`, se vuelve a cargar y se expone mediante una función que
puede usarse con datos nuevos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.base import BaseEstimator

from src import config, models


class ModeloNoDisponibleError(RuntimeError):
    """Se lanza cuando el artefacto del modelo no existe en disco."""


class DatosInvalidosError(ValueError):
    """Se lanza cuando los datos de entrada no cumplen el contrato del modelo."""


def load_final_model(
    path: Path | None = None,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Carga el modelo final y sus metadatos de trazabilidad."""
    model_path = Path(path) if path is not None else config.FINAL_MODEL_PATH
    if not model_path.exists():
        raise ModeloNoDisponibleError(
            f"No se encontró el modelo en {model_path}. "
            "Ejecuta `python -m src.sprint2` para generarlo."
        )

    metadata_path = model_path.with_suffix(".metadata.json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return models.load_model(model_path), metadata


def _expected_features(metadata: Mapping[str, Any] | None) -> list[str] | None:
    features = (metadata or {}).get("features")
    return list(features) if features else None


def to_feature_frame(
    input_data: Any,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Normaliza la entrada del usuario a un dataframe con el contrato del modelo.

    Acepta un diccionario, una serie, un dataframe o una lista de diccionarios.
    """
    try:
        if isinstance(input_data, Mapping):
            frame = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.Series):
            frame = input_data.to_frame().T
        elif isinstance(input_data, pd.DataFrame):
            frame = input_data.copy()
        elif isinstance(input_data, (list, tuple)):
            frame = pd.DataFrame(list(input_data))
        else:
            raise DatosInvalidosError(
                "Entrada no soportada. Usa dict, pandas.Series, pandas.DataFrame "
                "o lista de diccionarios."
            )
    except DatosInvalidosError:
        raise
    except (TypeError, ValueError) as error:
        raise DatosInvalidosError(
            f"No se pudo interpretar la entrada: {error}"
        ) from error

    if frame.shape[1] == 0:
        raise DatosInvalidosError(
            "La entrada no contiene variables. Envía las medidas geométricas del grano."
        )

    if frame.empty:
        raise DatosInvalidosError("La entrada no contiene filas para predecir.")

    if features:
        missing = [column for column in features if column not in frame.columns]
        extra = [column for column in frame.columns if column not in features]
        if missing:
            raise DatosInvalidosError(
                f"Faltan variables requeridas: {missing}. "
                f"Se esperaban {len(features)} variables geométricas."
            )
        if extra:
            raise DatosInvalidosError(
                f"Hay variables no reconocidas: {extra}. Revisa los nombres."
            )
        frame = frame[features]

    if frame.isna().any().any():
        raise DatosInvalidosError("La entrada contiene valores nulos.")

    non_numeric = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise DatosInvalidosError(
            f"Las variables deben ser numéricas. Revisa: {non_numeric}"
        )
    return frame.reset_index(drop=True)


def _model_features(model: BaseEstimator, features: list[str] | None) -> list[str] | None:
    """Contrato de variables: el explícito o el que recuerda el modelo."""
    if features:
        return list(features)
    in_model = getattr(model, "feature_names_in_", None)
    return [str(name) for name in in_model] if in_model is not None else None


def predict_bean_class(
    model: BaseEstimator,
    input_data: Any,
    features: list[str] | None = None,
) -> str:
    """Predice la variedad de un solo grano (paso 18 del laboratorio).

    Acepta un diccionario con las 16 medidas geométricas o una fila de dataframe.
    Si el modelo no recuerda los nombres de sus variables (por ejemplo, si se
    entrenó con un arreglo de NumPy), pasa `features` con el orden correcto.
    """
    frame = to_feature_frame(input_data, _model_features(model, features))
    prediction = model.predict(frame)
    return str(prediction[0])


def predict_beans(
    model: BaseEstimator,
    input_data: Any,
    features: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Predice uno o varios granos e incluye la probabilidad de la clase elegida."""
    frame = to_feature_frame(input_data, _model_features(model, features))

    predictions = model.predict(frame)
    probabilities = (
        model.predict_proba(frame) if hasattr(model, "predict_proba") else None
    )
    classes = [str(label) for label in getattr(model, "classes_", [])]

    results = []
    for index, prediction in enumerate(predictions):
        record = {"prediccion": str(prediction)}
        if probabilities is not None and classes:
            proba = probabilities[index]
            chosen = classes.index(str(prediction))
            record["probabilidad"] = round(float(proba[chosen]), 4)
            record["clase_mas_probable"] = classes[int(proba.argmax())]
        results.append(record)
    return results
