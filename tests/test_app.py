"""Pruebas de humo para la aplicación Streamlit (`app.py`)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import app

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_and_predicts() -> None:
    """La app arranca, carga el modelo y ejecuta una predicción individual."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()

    assert not at.exception
    assert len(at.title) == 1
    assert len(at.tabs) == 2

    # Enviar el formulario de predicción individual con la muestra por defecto
    assert len(at.button) >= 1
    at.button[0].click().run()

    assert not at.exception
    # Debe mostrar la métrica de variedad predicha además de las métricas del sidebar
    metric_labels = [m.label for m in at.metric]
    assert "Variedad predicha" in metric_labels


def test_build_class_presets_fallback() -> None:
    """Cuando no hay y_test, devuelve el ejemplo por defecto."""
    df = pd.DataFrame([app.DEFAULT_SAMPLE])
    presets = app.build_class_presets(df, None)
    assert "Ejemplo por defecto" in presets
    assert presets["Ejemplo por defecto"]["true_class"] is None


def test_build_bean_simulation_figure() -> None:
    """La simulación geométrica genera una figura válida con los parámetros."""
    fig = app.build_bean_simulation_figure(app.DEFAULT_SAMPLE)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].patches) == 3  # caja envolvente, círculo equiv. y contorno
