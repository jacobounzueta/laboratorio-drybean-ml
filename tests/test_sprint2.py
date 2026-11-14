"""Pruebas del Sprint 2: construcción, evaluación y persistencia de modelos.

Se usa una muestra pequeña del dataset para mantener las pruebas rápidas; la
validación completa se hace con `python -m src.sprint2`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import config, data, models, prepare

METRIC_KEYS = {
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "train_seconds",
    "predict_seconds",
}


@pytest.fixture(scope="module")
def small_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Muestra estratificada pequeña construida desde el dataset versionado."""
    df = data.load_raw_data()
    df, _ = prepare.clean_data(df)
    X, y = prepare.split_features_target(df)
    X_train, X_test, y_train, y_test = prepare.make_train_test_split(X, y)

    sample_index = X_train.sample(n=1500, random_state=0).index
    return (
        X_train.loc[sample_index],
        X_test.head(700),
        y_train.loc[sample_index],
        y_test.head(700),
    )


@pytest.fixture(scope="module")
def baseline_fitted(small_split):
    X_train, _, y_train, _ = small_split
    model, seconds = models.train_model(models.build_baseline_pipeline(), X_train, y_train)
    return model, seconds


@pytest.fixture(scope="module")
def forest_fitted(small_split):
    X_train, _, y_train, _ = small_split
    model, seconds = models.train_model(models.build_forest_model(), X_train, y_train)
    return model, seconds


# ---------------------------------------------------------------------------
# Construcción
# ---------------------------------------------------------------------------
def test_baseline_pipeline_structure() -> None:
    pipeline = models.build_baseline_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["scaler", "classifier"]
    assert pipeline.named_steps["classifier"].random_state == config.RANDOM_STATE
    assert pipeline.named_steps["classifier"].max_iter == 1000


def test_forest_configuration() -> None:
    forest = models.build_forest_model()
    assert forest.n_estimators == 200
    assert forest.class_weight == "balanced"
    assert forest.random_state == config.RANDOM_STATE


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def test_evaluate_model_metrics_and_matrix(small_split, baseline_fitted) -> None:
    _, X_test, _, y_test = small_split
    model, seconds = baseline_fitted

    result = models.evaluate_model(
        model, X_test, y_test, models.BASELINE_NAME, seconds
    )

    assert set(result["metrics"]) == METRIC_KEYS
    assert 0.8 <= result["metrics"]["accuracy"] <= 1.0
    assert 0.8 <= result["metrics"]["f1_macro"] <= 1.0
    assert result["metrics"]["train_seconds"] is not None

    matrix = np.asarray(result["confusion_matrix"])
    assert matrix.shape == (len(result["labels"]), len(result["labels"]))
    assert matrix.sum() == len(y_test)
    assert np.trace(matrix) == int(
        round(result["metrics"]["accuracy"] * len(y_test))
    )

    assert len(result["y_pred"]) == len(y_test)
    assert {"precision", "recall", "f1-score", "support"} <= set(result["per_class"].columns)


def test_baseline_beats_dummy_accuracy(small_split, baseline_fitted) -> None:
    _, X_test, _, y_test = small_split
    model, seconds = baseline_fitted
    result = models.evaluate_model(model, X_test, y_test, models.BASELINE_NAME, seconds)

    most_frequent = y_test.value_counts(normalize=True).max()
    assert result["metrics"]["accuracy"] > most_frequent


def test_comparison_table_has_both_models(small_split, baseline_fitted, forest_fitted) -> None:
    _, X_test, _, y_test = small_split
    baseline, baseline_seconds = baseline_fitted
    forest, forest_seconds = forest_fitted

    results = {
        "baseline": models.evaluate_model(
            baseline, X_test, y_test, models.BASELINE_NAME, baseline_seconds
        ),
        "forest": models.evaluate_model(
            forest, X_test, y_test, models.FOREST_NAME, forest_seconds
        ),
    }
    comparison = models.build_comparison_table(results)

    assert comparison.index.tolist() == ["baseline", "forest"]
    assert {"accuracy", "f1_macro", "balanced_accuracy"} <= set(comparison.columns)
    assert comparison["accuracy"].between(0, 1).all()


def test_select_best_model_prefers_simple_when_tied() -> None:
    comparison = pd.DataFrame(
        {
            "f1_macro": [0.9300, 0.9303],
            "train_seconds": [0.1, 0.9],
        },
        index=["baseline", "forest"],
    )
    selection = models.select_best_model(comparison, tolerance=0.005)

    assert selection["selected"] == "baseline"
    assert selection["f1_macro_gap"] == 0.0 or selection["f1_macro_gap"] < 0.005
    assert "mas simple" in selection["rationale"]


def test_select_best_model_picks_clear_winner() -> None:
    comparison = pd.DataFrame(
        {
            "f1_macro": [0.800, 0.950],
            "train_seconds": [0.1, 0.9],
        },
        index=["baseline", "forest"],
    )
    selection = models.select_best_model(comparison, tolerance=0.005)

    assert selection["selected"] == "forest"
    assert selection["f1_macro_gap"] == 0.15


# ---------------------------------------------------------------------------
# Importancia y persistencia
# ---------------------------------------------------------------------------
def test_feature_importance_is_sorted_and_normalized(small_split, forest_fitted) -> None:
    X_train = small_split[0]
    forest, _ = forest_fitted

    importance = models.feature_importance(forest, list(X_train.columns))

    assert len(importance) == X_train.shape[1]
    assert importance["importancia"].is_monotonic_decreasing
    assert importance["importancia"].sum() == pytest.approx(1.0, abs=1e-6)
    assert (importance["importancia"] >= 0).all()


def test_feature_importance_supports_linear_models(small_split, baseline_fitted) -> None:
    X_train = small_split[0]
    baseline, _ = baseline_fitted

    importance = models.feature_importance(baseline, list(X_train.columns))

    assert len(importance) == X_train.shape[1]
    assert importance["importancia"].sum() > 0


def test_save_and_load_model_roundtrip(tmp_path, small_split, baseline_fitted) -> None:
    _, X_test, _, _ = small_split
    model, _ = baseline_fitted
    path = tmp_path / "modelo.joblib"

    artifact = models.save_model(model, path, metadata={"prueba": True})

    assert path.exists()
    assert path.with_suffix(".metadata.json").exists()

    loaded = models.load_model(path)
    np.testing.assert_array_equal(loaded.predict(X_test), model.predict(X_test))
    assert artifact["model"].endswith("modelo.joblib")
