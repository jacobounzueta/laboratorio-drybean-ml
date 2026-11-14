"""Pruebas del Sprint 2: construcción, validación cruzada, evaluación y persistencia.

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


def test_hgb_configuration() -> None:
    hgb = models.build_hgb_model()
    assert hgb.random_state == config.RANDOM_STATE
    assert hgb.l2_regularization == 1.0
    assert hgb.learning_rate == 0.1


# ---------------------------------------------------------------------------
# Validación cruzada
# ---------------------------------------------------------------------------
def test_cross_validate_model_returns_fold_scores(small_split) -> None:
    X_train, _, y_train, _ = small_split
    result = models.cross_validate_model(
        models.build_baseline_pipeline(), X_train, y_train, cv=3
    )

    assert result["cv"] == 3
    assert len(result["scores"]) == 3
    assert 0.5 <= result["mean"] <= 1.0
    assert result["std"] >= 0
    assert result["seconds"] >= 0


def test_cross_validation_is_stable_for_reference_model(small_split) -> None:
    X_train, _, y_train, _ = small_split
    first = models.cross_validate_model(
        models.build_baseline_pipeline(), X_train, y_train, cv=3
    )
    second = models.cross_validate_model(
        models.build_baseline_pipeline(), X_train, y_train, cv=3
    )
    assert first["scores"] == second["scores"]


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


def test_comparison_table_merges_cross_validation(small_split, baseline_fitted) -> None:
    _, X_test, _, y_test = small_split
    baseline, seconds = baseline_fitted
    name = models.BASELINE_NAME

    results = {
        name: models.evaluate_model(baseline, X_test, y_test, name, seconds)
    }
    cv_results = {
        name: {
            "cv": 5,
            "scores": [0.9] * 5,
            "mean": 0.9,
            "std": 0.01,
            "seconds": 1.0,
        }
    }
    comparison = models.build_comparison_table(results, cv_results)

    assert comparison.index.tolist() == [name]
    assert {"accuracy", "f1_macro", "cv_f1_macro_mean", "cv_f1_macro_std"} <= set(
        comparison.columns
    )
    assert comparison.loc[name, "cv_f1_macro_mean"] == pytest.approx(0.9)


def test_select_best_model_uses_cross_validation_when_tied() -> None:
    comparison = pd.DataFrame(
        {
            "f1_macro": [0.9310, 0.9309],
            "cv_f1_macro_mean": [0.9360, 0.9362],
            "train_seconds": [0.1, 0.9],
        },
        index=["Logistic Regression", "Random Forest"],
    )
    selection = models.select_best_model(comparison, tolerance=0.005)

    assert selection["metric"] == "cv_f1_macro_mean"
    assert selection["selected"] == "Logistic Regression"
    assert "mas simple" in selection["rationale"]


def test_select_best_model_picks_clear_winner() -> None:
    comparison = pd.DataFrame(
        {
            "f1_macro": [0.92, 0.93],
            "cv_f1_macro_mean": [0.9360, 0.9429],
            "train_seconds": [0.1, 0.4],
        },
        index=["Logistic Regression", "HistGradientBoosting"],
    )
    selection = models.select_best_model(comparison, tolerance=0.005)

    assert selection["selected"] == "HistGradientBoosting"
    assert selection["gap"] == pytest.approx(0.0069)


def test_select_best_model_falls_back_to_test_metric() -> None:
    comparison = pd.DataFrame(
        {
            "f1_macro": [0.92, 0.93],
            "train_seconds": [0.1, 0.4],
        },
        index=["Logistic Regression", "Random Forest"],
    )
    selection = models.select_best_model(comparison)

    assert selection["metric"] == "f1_macro"
    assert selection["selected"] == "Random Forest"


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
