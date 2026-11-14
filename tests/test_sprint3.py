"""Pruebas del Sprint 3: carga del artefacto, contrato de entrada y predicción.

El modelo pequeño se entrena con una muestra del dataset para que las pruebas
sean rápidas; la verificación del artefacto real se hace con
`python -m src.sprint3`.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config, data, deploy, models, prepare

REQUIRED_KEYS = {"selected_model", "model_type", "metrics", "features", "classes"}


@pytest.fixture(scope="module")
def small_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = data.load_raw_data()
    df, _ = prepare.clean_data(df)
    X, y = prepare.split_features_target(df)
    X_train, X_test, y_train, y_test = prepare.make_train_test_split(X, y)

    sample_index = X_train.sample(n=800, random_state=0).index
    return (
        X_train.loc[sample_index],
        X_test.head(200),
        y_train.loc[sample_index],
        y_test.head(200),
    )


@pytest.fixture(scope="module")
def fitted_hgb(small_split):
    X_train, _, y_train, _ = small_split
    model, _ = models.train_model(models.build_hgb_model(), X_train, y_train)
    return model


@pytest.fixture()
def saved_artifact(tmp_path, fitted_hgb):
    """Artefacto completo (joblib + metadatos) igual al del Sprint 2."""
    path = tmp_path / "modelo_prueba.joblib"
    metadata = {
        "selected_model": models.HGB_NAME,
        "model_type": type(fitted_hgb).__name__,
        "metrics": {"accuracy": 0.9, "f1_macro": 0.91},
        "features": list(fitted_hgb.feature_names_in_),
        "classes": [str(label) for label in fitted_hgb.classes_],
    }
    models.save_model(fitted_hgb, path, metadata=metadata)
    return path, metadata, fitted_hgb


@pytest.fixture()
def sample_grain(small_split) -> dict[str, float]:
    X_test = small_split[1]
    return {column: float(value) for column, value in X_test.iloc[0].items()}


# ---------------------------------------------------------------------------
# Carga del artefacto
# ---------------------------------------------------------------------------
def test_load_final_model_raises_when_missing(tmp_path) -> None:
    with pytest.raises(deploy.ModeloNoDisponibleError):
        deploy.load_final_model(tmp_path / "no_existe.joblib")


def test_load_final_model_roundtrip(saved_artifact, small_split) -> None:
    path, metadata, original = saved_artifact
    X_test = small_split[1]

    model, loaded_metadata = deploy.load_final_model(path)

    assert loaded_metadata["selected_model"] == models.HGB_NAME
    assert REQUIRED_KEYS <= set(loaded_metadata)
    assert {"artifact", "saved_at_utc", "sklearn_version"} <= set(loaded_metadata)
    assert [str(label) for label in model.classes_] == metadata["classes"]
    assert list(model.feature_names_in_) == metadata["features"]
    assert list(model.predict(X_test)) == list(original.predict(X_test))


def test_load_final_model_without_metadata(tmp_path, fitted_hgb) -> None:
    path = tmp_path / "solo_modelo.joblib"
    models.save_model(fitted_hgb, path)

    _, metadata = deploy.load_final_model(path)
    assert metadata["artifact"] == path.name
    assert "features" not in metadata


# ---------------------------------------------------------------------------
# Contrato de entrada
# ---------------------------------------------------------------------------
def test_to_feature_frame_accepts_dict_and_orders_columns(
    sample_grain, fitted_hgb
) -> None:
    features = list(fitted_hgb.feature_names_in_)
    shuffled = {name: sample_grain[name] for name in reversed(features)}

    frame = deploy.to_feature_frame(shuffled, features)

    assert frame.shape == (1, len(features))
    assert list(frame.columns) == features


def test_to_feature_frame_accepts_series_and_dataframe(sample_grain, fitted_hgb) -> None:
    features = list(fitted_hgb.feature_names_in_)

    from_series = deploy.to_feature_frame(pd.Series(sample_grain), features)
    from_frame = deploy.to_feature_frame(pd.DataFrame([sample_grain]), features)

    assert from_series.equals(from_frame)
    assert from_series.iloc[0][features].tolist() == [
        sample_grain[name] for name in features
    ]


def test_to_feature_frame_accepts_list_of_dicts(
    small_split, fitted_hgb
) -> None:
    features = list(fitted_hgb.feature_names_in_)
    rows = [
        {column: float(value) for column, value in row.items()}
        for row in small_split[1].head(3).to_dict(orient="records")
    ]

    frame = deploy.to_feature_frame(rows, features)
    assert frame.shape == (3, len(features))


def test_to_feature_frame_rejects_missing_columns(sample_grain, fitted_hgb) -> None:
    features = list(fitted_hgb.feature_names_in_)
    incomplete = dict(list(sample_grain.items())[:5])

    with pytest.raises(deploy.DatosInvalidosError, match="Faltan variables"):
        deploy.to_feature_frame(incomplete, features)


def test_to_feature_frame_rejects_unknown_columns(sample_grain, fitted_hgb) -> None:
    features = list(fitted_hgb.feature_names_in_)
    extra = {**sample_grain, "Color": 3}

    with pytest.raises(deploy.DatosInvalidosError, match="no reconocidas"):
        deploy.to_feature_frame(extra, features)


def test_to_feature_frame_rejects_nulls(sample_grain, fitted_hgb) -> None:
    features = list(fitted_hgb.feature_names_in_)
    with_nan = {**sample_grain, "Area": None}

    with pytest.raises(deploy.DatosInvalidosError, match="nulos"):
        deploy.to_feature_frame(with_nan, features)


def test_to_feature_frame_rejects_non_numeric(sample_grain, fitted_hgb) -> None:
    features = list(fitted_hgb.feature_names_in_)
    with_text = {**sample_grain, "Area": "grande"}

    with pytest.raises(deploy.DatosInvalidosError, match="numéricas"):
        deploy.to_feature_frame(with_text, features)


def test_to_feature_frame_rejects_empty_frame(fitted_hgb) -> None:
    features = list(fitted_hgb.feature_names_in_)

    with pytest.raises(deploy.DatosInvalidosError, match="filas"):
        deploy.to_feature_frame(pd.DataFrame(columns=features), features)


def test_to_feature_frame_rejects_empty_dict(fitted_hgb) -> None:
    with pytest.raises(deploy.DatosInvalidosError, match="variables"):
        deploy.to_feature_frame({}, list(fitted_hgb.feature_names_in_))


def test_to_feature_frame_rejects_unsupported_type() -> None:
    with pytest.raises(deploy.DatosInvalidosError, match="no soportada"):
        deploy.to_feature_frame(42)


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def test_predict_bean_class_returns_known_class(fitted_hgb, sample_grain) -> None:
    prediction = deploy.predict_bean_class(fitted_hgb, sample_grain)

    assert isinstance(prediction, str)
    assert prediction in {str(label) for label in fitted_hgb.classes_}


def test_predict_bean_class_accepts_dataframe_row(fitted_hgb, small_split) -> None:
    row = small_split[1].iloc[[0]]

    from_frame = deploy.predict_bean_class(fitted_hgb, row)
    from_dict = deploy.predict_bean_class(
        fitted_hgb, {column: float(value) for column, value in row.iloc[0].items()}
    )

    assert from_frame == from_dict


def test_predict_beans_returns_probabilities(fitted_hgb, small_split) -> None:
    rows = small_split[1].head(4)

    results = deploy.predict_beans(fitted_hgb, rows)

    assert len(results) == len(rows)
    for item in results:
        assert item["prediccion"] in {str(label) for label in fitted_hgb.classes_}
        assert 0.0 <= item["probabilidad"] <= 1.0
        assert item["clase_mas_probable"] == item["prediccion"]
        assert set(item) == {"prediccion", "probabilidad", "clase_mas_probable"}


def test_predictions_are_deterministic(fitted_hgb, small_split) -> None:
    rows = small_split[1].head(2)

    first = deploy.predict_beans(fitted_hgb, rows)
    second = deploy.predict_beans(fitted_hgb, rows)

    assert first == second


def test_predict_bean_class_uses_explicit_features(small_split) -> None:
    """Un modelo entrenado con NumPy no recuerda nombres: se pasan explícitos."""
    import warnings

    import numpy as np

    X_train, X_test, y_train, _ = small_split
    plain_model = models.build_hgb_model()
    plain_model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train))
    assert not hasattr(plain_model, "feature_names_in_")

    row = X_test.iloc[0]
    as_array = {column: float(value) for column, value in row.items()}

    # El modelo sin nombres avisa al recibir un DataFrame: es lo esperado aquí,
    # porque el orden de las columnas lo garantiza `to_feature_frame`.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        prediction = deploy.predict_bean_class(
            plain_model, as_array, features=list(X_test.columns)
        )
        expected = deploy.predict_bean_class(
            models.build_hgb_model().fit(X_train, y_train), as_array
        )

    assert prediction in {str(label) for label in plain_model.classes_}
    assert prediction == expected


# ---------------------------------------------------------------------------
# Verificación del artefacto real
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not config.FINAL_MODEL_PATH.exists(),
    reason="El modelo final se genera con `python -m src.sprint2`.",
)
def test_final_artifact_matches_metadata() -> None:
    model, metadata = deploy.load_final_model()

    assert metadata["selected_model"] == models.HGB_NAME
    assert len(metadata["features"]) == 16
    assert len(metadata["classes"]) == 7
    assert list(model.feature_names_in_) == list(metadata["features"])

    _, X_test, _, y_test = prepare.load_processed_data()
    from sklearn.metrics import accuracy_score, f1_score

    predictions = model.predict(X_test)
    assert accuracy_score(y_test, predictions) == pytest.approx(
        metadata["metrics"]["accuracy"], abs=1e-9
    )
    assert f1_score(y_test, predictions, average="macro") == pytest.approx(
        metadata["metrics"]["f1_macro"], abs=1e-9
    )


@pytest.mark.skipif(
    not (config.SPRINT3_REPORTS_DIR / "verificacion_modelo.json").exists(),
    reason="El reporte se genera con `python -m src.sprint3`.",
)
def test_sprint3_verification_report_is_consistent() -> None:
    payload = json.loads(
        (config.SPRINT3_REPORTS_DIR / "verificacion_modelo.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["verificacion_ok"] is True
    assert payload["accuracy_delta"] < 1e-6
    assert payload["f1_macro_delta"] < 1e-6
    assert payload["features_coinciden"] is True
    assert payload["clases_coinciden"] is True
