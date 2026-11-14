"""Pruebas del Sprint 1: carga, calidad, split y pipeline de preprocesamiento.

Ejecutar con:
    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data, prepare

EXPECTED_ROWS = 13_611
EXPECTED_FEATURES = 16
EXPECTED_CLASSES = 7
EXPECTED_DUPLICATES = 68


@pytest.fixture(scope="module")
def df_raw() -> pd.DataFrame:
    """Dataset original (usa la copia local de data/raw si ya existe)."""
    return data.load_raw_data()


@pytest.fixture(scope="module")
def cleaned(df_raw: pd.DataFrame) -> pd.DataFrame:
    df, _ = prepare.clean_data(df_raw)
    return df


def test_raw_dataset_shape_and_target(df_raw: pd.DataFrame) -> None:
    assert df_raw.shape == (EXPECTED_ROWS, EXPECTED_FEATURES + 1)
    assert config.TARGET_COLUMN in df_raw.columns
    assert df_raw[config.TARGET_COLUMN].nunique() == EXPECTED_CLASSES
    assert df_raw.select_dtypes(include="number").shape[1] == EXPECTED_FEATURES
    assert not df_raw.isna().any().any()


def test_raw_metadata_tracks_download(df_raw: pd.DataFrame) -> None:
    import json

    assert config.RAW_METADATA_PATH.exists(), "Faltan los metadatos de trazabilidad."
    metadata = json.loads(config.RAW_METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["uci_id"] == config.UCI_DATASET_ID
    assert metadata["rows"] == EXPECTED_ROWS
    assert metadata["columns"] == EXPECTED_FEATURES + 1
    assert metadata["target_column"] == config.TARGET_COLUMN
    assert len(metadata["classes"]) == EXPECTED_CLASSES
    assert len(metadata["sha256"]) == 64


def test_raw_metadata_hash_matches_versioned_file() -> None:
    """El hash de los metadatos debe corresponder al archivo que está en Git."""
    import hashlib
    import json

    metadata = json.loads(config.RAW_METADATA_PATH.read_text(encoding="utf-8"))
    actual = hashlib.sha256(config.RAW_DATASET_PATH.read_bytes()).hexdigest()
    assert actual == metadata["sha256"], (
        "El CSV versionado no coincide con el hash de los metadatos."
    )


def test_clean_data_drops_duplicates(df_raw: pd.DataFrame) -> None:
    cleaned, report = prepare.clean_data(df_raw)

    assert report["duplicated_rows"] == EXPECTED_DUPLICATES
    assert report["rows_removed"] == EXPECTED_DUPLICATES
    assert cleaned.duplicated().sum() == 0
    assert len(cleaned) == EXPECTED_ROWS - EXPECTED_DUPLICATES
    assert report["missing_total"] == 0


def test_split_is_stratified_and_disjoint(cleaned: pd.DataFrame) -> None:
    X, y = prepare.split_features_target(cleaned)
    X_train, X_test, y_train, y_test = prepare.make_train_test_split(X, y)

    assert X_train.shape[1] == EXPECTED_FEATURES
    assert len(X_train) + len(X_test) == len(cleaned)
    assert len(X_test) == pytest.approx(len(cleaned) * config.TEST_SIZE, abs=2)
    assert len(set(X_train.index) & set(X_test.index)) == 0
    assert set(y_train.unique()) == set(y.unique())

    train_share = y_train.value_counts(normalize=True).sort_index()
    test_share = y_test.value_counts(normalize=True).sort_index()
    assert (train_share - test_share).abs().max() < 0.01


def test_split_is_reproducible(cleaned: pd.DataFrame) -> None:
    X, y = prepare.split_features_target(cleaned)
    first = prepare.make_train_test_split(X, y)
    second = prepare.make_train_test_split(X, y)

    assert first[0].equals(second[0])
    assert first[1].equals(second[1])
    assert first[2].equals(second[2])
    assert first[3].equals(second[3])


def test_preprocessing_pipeline_standardizes(cleaned: pd.DataFrame) -> None:
    X, y = prepare.split_features_target(cleaned)
    X_train, X_test, _, _ = prepare.make_train_test_split(X, y)

    pipeline = prepare.build_preprocessing_pipeline()
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert not np.isnan(X_train_scaled).any()
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-9)


def test_save_and_load_processed_data_roundtrip(
    cleaned: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    X, y = prepare.split_features_target(cleaned)
    X_train, X_test, y_train, y_test = prepare.make_train_test_split(X, y)

    monkeypatch.setattr(config, "X_TRAIN_PATH", tmp_path / "X_train.csv")
    monkeypatch.setattr(config, "X_TEST_PATH", tmp_path / "X_test.csv")
    monkeypatch.setattr(config, "Y_TRAIN_PATH", tmp_path / "y_train.csv")
    monkeypatch.setattr(config, "Y_TEST_PATH", tmp_path / "y_test.csv")

    prepare.save_processed_data(X_train, X_test, y_train, y_test)
    X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = (
        prepare.load_processed_data()
    )

    assert X_train_loaded.shape == X_train.shape
    assert X_test_loaded.shape == X_test.shape
    assert y_train_loaded.tolist() == y_train.tolist()
    assert y_test_loaded.tolist() == y_test.tolist()
