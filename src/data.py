"""Carga del Dry Bean Dataset desde UCI con caché local (Sprint 1, PB-01).

Reglas del proyecto:
* `data/raw/dry_bean.csv` es la copia inmutable del dataset original.
* Si el archivo existe se lee del disco (no se vuelve a descargar).
* Toda descarga deja metadatos de trazabilidad en
  `data/raw/dry_bean_metadata.json` (fecha, fuente, hash, dimensiones).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src import config


def _sha256(path) -> str:
    """Hash del archivo para verificar que el dataset no cambió."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_dry_bean_from_uci() -> pd.DataFrame:
    """Descarga el dataset oficial (id=602) y devuelve features + objetivo."""
    dataset = fetch_ucirepo(id=config.UCI_DATASET_ID)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    if targets.shape[1] != 1:
        raise ValueError(
            f"Se esperaba una sola columna objetivo, se recibieron {targets.shape[1]}."
        )

    targets.columns = [config.TARGET_COLUMN]
    df = pd.concat([features, targets], axis=1)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    """Devuelve el dataframe original, descargándolo solo si hace falta."""
    if config.RAW_DATASET_PATH.exists() and not force_download:
        df = pd.read_csv(config.RAW_DATASET_PATH)
        df.columns = [str(column).strip() for column in df.columns]
        return df

    df = fetch_dry_bean_from_uci()
    df.to_csv(config.RAW_DATASET_PATH, index=False, lineterminator="\n")

    metadata = {
        "dataset": "Dry Bean Dataset",
        "uci_id": config.UCI_DATASET_ID,
        "source": config.UCI_DATASET_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": config.TARGET_COLUMN,
        "classes": sorted(df[config.TARGET_COLUMN].astype(str).unique().tolist()),
        "sha256": _sha256(config.RAW_DATASET_PATH),
        "sha256_scope": "contenido del archivo versionado en Git (finales de linea LF)",
    }
    config.RAW_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    return df
