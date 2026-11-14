"""Ejecuta el Sprint 1 completo: carga, EDA, limpieza, split y pipeline.

Uso:
    python -m src.sprint1              # usa data/raw/dry_bean.csv si existe
    python -m src.sprint1 --refresh    # fuerza la descarga desde UCI

Genera:
    data/raw/dry_bean.csv
    data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
    outputs/reports/sprint1/reporte_sprint1.md
    outputs/reports/sprint1/calidad_datos.json
    outputs/reports/sprint1/resumen_sprint1.json
    outputs/reports/sprint1/figuras/*.png
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from src import config, data, eda, prepare


def _log(step: str, message: str) -> None:
    print(f"[{step}] {message}")


def run_sprint1(refresh: bool = False) -> dict[str, Any]:
    # ------------------------------------------------------------------ 1/6
    _log("1/6", "Cargando dataset (UCI id=602)...")
    df_raw = data.load_raw_data(force_download=refresh)
    _log("1/6", f"Dataset cargado: {df_raw.shape[0]} filas x {df_raw.shape[1]} columnas")

    # ------------------------------------------------------------------ 2/6
    _log("2/6", "Evaluando calidad y limpiando duplicados...")
    df, quality = prepare.clean_data(df_raw)
    _log(
        "2/6",
        f"Nulos: {quality['missing_total']} · Duplicados: {quality['duplicated_rows']} "
        f"· Registros finales: {quality['rows_after_cleaning']}",
    )

    # ------------------------------------------------------------------ 3/6
    _log("3/6", "Separando X/y y aplicando split estratificado...")
    X, y = prepare.split_features_target(df)
    X_train, X_test, y_train, y_test = prepare.make_train_test_split(X, y)
    split_summary = prepare.describe_split(X_train, X_test, y_train, y_test)
    _log(
        "3/6",
        f"Train: {split_summary['train_rows']} ({split_summary['train_share']:.1%}) "
        f"· Test: {split_summary['test_rows']} ({split_summary['test_share']:.1%}) "
        f"· Desvio max. de estratificacion: {split_summary['max_class_share_diff']:.4f}",
    )

    # ------------------------------------------------------------------ 4/6
    _log("4/6", "Validando pipeline de preprocesamiento (solo con train)...")
    pipeline = prepare.build_preprocessing_pipeline()
    X_train_scaled = pipeline.fit_transform(X_train)
    _log(
        "4/6",
        f"StandardScaler OK: salida {X_train_scaled.shape} · "
        f"media {X_train_scaled.mean():.2e} · std {X_train_scaled.std():.4f}",
    )
    split_summary["preprocessing_pipeline"] = str(pipeline)

    # ------------------------------------------------------------------ 5/6
    _log("5/6", "Guardando conjuntos procesados en data/processed/...")
    processed_paths = prepare.save_processed_data(X_train, X_test, y_train, y_test)
    for name, path in processed_paths.items():
        _log("5/6", f"{name}: {path}")

    # ------------------------------------------------------------------ 6/6
    _log("6/6", "Generando figuras y reporte del Sprint 1...")
    figures = eda.generate_figures(df)
    markdown = eda.build_eda_markdown(df, quality, split_summary, figures)
    report_path = eda.save_eda_report(
        markdown, config.SPRINT1_REPORTS_DIR / "reporte_sprint1.md"
    )

    if config.RAW_METADATA_PATH.exists():
        _log("6/6", f"Metadatos del dataset: {config.RAW_METADATA_PATH}")
    _log("6/6", f"Reporte: {report_path}")

    (config.SPRINT1_REPORTS_DIR / "calidad_datos.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    (config.SPRINT1_REPORTS_DIR / "resumen_sprint1.json").write_text(
        json.dumps(
            {
                "dataset": "Dry Bean Dataset (UCI 602)",
                "random_state": config.RANDOM_STATE,
                "test_size": config.TEST_SIZE,
                "quality": quality,
                "split": split_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    _log("6/6", f"Artefactos JSON: {config.SPRINT1_REPORTS_DIR}")

    return {
        "quality": quality,
        "split": split_summary,
        "report": str(report_path),
        "figures": {name: str(path) for name, path in figures.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 1 del laboratorio Dry Bean")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fuerza la descarga del dataset desde UCI (sobrescribe data/raw).",
    )
    args = parser.parse_args()

    result = run_sprint1(refresh=args.refresh)
    print("\nSprint 1 finalizado correctamente.")
    print(f"Registros para modelado: {result['split']['train_rows'] + result['split']['test_rows']}")
    print(f"Features: {result['split']['n_features']}")
    print(f"Reporte: {result['report']}")


if __name__ == "__main__":
    main()
