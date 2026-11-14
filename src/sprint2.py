"""Ejecuta el Sprint 2 completo: baseline, Random Forest, comparación y artefacto.

Uso:
    python -m src.sprint2

Requiere los artefactos del Sprint 1 en `data/processed/` (los genera
`python -m src.sprint1`).

Genera:
    outputs/models/modelo_final_drybean.joblib (+ metadatos JSON)
    outputs/reports/sprint2/reporte_sprint2.md
    outputs/reports/sprint2/comparacion_modelos.csv
    outputs/reports/sprint2/metricas_sprint2.json
    outputs/reports/sprint2/predicciones_test.csv
    outputs/reports/sprint2/figuras/*.png
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src import config, evaluate, models, prepare


def _log(step: str, message: str) -> None:
    print(f"[{step}] {message}")


def run_sprint2() -> dict[str, Any]:
    # ------------------------------------------------------------------ 1/6
    _log("1/6", "Cargando conjuntos procesados del Sprint 1...")
    X_train, X_test, y_train, y_test = prepare.load_processed_data()
    _log(
        "1/6",
        f"Train: {X_train.shape} · Test: {X_test.shape} · Clases: {y_train.nunique()}",
    )

    # ------------------------------------------------------------------ 2/6
    _log("2/6", "Entrenando baseline (Logistic Regression + StandardScaler)...")
    baseline, baseline_seconds = models.train_model(
        models.build_baseline_pipeline(), X_train, y_train
    )
    baseline_result = models.evaluate_model(
        baseline, X_test, y_test, models.BASELINE_NAME, baseline_seconds
    )
    _log(
        "2/6",
        f"accuracy={baseline_result['metrics']['accuracy']:.4f} · "
        f"f1_macro={baseline_result['metrics']['f1_macro']:.4f} · "
        f"tiempo={baseline_seconds:.2f}s",
    )

    # ------------------------------------------------------------------ 3/6
    _log("3/6", "Entrenando Random Forest balanceado (200 árboles)...")
    forest, forest_seconds = models.train_model(
        models.build_forest_model(), X_train, y_train
    )
    forest_result = models.evaluate_model(
        forest, X_test, y_test, models.FOREST_NAME, forest_seconds
    )
    _log(
        "3/6",
        f"accuracy={forest_result['metrics']['accuracy']:.4f} · "
        f"f1_macro={forest_result['metrics']['f1_macro']:.4f} · "
        f"tiempo={forest_seconds:.2f}s",
    )

    # ------------------------------------------------------------------ 4/6
    _log("4/6", "Comparando modelos y aplicando la decisión CRISP-DM...")
    results = {"baseline": baseline_result, "forest": forest_result}
    comparison = models.build_comparison_table(results)
    selection = models.select_best_model(comparison)
    print(comparison.to_string())
    _log("4/6", f"Modelo seleccionado: {selection['selected']}")
    _log("4/6", selection["rationale"])

    # ------------------------------------------------------------------ 5/6
    _log("5/6", "Calculando importancia de variables y figuras...")
    importance = models.feature_importance(forest, list(X_train.columns))
    figures = evaluate.generate_figures(comparison, results, importance)
    agreement = float((baseline_result["y_pred"] == forest_result["y_pred"]).mean())
    _log("5/6", f"Coincidencia entre modelos: {agreement:.2%} de las predicciones")
    for key, path in figures.items():
        _log("5/6", f"{key}: {path.name}")

    # ------------------------------------------------------------------ 6/6
    _log("6/6", "Guardando modelo, métricas y reporte...")
    best_key = "baseline" if selection["selected"] == models.BASELINE_NAME else "forest"
    best_model = baseline if best_key == "baseline" else forest
    best_metrics = results[best_key]["metrics"]
    artifact = models.save_model(
        best_model,
        config.FINAL_MODEL_PATH,
        metadata={
            "selected_model": selection["selected"],
            "decision_rationale": selection["rationale"],
            "metrics": {
                "accuracy": best_metrics["accuracy"],
                "f1_macro": best_metrics["f1_macro"],
            },
            "features": list(X_train.columns),
            "classes": sorted(y_train.unique().tolist()),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
        },
    )

    comparison.to_csv(config.SPRINT2_REPORTS_DIR / "comparacion_modelos.csv")
    predictions = X_test.copy()
    predictions["real_class"] = y_test.to_numpy()
    predictions["predicted_class"] = forest_result["y_pred"]
    predictions.to_csv(
        config.SPRINT2_REPORTS_DIR / "predicciones_test.csv", index=False
    )

    metrics_payload = {
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "prediction_agreement": round(agreement, 4),
        "models": {
            name: {
                "metrics": result["metrics"],
                "per_class": result["per_class"]
                .drop(index=["accuracy"], errors="ignore")
                .round(4)
                .to_dict(orient="index"),
                "top_confusions": evaluate.top_confusions(result),
            }
            for name, result in results.items()
        },
        "comparison": comparison.round(4).to_dict(orient="index"),
        "selection": selection,
        "feature_importance": {
            "kind": importance.attrs.get("kind"),
            "top_10": importance.head(10).round(4).to_dict(orient="records"),
        },
        "artifacts": artifact,
    }
    (config.SPRINT2_REPORTS_DIR / "metricas_sprint2.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    markdown = evaluate.build_sprint2_markdown(
        comparison, results, selection, importance, figures, artifact, agreement
    )
    report_path = evaluate.save_report(
        markdown, config.SPRINT2_REPORTS_DIR / "reporte_sprint2.md"
    )
    _log("6/6", f"Reporte: {report_path}")
    _log("6/6", f"Modelo: {artifact['model']}")

    return {
        "comparison": comparison,
        "selection": selection,
        "report": str(report_path),
        "artifact": artifact,
    }


def main() -> None:
    result = run_sprint2()
    print("\nSprint 2 finalizado correctamente.")
    print(f"Modelo seleccionado: {result['selection']['selected']}")
    print(f"Reporte: {result['report']}")


if __name__ == "__main__":
    main()
