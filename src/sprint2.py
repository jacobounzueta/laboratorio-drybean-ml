"""Ejecuta el Sprint 2 completo: baseline, Random Forest, gradient boosting,
validación cruzada, comparación y artefacto final.

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
from typing import Any, Callable

from sklearn.base import BaseEstimator

from src import config, evaluate, models, prepare

MODEL_BUILDERS: dict[str, Callable[[], BaseEstimator]] = {
    models.BASELINE_NAME: models.build_baseline_pipeline,
    models.FOREST_NAME: models.build_forest_model,
    models.HGB_NAME: models.build_hgb_model,
}


def _log(step: str, message: str) -> None:
    print(f"[{step}] {message}")


def _pairwise_agreement(predictions: dict[str, Any]) -> dict[str, float]:
    names = list(predictions)
    agreements = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            agreements[f"{names[i]} vs {names[j]}"] = round(
                float((predictions[names[i]] == predictions[names[j]]).mean()), 4
            )
    return agreements


def run_sprint2() -> dict[str, Any]:
    # ------------------------------------------------------------------ 1/8
    _log("1/8", "Cargando conjuntos procesados del Sprint 1...")
    X_train, X_test, y_train, y_test = prepare.load_processed_data()
    _log(
        "1/8",
        f"Train: {X_train.shape} · Test: {X_test.shape} · Clases: {y_train.nunique()}",
    )

    # ------------------------------------------------------------------ 2/8
    _log("2/8", "Entrenando los tres modelos...")
    fitted: dict[str, BaseEstimator] = {}
    train_seconds: dict[str, float] = {}
    for name, builder in MODEL_BUILDERS.items():
        model, seconds = models.train_model(builder(), X_train, y_train)
        fitted[name] = model
        train_seconds[name] = seconds
        _log("2/8", f"{name}: entrenado en {seconds:.2f}s")

    # ------------------------------------------------------------------ 3/8
    _log("3/8", "Evaluando en el conjunto de prueba...")
    results: dict[str, dict[str, Any]] = {}
    for name in MODEL_BUILDERS:
        result = models.evaluate_model(
            fitted[name], X_test, y_test, name, train_seconds[name]
        )
        results[name] = result
        _log(
            "3/8",
            f"{name}: accuracy={result['metrics']['accuracy']:.4f} · "
            f"f1_macro={result['metrics']['f1_macro']:.4f}",
        )

    # ------------------------------------------------------------------ 4/8
    _log("4/8", f"Validando con {models.CV_FOLDS}-fold sobre train...")
    cv_results: dict[str, dict[str, Any]] = {}
    for name, builder in MODEL_BUILDERS.items():
        cv_result = models.cross_validate_model(builder(), X_train, y_train)
        cv_results[name] = cv_result
        _log(
            "4/8",
            f"{name}: F1 macro CV = {cv_result['mean']:.4f} "
            f"(±{cv_result['std']:.4f}) en {cv_result['seconds']:.1f}s",
        )

    # ------------------------------------------------------------------ 5/8
    _log("5/8", "Comparando modelos y aplicando la decisión CRISP-DM...")
    comparison = models.build_comparison_table(results, cv_results)
    selection = models.select_best_model(comparison)
    print(comparison.to_string())
    _log("5/8", f"Modelo seleccionado: {selection['selected']}")
    _log("5/8", selection["rationale"])

    # ------------------------------------------------------------------ 6/8
    _log("6/8", "Calculando importancia de variables y figuras...")
    importance = models.feature_importance(
        fitted[models.FOREST_NAME], list(X_train.columns)
    )
    figures = evaluate.generate_figures(comparison, results, importance)

    predictions = {name: result["y_pred"] for name, result in results.items()}
    pairwise = _pairwise_agreement(predictions)
    mean_agreement = sum(pairwise.values()) / len(pairwise)
    _log("6/8", f"Coincidencia media entre modelos: {mean_agreement:.2%}")
    for key, path in figures.items():
        _log("6/8", f"{key}: {path.name}")

    # ------------------------------------------------------------------ 7/8
    _log("7/8", "Guardando el modelo seleccionado...")
    selected_name = selection["selected"]
    best_result = results[selected_name]
    artifact = models.save_model(
        fitted[selected_name],
        config.FINAL_MODEL_PATH,
        metadata={
            "selected_model": selected_name,
            "selection_metric": selection["metric"],
            "decision_rationale": selection["rationale"],
            "metrics": {
                "accuracy": best_result["metrics"]["accuracy"],
                "f1_macro": best_result["metrics"]["f1_macro"],
                "cv_f1_macro_mean": cv_results[selected_name]["mean"],
                "cv_f1_macro_std": cv_results[selected_name]["std"],
            },
            "features": list(X_train.columns),
            "classes": sorted(y_train.unique().tolist()),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "model_type": type(fitted[selected_name]).__name__,
        },
    )
    _log("7/8", f"Modelo: {artifact['model']}")

    # ------------------------------------------------------------------ 8/8
    _log("8/8", "Guardando métricas, predicciones y reporte...")
    comparison.to_csv(
        config.SPRINT2_REPORTS_DIR / "comparacion_modelos.csv", lineterminator="\n"
    )

    predictions_df = X_test.copy()
    predictions_df["real_class"] = y_test.to_numpy()
    predictions_df["predicted_class"] = best_result["y_pred"]
    predictions_df["acierto"] = (
        predictions_df["real_class"] == predictions_df["predicted_class"]
    )
    predictions_df.to_csv(
        config.SPRINT2_REPORTS_DIR / "predicciones_test.csv",
        index=False,
        lineterminator="\n",
    )

    metrics_payload = {
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "cv_folds": models.CV_FOLDS,
        "models": {
            name: {
                "metrics": result["metrics"],
                "cross_validation": cv_results[name],
                "per_class": result["per_class"]
                .drop(index=["accuracy"], errors="ignore")
                .round(4)
                .to_dict(orient="index"),
                "top_confusions": evaluate.top_confusions(result),
            }
            for name, result in results.items()
        },
        "comparison": comparison.round(4).to_dict(orient="index"),
        "prediction_agreement": pairwise,
        "selection": selection,
        "feature_importance": {
            "kind": importance.attrs.get("kind"),
            "source_model": models.FOREST_NAME,
            "top_10": importance.head(10).round(4).to_dict(orient="records"),
        },
        "artifacts": artifact,
    }
    (config.SPRINT2_REPORTS_DIR / "metricas_sprint2.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    markdown = evaluate.build_sprint2_markdown(
        comparison,
        results,
        cv_results,
        selection,
        importance,
        figures,
        artifact,
        mean_agreement,
    )
    report_path = evaluate.save_report(
        markdown, config.SPRINT2_REPORTS_DIR / "reporte_sprint2.md"
    )
    _log("8/8", f"Reporte: {report_path}")

    return {
        "comparison": comparison,
        "cv_results": cv_results,
        "selection": selection,
        "report": str(report_path),
        "artifact": artifact,
        "agreement": pairwise,
    }


def main() -> None:
    result = run_sprint2()
    print("\nSprint 2 finalizado correctamente.")
    print(f"Modelo seleccionado: {result['selection']['selected']}")
    print(f"Reporte: {result['report']}")


if __name__ == "__main__":
    main()
