"""Figuras y reporte del Sprint 2: comparación de modelos y evaluación.

Genera las gráficas en `outputs/reports/sprint2/figuras/` y un reporte en
Markdown con la fase de evaluación de CRISP-DM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src import config
from src.eda import dataframe_to_markdown

_FIGURE_TITLES = {
    "comparacion": "Comparación de métricas",
    "confusion_baseline": "Matriz de confusión - Logistic Regression",
    "confusion_forest": "Matriz de confusión - Random Forest",
    "importancia": "Top 10 variables más importantes",
}


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> Path:
    metrics = ["accuracy", "f1_macro", "balanced_accuracy"]
    labels = [metric.replace("_", " ").title() for metric in metrics]
    models = comparison.index.tolist()

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#3b6ea5", "#d98d3b", "#5a9367", "#a35b8f"]
    for index, model in enumerate(models):
        values = comparison.loc[model, metrics].astype(float).tolist()
        offsets = x - 0.4 + width / 2 + index * width
        bars = ax.bar(offsets, values, width, label=model, color=colors[index % len(colors)])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Puntaje")
    ax.set_title("Comparación de modelos - conjunto de prueba")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(
    result: dict[str, Any], path: Path, normalize: bool = False
) -> Path:
    matrix = np.asarray(result["confusion_matrix"], dtype=float)
    if normalize:
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        values_format = ".2f"
    else:
        matrix = matrix.astype(int)
        values_format = "d"

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=result["labels"],
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format=values_format,
    )
    title = f"Matriz de confusión - {result['model_name']}"
    if normalize:
        title += " (normalizada por clase real)"
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(
    importance: pd.DataFrame, path: Path, top_n: int = 10
) -> Path:
    top = importance.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top["variable"], top["importancia"], color="#3b6ea5")
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xlabel(f"Importancia ({importance.attrs.get('kind', 'modelo')})")
    ax.set_title(f"Top {top_n} variables más importantes - Random Forest")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_figures(
    comparison: pd.DataFrame,
    results: dict[str, dict[str, Any]],
    importance: pd.DataFrame,
) -> dict[str, Path]:
    """Genera todas las figuras del Sprint 2 y devuelve sus rutas."""
    figures_dir = config.SPRINT2_FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "comparacion": plot_model_comparison(
            comparison, figures_dir / "comparacion_modelos.png"
        ),
        "confusion_baseline": plot_confusion(
            results["baseline"], figures_dir / "confusion_baseline.png"
        ),
        "confusion_forest": plot_confusion(
            results["forest"], figures_dir / "confusion_forest.png"
        ),
        "importancia": plot_feature_importance(
            importance, figures_dir / "importancia_variables.png"
        ),
    }


# ---------------------------------------------------------------------------
# Análisis de errores
# ---------------------------------------------------------------------------
def top_confusions(
    result: dict[str, Any], n: int = 3
) -> list[dict[str, Any]]:
    """Pares de clases que más se confunden (fuera de la diagonal)."""
    matrix = np.array(result["confusion_matrix"], dtype=int)
    labels = result["labels"]
    pairs = []
    for i, real in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i == j:
                continue
            count = int(matrix[i, j])
            if count > 0:
                pairs.append(
                    {
                        "real": real,
                        "predicho": predicted,
                        "casos": count,
                        "tasa": round(count / int(matrix[i].sum()), 4),
                    }
                )
    return sorted(pairs, key=lambda item: item["casos"], reverse=True)[:n]


def worst_classes(result: dict[str, Any], n: int = 2) -> list[dict[str, Any]]:
    """Clases con menor F1 en el conjunto de prueba."""
    per_class = result["per_class"]
    classes = per_class.loc[result["labels"], ["precision", "recall", "f1-score", "support"]]
    classes = classes.sort_values("f1-score").head(n)
    return [
        {
            "clase": label,
            "precision": round(float(row["precision"]), 4),
            "recall": round(float(row["recall"]), 4),
            "f1": round(float(row["f1-score"]), 4),
            "soporte": int(row["support"]),
        }
        for label, row in classes.iterrows()
    ]


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
def _per_class_table(result: dict[str, Any]) -> str:
    per_class = result["per_class"]
    rows = result["labels"] + ["macro avg"]
    table = per_class.loc[rows, ["precision", "recall", "f1-score", "support"]].copy()
    table["support"] = table["support"].astype(int)
    table.index.name = "clase"
    return dataframe_to_markdown(table)


def _relative(path: str | Path) -> str:
    """Ruta relativa al repositorio (para reportes portables)."""
    try:
        return Path(path).resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def build_sprint2_markdown(
    comparison: pd.DataFrame,
    results: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    importance: pd.DataFrame,
    figures: dict[str, Path],
    artifact: dict[str, str] | None,
    agreement: float | None = None,
) -> str:
    comparison_table = comparison.copy()
    comparison_table.index.name = "modelo"

    figures_section = "\n".join(
        f"### {_FIGURE_TITLES.get(key, key)}\n\n"
        f"![{_FIGURE_TITLES.get(key, key)}](figuras/{path.name})\n"
        for key, path in figures.items()
    )

    confusions = {
        name: top_confusions(result) for name, result in results.items()
    }
    confusion_lines = []
    for name, pairs in confusions.items():
        label = "Baseline" if name == "baseline" else "Random Forest"
        detail = ", ".join(
            f"{pair['real']} → {pair['predicho']} ({pair['casos']} casos, "
            f"{pair['tasa']:.1%})"
            for pair in pairs
        )
        confusion_lines.append(f"- **{label}:** {detail}")

    worst = worst_classes(results["forest"])
    worst_lines = "\n".join(
        f"- `{item['clase']}`: F1 = {item['f1']} · recall = {item['recall']} · "
        f"soporte = {item['soporte']}"
        for item in worst
    )

    forest_per_class = results["forest"]["per_class"]
    support = forest_per_class.loc[results["forest"]["labels"], "support"]
    minority = support.sort_values().index[:2]
    minority_lines = " · ".join(
        f"`{label}`: recall = {forest_per_class.loc[label, 'recall']:.3f} "
        f"(soporte {int(support[label])})"
        for label in minority
    )
    main_pair = top_confusions(results["forest"], n=1)[0]
    second_pair = top_confusions(results["forest"], n=2)[1]

    importance_table = importance.head(10).copy()
    importance_table.columns = ["Variable", "Importancia"]
    importance_table.index = range(1, len(importance_table) + 1)
    importance_table.index.name = "#"

    top_features = ", ".join(f"`{name}`" for name in importance.head(4)["variable"])

    artifact_section = (
        f"- Modelo: `{_relative(artifact['model'])}`\n"
        f"- Metadatos: `{_relative(artifact['metadata'])}`"
        if artifact
        else "- No se guardó el modelo en este sprint."
    )

    agreement_note = ""
    if agreement is not None:
        total = len(results["forest"]["y_pred"])
        agreement_note = (
            f"\nAmbos modelos coinciden en **{agreement:.2%}** de las predicciones del "
            f"conjunto de prueba ({total} filas) y aciertan la misma cantidad de casos "
            f"({int(round(results['forest']['metrics']['accuracy'] * total))} de {total}): "
            "el desempate lo da el F1 macro y el costo computacional.\n"
        )

    return f"""# Reporte Sprint 2 - Modelado y evaluación

**Proyecto:** Clasificación de variedades de frijol seco (Dry Bean Dataset)
**Metodologías:** CRISP-DM · TDSP · Scrum ML
**Semilla:** `random_state={config.RANDOM_STATE}` · **Train/Test:** 80% / 20% estratificado

## 1. Resumen del sprint

Se entrenaron dos modelos sobre los datos preparados en el Sprint 1:

| Modelo | Configuración |
|---|---|
| **Baseline** | `StandardScaler` + `LogisticRegression(max_iter=1000)` |
| **Alternativo** | `RandomForestClassifier(n_estimators=200, class_weight="balanced")` |

El escalamiento se ajustó solo con train (dentro del pipeline) para no filtrar
información del conjunto de prueba.

## 2. Comparación de métricas (test)

{dataframe_to_markdown(comparison_table)}

![Comparación de métricas](figuras/{figures["comparacion"].name})
{agreement_note}
## 3. Evaluación por clase

### Logistic Regression (baseline)

{_per_class_table(results["baseline"])}

### Random Forest (alternativo)

{_per_class_table(results["forest"])}

Clases con menor desempeño en Random Forest:

{worst_lines}

## 4. Matriz de confusión y errores

{chr(10).join(confusion_lines)}

![Matriz de confusión - baseline](figuras/{figures["confusion_baseline"].name})

![Matriz de confusión - Random Forest](figuras/{figures["confusion_forest"].name})

**Pregunta del laboratorio:** ¿qué clases se confunden más entre sí?

La confusión principal se da entre `SIRA` y `DERMASON`, variedades de tamaño y
forma cercanos, en ambos sentidos. En contraste, `BOMBAY` (la clase más pequeña y
minoritaria) se clasifica sin errores: es tan distinta en tamaño que las medidas
geométricas la separan fácilmente. El desbalance no oculta los errores de las
clases minoritarias, pero sí hace que accuracy por sí sola no sea suficiente para
evaluar el modelo.

## 5. Importancia de variables

{dataframe_to_markdown(importance_table)}

![Importancia de variables](figuras/{figures["importancia"].name})

La importancia indica qué variables usó el bosque para separar clases; **no
prueba causalidad**. En este entrenamiento lideran {top_features}, lo que
refuerza el hallazgo del EDA: la combinación de tamaño y forma es lo que
discrimina las variedades, no una única medida.

## 6. Decisión CRISP-DM

- **Modelo seleccionado:** `{selection["selected"]}`
- **Diferencia de F1 macro:** {selection["f1_macro_gap"]}
- **Criterio:** priorizar `f1_macro`, revisar errores por clase y preferir el
  modelo más simple cuando el desempeño es equivalente.
- **Justificación:** {selection["rationale"]}

## 7. Artefacto generado

{artifact_section}

## 8. Conclusiones del sprint

1. Ambos modelos superan el 91% de accuracy, lo que confirma que las variables
   geométricas son informativas para separar las 7 variedades.
2. El F1 macro es la métrica de decisión porque las clases están desbalanceadas
   (6,79:1), tal como se identificó en el Sprint 1.
3. Con `class_weight="balanced"`, el Random Forest alcanza estos niveles en las
   dos clases minoritarias: {minority_lines}.
4. La confusión más frecuente del bosque es `{main_pair["real"]}` → `{main_pair["predicho"]}`
   ({main_pair["casos"]} casos, {main_pair["tasa"]:.1%} de esa clase), seguida de
   `{second_pair["real"]}` → `{second_pair["predicho"]}` ({second_pair["casos"]} casos).
5. El modelo seleccionado queda guardado con `joblib` y metadatos, listo para la
   demo de despliegue del Sprint 3.
"""


def save_report(markdown: str, path: Path) -> Path:
    path.write_text(markdown, encoding="utf-8")
    return path
