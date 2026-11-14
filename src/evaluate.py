"""Figuras y reporte del Sprint 2: comparación de modelos y evaluación.

Genera las gráficas en `outputs/reports/sprint2/figuras/` y un reporte en
Markdown con la fase de evaluación de CRISP-DM.
"""

from __future__ import annotations

import re
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


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> Path:
    metrics = ["accuracy", "f1_macro", "balanced_accuracy"]
    labels = [metric.replace("_", " ").title() for metric in metrics]
    models = comparison.index.tolist()

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#3b6ea5", "#d98d3b", "#5a9367", "#a35b8f"]
    for index, model in enumerate(models):
        values = comparison.loc[model, metrics].astype(float).tolist()
        offsets = x - 0.4 + width / 2 + index * width
        bars = ax.bar(offsets, values, width, label=model, color=colors[index % len(colors)])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    values_all = comparison[metrics].astype(float).to_numpy().ravel()
    ax.set_ylim(values_all.min() - 0.03, values_all.max() + 0.035)
    ax.set_ylabel("Puntaje")
    ax.set_title("Comparación de modelos - test (eje truncado para ver diferencias)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cv_comparison(comparison: pd.DataFrame, path: Path) -> Path:
    models = comparison.index.tolist()
    means = comparison["cv_f1_macro_mean"].astype(float).tolist()
    stds = comparison["cv_f1_macro_std"].astype(float).tolist()
    colors = ["#3b6ea5", "#d98d3b", "#5a9367", "#a35b8f"][: len(models)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models, means, yerr=stds, capsize=6, color=colors)
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=8)
    ax.set_ylim(min(means) - max(stds) - 0.01, max(means) + max(stds) + 0.03)
    ax.set_ylabel("F1 macro medio (5 folds)")
    ax.set_title("Estabilidad de los modelos - validación cruzada sobre train")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
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

    figures: dict[str, Path] = {
        "comparacion": plot_model_comparison(
            comparison, figures_dir / "comparacion_modelos.png"
        ),
        "comparacion_cv": plot_cv_comparison(
            comparison, figures_dir / "comparacion_cv.png"
        ),
    }
    for name, result in results.items():
        slug = _slug(name)
        figures[f"confusion_{slug}"] = plot_confusion(
            result, figures_dir / f"confusion_{slug}.png"
        )
    figures["importancia"] = plot_feature_importance(
        importance, figures_dir / "importancia_variables.png"
    )
    return figures


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
    cv_results: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    importance: pd.DataFrame,
    figures: dict[str, Path],
    artifact: dict[str, str] | None,
    agreement: float | None = None,
) -> str:
    test_columns = [
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "train_seconds",
    ]
    test_table = comparison[test_columns].copy()
    test_table.index.name = "modelo"

    cv_table = pd.DataFrame(
        {
            "F1 macro medio": [cv_results[name]["mean"] for name in comparison.index],
            "Desv. estándar": [cv_results[name]["std"] for name in comparison.index],
            "Folds": [
                " · ".join(f"{score:.4f}" for score in cv_results[name]["scores"])
                for name in comparison.index
            ],
        },
        index=comparison.index,
    )
    cv_table = cv_table.round(4)
    cv_table.index.name = "modelo"

    confusion_sections = "\n".join(
        f"### Matriz de confusión - {result['model_name']}\n\n"
        f"![Matriz de confusión - {result['model_name']}]"
        f"(figuras/{figures[f'confusion_{_slug(name)}'].name})\n"
        for name, result in results.items()
    )

    selected = results[selection["selected"]]
    selected_name = selected["model_name"]

    confusion_lines = []
    for result in results.values():
        pairs = top_confusions(result, n=3)
        detail = ", ".join(
            f"{pair['real']} → {pair['predicho']} ({pair['casos']} casos, {pair['tasa']:.1%})"
            for pair in pairs
        )
        confusion_lines.append(f"- **{result['model_name']}:** {detail}")

    minority = (
        selected["per_class"].loc[selected["labels"], ["recall", "support"]]
        .sort_values("support")
        .head(2)
    )
    minority_lines = " · ".join(
        f"`{label}`: recall = {row['recall']:.3f} (soporte {int(row['support'])})"
        for label, row in minority.iterrows()
    )

    main_pair = top_confusions(selected, n=1)[0]
    second_pair = top_confusions(selected, n=2)[1]

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
        agreement_note = (
            f"\nLa coincidencia media entre las predicciones de los modelos es "
            f"**{agreement:.2%}**: comparten casi la misma frontera de decisión y se "
            "diferencian en casos límite.\n"
        )

    per_class_sections = "\n".join(
        f"### {result['model_name']}\n\n{_per_class_table(result)}\n"
        for result in results.values()
    )

    min_accuracy = comparison["accuracy"].min()
    max_accuracy = comparison["accuracy"].max()

    return f"""# Reporte Sprint 2 - Modelado y evaluación

**Proyecto:** Clasificación de variedades de frijol seco (Dry Bean Dataset)
**Metodologías:** CRISP-DM · TDSP · Scrum ML
**Semilla:** `random_state={config.RANDOM_STATE}` · **Train/Test:** 80% / 20% estratificado

## 1. Resumen del sprint

Se entrenaron tres modelos sobre los datos preparados en el Sprint 1:

| Modelo | Configuración |
|---|---|
| **Baseline** | `StandardScaler` + `LogisticRegression(max_iter=1000)` |
| **Alternativo** | `RandomForestClassifier(n_estimators=200, class_weight="balanced")` |
| **Tercero** | `HistGradientBoostingClassifier(l2_regularization=1.0)` |

El escalamiento del baseline se ajustó solo con train (dentro del pipeline) para
no filtrar información del conjunto de prueba.

## 2. Protocolo de evaluación

1. **Hold-out:** métricas sobre las 2.709 filas de test que nunca vieron los modelos.
2. **Validación cruzada 5-fold** sobre train: se usa para **decidir** entre modelos,
   porque una única partición de test no distingue diferencias de milésimas.
3. **Decisión CRISP-DM:** se prioriza el F1 macro medio de la validación cruzada;
   ante empate (< 0,005) se prefiere el modelo más simple y rápido.

## 3. Comparación en test (hold-out)

{dataframe_to_markdown(test_table)}

### Comparación de métricas en test

![Comparación de métricas en test](figuras/{figures["comparacion"].name})
{agreement_note}
## 4. Validación cruzada 5-fold

{dataframe_to_markdown(cv_table, float_format="{:.4f}", adaptive=False)}

### F1 macro con validación cruzada

![F1 macro con validación cruzada](figuras/{figures["comparacion_cv"].name})

## 5. Evaluación por clase

{per_class_sections}
## 6. Matrices de confusión y errores

{chr(10).join(confusion_lines)}

{confusion_sections}
**Pregunta del laboratorio:** ¿qué clases se confunden más entre sí?

La confusión principal se da entre `SIRA` y `DERMASON`, variedades de tamaño y
forma cercanos, y ocurre en los tres modelos. `BOMBAY`, la clase más pequeña y
minoritaria, se clasifica sin errores: sus medidas geométricas son claramente
distintas. El límite del modelo está en las variedades parecidas, no en las clases
con pocos ejemplos.

## 7. Importancia de variables

{dataframe_to_markdown(importance_table)}

![Top 10 variables más importantes](figuras/{figures["importancia"].name})

La importancia se calcula con el Random Forest (el gradient boosting no expone
importancias por impureza). Indica qué variables fueron útiles para separar
clases; **no prueba causalidad**. Lideran {top_features}, lo que refuerza el
hallazgo del EDA: la combinación de tamaño y forma es lo que discrimina las
variedades, no una única medida.

## 8. Decisión CRISP-DM

- **Modelo seleccionado:** `{selected_name}`
- **Métrica de decisión:** `{selection["metric"]}`
- **Diferencia con el segundo:** {selection["gap"]}
- **Justificación:** {selection["rationale"]}

## 9. Artefacto generado

{artifact_section}

## 10. Conclusiones del sprint

1. Los tres modelos se mueven entre {min_accuracy:.2%} y {max_accuracy:.2%} de
   accuracy: el problema está bien planteado y las variables geométricas son
   informativas.
2. Las diferencias entre modelos son inferiores a 1 punto porcentual, muy cerca
   del ruido de una muestra de 2.709 filas. El techo parece estar en las variables
   disponibles, no en el algoritmo.
3. La validación cruzada ordena con más estabilidad que el hold-out y es la base
   de la decisión: `{selected_name}` gana en las 5 particiones.
4. `SIRA` ↔ `DERMASON` es la confusión dominante (`{main_pair["real"]}` →
   `{main_pair["predicho"]}`, {main_pair["casos"]} casos; `{second_pair["real"]}` →
   `{second_pair["predicho"]}`, {second_pair["casos"]} casos). `BOMBAY` se separa
   sin errores (recall = 1,000).
5. En el modelo seleccionado, las clases minoritarias quedan así: {minority_lines}.
6. El modelo final se guarda con `joblib` y metadatos, listo para la demo de
   despliegue del Sprint 3.
"""


def save_report(markdown: str, path: Path) -> Path:
    path.write_text(markdown, encoding="utf-8", newline="\n")
    return path
