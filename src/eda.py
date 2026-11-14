"""Análisis exploratorio de datos y generación del reporte del Sprint 1.

Genera las figuras en `outputs/reports/sprint1/figuras/` y un reporte en
Markdown con los hallazgos que exige CRISP-DM en la fase de comprensión.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: permite ejecutar en scripts y CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config


def _format_value(value, float_format: str = "{:.3f}") -> str:
    """Formatea valores conservando la precisión de variables pequeñas."""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value != 0 and abs(value) < 0.01:
            return f"{value:.6f}"
        return float_format.format(value)
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """Convierte un dataframe pequeño a tabla Markdown sin depender de tabulate."""
    display = df.astype(object)
    index_name = display.index.name or ""
    header = f"| {index_name} | " + " | ".join(str(c) for c in display.columns) + " |"
    separator = "|" + "---|" * (len(display.columns) + 1)
    lines = [header, separator]
    for label, row in zip(display.index, display.to_numpy(dtype=object)):
        values = [_format_value(value, float_format) for value in row]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_class_distribution(df: pd.DataFrame, path: Path) -> Path:
    counts = df[config.TARGET_COLUMN].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index, counts.values, color="#3b6ea5")
    ax.set_title("Distribución de clases - Dry Bean Dataset")
    ax.set_xlabel("Variedad de frijol")
    ax.set_ylabel("Cantidad de registros")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.bar_label(bars, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_histograms(df: pd.DataFrame, path: Path) -> Path:
    features = [column for column in df.columns if column != config.TARGET_COLUMN]
    n_cols = 4
    n_rows = int(np.ceil(len(features) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    for ax, column in zip(axes.ravel(), features):
        ax.hist(df[column].astype(float), bins=40, color="#5c8dc4", edgecolor="white")
        ax.set_title(column, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)
    fig.suptitle("Distribución de las 16 variables geométricas", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> Path:
    numeric = df.select_dtypes(include="number").astype(float)
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Correlación de Pearson entre variables")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_area_by_class(df: pd.DataFrame, path: Path) -> Path:
    order = df.groupby(config.TARGET_COLUMN)["Area"].median().sort_values().index
    data = [df.loc[df[config.TARGET_COLUMN] == label, "Area"].astype(float) for label in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=list(order))
    ax.set_title("Área del grano por variedad (posible variable discriminante)")
    ax.set_xlabel("Variedad de frijol")
    ax.set_ylabel("Area")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_figures(df: pd.DataFrame) -> dict[str, Path]:
    """Genera todas las figuras del Sprint 1 y devuelve sus rutas."""
    figures_dir = config.SPRINT1_FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "distribucion_clases": plot_class_distribution(
            df, figures_dir / "distribucion_clases.png"
        ),
        "histogramas": plot_feature_histograms(df, figures_dir / "histogramas_variables.png"),
        "correlacion": plot_correlation_heatmap(df, figures_dir / "correlacion_variables.png"),
        "area_por_clase": plot_area_by_class(df, figures_dir / "area_por_clase.png"),
    }


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------
_FIGURE_TITLES = {
    "distribucion_clases": "Distribución de clases",
    "histogramas": "Histogramas de las variables",
    "correlacion": "Matriz de correlación",
    "area_por_clase": "Área por variedad",
}


def build_eda_markdown(
    df: pd.DataFrame,
    quality: dict[str, Any],
    split_summary: dict[str, Any],
    figures: dict[str, Path],
) -> str:
    class_counts = pd.Series(quality["class_counts"], name="registros")
    class_share = (class_counts / class_counts.sum() * 100).round(2)
    other_non_numeric = [
        column
        for column in quality["non_numeric_columns"]
        if column != config.TARGET_COLUMN
    ]
    class_table = pd.DataFrame(
        {"registros": class_counts, "% del total": class_share}
    )
    class_table.index.name = "Clase"

    describe = df.select_dtypes(include="number").astype(float).describe().T
    describe = describe[["mean", "std", "min", "50%", "max"]]
    describe.index.name = "Variable"

    correlations = df.select_dtypes(include="number").astype(float).corr()
    upper = correlations.where(~np.tril(np.ones(correlations.shape)).astype(bool))
    top_corr = (
        upper.stack()
        .sort_values(key=np.abs, ascending=False)
        .head(5)
    )
    top_corr_lines = "\n".join(
        f"- `{a}` y `{b}`: r = {value:.3f}" for (a, b), value in top_corr.items()
    )

    figures_section = "\n".join(
        f"### {_FIGURE_TITLES.get(title, title)}\n\n"
        f"![{_FIGURE_TITLES.get(title, title)}](figuras/{path.name})\n"
        for title, path in figures.items()
    )

    return f"""# Reporte Sprint 1 - Comprensión y preparación de datos

**Proyecto:** Clasificación de variedades de frijol seco (Dry Bean Dataset)
**Metodologías:** CRISP-DM · TDSP · Scrum ML
**Fuente:** {config.UCI_DATASET_URL}
**Semilla:** `random_state={config.RANDOM_STATE}` · **Test:** {config.TEST_SIZE:.0%} estratificado

## 1. Comprensión del negocio

Una empresa agrícola quiere automatizar el control de calidad de granos de
frijol. El objetivo es predecir la variedad (`Class`) a partir de 16 medidas
geométricas obtenidas por visión por computador, reduciendo la clasificación
manual. Es un problema de **clasificación multiclase supervisada**.

## 2. Comprensión de los datos

- Registros originales: **{quality["rows"]}**
- Columnas: **{quality["columns"]}** (16 variables + `{config.TARGET_COLUMN}`)
- Variables numéricas: **{quality["numeric_columns"]}** · columnas no numéricas distintas del objetivo: {", ".join(other_non_numeric) or "ninguna"}
- Clases: **{quality["n_classes"]}** · relación entre la clase mayor y la menor: **{quality["class_balance_ratio"]}:1**

### Distribución de clases

{dataframe_to_markdown(class_table, float_format="{:.2f}")}

Las clases **no están balanceadas**: `DERMASON` concentra la mayor cantidad de
registros y `BOMBAY` la menor. Esto justifica reportar **F1 macro** además de
accuracy y usar `class_weight="balanced"` en el modelo alternativo.

### Estadísticas descriptivas

{dataframe_to_markdown(describe)}

### Correlaciones más fuertes

{top_corr_lines}

Se observan correlaciones muy altas entre variables de tamaño (`Area`,
`Perimeter`, `ConvexArea`, `EquivDiameter`). Para modelos lineales conviene el
escalamiento; para modelos de árboles no es obligatorio.

## 3. Calidad de datos

| Chequeo | Resultado |
|---|---|
| Valores nulos totales | {quality["missing_total"]} |
| Columnas con nulos | {len(quality["missing_by_column"])} |
| Filas duplicadas | {quality["duplicated_rows"]} |
| Filas eliminadas en limpieza | {quality["rows_removed"]} |
| Registros finales | {quality["rows_after_cleaning"]} |

Acciones aplicadas:

{chr(10).join(f"- {action}" for action in quality["cleaning_actions"])}

## 4. Preparación

- **X:** {split_summary["n_features"]} variables numéricas · **y:** `{config.TARGET_COLUMN}`
- **Train:** {split_summary["train_rows"]} registros ({split_summary["train_share"]:.1%})
- **Test:** {split_summary["test_rows"]} registros ({split_summary["test_share"]:.1%})
- **Estratificación:** proporciones de clase equivalentes en train y test (ver JSON de resumen).
- **Pipeline:** `{split_summary["preprocessing_pipeline"]}` — se ajusta solo con train
  para evitar fuga de información.

## 5. Figuras generadas

{figures_section}
## 6. Conclusiones del Sprint 1

1. El dataset se carga de forma reproducible desde UCI y queda versionado en `data/raw/`.
2. No hay valores nulos y los duplicados se eliminaron de forma documentada.
3. El desbalance de clases obliga a mirar F1 macro y no solo accuracy.
4. Las variables de tamaño dominan la estructura de correlación, lo que anticipa
   un buen desempeño en clasificación.
5. El split estratificado y el pipeline dejan el terreno listo para el Sprint 2.
"""


def save_eda_report(markdown: str, path: Path) -> Path:
    path.write_text(markdown, encoding="utf-8")
    return path
