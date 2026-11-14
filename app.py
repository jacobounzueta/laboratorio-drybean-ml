"""Aplicación Streamlit para despliegue del clasificador Dry Bean ML.

Uso:
    streamlit run app.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle

from src import config, deploy

DEFAULT_FEATURES: list[str] = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

DEFAULT_SAMPLE: dict[str, float] = {
    "Area": 40512.0,
    "Perimeter": 737.636,
    "MajorAxisLength": 248.20203,
    "MinorAxisLength": 208.52515,
    "AspectRatio": 1.19027,
    "Eccentricity": 0.54237,
    "ConvexArea": 40973.0,
    "EquivDiameter": 227.11557,
    "Extent": 0.74427,
    "Solidity": 0.98875,
    "Roundness": 0.93564,
    "Compactness": 0.91504,
    "ShapeFactor1": 0.006127,
    "ShapeFactor2": 0.002650,
    "ShapeFactor3": 0.837304,
    "ShapeFactor4": 0.996621,
}

HIGH_PRECISION_FEATURES = {
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "Eccentricity",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
}


@st.cache_resource
def load_model_bundle() -> tuple[Any, dict[str, Any]]:
    """Carga el modelo final y sus metadatos una sola vez."""
    return deploy.load_final_model()


@st.cache_data
def load_test_samples() -> tuple[pd.DataFrame, pd.Series | None]:
    """Carga ejemplos reales del conjunto de test si están disponibles."""
    if not config.X_TEST_PATH.exists():
        return pd.DataFrame([DEFAULT_SAMPLE]), None

    X_test = pd.read_csv(config.X_TEST_PATH)
    y_test = (
        pd.read_csv(config.Y_TEST_PATH).iloc[:, 0]
        if config.Y_TEST_PATH.exists()
        else None
    )
    return X_test, y_test


def build_class_presets(
    X_test: pd.DataFrame,
    y_test: pd.Series | None,
) -> dict[str, dict[str, Any]]:
    """Construye un diccionario de ejemplos representativos por clase."""
    presets: dict[str, dict[str, Any]] = {}
    if y_test is None or X_test.empty:
        presets["Ejemplo por defecto"] = {
            "features": DEFAULT_SAMPLE,
            "true_class": None,
        }
        return presets

    for label in sorted(y_test.astype(str).unique()):
        idx = y_test[y_test.astype(str) == label].index[0]
        row = X_test.loc[idx].to_dict()
        presets[f"Muestra real: {label}"] = {
            "features": {k: float(v) for k, v in row.items()},
            "true_class": label,
        }
    return presets


def render_sidebar(metadata: dict[str, Any]) -> None:
    """Muestra información y trazabilidad del modelo en la barra lateral."""
    st.sidebar.header("Modelo en producción")
    st.sidebar.write(f"**Modelo:** {metadata.get('selected_model', 'N/D')}")
    st.sidebar.write(f"**Tipo:** `{metadata.get('model_type', 'N/D')}`")
    if metadata.get("saved_at_utc"):
        st.sidebar.write(f"**Guardado (UTC):** `{metadata['saved_at_utc']}`")

    metrics = metadata.get("metrics", {})
    if metrics:
        st.sidebar.subheader("Métricas de evaluación")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Accuracy (test)", f"{metrics.get('accuracy', 0):.4f}")
        col2.metric("F1 macro (test)", f"{metrics.get('f1_macro', 0):.4f}")
        if "cv_f1_macro_mean" in metrics:
            st.sidebar.metric(
                "F1 macro CV (5-fold)",
                f"{metrics['cv_f1_macro_mean']:.4f}",
            )

    classes = metadata.get("classes", [])
    if classes:
        st.sidebar.subheader("Variedades soportadas")
        st.sidebar.write(", ".join(f"`{c}`" for c in classes))

    st.sidebar.divider()
    st.sidebar.caption(
        "Dataset: Dry Bean (UCI 602) · 16 variables geométricas · "
        f"Train: {metadata.get('train_rows', 'N/D')} · Test: {metadata.get('test_rows', 'N/D')}"
    )


def build_bean_simulation_figure(params: dict[str, float]) -> Figure:
    """Genera una simulación 2D de la geometría del grano a partir de sus parámetros."""
    major = max(float(params.get("MajorAxisLength", 200.0)), 1.0)
    minor = max(float(params.get("MinorAxisLength", 150.0)), 1.0)
    equiv_d = max(float(params.get("EquivDiameter", 170.0)), 1.0)
    solidity = float(np.clip(params.get("Solidity", 0.98), 0.5, 1.0))
    roundness = float(np.clip(params.get("Roundness", 0.85), 0.2, 1.0))

    a = major / 2.0
    b = minor / 2.0
    theta = np.linspace(0, 2 * np.pi, 360)

    # Ligera concavidad en el hilio/borde modulada por Solidity y Roundness
    concavity = (1.0 - solidity) * 2.2
    lobe = (1.0 - roundness) * 0.06
    hilum = concavity * np.exp(-((theta - np.pi / 2.0) ** 2) / 0.18)
    r_scale = 1.0 - hilum + lobe * np.cos(2.0 * theta)

    x = a * r_scale * np.cos(theta)
    y = b * r_scale * np.sin(theta)
    coords = np.column_stack([x, y])

    fig = Figure(figsize=(5.2, 4.2), dpi=130)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    # Caja delimitadora (MajorAxisLength x MinorAxisLength)
    bbox = Rectangle(
        (-a, -b),
        major,
        minor,
        linewidth=1.0,
        linestyle="--",
        edgecolor="#94a3b8",
        facecolor="none",
        label="Caja envolvente",
    )
    ax.add_patch(bbox)

    # Círculo de diámetro equivalente
    equiv_circle = Circle(
        (0.0, 0.0),
        radius=equiv_d / 2.0,
        linewidth=1.1,
        linestyle=":",
        edgecolor="#0284c7",
        facecolor="none",
        label=f"Diámetro equiv. ({equiv_d:.1f} px)",
    )
    ax.add_patch(equiv_circle)

    # Contorno simulado del grano
    bean_poly = Polygon(
        coords,
        closed=True,
        facecolor="#c68b59",
        edgecolor="#7c471b",
        linewidth=2.0,
        alpha=0.72,
        label="Contorno simulado",
    )
    ax.add_patch(bean_poly)

    # Ejes mayor y menor
    ax.plot([-a, a], [0, 0], color="#b91c1c", linewidth=1.4, label=f"Eje mayor ({major:.1f} px)")
    ax.plot([0, 0], [-b, b], color="#15803d", linewidth=1.4, label=f"Eje menor ({minor:.1f} px)")

    half_span = max(a, b, equiv_d / 2.0, 260.0) * 1.18
    ax.set_xlim(-half_span, half_span)
    ax.set_ylim(-half_span * 0.82, half_span * 0.82)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.6, color="#cbd5e1", alpha=0.8)
    ax.set_xlabel("px")
    ax.set_ylabel("px")
    ax.set_title("Silueta geométrica simulada", fontsize=11, fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    return fig


def render_single_prediction_tab(
    model: Any,
    features: list[str],
    presets: dict[str, dict[str, Any]],
) -> None:
    """Pestaña para predecir la variedad de un grano individual."""
    st.subheader("Predicción de un grano individual")
    st.write(
        "Selecciona una muestra de ejemplo o ajusta las 16 medidas geométricas; "
        "la simulación se actualiza en tiempo real."
    )

    col_config, col_sim = st.columns([3, 2], gap="large")

    with col_config:
        preset_names = list(presets.keys())
        selected_preset = st.selectbox(
            "Cargar valores de ejemplo",
            options=preset_names,
            index=0,
        )
        preset_data = presets[selected_preset]
        initial_values = preset_data["features"]
        true_class = preset_data["true_class"]

        cols = st.columns(2)
        user_inputs: dict[str, float] = {}
        for idx, feature in enumerate(features):
            col = cols[idx % 2]
            default_val = float(initial_values.get(feature, DEFAULT_SAMPLE.get(feature, 0.0)))
            is_small = feature in HIGH_PRECISION_FEATURES
            user_inputs[feature] = col.number_input(
                label=feature,
                value=default_val,
                step=0.0001 if is_small else 1.0,
                format="%.6f" if is_small else "%.3f",
                key=f"input_{selected_preset}_{feature}",
            )

        submitted = st.button("Predecir variedad", type="primary")

    with col_sim:
        st.markdown("**Simulación de parámetros**")
        fig = build_bean_simulation_figure(user_inputs)
        st.pyplot(fig, use_container_width=True)

        sim_m1, sim_m2 = st.columns(2)
        sim_m1.metric("Area", f"{user_inputs.get('Area', 0):.0f} px²")
        sim_m2.metric("AspectRatio", f"{user_inputs.get('AspectRatio', 0):.3f}")
        sim_m3, sim_m4 = st.columns(2)
        sim_m3.metric("Roundness", f"{user_inputs.get('Roundness', 0):.4f}")
        sim_m4.metric("Solidity", f"{user_inputs.get('Solidity', 0):.4f}")

    if submitted:
        try:
            result = deploy.predict_beans(model, user_inputs, features=features)[0]
            frame = deploy.to_feature_frame(user_inputs, features=features)
        except deploy.DatosInvalidosError as err:
            st.error(f"Datos inválidos: {err}")
            return

        pred = result["prediccion"]
        proba = result.get("probabilidad")

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Variedad predicha", pred)
        if proba is not None:
            res_col2.metric("Confianza", f"{proba * 100:.2f}%")
        if true_class is not None:
            res_col3.metric(
                "Clase real (referencia)",
                true_class,
                delta="Acierto" if pred == true_class else "Error",
                delta_color="normal" if pred == true_class else "inverse",
            )

        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            probs = model.predict_proba(frame)[0]
            prob_df = pd.DataFrame(
                {
                    "Variedad": [str(c) for c in model.classes_],
                    "Probabilidad": probs,
                }
            ).sort_values("Probabilidad", ascending=False)
            st.write("**Distribución de probabilidades por variedad**")
            st.bar_chart(prob_df.set_index("Variedad"))


def render_batch_prediction_tab(
    model: Any,
    features: list[str],
    X_test: pd.DataFrame,
) -> None:
    """Pestaña para predicción por lotes mediante carga de CSV."""
    st.subheader("Predicción por lotes (CSV)")
    st.write(
        "Sube un archivo `.csv` con las 16 columnas geométricas exactas para "
        "obtener las predicciones de múltiples granos."
    )

    sample_csv = X_test[features].head(10).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar CSV de ejemplo (10 filas)",
        data=sample_csv,
        file_name="ejemplo_dry_bean.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Archivo CSV de entrada", type=["csv"])
    if uploaded_file is None:
        return

    try:
        input_df = pd.read_csv(uploaded_file)
        predictions = deploy.predict_beans(model, input_df, features=features)
    except deploy.DatosInvalidosError as err:
        st.error(f"Error de validación en el CSV: {err}")
        return
    except Exception as err:
        st.error(f"No se pudo procesar el archivo CSV: {err}")
        return

    pred_df = pd.DataFrame(predictions)
    output_df = pd.concat([input_df.reset_index(drop=True), pred_df], axis=1)

    col1, col2 = st.columns(2)
    col1.metric("Granos procesados", len(output_df))
    top_class = output_df["prediccion"].mode().iloc[0]
    col2.metric("Variedad más frecuente", top_class)

    st.write("**Conteo por variedad predicha**")
    counts = output_df["prediccion"].value_counts().rename_axis("Variedad").to_frame("Cantidad")
    st.bar_chart(counts)

    st.write("**Resultados (vista previa)**")
    st.dataframe(output_df.head(50), use_container_width=True)

    result_csv = output_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar predicciones completas (CSV)",
        data=result_csv,
        file_name="predicciones_dry_bean.csv",
        mime="text/csv",
        type="primary",
    )


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(
        page_title="Dry Bean ML - Clasificador",
        page_icon="🫘",
        layout="wide",
    )
    st.title("Clasificador de Variedades de Frijol Seco (Dry Bean ML)")
    st.caption(
        "Despliegue interactivo del modelo entrenado en el laboratorio "
        "(CRISP-DM · TDSP · Scrum ML)."
    )

    try:
        model, metadata = load_model_bundle()
    except deploy.ModeloNoDisponibleError as err:
        st.error(str(err))
        st.stop()

    features = metadata.get("features") or DEFAULT_FEATURES
    X_test, y_test = load_test_samples()
    presets = build_class_presets(X_test, y_test)

    render_sidebar(metadata)

    tab_single, tab_batch = st.tabs(
        ["Predicción individual", "Predicción por lotes (CSV)"]
    )
    with tab_single:
        render_single_prediction_tab(model, features, presets)
    with tab_batch:
        render_batch_prediction_tab(model, features, X_test)


if __name__ == "__main__":
    main()
