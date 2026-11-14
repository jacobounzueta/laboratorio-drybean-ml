"""Sprint 3: despliegue del modelo y verificación del artefacto final.

Uso:
    python -m src.sprint3

Requiere el modelo del Sprint 2 en `outputs/models/` (lo genera
`python -m src.sprint2`).

Genera:
    outputs/reports/sprint3/reporte_sprint3.md
    outputs/reports/sprint3/demo_predicciones.csv
    outputs/reports/sprint3/verificacion_modelo.json
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src import config, deploy, prepare

DEMO_ROWS = 5


def _log(step: str, message: str) -> None:
    print(f"[{step}] {message}")


def _relative(path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _verify_model(model, metadata: dict[str, Any], X_test, y_test) -> dict[str, Any]:
    """Recalcula las métricas del artefacto para detectar modelos corruptos."""
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    f1_macro = float(f1_score(y_test, predictions, average="macro"))

    stored = metadata.get("metrics", {})
    accuracy_delta = abs(accuracy - float(stored.get("accuracy", 0.0)))
    f1_delta = abs(f1_macro - float(stored.get("f1_macro", 0.0)))

    tolerance = 1e-6
    return {
        "accuracy_recalculada": round(accuracy, 6),
        "accuracy_guardada": stored.get("accuracy"),
        "accuracy_delta": round(accuracy_delta, 9),
        "f1_macro_recalculado": round(f1_macro, 6),
        "f1_macro_guardado": stored.get("f1_macro"),
        "f1_macro_delta": round(f1_delta, 9),
        "features_coinciden": list(getattr(model, "feature_names_in_", []))
        == list(metadata.get("features", [])),
        "clases_coinciden": [str(c) for c in getattr(model, "classes_", [])]
        == [str(c) for c in metadata.get("classes", [])],
        "verificacion_ok": accuracy_delta < tolerance
        and f1_delta < tolerance
        and list(getattr(model, "feature_names_in_", []))
        == list(metadata.get("features", [])),
    }


def _demo_table(model, X_test, y_test) -> pd.DataFrame:
    """Demo honesta: mezcla aciertos y los errores más frecuentes del modelo.

    Se muestran tres aciertos y un caso de cada una de las dos confusiones más
    frecuentes, para que el reporte enseñe también dónde falla el clasificador.
    """
    predicted = pd.Series(model.predict(X_test), index=X_test.index)
    pairs = pd.DataFrame(
        {"real": y_test.to_numpy(), "predicho": predicted.to_numpy()},
        index=y_test.index,
    )
    errors = pairs[pairs["real"] != pairs["predicho"]]
    top_pairs = errors.value_counts().head(2).index.tolist()

    hits = predicted[predicted == y_test].index[: DEMO_ROWS - len(top_pairs)]
    misses = [errors[(errors["real"] == real) & (errors["predicho"] == pred)].index[0]
              for real, pred in top_pairs]
    sample_index = list(hits) + misses

    sample = X_test.loc[sample_index]
    labels = y_test.loc[sample_index]
    results = deploy.predict_beans(model, sample)

    table = sample.copy()
    table["clase_real"] = labels.to_numpy()
    table["clase_predicha"] = [item["prediccion"] for item in results]
    table["probabilidad"] = [item.get("probabilidad") for item in results]
    table["acierto"] = table["clase_real"] == table["clase_predicha"]
    return table


def _dict_demo(model, X_test) -> dict[str, Any]:
    """Simula un grano nuevo: dict con las 16 medidas redondeadas."""
    row = X_test.iloc[0]
    nuevo_grano = {column: round(float(value), 2) for column, value in row.items()}

    clase = deploy.predict_bean_class(model, nuevo_grano)
    detalle = deploy.predict_beans(model, nuevo_grano)[0]
    return {"entrada": nuevo_grano, "prediccion": clase, "detalle": detalle}


def build_markdown(
    metadata: dict[str, Any],
    verification: dict[str, Any],
    demo: pd.DataFrame,
    dict_demo: dict[str, Any],
    artifact_paths: dict[str, str],
) -> str:
    demo_view = demo[
        ["clase_real", "clase_predicha", "probabilidad", "acierto"]
    ].copy()
    demo_view.index = range(1, len(demo_view) + 1)
    demo_view.index.name = "muestra"

    from src.eda import dataframe_to_markdown

    demo_table = dataframe_to_markdown(demo_view)
    error_pairs = " y ".join(
        f"`{row['clase_real']}` → `{row['clase_predicha']}`"
        for _, row in demo[~demo["acierto"]].iterrows()
    ) or "ninguno"
    models_table = pd.DataFrame(
        {
            "Métrica": ["Accuracy (test)", "F1 macro (test)", "F1 macro CV 5-fold"],
            "Valor guardado": [
                f"{float(metadata['metrics']['accuracy']):.4f}",
                f"{float(metadata['metrics']['f1_macro']):.4f}",
                f"{float(metadata['metrics']['cv_f1_macro_mean']):.4f} "
                f"± {float(metadata['metrics']['cv_f1_macro_std']):.4f}",
            ],
        }
    )
    models_table.index = models_table.index + 1
    models_table.index.name = "#"

    input_example = json.dumps(dict_demo["entrada"], indent=4, ensure_ascii=False)

    return f"""# Reporte Sprint 3 - Despliegue y verificación

**Proyecto:** Clasificación de variedades de frijol seco (Dry Bean Dataset)
**Metodologías:** CRISP-DM · TDSP · Scrum ML
**Objetivo del sprint:** cargar el modelo guardado, predecir con datos nuevos y
dejar el artefacto listo para una demo reproducible.

## 1. Artefacto cargado

| Campo | Valor |
|---|---|
| Modelo | `{metadata.get("selected_model")}` |
| Tipo | `{metadata.get("model_type")}` |
| Archivo | `{artifact_paths["model"]}` |
| Metadatos | `{artifact_paths["metadata"]}` |
| Guardado (UTC) | {metadata.get("saved_at_utc")} |
| scikit-learn | {metadata.get("sklearn_version")} |
| Features | {len(metadata.get("features", []))} |
| Clases | {", ".join(metadata.get("classes", []))} |

{dataframe_to_markdown(models_table)}

## 2. Verificación de integridad

Se volvió a predecir el conjunto de prueba con el modelo cargado desde disco:

| Verificación | Resultado |
|---|---|
| Accuracy guardada vs recalculada | {verification["accuracy_guardada"]:.6f} vs {verification["accuracy_recalculada"]:.6f} |
| Delta de accuracy | {verification["accuracy_delta"]:.2e} |
| Delta de F1 macro | {verification["f1_macro_delta"]:.2e} |
| Features coinciden con los metadatos | {"sí" if verification["features_coinciden"] else "no"} |
| Clases coinciden con los metadatos | {"sí" if verification["clases_coinciden"] else "no"} |
| Resultado | {"✅ artefacto íntegro" if verification["verificacion_ok"] else "❌ revisar artefacto"} |

## 3. Demo de predicción

Cinco casos del conjunto de prueba que los modelos nunca vieron: **tres aciertos**
y **un caso de cada una de las dos confusiones más frecuentes**, para que la demo
muestre también los límites reales del clasificador.

{demo_table}

Aciertos en la demo: **{int(demo["acierto"].sum())} de {len(demo)}**.

Errores incluidos: {error_pairs}. Son variedades de tamaño y forma parecidos
donde el modelo predice con alta confianza y aun así se equivoca: ese es el riesgo
que hay que monitorear en producción.

## 4. Predicción de un grano nuevo

Se simula la llegada de un grano con medidas redondeadas a 2 decimales:

```json
{input_example}
```

Resultado:

```python
from src import deploy

modelo, metadatos = deploy.load_final_model()
clase = deploy.predict_bean_class(modelo, nuevo_grano)
```

- **Predicción:** `{dict_demo["prediccion"]}`
- **Probabilidad:** {dict_demo["detalle"].get("probabilidad")}

## 5. Función reutilizable

`src/deploy.py` expone tres piezas:

| Función | Uso |
|---|---|
| `load_final_model()` | Carga el `.joblib` y sus metadatos |
| `predict_bean_class(model, datos)` | Predice **un** grano (dict, Series o fila de DataFrame) |
| `predict_beans(model, datos)` | Predice **varios** granos e incluye probabilidad |

Validaciones incluidas: variables faltantes, variables desconocidas, valores
nulos, columnas no numéricas y entrada vacía. Cada error explica qué corregir.

## 6. Cómo llevarlo a producción

1. Reentrenar con `python -m src.sprint2` cuando cambien los datos (semilla fija).
2. Versionar el `.joblib` y sus metadatos junto con la versión de scikit-learn.
3. Exponer `predict_bean_class` detrás de un endpoint o un script de lote.
4. Monitorear la distribución de entrada: si aparecen medidas fuera del rango
   histórico, el modelo puede degradarse sin avisar.

## 7. Limitaciones conocidas

- El modelo se entrenó con 16 medidas de visión por computador: si el sistema de
  captura cambia, las variables dejan de ser comparables.
- `SIRA` y `DERMASON` se confunden de forma sistemática (~9% de `SIRA`): para
  control de calidad estricto conviene revisión humana en esos casos.
- No se modela el costo de equivocarse: un falso negativo en `BOMBAY` (clase
  minoritaria y valiosa) puede costar más que un error entre variedades comunes.
- El desempeño reportado (~92% accuracy) es sobre una partición fija; en
  producción debe monitorearse con datos nuevos.

## 8. Conexión CRISP-DM

Este sprint cierra la fase de **despliegue**: el modelo ya no es un notebook, es
un artefacto versionado que recibe datos nuevos y devuelve una predicción con
probabilidad. La verificación automática de métricas evita el riesgo típico de
desplegar un archivo equivocado.

## 9. Entregables del sprint

- `outputs/models/modelo_final_drybean.joblib` + metadatos
- `src/deploy.py` con la función de predicción
- `notebooks/03_sprint3_despliegue.ipynb` (ejecutado)
- Este reporte, `demo_predicciones.csv` y `verificacion_modelo.json`
- Documentación final en `docs/reporte_final.md`
"""


def run_sprint3() -> dict[str, Any]:
    # ------------------------------------------------------------------ 1/6
    _log("1/6", "Cargando el modelo final y sus metadatos...")
    model, metadata = deploy.load_final_model()
    _log(
        "1/6",
        f"{metadata.get('selected_model')} ({metadata.get('model_type')}) "
        f"· {len(metadata.get('features', []))} features "
        f"· {len(metadata.get('classes', []))} clases",
    )

    # ------------------------------------------------------------------ 2/6
    _log("2/6", "Cargando datos de prueba y verificando integridad...")
    X_train, X_test, y_train, y_test = prepare.load_processed_data()
    verification = _verify_model(model, metadata, X_test, y_test)
    _log(
        "2/6",
        f"accuracy {verification['accuracy_recalculada']:.6f} "
        f"(delta {verification['accuracy_delta']:.2e}) · "
        f"verificacion_ok={verification['verificacion_ok']}",
    )
    if not verification["verificacion_ok"]:
        raise RuntimeError("El artefacto no reproduce las métricas guardadas.")

    # ------------------------------------------------------------------ 3/6
    _log("3/6", f"Prediciendo {DEMO_ROWS} muestras de test...")
    demo = _demo_table(model, X_test, y_test)
    _log("3/6", f"Aciertos: {int(demo['acierto'].sum())}/{len(demo)}")

    # ------------------------------------------------------------------ 4/6
    _log("4/6", "Simulando un grano nuevo con predict_bean_class...")
    dict_demo = _dict_demo(model, X_test)
    _log(
        "4/6",
        f"Predicción: {dict_demo['prediccion']} "
        f"(probabilidad {dict_demo['detalle'].get('probabilidad')})",
    )

    # ------------------------------------------------------------------ 5/6
    _log("5/6", "Guardando demo y verificación...")
    demo.to_csv(config.SPRINT3_REPORTS_DIR / "demo_predicciones.csv", index=False)
    (config.SPRINT3_REPORTS_DIR / "verificacion_modelo.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ------------------------------------------------------------------ 6/6
    _log("6/6", "Generando el reporte del sprint...")
    artifact_paths = {
        "model": _relative(config.FINAL_MODEL_PATH),
        "metadata": _relative(config.FINAL_MODEL_PATH.with_suffix(".metadata.json")),
    }
    markdown = build_markdown(metadata, verification, demo, dict_demo, artifact_paths)
    report_path = config.SPRINT3_REPORTS_DIR / "reporte_sprint3.md"
    report_path.write_text(markdown, encoding="utf-8")
    _log("6/6", f"Reporte: {report_path}")

    return {
        "metadata": metadata,
        "verification": verification,
        "report": str(report_path),
        "demo": demo,
    }


def main() -> None:
    result = run_sprint3()
    print("\nSprint 3 finalizado correctamente.")
    print(f"Artefacto íntegro: {result['verification']['verificacion_ok']}")
    print(f"Reporte: {result['report']}")


if __name__ == "__main__":
    main()
