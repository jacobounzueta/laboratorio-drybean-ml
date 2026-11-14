# Laboratorio Dry Bean - CRISP-DM + TDSP + Scrum ML

Proyecto de clasificación de variedades de frijol seco (**Dry Bean Dataset**, UCI 602)
desarrollado como laboratorio práctico de Machine Learning con metodologías
profesionales de ciencia de datos.

| | |
|---|---|
| **Equipo** | Jacobo Unzueta y Rommel Prieto |
| **Programa** | Maestría en IA y Ciencia de Datos - Universidad Autónoma de Occidente |
| **Metodologías** | CRISP-DM · TDSP · Scrum ML |
| **Estado** | Sprint 1 y 2 finalizados · Sprint 3 en curso |

---

## 1. Problema de negocio

Una empresa agrícola quiere automatizar el control de calidad de granos de frijol.
Hoy parte de la clasificación se hace a mano; el objetivo es predecir la variedad
del grano (`Class`) a partir de 16 medidas geométricas obtenidas por visión por
computador. Es un problema de **clasificación multiclase supervisada** con 7 clases.

## 2. Dataset

- Fuente oficial: <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset>
- 13.611 instancias · 16 variables numéricas · 7 clases (`DERMASON`, `SIRA`, `SEKER`, `HOROZ`, `CALI`, `BARBUNYA`, `BOMBAY`)
- Copia versionada en `data/raw/dry_bean.csv` + metadatos de trazabilidad (fecha, hash SHA-256, dimensiones)

## 3. Estructura del repositorio (TDSP)

```
laboratorio_drybean_ml/
├── data/
│   ├── raw/                 # dataset original sin modificar (versionado)
│   └── processed/           # X_train, X_test, y_train, y_test (generados)
├── notebooks/
│   ├── 01_laboratorio_drybean.ipynb
│   └── 02_sprint2_modelado.ipynb
├── outputs/
│   ├── models/              # modelo_final_drybean.joblib + metadatos
│   └── reports/
│       ├── sprint1/         # reporte, JSON de calidad y figuras
│       └── sprint2/         # reporte, métricas, predicciones y figuras
├── src/                     # código reutilizable
│   ├── config.py            # rutas, semilla y parámetros
│   ├── data.py              # descarga y caché del dataset
│   ├── prepare.py           # calidad, limpieza, split y pipeline
│   ├── eda.py               # figuras y reporte exploratorio
│   ├── models.py            # construcción, evaluación y persistencia de modelos
│   ├── evaluate.py          # figuras y reporte de evaluación
│   ├── sprint1.py           # ejecución completa del Sprint 1
│   └── sprint2.py           # ejecución completa del Sprint 2
├── tests/                   # pruebas automáticas (pytest)
├── conftest.py
├── requirements.txt
└── README.md
```

## 4. Entorno

Requiere Python 3.12+.

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Cómo ejecutar

```bash
# Sprint 1: carga, calidad, EDA, split y pipeline
python -m src.sprint1

# Sprint 2: baseline, Random Forest, comparación, figuras y modelo final
python -m src.sprint2

# Pruebas automáticas
pytest -q
```

Ambos notebooks se ejecutan desde la raíz del repositorio y están guardados **con
salidas**, como exige el entregable de notebook ejecutado. El Sprint 2 requiere
los artefactos del Sprint 1 en `data/processed/`.

## 6. Resultados del Sprint 1

| Métrica | Valor |
|---|---|
| Registros originales | 13.611 |
| Nulos | 0 |
| Filas duplicadas eliminadas | 68 |
| Registros para modelado | 13.543 |
| Variables predictoras | 16 |
| Clases | 7 (relación mayor/menor 6,79:1) |
| Train / Test | 10.834 (80%) / 2.709 (20%) |
| Desvío máximo de estratificación | 0,0003 |
| Pipeline | `StandardScaler` (ajustado solo con train) |

Artefactos: `outputs/reports/sprint1/reporte_sprint1.md`, `calidad_datos.json`,
`resumen_sprint1.json` y 4 figuras.

## 7. Resultados del Sprint 2

| Modelo | Accuracy | F1 macro | Balanced accuracy | Entrenamiento |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0,9195 | 0,9306 | 0,9303 | 0,11 s |
| Random Forest (200 árboles, balanced) | 0,9195 | 0,9310 | 0,9310 | 0,38 s |

- Ambos modelos coinciden en **96%** de las predicciones y aciertan 2.491 de 2.709 casos.
- **Decisión CRISP-DM:** el F1 macro difiere solo 0,0004 (< 0,005), por lo que se
  prefiere el modelo más simple y rápido: **Logistic Regression** (criterio del
  laboratorio de priorizar F1 macro y simplicidad).
- Errores principales: `SIRA` ↔ `DERMASON`. `BOMBAY` se clasifica sin errores.
- Artefacto: `outputs/models/modelo_final_drybean.joblib` + `.metadata.json`.

Artefactos: `outputs/reports/sprint2/reporte_sprint2.md`, `metricas_sprint2.json`,
`comparacion_modelos.csv`, `predicciones_test.csv` y 4 figuras.

## 8. Estado del backlog (Scrum ML)

| ID | Historia / tarea | Responsable | Sprint | Estado |
|---|---|---|---|---|
| PB-01 | Descargar y cargar el dataset | Data Engineer | 1 | ✅ Hecho |
| PB-02 | Analizar calidad de datos | Data Analyst | 1 | ✅ Hecho |
| - | Split estratificado + pipeline | ML Engineer | 1 | ✅ Hecho |
| PB-03 | Baseline `LogisticRegression` | ML Engineer | 2 | ✅ Hecho |
| PB-04 | `RandomForestClassifier` balanceado | ML Engineer | 2 | ✅ Hecho |
| PB-05 | Comparación de métricas + matriz de confusión | Equipo | 2 | ✅ Hecho |
| PB-06 | README y documentación final | Scrum Master | 3 | 🟡 En progreso |

**Definition of Done del Sprint 1:** cumplido.
**Definition of Done del Sprint 2:** cumplido (dos modelos, métricas comparadas,
matriz de confusión, importancia de variables y modelo guardado).

## 9. Reproducibilidad

- Semilla fija `RANDOM_STATE = 42` y `TEST_SIZE = 0.2` en `src/config.py`.
- El dataset se versiona con hash SHA-256 en `data/raw/dry_bean_metadata.json`.
- Toda la lógica vive en `src/`; los notebooks solo la invocan y la explican.
- Las pruebas (`pytest`, 17 en total) verifican datos, split, pipeline, modelos,
  métricas, importancia de variables y persistencia.

## 10. Roadmap

- **Sprint 3:** cargar el modelo guardado, función de predicción reutilizable,
  demo local, reporte final de 7 secciones y evidencias Scrum (daily, retrospectiva,
  tablero).

## 11. Créditos

Laboratorio diseñado por Juan Manuel Núñez Velasco. Dataset: Dry Bean Dataset,
UCI Machine Learning Repository.
