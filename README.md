# Laboratorio Dry Bean - CRISP-DM + TDSP + Scrum ML

Proyecto de clasificación de variedades de frijol seco (**Dry Bean Dataset**, UCI 602)
desarrollado como laboratorio práctico de Machine Learning con metodologías
profesionales de ciencia de datos.

| | |
|---|---|
| **Equipo** | Jacobo Unzueta y Rommel Prieto |
| **Programa** | Maestría en IA y Ciencia de Datos - Universidad Autónoma de Occidente |
| **Metodologías** | CRISP-DM · TDSP · Scrum ML |
| **Estado** | Sprints 1, 2 y 3 finalizados |

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
│   ├── 02_sprint2_modelado.ipynb
│   └── 03_sprint3_despliegue.ipynb
├── outputs/
│   ├── models/              # modelo_final_drybean.joblib + metadatos
│   └── reports/
│       ├── sprint1/         # reporte, JSON de calidad y figuras
│       ├── sprint2/         # reporte, métricas, predicciones y figuras
│       └── sprint3/         # reporte, demo y verificación del artefacto
├── docs/
│   ├── reporte_final.md     # reporte CRISP-DM de 7 secciones
│   ├── backlog.md           # product backlog, sprint backlogs y DoD
│   └── sprints.md           # tablero, sprint reviews y retrospectivas
├── src/                     # código reutilizable
│   ├── config.py            # rutas, semilla y parámetros
│   ├── data.py              # descarga y caché del dataset
│   ├── prepare.py           # calidad, limpieza, split y pipeline
│   ├── eda.py               # figuras y reporte exploratorio
│   ├── models.py            # construcción, evaluación y persistencia de modelos
│   ├── evaluate.py          # figuras y reporte de evaluación
│   ├── deploy.py            # carga del modelo y predicción con datos nuevos
│   ├── sprint1.py           # ejecución completa del Sprint 1
│   ├── sprint2.py           # ejecución completa del Sprint 2
│   └── sprint3.py           # ejecución completa del Sprint 3
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

# Sprint 2: tres modelos, validación cruzada, figuras y modelo final
python -m src.sprint2

# Sprint 3: carga del modelo, verificación de integridad y demo
python -m src.sprint3

# Pruebas automáticas
pytest -q
```

Los tres notebooks se ejecutan desde la raíz del repositorio y están guardados
**con salidas**, como exige el entregable de notebook ejecutado. Cada sprint
requiere los artefactos del anterior (`data/processed/` para el Sprint 2 y
`outputs/models/` para el Sprint 3).

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

Tres modelos sobre el mismo split, con validación cruzada 5-fold para decidir
(el test tiene 2.709 filas: diferencias de milésimas son ruido).

| Modelo | Accuracy (test) | F1 macro (test) | F1 macro CV 5-fold | Entrenamiento |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0,9195 | 0,9306 | 0,9360 ± 0,0031 | 0,10 s |
| Random Forest (200, balanced) | 0,9195 | 0,9310 | 0,9352 ± 0,0016 | 0,36 s |
| **HistGradientBoosting (l2=1.0)** | 0,9195 | **0,9315** | **0,9429 ± 0,0032** | 0,43 s |

- Los tres modelos coinciden en ~96% de las predicciones.
- **Decisión CRISP-DM:** `HistGradientBoosting` supera al baseline por 0,0069 de
  F1 macro medio en CV (gana en las 5 particiones), así que pasa a despliegue.
- Errores principales: `SIRA` ↔ `DERMASON`. `BOMBAY` se clasifica sin errores.
- Artefacto: `outputs/models/modelo_final_drybean.joblib` + `.metadata.json`.

Artefactos: `outputs/reports/sprint2/reporte_sprint2.md`, `metricas_sprint2.json`,
`comparacion_modelos.csv`, `predicciones_test.csv` y 6 figuras.

## 8. Resultados del Sprint 3

El modelo se carga desde disco, se verifica y se usa con datos nuevos a través de
`src/deploy.py`.

| Verificación | Resultado |
|---|---|
| Accuracy guardada vs recalculada | 0,919528 vs 0,919528 (delta 0,00e+00) |
| Delta de F1 macro | 0,00e+00 |
| Variables y clases vs metadatos | coinciden (16 variables, 7 clases) |
| Modelo en disco vs reentrenado | 100% de predicciones idénticas |
| Demo con casos reales | 3 aciertos y los 2 errores más frecuentes (`SIRA` ↔ `DERMASON`) |

```python
from src import deploy

modelo, metadatos = deploy.load_final_model()
clase = deploy.predict_bean_class(modelo, medidas_del_grano)   # un grano
lote = deploy.predict_beans(modelo, dataframe_o_lista)         # varios, con probabilidad
```

La entrada se valida antes de predecir: variables faltantes, variables
desconocidas, nulos, valores no numéricos y entrada vacía lanzan
`DatosInvalidosError` con un mensaje que dice qué corregir.

El `.joblib` y sus metadatos **están versionados en Git** (~1,15 MB), así que el
modelo se puede cargar desde un clon limpio sin reentrenar. `data/processed/`
sigue siendo derivado y se regenera con `python -m src.sprint1`.

Artefactos: `outputs/reports/sprint3/reporte_sprint3.md`,
`demo_predicciones.csv`, `verificacion_modelo.json` y el notebook 03.

## 9. Documentación

- [`docs/reporte_final.md`](docs/reporte_final.md): reporte CRISP-DM de 7 secciones,
  con trazabilidad a TDSP y Scrum y las evidencias de cada fase.
- [`docs/backlog.md`](docs/backlog.md): product backlog, sprint backlogs, velocidad,
  Definition of Ready/Done, impedimentos y riesgos.
- [`docs/sprints.md`](docs/sprints.md): tablero por sprint, sprint reviews,
  retrospectivas y acciones de mejora.
- [`docs/presentacion.md`](docs/presentacion.md): guion del pitch de 15 minutos,
  reparto de intervenciones entre los dos integrantes, demo en vivo y preguntas
  probables.

## 10. Estado del backlog (Scrum ML)

| ID | Historia / tarea | Responsable | Sprint | Estado |
|---|---|---|---|---|
| PB-01 | Descargar y cargar el dataset | Data Engineer | 1 | ✅ Hecho |
| PB-02 | Analizar calidad de datos | Data Analyst | 1 | ✅ Hecho |
| T-01 | Split estratificado + pipeline | ML Engineer | 1 | ✅ Hecho |
| PB-03 | Baseline `LogisticRegression` | ML Engineer | 2 | ✅ Hecho |
| PB-04 | `RandomForestClassifier` balanceado | ML Engineer | 2 | ✅ Hecho |
| PB-05 | Comparación de métricas + matriz de confusión | Equipo | 2 | ✅ Hecho |
| PB-07 | Tercer modelo (HGB) + validación cruzada | ML Engineer | 2 | ✅ Hecho |
| PB-08 | Despliegue y función de predicción | ML Engineer | 3 | ✅ Hecho |
| PB-06 | README y documentación final | Scrum Master | 3 | ✅ Hecho |

**Definition of Done del Sprint 1:** cumplido.
**Definition of Done del Sprint 2:** cumplido (tres modelos, métricas comparadas,
validación cruzada, matriz de confusión, importancia de variables y modelo guardado).
**Definition of Done del Sprint 3:** cumplido (modelo cargado y verificado, función
de predicción con validación de entrada, demo, reporte final y evidencias Scrum).

## 11. Reproducibilidad

- Semilla fija `RANDOM_STATE = 42` y `TEST_SIZE = 0.2` en `src/config.py`.
- El dataset se versiona con hash SHA-256 en `data/raw/dry_bean_metadata.json`.
- El modelo final (`outputs/models/modelo_final_drybean.joblib`) y sus metadatos
  se versionan, de modo que se puede cargar desde un clon limpio sin reentrenar.
- Toda la lógica vive en `src/`; los notebooks solo la invocan y la explican.
- El artefacto guarda sus metadatos: modelo, métricas, variables, clases, filas de
  train/test, semilla y versión de scikit-learn.
- Las pruebas (`pytest`, 40 en total) verifican datos, split, pipeline, modelos,
  validación cruzada, métricas, importancia de variables, persistencia, contrato
  de entrada y verificación del artefacto desplegado.

## 12. Trabajo futuro

1. Validar con granos de otros lotes y equipos de captura para medir degradación.
2. Incorporar la matriz de costos del negocio y calibrar probabilidades.
3. Explicabilidad local (SHAP) por predicción.
4. Endpoint de inferencia con validación de esquema y registro de predicciones.
5. Reentrenamiento automatizado que solo publique si mejora el F1 macro.

## 13. Créditos

Laboratorio diseñado por Juan Manuel Núñez Velasco. Dataset: Dry Bean Dataset,
UCI Machine Learning Repository.
