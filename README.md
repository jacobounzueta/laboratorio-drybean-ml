# Laboratorio Dry Bean - CRISP-DM + TDSP + Scrum ML

Proyecto de clasificación de variedades de frijol seco (**Dry Bean Dataset**, UCI 602)
desarrollado como laboratorio práctico de Machine Learning con metodologías
profesionales de ciencia de datos.

| | |
|---|---|
| **Equipo** | Jacobo Unzueta y Rommel Prieto |
| **Programa** | Maestría en IA y Ciencia de Datos - Universidad Autónoma de Occidente |
| **Metodologías** | CRISP-DM · TDSP · Scrum ML |
| **Estado** | Sprint 1 finalizado · Sprint 2 pendiente |

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
│   └── 01_laboratorio_drybean.ipynb
├── outputs/
│   ├── models/              # modelos entrenados (.joblib)
│   └── reports/
│       └── sprint1/         # reporte, JSON de calidad y figuras
├── src/                     # código reutilizable
│   ├── config.py            # rutas, semilla y parámetros
│   ├── data.py              # descarga y caché del dataset
│   ├── prepare.py           # calidad, limpieza, split y pipeline
│   ├── eda.py               # figuras y reporte exploratorio
│   └── sprint1.py           # ejecución completa del Sprint 1
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
# Sprint 1 completo: descarga/carga, calidad, EDA, split y pipeline
python -m src.sprint1

# Forzar la descarga desde UCI (sobrescribe data/raw)
python -m src.sprint1 --refresh

# Pruebas automáticas
pytest -q
```

El notebook `notebooks/01_laboratorio_drybean.ipynb` se ejecuta desde la raíz del
repositorio y ya está guardado **con salidas**, como exige el entregable 1.

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

Artefactos generados:

- `outputs/reports/sprint1/reporte_sprint1.md` - reporte CRISP-DM del sprint
- `outputs/reports/sprint1/calidad_datos.json` y `resumen_sprint1.json`
- `outputs/reports/sprint1/figuras/` - 4 figuras (clases, histogramas, correlación, área por clase)

Hallazgo principal: las clases están **desbalanceadas** (DERMASON 26% vs BOMBAY 3,8%),
por lo que el Sprint 2 evaluará con **F1 macro** además de accuracy y usará
`class_weight="balanced"` en el modelo alternativo.

## 7. Estado del backlog (Scrum ML)

| ID | Historia / tarea | Responsable | Sprint | Estado |
|---|---|---|---|---|
| PB-01 | Descargar y cargar el dataset | Data Engineer | 1 | ✅ Hecho |
| PB-02 | Analizar calidad de datos (nulos, duplicados, clases) | Data Analyst | 1 | ✅ Hecho |
| - | Split train/test estratificado + pipeline | ML Engineer | 1 | ✅ Hecho |
| PB-03 | Baseline `LogisticRegression` | ML Engineer | 2 | ⏳ Por hacer |
| PB-04 | `RandomForestClassifier` balanceado | ML Engineer | 2 | ⏳ Por hacer |
| PB-05 | Comparación de métricas + matriz de confusión | Equipo | 2 | ⏳ Por hacer |
| PB-06 | README y documentación final | Scrum Master | 3 | 🟡 En progreso |

**Definition of Done del Sprint 1:** cumplido (dataset reproducible, calidad
reportada, split reproducible, pipeline listo, pruebas en verde).

## 8. Reproducibilidad

- Semilla fija `RANDOM_STATE = 42` y `TEST_SIZE = 0.2` en `src/config.py`.
- El dataset se versiona con hash SHA-256 en `data/raw/dry_bean_metadata.json`.
- Toda la lógica vive en `src/`; el notebook solo la invoca y la explica.
- Las pruebas (`pytest`) verifican forma del dataset, limpieza, estratificación,
  reproducibilidad del split y del pipeline.

## 9. Roadmap

- **Sprint 2:** baseline + Random Forest, comparación de métricas, matriz de
  confusión, importancia de variables.
- **Sprint 3:** guardado del modelo (`joblib`), demo de predicción, reporte final
  de 7 secciones y evidencias Scrum (daily, retrospectiva, tablero).

## 10. Créditos

Laboratorio diseñado por Juan Manuel Núñez Velasco. Dataset: Dry Bean Dataset,
UCI Machine Learning Repository.
