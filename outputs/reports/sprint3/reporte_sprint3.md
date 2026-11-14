# Reporte Sprint 3 - Despliegue y verificación

**Proyecto:** Clasificación de variedades de frijol seco (Dry Bean Dataset)
**Metodologías:** CRISP-DM · TDSP · Scrum ML
**Objetivo del sprint:** cargar el modelo guardado, predecir con datos nuevos y
dejar el artefacto listo para una demo reproducible.

## 1. Artefacto cargado

| Campo | Valor |
|---|---|
| Modelo | `HistGradientBoosting` |
| Tipo | `HistGradientBoostingClassifier` |
| Archivo | `outputs/models/modelo_final_drybean.joblib` |
| Metadatos | `outputs/models/modelo_final_drybean.metadata.json` |
| Guardado (UTC) | 2026-11-14T04:16:52+00:00 |
| scikit-learn | 1.9.1 |
| Features | 16 |
| Clases | BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |

| # | Métrica | Valor guardado |
|---|---|---|
| 1 | Accuracy (test) | 0.9195 |
| 2 | F1 macro (test) | 0.9315 |
| 3 | F1 macro CV 5-fold | 0.9429 ± 0.0032 |

El `.joblib` y sus metadatos están versionados en Git (~1,15 MB), así que el
modelo se puede cargar desde un clon limpio sin reentrenar.

## 2. Verificación de integridad

Se volvió a predecir el conjunto de prueba con el modelo cargado desde disco:

| Verificación | Resultado |
|---|---|
| Accuracy guardada vs recalculada | 0.919528 vs 0.919528 |
| Delta de accuracy | 0.00e+00 |
| Delta de F1 macro | 0.00e+00 |
| Features coinciden con los metadatos | sí |
| Clases coinciden con los metadatos | sí |
| Resultado | ✅ artefacto íntegro |

## 3. Demo de predicción

Cinco casos del conjunto de prueba que los modelos nunca vieron: **tres aciertos**
y **un caso de cada una de las dos confusiones más frecuentes**, para que la demo
muestre también los límites reales del clasificador.

| muestra | clase_real | clase_predicha | probabilidad | acierto |
|---|---|---|---|---|
| 1 | SEKER | SEKER | 0.997 | True |
| 2 | DERMASON | DERMASON | 0.999 | True |
| 3 | DERMASON | DERMASON | 0.992 | True |
| 4 | SIRA | DERMASON | 0.668 | False |
| 5 | DERMASON | SIRA | 0.918 | False |

Aciertos en la demo: **3 de 5**.

Errores incluidos: `SIRA` → `DERMASON` y `DERMASON` → `SIRA`. Son variedades de tamaño y forma parecidos
donde el modelo predice con alta confianza y aun así se equivoca: ese es el riesgo
que hay que monitorear en producción.

## 4. Predicción de un grano nuevo

Se simula la llegada de un grano con medidas redondeadas a 2 decimales:

```json
{
    "Area": 40512.0,
    "Perimeter": 737.64,
    "MajorAxisLength": 248.2,
    "MinorAxisLength": 208.53,
    "AspectRatio": 1.19,
    "Eccentricity": 0.54,
    "ConvexArea": 40973.0,
    "EquivDiameter": 227.12,
    "Extent": 0.74,
    "Solidity": 0.99,
    "Roundness": 0.94,
    "Compactness": 0.92,
    "ShapeFactor1": 0.01,
    "ShapeFactor2": 0.0,
    "ShapeFactor3": 0.84,
    "ShapeFactor4": 1.0
}
```

Resultado:

```python
from src import deploy

modelo, metadatos = deploy.load_final_model()
clase = deploy.predict_bean_class(modelo, nuevo_grano)
```

- **Predicción:** `SEKER`
- **Probabilidad:** 0.8927

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
