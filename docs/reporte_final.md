# Reporte final - Laboratorio Dry Bean

**Proyecto:** Clasificación automática de variedades de frijol seco (Dry Bean Dataset, UCI 602)
**Equipo:** Jacobo Unzueta y Rommel Prieto
**Programa:** Maestría en Inteligencia Artificial y Ciencia de Datos - Universidad Autónoma de Occidente
**Metodologías:** CRISP-DM · TDSP · Scrum ML
**Repositorio:** <https://github.com/jacobounzueta/laboratorio-drybean-ml>

| Entregable | Dónde está |
|---|---|
| 1. Notebook ejecutado | `notebooks/01_laboratorio_drybean.ipynb`, `02_sprint2_modelado.ipynb`, `03_sprint3_despliegue.ipynb` |
| 2. Reporte breve de 7 secciones | este documento (secciones 1 a 7) |
| 3. Evidencias Scrum | `docs/backlog.md` y `docs/sprints.md` (tablero, review y retrospectiva) |

---

## 1. Comprensión del negocio

**Situación.** Una empresa agrícola clasifica a mano muestras de frijol para
control de calidad. La clasificación manual es lenta, subjetiva entre operarios y
difícil de auditar.

**Oportunidad.** El sistema de visión por computador ya captura 16 medidas
geométricas de cada grano (área, perímetro, ejes, factores de forma). Si un modelo
predice la variedad a partir de esas medidas, la clasificación se vuelve inmediata
y reproducible.

**Pregunta de negocio.** ¿Se puede determinar la variedad de un grano de frijol a
partir de sus medidas geométricas con una precisión suficiente para apoyar el
control de calidad?

**Objetivo del proyecto.** Construir, evaluar y desplegar un clasificador
multiclase que prediga la variedad (`Class`) de un grano, con trazabilidad
completa del proceso analítico.

**Criterios de éxito acordados con el negocio**

| Criterio | Umbral | Resultado |
|---|---|---|
| F1 macro en test | ≥ 0,90 | **0,9315** ✅ |
| Accuracy en test | ≥ 0,90 | **0,9195** ✅ |
| Ninguna clase con F1 < 0,80 | obligatorio | mínimo 0,867 (`SIRA`) ✅ |
| El modelo debe cargarse de disco y predecir sin reentrenar | obligatorio | verificado, delta de métricas 0 ✅ |

**¿Por qué F1 macro y no solo accuracy?** La clase mayoritaria (`DERMASON`,
3.546 granos) es 6,79 veces más frecuente que la minoritaria (`BOMBAY`, 522). Un
modelo que ignore las clases pequeñas puede tener accuracy alta y ser inútil en
planta. F1 macro promedia el F1 de las 7 clases con el mismo peso, así que castiga
al modelo que abandona variedades.

**Relación con las metodologías**

| CRISP-DM | TDSP | Scrum ML |
|---|---|---|
| Fase 1: Business Understanding | Business Understanding | Product Goal, Product Backlog (PB-01 a PB-08) |
| — | — | Roles: Product Owner (negocio), Scrum Master (Jacobo), ML Engineer (Rommel) |

---

## 2. Comprensión de los datos

**Fuente.** Dry Bean Dataset (UCI 602), donado por el grupo de investigación
Selcuk de visión por computador. Se descarga con `ucimlrepo` y se versiona en
`data/raw/dry_bean.csv` con hash SHA-256 en `data/raw/dry_bean_metadata.json`.

**Estructura.** 13.611 instancias × 17 columnas: 16 variables numéricas continuas
y la etiqueta `Class` con 7 variedades.

| Clase | Granos | % |
|---|---|---|
| DERMASON | 3.546 | 26,05 |
| SIRA | 2.636 | 19,37 |
| SEKER | 2.027 | 14,89 |
| HOROZ | 1.928 | 14,17 |
| CALI | 1.630 | 11,98 |
| BARBUNYA | 1.322 | 9,71 |
| BOMBAY | 522 | 3,84 |

**Hallazgos del análisis exploratorio**

1. **Desbalance moderado** (6,79:1). No exige remuestreo, pero sí F1 macro y
   `class_weight="balanced"` en el modelo de árboles.
2. **Escalas muy distintas**: `Area` va de 20.420 a 254.616, mientras
   `Solidity` vive entre 0,919 y 0,995. Los modelos lineales necesitan
   escalamiento; los árboles, no.
3. **Correlaciones casi perfectas dentro del bloque de tamaño**:
   `Area`–`ConvexArea` r = 1,000; `Compactness`–`ShapeFactor3` r = 0,999;
   `Perimeter`–`EquivDiameter` r = 0,991. Hay redundancia: 16 variables no
   aportan 16 dimensiones reales de información.
4. **`BOMBAY` es un outlier geométrico**: es la clase más pequeña y alargada,
   claramente separada del resto en los diagramas de caja. Esto anticipa que será
   la clase más fácil de reconocer, aunque sea la minoritaria.
5. **`SIRA` y `DERMASON` se solapan** en tamaño y forma, lo que anticipa la
   confusión que después aparece en todos los modelos.

**Calidad de los datos**

| Chequeo | Resultado |
|---|---|
| Valores nulos | 0 |
| Columnas con nulos | 0 |
| Filas duplicadas exactas | 68 |
| Tipos incorrectos | 0 |
| Registros tras limpieza | **13.543** |

Evidencias: `outputs/reports/sprint1/reporte_sprint1.md`,
`outputs/reports/sprint1/calidad_datos.json` y 4 figuras en
`outputs/reports/sprint1/figuras/`.

---

## 3. Preparación de los datos

| Decisión | Justificación |
|---|---|
| Eliminar 68 duplicados exactos | Son copias idénticas de la misma medición; inflarían el test y el train con el mismo grano. Se documenta el conteo, no se oculta. |
| No imputar | No hay nulos: imputar sería inventar información. |
| No eliminar outliers | Los valores extremos son granos reales (p. ej. `BOMBAY`); recortarlos borraría justamente la clase minoritaria. Los árboles son robustos a ellos. |
| `StandardScaler` dentro del pipeline | Se ajusta **solo con train**, así que la media y la desviación del test no contaminan el entrenamiento (sin fuga de información). |
| Split 80/20 estratificado | 10.834 train / 2.709 test, con desviación máxima de estratificación de 0,0003. La estratificación conserva la proporción de las 7 clases en ambos conjuntos. |
| Semilla `RANDOM_STATE = 42` | Cualquier integrante reproduce exactamente el mismo split y los mismos modelos. |
| El escalamiento solo se aplica a la regresión logística | Random Forest y HistGradientBoosting no lo necesitan; aplicarlo a todos impediría comparar el efecto del algoritmo y no del preprocesamiento. |

**Salidas versionadas**: `data/processed/X_train.csv`, `X_test.csv`,
`y_train.csv`, `y_test.csv` y `split_summary.json` (generados por
`python -m src.sprint1`).

**Relación con las metodologías.** Esta fase cierra la etapa *Data Preparation*
de TDSP y la fase 3 de CRISP-DM; en Scrum corresponde a las tareas técnicas del
Sprint 1 (T-01) y a la Definition of Done de PB-01 y PB-02.

---

## 4. Modelado

Se entrenaron tres modelos con el mismo split, la misma semilla y las mismas 16
variables. La comparación es, por tanto, atribuible al algoritmo.

| Modelo | Configuración | Rol |
|---|---|---|
| **Logistic Regression** | `StandardScaler` + `LogisticRegression(max_iter=1000)` | Baseline interpretable y barato |
| **Random Forest** | `n_estimators=200`, `class_weight="balanced"` | No lineal, robusto al desbalance, da importancia de variables |
| **HistGradientBoosting** | `learning_rate=0.1`, `max_iter=100`, `l2_regularization=1.0` | Boosting regularizado, buen desempeño con datos tabulares |

**Protocolo de selección.** El conjunto de prueba tiene 2.709 filas: una sola
partición no distingue diferencias de milésimas (el intervalo de confianza de una
accuracy de 0,92 con n = 2.709 es de aproximadamente ±0,01). Por eso:

1. **Hold-out** para reportar el desempeño final honesto.
2. **Validación cruzada estratificada 5-fold sobre train** para **decidir** entre
   modelos: 5 estimaciones independientes en lugar de una.
3. **Criterio**: mayor F1 macro medio en CV; si la diferencia es menor que 0,005,
   se prefiere el modelo más simple y rápido de entrenar.

**Ajuste de hiperparámetros.** Se exploraron variantes de HGB con CV 5-fold:

| Variante | F1 macro CV | F1 macro test |
|---|---|---|
| **`l2_regularization=1.0`** | **0,9429 ± 0,0032** | **0,9315** |
| `max_leaf_nodes=63` | 0,9427 ± 0,0020 | 0,9319 |
| Configuración por defecto | 0,9419 ± 0,0026 | 0,9303 |
| `learning_rate=0.05`, `max_iter=400` | 0,9415 ± 0,0029 | 0,9307 |
| `l2=1.0` + `max_leaf_nodes=63` | 0,9423 ± 0,0025 | 0,9312 |

Se eligió `l2_regularization=1.0` por ser la mejor en CV y la más simple. Las
diferencias entre variantes son menores que su propia desviación estándar: no hay
evidencia de que una configuración sea claramente superior, y se documenta en vez
de afirmar lo contrario.

**Experimentos descartados (no incorporados al repositorio)**

| Candidato | Accuracy | F1 macro | F1 macro CV |
|---|---|---|---|
| ExtraTrees (300, balanced) | 0,9162 | 0,9285 | 0,9325 |
| KNN (k=5, escalado) | 0,9155 | 0,9270 | 0,9344 |

Ambos quedaron por debajo de los tres modelos finales en F1 macro. Se reportan
como análisis exploratorio para mostrar que la elección no fue arbitraria.

**Relación con las metodologías.** Fase 4 de CRISP-DM (*Modeling*) y etapa
*Modeling* de TDSP; en Scrum son PB-03, PB-04 y PB-07 más las tareas de
experimentación del Sprint 2.

---

## 5. Evaluación

### 5.1 Resultados en el conjunto de prueba (2.709 granos no vistos)

| Modelo | Accuracy | F1 macro | F1 weighted | Balanced accuracy | Entrenamiento |
|---|---|---|---|---|---|
| Logistic Regression | 0,9195 | 0,9306 | 0,9197 | 0,9303 | 0,10 s |
| Random Forest | 0,9195 | 0,9310 | 0,9195 | 0,9310 | 0,36 s |
| **HistGradientBoosting** | **0,9195** | **0,9315** | 0,9195 | 0,9307 | 0,43 s |

Los tres modelos aciertan exactamente los mismos 2.491 de 2.709 granos
(218 errores) y coinciden entre el 95,4% y el 96,7% de las predicciones. Ese
empate no es casualidad: indica que **el límite lo ponen las variables, no el
algoritmo**.

### 5.2 Validación cruzada 5-fold sobre train (criterio de decisión)

| Modelo | F1 macro medio | Desv. estándar | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|---|---|
| **HistGradientBoosting** | **0,9429** | 0,0032 | 0,9422 | 0,9432 | 0,9434 | 0,9368 | 0,9440 |
| Logistic Regression | 0,9360 | 0,0031 | 0,9378 | 0,9337 | 0,9336 | 0,9336 | 0,9412 |
| Random Forest | 0,9352 | 0,0016 | 0,9373 | 0,9337 | 0,9344 | 0,9337 | 0,9368 |

HGB gana en **las 5 particiones**, no solo en el promedio. La diferencia con el
baseline (0,0069) es mayor que la desviación estándar de ambos, así que la
decisión es defendible.

### 5.3 Desempeño por clase (HistGradientBoosting)

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BARBUNYA | 0,923 | 0,902 | 0,912 | 265 |
| BOMBAY | 1,000 | 1,000 | **1,000** | 104 |
| CALI | 0,922 | 0,945 | 0,933 | 326 |
| DERMASON | 0,907 | 0,918 | 0,912 | 709 |
| HOROZ | 0,959 | 0,941 | 0,950 | 372 |
| SEKER | 0,946 | 0,946 | 0,946 | 406 |
| SIRA | 0,870 | 0,863 | **0,867** | 527 |

### 5.4 ¿Qué clases se confunden más entre sí?

| Real | Predicho | Casos | % de la clase real |
|---|---|---|---|
| SIRA | DERMASON | 53 | 10,1% |
| DERMASON | SIRA | 47 | 6,6% |
| BARBUNYA | CALI | 16 | 6,0% |
| DERMASON | SEKER | 10 | 1,4% |
| HOROZ | CALI | 9 | 2,4% |

**Respuesta a la pregunta del laboratorio.** La confusión dominante es
`SIRA` ↔ `DERMASON` en ambos sentidos (100 de los 218 errores, el 45,9%).
Son variedades de tamaño y forma muy parecidos, y el solapamiento ya se veía en el
EDA. En el extremo opuesto, `BOMBAY` se clasifica sin un solo error pese a ser la
clase con menos ejemplos: sus medidas geométricas son inconfundibles. Es decir,
**el problema no es el desbalance sino la similitud entre variedades concretas**.

### 5.5 Importancia de variables (Random Forest, impureza de Gini)

| # | Variable | Importancia |
|---|---|---|
| 1 | Perimeter | 0,1092 |
| 2 | ShapeFactor1 | 0,1021 |
| 3 | Compactness | 0,0952 |
| 4 | ShapeFactor3 | 0,0939 |
| 5 | MinorAxisLength | 0,0929 |

Ninguna variable domina (la mayor pesa 10,9%): el modelo combina tamaño y forma.
Esto es coherente con el EDA y con la existencia de correlaciones altas dentro del
bloque de tamaño. La importancia por impureza **no prueba causalidad** ni el signo
del efecto; solo indica qué variables usó el bosque para separar clases.

### 5.6 Decisión CRISP-DM

- **Modelo seleccionado:** `HistGradientBoostingClassifier(l2_regularization=1.0)`.
- **Métrica de decisión:** F1 macro medio de la validación cruzada 5-fold.
- **Diferencia con el segundo:** +0,0069 (0,9429 vs 0,9360).
- **Justificación:** gana en las 5 particiones y también en F1 macro de test; su
  costo de entrenamiento (0,43 s) es irrelevante para este tamaño de datos.

### 5.7 Riesgos de esta evaluación

1. **Una sola partición de test.** Mitigado con CV, no eliminado: no hay conjunto
   de validación externo ni corte temporal.
2. **Mismo origen de datos.** Todas las imágenes vienen del mismo montaje; un
   cambio de cámara o iluminación puede alterar las medidas.
3. **Sin costo de error.** El negocio no definió cuánto cuesta confundir `SIRA`
   con `DERMASON` frente a confundir `BOMBAY`; sin esa matriz de costos no se puede
   optimizar el umbral.
4. **Sin calibración de probabilidades.** Las probabilidades son útiles para
   priorizar revisión humana, pero no se calibraron (`CalibratedClassifierCV`).

Evidencias: `outputs/reports/sprint2/reporte_sprint2.md`,
`metricas_sprint2.json`, `comparacion_modelos.csv`, `predicciones_test.csv` y 6
figuras.

---

## 6. Despliegue

**Qué se entrega.** Un artefacto versionado, no un notebook:

| Elemento | Ruta |
|---|---|
| Modelo serializado | `outputs/models/modelo_final_drybean.joblib` (~1,15 MB, versionado en Git) |
| Metadatos de trazabilidad | `outputs/models/modelo_final_drybean.metadata.json` (versionado en Git) |
| Funciones de inferencia | `src/deploy.py` |
| Verificación automática | `src/sprint3.py` → `outputs/reports/sprint3/` |

**Contrato del artefacto**

- Entrada: 16 variables numéricas en el orden registrado en los metadatos
  (`Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRatio`,
  `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `Roundness`,
  `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4`).
- Salida: una de 7 clases (`BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`,
  `SEKER`, `SIRA`) y la probabilidad de la clase elegida.

**API de inferencia**

```python
from src import deploy

modelo, metadatos = deploy.load_final_model()

# Un grano (dict, Series o fila de DataFrame)
clase = deploy.predict_bean_class(modelo, {
    "Area": 40512.0, "Perimeter": 737.64, "MajorAxisLength": 248.2,
    "MinorAxisLength": 208.53, "AspectRatio": 1.19, "Eccentricity": 0.54,
    "ConvexArea": 40973.0, "EquivDiameter": 227.12, "Extent": 0.74,
    "Solidity": 0.99, "Roundness": 0.94, "Compactness": 0.92,
    "ShapeFactor1": 0.01, "ShapeFactor2": 0.0, "ShapeFactor3": 0.84,
    "ShapeFactor4": 1.0,
})
# 'SEKER'

# Varios granos, con probabilidad
lotes = deploy.predict_beans(modelo, lista_o_dataframe)
# [{'prediccion': 'SEKER', 'probabilidad': 0.8927, 'clase_mas_probable': 'SEKER'}, ...]
```

**Controles implementados**

| Riesgo operativo | Control |
|---|---|
| Modelo inexistente | `ModeloNoDisponibleError` con instrucción de cómo generarlo |
| Variables faltantes o mal escritas | `DatosInvalidosError` que lista los nombres exactos |
| Valores nulos o no numéricos | Rechazo explícito antes de llamar a scikit-learn |
| Archivo corrupto o de otro entrenamiento | Recálculo de métricas en test y comparación contra los metadatos (delta 0,00e+00) |
| Pérdida de reproducibilidad | Metadatos con semilla, versión de scikit-learn, filas de train/test, features y clases |

**Verificación ejecutada.** `python -m src.sprint3` recarga el `.joblib`,
recalcula accuracy y F1 macro sobre las 2.709 filas de test y confirma que
coinciden exactamente con lo guardado. Después predice un lote y un grano nuevo, y
guarda `outputs/reports/sprint3/reporte_sprint3.md`, `demo_predicciones.csv` y
`verificacion_modelo.json`.

**Cómo llevarlo a producción**

1. Reentrenar con `python -m src.sprint2` cuando cambien los datos o el montaje de
   captura; la semilla fija garantiza comparabilidad.
2. Versionar el `.joblib` junto con sus metadatos y la versión de scikit-learn.
3. Exponer `predict_bean_class` detrás de un endpoint o un proceso por lote.
4. Monitorear la distribución de entrada (media y varianza de las 16 variables):
   un cambio de cámara puede degradar el modelo sin que las métricas históricas lo
   detecten.
5. Enviar a revisión humana los casos `SIRA`/`DERMASON` con probabilidad baja.

**Relación con las metodologías.** Fase 6 de CRISP-DM (*Deployment*), etapa
*Deployment* de TDSP y PB-08 en Scrum. El ciclo no se cierra del todo: en CRISP-DM
el despliegue alimenta una nueva vuelta de comprensión del negocio, que aquí queda
como trabajo futuro.

---

## 7. Conclusiones, limitaciones y trabajo futuro

### 7.1 Conclusiones

1. **El problema es resoluble con las variables disponibles**: los tres modelos
   superan el 91,9% de accuracy y el 93,0% de F1 macro, por encima del umbral de
   éxito pactado.
2. **El algoritmo no es el cuello de botella**: los tres modelos aciertan los
   mismos 2.491 granos. Probar arquitecturas más complejas no mejoraría el
   resultado de forma defendible.
3. **La validación cruzada cambió la decisión**: con una sola partición de test
   los tres modelos empataban; con 5-fold, HGB gana en todas las particiones. Es
   más honesto decidir con 5 estimaciones que con una.
4. **El error está localizado**: `SIRA` ↔ `DERMASON` concentra el 45,9% de los
   errores, y `BOMBAY` se clasifica perfectamente pese a ser la clase minoritaria.
   El desbalance no era el problema; la similitud geométrica sí.
5. **La información está en la combinación de medidas**, no en una variable
   estrella: la más importante apenas pesa 10,9%.
6. **El artefacto es verificable**: se recarga de disco y reproduce las métricas
   con delta cero, lo que elimina el riesgo de desplegar el modelo equivocado.
7. **La trazabilidad se sostiene sola**: cada decisión tiene un commit, un test y
   un reporte asociado; cualquier integrante puede reproducir el laboratorio con
   tres comandos.

### 7.2 Limitaciones

| Limitación | Impacto | Mitigación actual |
|---|---|---|
| 16 variables de un único montaje de visión | El modelo no es portable a otra cámara o iluminación | Documentado en el contrato y en las limitaciones del reporte de despliegue |
| Sin matriz de costos del negocio | No se puede elegir el umbral óptimo | Se reportan probabilidades para priorizar revisión humana |
| Test de una sola partición | Intervalos de confianza anchos (±1 pp) | CV 5-fold como criterio de decisión |
| Sin calibración de probabilidades | Las probabilidades no son frecuencias fiables | Se usan solo para ordenar casos dudosos |
| Sin datos nuevos post-entrenamiento | No se midió degradación (*drift*) | Recomendación de monitoreo de distribución |
| Equipo de dos personas en tres sprints | No hubo tiempo para *tuning* exhaustivo ni SHAP | Se priorizó el ciclo completo sobre la optimización fina |

### 7.3 Trabajo futuro

1. **Ampliar la captura**: granos de otros lotes, cosechas y equipos para medir
   generalización real y *data drift*.
2. **Modelo sensible al costo**: incorporar la matriz de costos del negocio y
   optimizar el umbral de decisión por clase.
3. **Calibración**: `CalibratedClassifierCV` para que la probabilidad sea
   interpretable como confianza operativa.
4. **Explicabilidad local**: SHAP por predicción, para que el operario vea qué
   medida inclinó la decisión.
5. **Servicio de inferencia**: endpoint con validación de esquema (p. ej.
   FastAPI + Pydantic) y registro de predicciones para auditoría.
6. **Automatización del reentrenamiento**: pipeline que reentrene, valide contra
   un umbral mínimo de F1 macro y solo publique si mejora.

### 7.4 Lecciones del equipo

- **Medir antes de optimizar**: el experimento con ExtraTrees y KNN evitó una hora
  de trabajo en modelos que no iban a ganar.
- **El test pequeño engaña**: decidir con una sola partición nos habría hecho
  elegir el modelo más simple por una diferencia de 0,0004 que era ruido.
- **Documentar los descartes** da más credibilidad que presentar solo el ganador.
- **Guardar metadatos** cuesta cinco minutos y evita el error clásico de cargar un
  `.joblib` que espera otras columnas.

---

## Anexo A. Trazabilidad CRISP-DM ↔ TDSP ↔ Scrum

| Fase CRISP-DM | Etapa TDSP | Artefacto Scrum | Evidencia |
|---|---|---|---|
| 1. Comprensión del negocio | Business Understanding | Product Goal y backlog | `docs/backlog.md`, sección 1 de este reporte |
| 2. Comprensión de los datos | Data Understanding | PB-01, PB-02 | `outputs/reports/sprint1/`, notebook 01 |
| 3. Preparación de los datos | Data Preparation | T-01 (split y pipeline) | `data/processed/`, `src/prepare.py` |
| 4. Modelado | Modeling | PB-03, PB-04, PB-07 | `src/models.py`, `src/sprint2.py` |
| 5. Evaluación | Modeling / Evaluation | PB-05, sprint review 2 | `outputs/reports/sprint2/` |
| 6. Despliegue | Deployment | PB-08, sprint review 3 | `src/deploy.py`, `outputs/reports/sprint3/` |
| Cierre y aprendizaje | — | Retrospectivas | `docs/sprints.md` |

## Anexo B. Cómo reproducir todo

```bash
python -m src.sprint1    # datos, calidad, EDA, split
python -m src.sprint2    # tres modelos, CV, evaluación, artefacto
python -m src.sprint3    # carga, verificación y demo de despliegue
pytest -q                # 40 pruebas
```

Los tres notebooks (`notebooks/01`, `02`, `03`) están guardados con sus salidas y
se ejecutan en orden desde la raíz del repositorio.

## Anexo C. Glosario

- **CRISP-DM**: proceso estándar de minería de datos en 6 fases iterativas.
- **TDSP**: Team Data Science Process de Microsoft; estructura el repositorio en
  datos, código, notebooks, modelos y reportes.
- **Scrum ML**: adaptación de Scrum a proyectos de machine learning, con sprints
  cortos, backlog priorizado y revisión de resultados con el negocio.
- **F1 macro**: promedio no ponderado del F1 de cada clase; trata todas las clases
  por igual.
- **Hold-out**: conjunto de prueba reservado que el modelo nunca ve durante el
  entrenamiento ni la selección.
- **Leakage (fuga de información)**: contaminación del entrenamiento con
  información del test; aquí se evita ajustando el escalador solo con train.
