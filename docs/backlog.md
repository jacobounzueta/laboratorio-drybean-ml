# Product Backlog - Laboratorio Dry Bean

**Product Goal:** disponer de un clasificador de variedades de frijol seco,
entrenado con las 16 medidas geométricas del dataset UCI 602, evaluado con
métricas adecuadas al desbalance y desplegado como artefacto reutilizable con
trazabilidad completa.

**Equipo:** Jacobo Unzueta y Rommel Prieto (Maestría en IA y Ciencia de Datos, UAO)
**Metodologías:** CRISP-DM (fases), TDSP (estructura del repositorio), Scrum ML (gestión)

---

## 1. Convenciones

| Elemento | Regla |
|---|---|
| Unidad de estimación | Puntos de historia en escala informal de Fibonacci (1, 2, 3, 5, 8) |
| Capacidad por sprint | 25 a 30 puntos para el equipo de dos personas |
| Prioridad | MoSCoW: **M**ust, **S**hould, **C**ould |
| Estados del tablero | `Backlog` → `To Do` → `In Progress` → `Review` → `Done` |
| Numeración | `PB-xx` historias de producto · `T-xx` tareas técnicas del sprint |
| Definition of Done | ver sección 5 |

---

## 2. Product Backlog

| ID | Historia de usuario | Criterio de aceptación | Pts | Prioridad | Sprint | Estado |
|---|---|---|---|---|---|---|
| PB-01 | Como analista quiero cargar el dataset desde su fuente oficial para trabajar con datos trazables | Dataset descargado con `ucimlrepo`, versionado en `data/raw/` con hash SHA-256 y metadatos de origen | 3 | M | 1 | ✅ Done |
| PB-02 | Como analista quiero conocer la calidad y la distribución de los datos para decidir cómo prepararlos | Reporte con nulos, duplicados, tipos, estadísticas, distribución de clases y correlaciones | 3 | M | 1 | ✅ Done |
| PB-03 | Como equipo quiero un modelo baseline interpretable para tener una referencia contra la cual comparar | `StandardScaler` + `LogisticRegression` entrenado y evaluado con accuracy y F1 macro | 5 | M | 2 | ✅ Done |
| PB-04 | Como equipo quiero un modelo alternativo no lineal para capturar relaciones que el baseline no ve | `RandomForestClassifier` balanceado, entrenado y evaluado con las mismas métricas y el mismo split | 5 | M | 2 | ✅ Done |
| PB-05 | Como negocio quiero comparar los modelos con matrices de confusión para saber dónde falla cada uno | Tabla comparativa, matriz de confusión por modelo y análisis de las clases más confundidas | 5 | M | 2 | ✅ Done |
| PB-07 | Como equipo queremos un tercer modelo y un criterio de selección más estable que una sola partición de test | `HistGradientBoosting` con ajuste por CV y decisión basada en F1 macro medio de 5-fold | 8 | S | 2 | ✅ Done |
| PB-08 | Como operario quiero predecir la variedad de un grano nuevo desde una función reutilizable | `load_final_model`, `predict_bean_class`, `predict_beans`, validación de entrada y verificación de integridad | 5 | M | 3 | ✅ Done |
| PB-06 | Como equipo quiero documentar el proyecto y las evidencias Scrum para poder entregarlo y defenderlo | Reporte final de 7 secciones, backlog, actas de sprint, README actualizado y notebooks ejecutados | 5 | M | 3 | ✅ Done |

**Total:** 39 puntos de historia.

> El orden de las filas refleja la prioridad, no la numeración: PB-07 y PB-08 se
> añadieron después de la primera planificación, cuando el sprint review del
> Sprint 1 mostró que hacía falta un criterio de selección más robusto y una fase
> de despliegue explícita.

---

## 3. Sprint Backlogs

### Sprint 1 - Comprensión y preparación de datos

**Sprint Goal:** dejar los datos listos y confiables para modelar.
**Duración:** 1 semana · **Capacidad:** 25 pts · **Comprometido:** 18 pts

| ID | Tarea | Responsable | Pts | Estado |
|---|---|---|---|---|
| PB-01 | Descargar, versionar y documentar el dataset | Data Engineer (Rommel) | 3 | ✅ Done |
| PB-02 | Perfilado de calidad, EDA y figuras | Data Analyst (Jacobo) | 3 | ✅ Done |
| T-01 | Split estratificado 80/20 + pipeline de escalamiento (sin fuga) | ML Engineer (Rommel) | 5 | ✅ Done |
| T-02 | Cuatro figuras del EDA (clases, cajas, correlaciones, áreas) | Data Analyst (Jacobo) | 2 | ✅ Done |
| T-03 | Notebook `01_laboratorio_drybean.ipynb` ejecutado | Jacobo | 2 | ✅ Done |
| T-04 | Pruebas de datos, split y pipeline (7) | Rommel | 3 | ✅ Done |

### Sprint 2 - Modelado y evaluación

**Sprint Goal:** comparar tres modelos y decidir con evidencia cuál pasa a despliegue.
**Duración:** 1 semana · **Capacidad:** 30 pts · **Comprometido:** 31 pts

| ID | Tarea | Responsable | Pts | Estado |
|---|---|---|---|---|
| PB-03 | Baseline `LogisticRegression` | ML Engineer (Rommel) | 5 | ✅ Done |
| PB-04 | `RandomForestClassifier` balanceado | ML Engineer (Rommel) | 5 | ✅ Done |
| PB-05 | Comparación, matrices de confusión e importancia de variables | Equipo | 5 | ✅ Done |
| PB-07 | Tercer modelo + validación cruzada 5-fold como criterio | Rommel | 8 | ✅ Done |
| T-05 | Figuras del sprint (comparación test, comparación CV, 3 matrices, importancia) | Jacobo | 3 | ✅ Done |
| T-06 | Notebook `02_sprint2_modelado.ipynb` ejecutado | Jacobo | 2 | ✅ Done |
| T-07 | Pruebas de modelos, CV, métricas y persistencia (13) | Rommel | 3 | ✅ Done |

### Sprint 3 - Despliegue y documentación

**Sprint Goal:** entregar el modelo como artefacto reutilizable y cerrar la documentación.
**Duración:** 1 semana · **Capacidad:** 30 pts · **Comprometido:** 26 pts

| ID | Tarea | Responsable | Pts | Estado |
|---|---|---|---|---|
| PB-08 | `src/deploy.py` con carga, contrato de entrada y predicción | Rommel | 5 | ✅ Done |
| T-08 | Verificación de integridad y demo (`src/sprint3.py`) | Rommel | 3 | ✅ Done |
| T-09 | Reporte de despliegue y artefactos JSON/CSV | Rommel | 3 | ✅ Done |
| T-10 | Notebook `03_sprint3_despliegue.ipynb` ejecutado | Jacobo | 2 | ✅ Done |
| T-11 | Pruebas de despliegue (20) | Rommel | 3 | ✅ Done |
| PB-06 | Reporte final, backlog, actas Scrum y README | Jacobo | 5 | ✅ Done |
| T-12 | Revisión cruzada del repositorio y de los PRs | Equipo | 5 | ✅ Done |

---

## 4. Velocidad

| Sprint | Comprometido | Completado | Velocidad |
|---|---|---|---|
| 1 | 18 | 18 | 18 |
| 2 | 31 | 31 | 31 |
| 3 | 26 | 26 | 26 |

Promedio: **25 puntos por sprint**. El Sprint 2 concentra el trabajo de modelado
(tres modelos, CV y cinco figuras); el Sprint 1 fue más liviano porque la descarga
y el EDA son actividades cortas con el dataset ya provisto.

---

## 5. Definition of Ready / Definition of Done

**Definition of Ready (una historia entra al sprint si):**

- Tiene un criterio de aceptación verificable.
- Está estimada en puntos.
- Se sabe qué archivo o artefacto produce.
- No depende de una historia que no esté cerrada.

**Definition of Done (una historia se cierra si):**

- [x] El código vive en `src/` (no solo en el notebook).
- [x] El notebook que la explica está ejecutado y guardado con salidas.
- [x] Hay al menos una prueba automática que la cubre.
- [x] `pytest -q` pasa completo (40 pruebas al cierre).
- [x] El resultado quedó escrito en un reporte dentro de `outputs/reports/`.
- [x] La decisión o el hallazgo está documentado con su justificación.
- [x] Los artefactos generados están versionados o correctamente ignorados.
- [x] El commit tiene un mensaje que describe el cambio, no "update".

---

## 6. Impedimentos registrados

| Sprint | Impedimento | Acción tomada |
|---|---|---|
| 1 | La consola de Windows no muestra tildes al imprimir | Los scripts imprimen logs sin tildes; los reportes Markdown sí las llevan |
| 1 | `pd.DataFrame.iterrows()` cambiaba los tipos en pandas 3 | `dataframe_to_markdown` usa `astype(object)` y formato adaptativo |
| 2 | `ConfusionMatrixDisplay.plot()` ya no acepta `normalize=` en scikit-learn 1.9 | La matriz se normaliza a mano antes de graficar |
| 2 | PowerShell devuelve código 1 en `git push` aunque el push funcione (escribe por stderr) | Se verifica con `git ls-remote` |
| 2 | El invitado `stiwardprieto` no había aceptado la invitación al repositorio | Se trabajó con PRs en el repositorio del propietario hasta la aceptación |
| 3 | El modelo final depende de la versión de scikit-learn al deserializar | La versión queda registrada en los metadatos del artefacto |

---

## 7. Riesgos del producto

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Cambio del montaje de captura invalida las 16 variables | Media | Alto | Documentado en las limitaciones; monitoreo de distribución |
| Confusión `SIRA`/`DERMASON` en producción | Alta | Medio | Revisión humana con probabilidad baja |
| El dataset no generaliza a otros lotes | Media | Alto | Validar con datos nuevos antes de automatizar al 100% |
| Pérdida de reproducibilidad por actualización de librerías | Media | Medio | Semilla fija, `requirements.txt` y versión en metadatos |
