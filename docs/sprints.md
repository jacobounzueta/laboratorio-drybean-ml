# Evidencias Scrum - Sprints, reviews y retrospectivas

**Proyecto:** Laboratorio Dry Bean · **Equipo:** Jacobo Unzueta y Rommel Prieto
**Marcos:** Scrum ML (gestión) · CRISP-DM (fases analíticas) · TDSP (repositorio)

Este documento reúne las evidencias de gestión: objetivo y tablero de cada
sprint, acta de sprint review, retrospectiva y acciones de mejora. El detalle de
las historias y su estimación está en [`backlog.md`](backlog.md).

---

## Sprint 1 - Comprensión y preparación de datos

**Sprint Goal:** dejar los datos listos, documentados y confiables para modelar.

### Tablero al cierre

| Backlog | To Do | In Progress | Review | Done |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 6 (PB-01, PB-02, T-01, T-02, T-03, T-04) |

### Sprint Review

**Demostración**

- `python -m src.sprint1` corre de principio a fin: descarga, calidad, EDA, split
  y pipeline.
- `outputs/reports/sprint1/` con reporte, JSON de calidad y 4 figuras.
- Notebook `01_laboratorio_drybean.ipynb` ejecutado con salidas.
- 7 pruebas en verde.

**Resultados presentados**

| Métrica | Valor |
|---|---|
| Registros originales | 13.611 |
| Nulos | 0 |
| Duplicados eliminados | 68 |
| Registros para modelado | 13.543 |
| Train / Test | 10.834 / 2.709 |
| Desviación máxima de estratificación | 0,0003 |

**Decisiones tomadas en la review**

1. Se acepta el dataset y la limpieza; no se imputa nada porque no hay nulos.
2. Se acuerda reportar **F1 macro** además de accuracy por el desbalance 6,79:1.
3. Se acuerda comparar al menos **dos modelos** y añadir **matriz de confusión**,
   no solo métricas agregadas.
4. **Nuevo requisito del negocio:** el modelo debe poder usarse con datos nuevos,
   no quedarse en el notebook → se crea PB-08 para el Sprint 3.

**Feedback recibido**

> "Está bien documentado, pero falta demostrar que el modelo sirve fuera del
> notebook y justificar la elección entre modelos con algo más que accuracy."

Ese comentario origina PB-07 (validación cruzada) y PB-08 (despliegue).

### Retrospectiva

| Categoría | Observación |
|---|---|
| ✅ Funcionó | Versionar el dataset con hash; separar lógica en `src/` desde el inicio; los reportes generados por código en vez de escritos a mano |
| ⚠️ No funcionó | Se perdió tiempo con acentos rotos en la consola de Windows; el primer `git push` no se verificó y pareció fallar |
| 💡 Idea | Añadir un `conftest.py` que resuelva el `sys.path` para que `pytest` corra sin configuración extra |

**Acciones comprometidas**

| Acción | Responsable | Estado |
|---|---|---|
| Logs sin tildes en scripts; tildes solo en Markdown | Jacobo | ✅ Hecho |
| `conftest.py` en la raíz | Rommel | ✅ Hecho |
| Verificar cada push con `git ls-remote` | Jacobo | ✅ Hecho |

---

## Sprint 2 - Modelado y evaluación

**Sprint Goal:** comparar tres modelos con el mismo split y decidir con evidencia
cuál pasa a despliegue.

### Tablero al cierre

| Backlog | To Do | In Progress | Review | Done |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 7 (PB-03, PB-04, PB-05, PB-07, T-05, T-06, T-07) |

### Sprint Review

**Demostración**

- Tres modelos entrenados y evaluados sobre las mismas 2.709 filas de test.
- Validación cruzada 5-fold sobre train como criterio de decisión.
- Tres matrices de confusión, comparación de métricas e importancia de variables.
- Modelo final guardado con `joblib` + metadatos de trazabilidad.
- Notebook `02_sprint2_modelado.ipynb` ejecutado con 6 figuras.
- 20 pruebas en verde.

**Resultados presentados**

| Modelo | Accuracy | F1 macro | F1 macro CV 5-fold |
|---|---|---|---|
| Logistic Regression | 0,9195 | 0,9306 | 0,9360 ± 0,0031 |
| Random Forest | 0,9195 | 0,9310 | 0,9352 ± 0,0016 |
| **HistGradientBoosting** | **0,9195** | **0,9315** | **0,9429 ± 0,0032** |

**Decisiones tomadas en la review**

1. Se acepta **HistGradientBoosting** como modelo final: gana en las 5
   particiones de la validación cruzada (+0,0069 sobre el baseline).
2. Se acepta la validación cruzada como **criterio de selección** y el hold-out
   como **verificación final**; queda escrito en el reporte y en el README.
3. Se descartan ExtraTrees y KNN: quedaron por debajo en F1 macro. Se documentan
   como análisis exploratorio, no se incorporan al código.
4. **Riesgo aceptado:** las diferencias entre modelos son menores que el error de
   muestreo del test; se comunica explícitamente en lugar de prometer una mejora
   que los datos no sostienen.

**Feedback recibido**

> "La comparación es honesta, pero si el modelo va a producción necesito ver que
> el archivo guardado se puede volver a cargar y que predice igual."

Ese comentario se convierte en el criterio de aceptación central de PB-08.

### Retrospectiva

| Categoría | Observación |
|---|---|
| ✅ Funcionó | Decidir con CV en vez de con una sola partición; los experimentos temporales en `%TEMP%` antes de tocar el repo; escribir el reporte desde código |
| ⚠️ No funcionó | El primer reporte mostraba las barras truncadas a 0–1 y las diferencias eran invisibles; se detectó solo al revisar la figura |
| 💡 Idea | Revisar siempre las figuras generadas, no solo que el script termine sin error |

**Acciones comprometidas**

| Acción | Responsable | Estado |
|---|---|---|
| Revisar visualmente cada figura antes de commitear | Jacobo | ✅ Hecho |
| Ajustar el eje de la comparación para que las diferencias se vean | Rommel | ✅ Hecho |
| Fijar la decisión por CV en el código (`select_best_model`) y no solo en el texto | Rommel | ✅ Hecho |

---

## Sprint 3 - Despliegue y documentación

**Sprint Goal:** entregar el modelo como artefacto reutilizable, verificado y
documentado, con el ciclo CRISP-DM cerrado.

### Tablero al cierre

| Backlog | To Do | In Progress | Review | Done |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 7 (PB-06, PB-08, T-08, T-09, T-10, T-11, T-12) |

### Sprint Review

**Demostración**

- `python -m src.sprint3` carga el `.joblib` desde disco, verifica que reproduce
  las métricas guardadas (delta 0,00e+00) y predice un grano nuevo.
- `src/deploy.py` con `load_final_model`, `predict_bean_class` y `predict_beans`,
  más validación de entrada (faltantes, desconocidas, nulos, no numéricos, vacío).
- Notebook `03_sprint3_despliegue.ipynb` ejecutado.
- 40 pruebas en verde (44 tras la extensión web).
- Reporte final de 7 secciones, backlog y este documento.

**Resultados presentados**

| Verificación | Resultado |
|---|---|
| Accuracy guardada vs recalculada | 0,919528 vs 0,919528 |
| Delta de F1 macro | 0,00e+00 |
| Variables y clases vs metadatos | coinciden |
| Modelo en disco vs reentrenado | 100% de predicciones idénticas |
| Demo con casos reales | 3 aciertos y los 2 errores más frecuentes (`SIRA` ↔ `DERMASON`) |

**Decisiones tomadas en la review**

1. Se acepta el artefacto como entregable: se puede cargar sin reentrenar y
   reproduce el desempeño reportado.
2. Se decide que la demo **incluya errores** en lugar de mostrar solo aciertos;
   un demo perfecto ocultaría el riesgo principal del modelo.
3. Se documenta la limitación de las 16 variables y la confusión `SIRA` ↔
   `DERMASON` como advertencia operativa.
4. Se cierra el proyecto con los tres PRs encadenados hacia `main`: cada sprint
   queda en su rama y con su revisión asociada.

**Feedback recibido**

> "Con esto se puede defender: hay notebook, reporte y evidencias de gestión, y
> el modelo se puede ejecutar sin el equipo presente."

### Retrospectiva

| Categoría | Observación |
|---|---|
| ✅ Funcionó | Verificar el artefacto recalculando métricas; documentar los errores de la demo; pruebas del contrato de entrada |
| ⚠️ No funcionó | La primera versión de la demo mostraba 5 aciertos de 5 y daba una impresión falsa de perfección |
| 💡 Idea | Cualquier verificación numérica debe ser una prueba automática, no solo una línea en el reporte |

**Acciones comprometidas**

| Acción | Responsable | Estado |
|---|---|---|
| Demo con aciertos y errores reales | Jacobo | ✅ Hecho |
| Prueba automática de integridad del artefacto | Rommel | ✅ Hecho |
| Mapear explícitamente CRISP-DM ↔ TDSP ↔ Scrum en el reporte final | Jacobo | ✅ Hecho |

---

## Resumen de ceremonias

| Ceremonia | Sprint 1 | Sprint 2 | Sprint 3 |
|---|---|---|---|
| Sprint Planning | Alcance: datos y preparación | Alcance: tres modelos y evaluación | Alcance: despliegue y documentación |
| Daily (asíncrono) | 3 sincronizaciones cortas | 3 sincronizaciones cortas | 3 sincronizaciones cortas |
| Sprint Review | Datos y split aceptados | Modelo final elegido | Artefacto y documentación aceptados |
| Retrospectiva | 3 acciones, 3 cerradas | 3 acciones, 3 cerradas | 3 acciones, 3 cerradas |
| Incremento entregado | Dataset preparado y EDA | Modelo entrenado y evaluado | Modelo desplegado y documentado |

## Lecciones de gestión

1. **El backlog cambió con la evidencia, no con la intuición.** PB-07 y PB-08
   nacieron de los sprint reviews, no de la planificación inicial.
2. **Las acciones de retrospectiva se cerraron dentro del sprint siguiente.** No
   quedó ninguna acción abierta al cierre.
3. **La Definition of Done evitó entregables a medias**: cada historia cerrada
   tiene código, prueba, reporte y notebook ejecutado.

---

## Extensión posterior al cierre: aplicación web

Después del Sprint 3, Rommel añadió una interfaz Streamlit (`app.py`) y una
simulación geométrica en vivo (PB-09 y PB-10). **No es un cuarto sprint:** no hubo
planning ni tablero propios, y se documenta como extensión para no falsear la
cronología de las actas.

**Qué se demostró**

- Predicción individual con presets por clase y probabilidad de las 7 variedades.
- Predicción por lotes desde un CSV, con conteo por variedad y descarga de resultados.
- Simulación de la silueta del grano a partir de 5 medidas, actualizada en vivo.
- 44 pruebas en verde en el proyecto; la app aporta 3 (dos de humo y una de la figura).

**Decisiones**

1. La app reutiliza `src.deploy`; no duplica lógica ni relaja las validaciones.
2. La simulación se declara **didáctica**: no participa en la predicción ni valida
   coherencia geométrica.
3. Los presets por clase dependen de `data/processed/`; sin esa carpeta la app
   funciona con un ejemplo por defecto.

**Lección**

La interfaz cambió la forma de presentar el proyecto: para una audiencia no
técnica, mover los ejes y ver la silueta comunica mejor que una tabla de métricas.
Pero una demo visual sin advertencias puede sobreprometer; por eso la simulación
se explica como lo que es.
