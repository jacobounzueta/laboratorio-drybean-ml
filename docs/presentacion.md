# Pitch del proyecto - Laboratorio Dry Bean

**Duración objetivo:** 13:30 de contenido + 1:30 de colchón para preguntas = **15:00 máximo**
**Formato:** presentación con demo en vivo
**Reparto:** Jacobo (negocio, datos, Scrum, demo) · Rommel (modelado, evaluación, despliegue)

---

## 1. Ficha rápida

| # | Bloque | Inicio | Duración | Quién | Diapositiva |
|---|---|---|---|---|---|
| 1 | Apertura: el problema y la promesa | 0:00 | 1:00 | Jacobo | 1. Título |
| 2 | Los datos y lo que ya anticipaban | 1:00 | 1:30 | Jacobo | 2. Dataset y hallazgos |
| 3 | Preparación sin trampas | 2:30 | 1:00 | Jacobo | 3. Decisiones de preparación |
| 4 | Tres modelos y un protocolo | 3:30 | 2:00 | Rommel | 4. Modelado |
| 5 | Resultados y el error localizado | 5:30 | 2:30 | Rommel | 5. Métricas y confusión |
| 6 | La decisión y por qué | 8:00 | 1:00 | Rommel | 6. Decisión CRISP-DM |
| 7 | Demo en vivo | 9:00 | 2:00 | Jacobo | 7. Demo |
| 8 | Qué queda funcionando y qué no | 11:00 | 1:30 | Rommel | 8. Despliegue y límites |
| 9 | Cierre: metodologías y aprendizaje | 12:30 | 1:00 | Jacobo | 9. Cierre |

**Regla de oro:** si van retrasados, se recorta el bloque 8 (nunca el 5 ni el 7). El 5 es la evidencia y el 7 es lo que se recuerda.

> **Ritmo real:** el texto hablado son ~1.360 palabras, es decir unos 10 minutos a ritmo normal. Los 3 minutos restantes son la demo, las pausas y los cambios de diapositiva. Si van rápido, **no agreguen contenido**: den más aire a la demo y a las preguntas.
>
> **Si les dan solo 10 minutos:** mantengan los bloques 1, 2 (solo los dos hallazgos), 5, 6 y 7. Recorten el 3 y el 8 a una frase cada uno.

---

## 2. Antes de empezar (checklist de 5 minutos)

- [ ] Terminal abierta **en la raíz** del repo (`laboratorio_drybean_ml`).
- [ ] Ejecutar una vez `python -m src.sprint3` para calentar el entorno y verificar que no falla.
- [ ] Tener abierto en pestañas: `outputs/reports/sprint3/reporte_sprint3.md` (plan B de la demo) y `notebooks/03_sprint3_despliegue.ipynb` ya ejecutado.
- [ ] Cerrar notificaciones y pestañas que no se usan.
- [ ] Probar el cambio de tamaño de fuente de la terminal (se verá proyectada).
- [ ] Acordar la señal de "te quedan 30 segundos": una mano cerrada sobre la mesa.

---

## 3. Guion

### 0:00 - 1:00 · Apertura — **Jacobo** (diapositiva 1: título)

> Buenas tardes. En una planta de selección de frijol, cada lote se clasifica a mano: es lento, depende del ojo del operario y no deja ningún registro.
>
> Nosotros no inventamos un proceso nuevo. Usamos algo que la planta ya tiene: una cámara que captura **16 medidas geométricas** de cada grano. La pregunta fue simple: **¿alcanzan esas 16 medidas para determinar la variedad?**
>
> Somos Jacobo y Rommel. En los próximos 15 minutos les mostramos qué encontramos, cómo lo decidimos y qué queda funcionando.
>
> Una advertencia honesta desde el inicio: **no encontramos un modelo que resuelva el problema al cien por ciento.** Encontramos el techo de estos datos, y lo demostramos. Eso, para nosotros, es el resultado.

### 1:00 - 2:30 · Los datos — **Jacobo** (diapositiva 2)

> Partimos del **Dry Bean Dataset** de UCI: **13.611 granos**, 16 mediciones continuas y **7 variedades**.
>
> En calidad encontramos dos cosas. Bien: cero nulos, cero tipos inconsistentes. Mal: **68 duplicados exactos**, el mismo grano medido dos veces, que eliminamos. Quedaron **13.543 registros**.
>
> El EDA nos dio dos pistas que después se cumplieron. Primera: hay **redundancia** entre variables de tamaño; `Area` y `ConvexArea` correlacionan 1,000. Dieciséis variables no son dieciséis dimensiones reales.
>
> Segunda: las clases están desbalanceadas **6,79 a 1**, entre `DERMASON` con 3.546 granos y `BOMBAY` con 522. Y mirando las cajas se veía algo más: `BOMBAY` está geométricamente aparte, mientras `SIRA` y `DERMASON` se solapan.
>
> Eso nos llevó a una decisión metodológica: **no evaluar solo con accuracy**, porque un modelo que ignore las clases pequeñas puede verse bien y ser inútil en planta. Reportamos **F1 macro**, que pesa igual a las 7 clases.

### 2:30 - 3:30 · Preparación — **Jacobo** (diapositiva 3)

> La preparación fue deliberadamente conservadora.
>
> No imputamos: no hay nulos, imputar sería inventar. No borramos outliers: esos valores extremos son granos reales, y muchos son `BOMBAY`; recortarlos habría borrado justo la clase minoritaria.
>
> El split es **80/20 estratificado**: 10.834 para entrenar, 2.709 para probar, con una desviación máxima de estratificación de 0,0003. Semilla fija **42**, así que cualquiera reproduce exactamente el mismo experimento.
>
> Y un detalle que suele pasarse por alto: el escalamiento vive **dentro del pipeline** de la regresión logística, así que se ajusta solo con train. **No hay fuga de información del test.**
>
> Todo esto vive en la estructura de TDSP: datos, código en `src/`, notebooks, modelos y reportes. Los notebooks solo invocan el código; no contienen la lógica.

### 3:30 - 5:30 · Modelado — **Rommel** (diapositiva 4)

> Gracias, Jacobo. Ya con los datos listos, entrenamos **tres modelos** con el mismo split y la misma semilla.
>
> Primero un **baseline**: regresión logística con escalamiento. Es interpretable y barato; sirve para saber si el problema necesita algo más complejo.
>
> Segundo, un **Random Forest** con 200 árboles y peso balanceado. No necesita escalamiento y nos da importancia de variables.
>
> Tercero, **HistGradientBoosting** con regularización L2. Es boosting sobre histogramas: buen desempeño en datos tabulares y rápido.
>
> Ahora, lo importante no es la lista de modelos, es **cómo decidimos**. El test tiene 2.709 filas: el intervalo de confianza de una accuracy del 92% es de más o menos un punto porcentual. Es decir, **diferencias de milésimas son ruido**.
>
> Por eso separamos dos papeles. El **hold-out** se usa para reportar el desempeño final, con honestidad. La **validación cruzada de 5 folds sobre train** se usa para **decidir**, porque son cinco estimaciones en lugar de una.
>
> El criterio quedó en el código, no en una diapositiva: gana el mayor F1 macro medio; si la diferencia es menor que 0,005, se prefiere el modelo más simple.
>
> También probamos ExtraTrees y KNN. Quedaron por debajo y **no los incorporamos**. Los documentamos como análisis exploratorio, porque los descartes también son evidencia.

### 5:30 - 8:00 · Resultados — **Rommel** (diapositiva 5)

> Estos son los números sobre las 2.709 filas que los modelos nunca vieron.

| Modelo | Accuracy | F1 macro | F1 macro CV 5-fold |
|---|---|---|---|
| Logistic Regression | 0,9195 | 0,9306 | 0,9360 ± 0,0031 |
| Random Forest | 0,9195 | 0,9310 | 0,9352 ± 0,0016 |
| **HistGradientBoosting** | **0,9195** | **0,9315** | **0,9429 ± 0,0032** |

> Fíjense en el dato incómodo: **los tres tienen exactamente la misma accuracy**, 0,9195. Aciertan los mismos **2.491 granos de 2.709**. Hay 218 errores y coinciden entre el 95 y el 97% de las predicciones.
>
> Si solo hubiéramos mirado accuracy, habríamos dicho "empataron". La validación cruzada desempata: **HGB gana en las cinco particiones**, con 0,9429 frente a 0,9360 del baseline.
>
> ¿Dónde está el error? Aquí está la respuesta a la pregunta del laboratorio.

| Real | Predicho | Casos |
|---|---|---|
| SIRA | DERMASON | 53 |
| DERMASON | SIRA | 47 |
| BARBUNYA | CALI | 16 |

> **`SIRA` y `DERMASON` concentran 100 de los 218 errores: el 46%.** Son variedades de tamaño y forma muy parecidos, y eso ya se veía en el EDA.
>
> Y el hallazgo que rompe el sentido común: **`BOMBAY`, la clase más pequeña, se clasifica perfecto**, F1 de 1,000. El problema no es el desbalance; es la similitud entre variedades concretas.
>
> Por clase, la peor es `SIRA` con F1 0,867; la mejor, `BOMBAY` con 1,000.
>
> En importancia de variables no hay una estrella: `Perimeter` pesa 10,9%, `ShapeFactor1` 10,2%, `Compactness` 9,5%. **La información está en la combinación de tamaño y forma**, no en una medida única. Y aclaramos: importancia por impureza no prueba causalidad.

### 8:00 - 9:00 · La decisión — **Rommel** (diapositiva 6)

> Con ese protocolo, el modelo que pasa a despliegue es **HistGradientBoosting regularizado**.
>
> La justificación es una sola frase: **gana en las cinco particiones de validación cruzada y también en F1 macro de test**, con una diferencia de 0,0069 sobre el baseline.
>
> No elegimos el más complejo por moda. Si la diferencia hubiera sido menor que 0,005, el código habría elegido el más simple y rápido. La regla está implementada y probada.
>
> El modelo se guarda con `joblib` y un archivo de metadatos: tipo de modelo, métricas, las 16 variables, las 7 clases, filas de train y test, semilla y versión de scikit-learn. **Un `.joblib` sin metadatos es una caja negra.**

### 9:00 - 11:00 · Demo en vivo — **Jacobo** (diapositiva 7)

**Qué ejecutar (terminal ya abierta en la raíz):**

```powershell
.\.venv\Scripts\python.exe -m src.sprint3
```

**Qué decir mientras corre (son ~5 segundos):**

> Esto no es un notebook: es un artefacto que se carga desde disco. El script hace tres cosas.
>
> Uno: **carga el modelo** y muestra que tiene 16 variables y 7 clases.
>
> Dos: **vuelve a predecir el test** y compara con las métricas guardadas. Miren la línea: accuracy guardada 0.919528, recalculada 0.919528, **delta cero**. Si alguien hubiera cambiado el archivo por error, aquí falla.
>
> Tres: **predice un grano nuevo**. Este grano no viene de Internet, viene de las 16 medidas de un grano real, con la función que usaría la planta.

**Cierre de la demo (señalando el reporte):**

> Y esto queda escrito: reporte, demo en CSV y verificación en JSON. La demo incluye **tres aciertos y los dos errores más frecuentes** a propósito. Un demo perfecto escondería el riesgo que acabamos de mostrar.

**Plan B si falla la terminal:** abrir `notebooks/03_sprint3_despliegue.ipynb` (ya ejecutado) y mostrar la celda de verificación; después `outputs/reports/sprint3/reporte_sprint3.md`. Nunca improvisar comandos en vivo.

### 11:00 - 12:30 · Qué queda funcionando y qué no — **Rommel** (diapositiva 8)

> El entregable técnico son tres funciones: cargar el modelo, predecir un grano y predecir un lote con probabilidad.
>
> La entrada se valida **antes** de predecir: si falta una variable, sobra una, hay un nulo o un texto, el sistema devuelve un error que dice exactamente qué corregir. Eso evita el `KeyError` de scikit-learn en producción.
>
> Y hay **41 pruebas automáticas** que cubren datos, split, modelos, validación cruzada, persistencia, contrato de entrada y verificación del artefacto.
>
> Ahora, los límites, que son del problema y no del despliegue. Primero: el modelo aprendió de **un solo montaje de cámara**; si cambia la iluminación o el lente, las medidas dejan de ser comparables y hay que reentrenar. Segundo: **no tenemos la matriz de costos del negocio**; no es lo mismo confundir dos variedades comunes que perder un `BOMBAY` valioso. Tercero: el desempeño está medido en **una partición fija**; en producción hay que monitorear la distribución de entrada.
>
> Por eso el trabajo futuro es concreto: calibrar probabilidades, explicar cada predicción con SHAP, exponer un endpoint y monitorear *drift*.

### 12:30 - 13:30 · Cierre — **Jacobo** (diapositiva 9)

> Para cerrar, tres ideas.
>
> **Metodología.** Cerramos las seis fases de CRISP-DM: negocio, datos, preparación, modelado, evaluación y despliegue. Lo hicimos en **tres sprints de Scrum ML**, con backlog priorizado, tablero, sprint review y retrospectiva; cada acción de mejora se cerró en el sprint siguiente. Y usamos TDSP para que el repositorio se explique solo: datos, código, notebooks, artefactos y reportes.
>
> **La decisión que más cambió el proyecto** fue dejar de decidir con una sola partición de test. Con el hold-out los tres modelos empataban; con validación cruzada, uno gana en las cinco particiones. Eso es más honesto, y es defendible.
>
> **El resultado no es un 92%.** El resultado es un sistema que se puede volver a ejecutar, auditar y mejorar, y un mapa claro de dónde está el límite: en la similitud geométrica entre variedades, no en el algoritmo.
>
> Gracias. Quedamos atentos a sus preguntas.

---

## 4. Preguntas probables (colchón de 1:30)

| Pregunta | Respuesta corta | Quién |
|---|---|---|
| ¿Por qué no una red neuronal? | Con 16 variables tabulares y 13.543 filas, un boosting regularizado es el estado del arte razonable. Una red aportaría complejidad y costo de ajuste sin mejorar un techo que ya demostramos que está en las variables. | Rommel |
| ¿Por qué HGB y no Random Forest, si empatan en test? | Porque el test no puede desempatar: 2.709 filas dan ±1 pp. La CV sí: HGB gana en las 5 particiones por 0,007. | Rommel |
| ¿92% es suficiente? | Depende del costo del error, y eso es una decisión del negocio. Lo que sí podemos afirmar es que el error está localizado: 46% es `SIRA` ↔ `DERMASON`. Con revisión humana en ese par, el sistema es usable. | Jacobo |
| ¿No hay sobreajuste? | El modelo se evalúa en 2.709 filas que nunca vio y la brecha con CV es de poco más de 1 punto porcentual, consistente. Además el baseline y el bosque llegan al mismo punto: no es un modelo memorizando. | Rommel |
| ¿Y si cambia la cámara? | El modelo deja de ser válido; por eso el contrato incluye las 16 variables y recomendamos monitorear su distribución. Reentrenar es un comando. | Rommel |
| ¿Cómo evitan fuga de información? | Escalador dentro del pipeline, ajustado solo con train; split antes de cualquier transformación; selección de modelo con CV sobre train y test reservado. | Rommel |
| ¿Qué aporta Scrum aquí? | El backlog cambió con la evidencia: de la review del Sprint 1 salieron el tercer modelo y el despliegue. Sin esas ceremonias habríamos entregado dos modelos y un notebook. | Jacobo |
| ¿Se puede usar mañana en planta? | Se puede integrar mañana como apoyo a la clasificación, no como reemplazo. Faltan calibración, endpoint y validación con datos del lote real. | Jacobo |

---

## 5. Frases para memorizar (no leer)

- "No encontramos el modelo perfecto: encontramos el techo de los datos."
- "Con una sola partición de test, los tres modelos empataban. La validación cruzada fue lo que nos permitió decidir."
- "El problema no es el desbalance: es que `SIRA` y `DERMASON` son casi el mismo grano."
- "El resultado no es un 92%: es un sistema que se puede volver a ejecutar y auditar."

## 6. Cifras que no se pueden equivocar

| Dato | Valor |
|---|---|
| Registros originales → tras limpieza | 13.611 → 13.543 (68 duplicados) |
| Clases / desbalance | 7 clases, 6,79:1 |
| Train / Test | 10.834 / 2.709 |
| Accuracy / F1 macro del modelo final | 0,9195 / 0,9315 |
| F1 macro CV 5-fold | 0,9429 ± 0,0032 |
| Aciertos en test | 2.491 de 2.709 (218 errores) |
| Error `SIRA` ↔ `DERMASON` | 100 casos, 46% del total |
| `BOMBAY` | F1 = 1,000 (104 casos) |
| Ventaja de HGB sobre el baseline en CV | +0,0069 |
| Pruebas automáticas | 41 |
| Delta de verificación del artefacto | 0,00e+00 |

## 7. Reparto en una línea

- **Jacobo:** por qué importa, qué dicen los datos, cómo preparamos, demo y cierre metodológico.
- **Rommel:** qué modelos, cómo se decide, qué dicen los números, qué queda desplegado y qué no.
