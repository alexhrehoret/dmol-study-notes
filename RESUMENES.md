# Resúmenes por capítulo — lo que hay que recordar

Una página por capítulo, solo las ideas core. El detalle está en `notebooks/` y la bitácora en
`PROGRESO.md`. Las cifras son las de nuestros notebooks (semillas fijadas), no las del libro.

---

## Capítulo 1 — Tensores y formas (`01_tensores.ipynb`)

1. **Un tensor = rank + shape.** `ndim` dice cuántos ejes; `shape`, cuánto mide cada uno. El primer
   eje suele ser el **batch**: `(32, 50, 3)` = 32 moléculas × 50 átomos × 3 coordenadas.
2. **Reducciones:** `axis=k` significa "el eje k desaparece". `keepdims=True` lo deja con tamaño 1.
3. **Broadcasting** alinea las formas **por el final**. Cada eje tiene que ser igual, valer 1 o no
   existir. `np.newaxis` (o `None`) añade un eje de tamaño 1 para forzar la alineación que quieres.
4. **`einsum`**: los índices que no aparecen en la salida se suman. Es la notación de los papers.
5. **`*` es elemento a elemento; `@` es producto matricial.**
6. **Vista frente a copia:** los slices y los reshapes comparten memoria. Usa `.copy()` si vas a
   modificar.
7. **Hábito:** ante cualquier bug, lo primero es `print(a.shape, b.shape)`.

Patrón que reaparece en todo el libro (matriz de distancias interatómicas):

```python
diferencias = coords[:, None, :] - coords[None, :, :]   # (N, N, 3)
D = np.sqrt(np.sum(diferencias ** 2, axis=-1))          # (N, N)
```

---

## Capítulo 2 — Introducción al ML (`02_introduccion.ipynb`, `02_ejercicios_resueltos.ipynb`)

**El esquema que sirve para todo el libro:**
datos $(\vec{x}_i, y_i)$ → modelo $\hat{f}(\vec{x}; \vec{w})$ → pérdida $L$ → optimizador → mejores $\vec{w}$.
Lo que irá cambiando es el modelo y la representación de la molécula; el bucle es siempre el mismo.

**Supervisado** (hay labels: aproximar $f$) frente a **no supervisado** (sin labels: buscar
estructura, sin un criterio objetivo de acierto).

**Ideas core:**

1. **Mira los datos antes de modelar.** La multicolinealidad, los 400 `BalabanJ = 0` falsos y las 573
   mezclas con descriptores sumados salieron de mirar, no de entrenar.
2. **Ten siempre un listón tonto.** Predecir la media da RMSE **2.37** (la desviación típica de la
   solubilidad). Sin esa referencia, cualquier otro RMSE no significa nada.
3. **Ninguna feature basta sola.** La mejor, `MolLogP`, tiene $r = -0.61$ ($r^2 = 0.37$).
4. **Estandariza las features** (media 0 y desviación 1). Con escalas incompatibles el descenso de
   gradiente se ahoga: con $\eta = 10^{-6}$ se queda en RMSE 2.06, y con $10^{-5}$ diverge. Estandarizado,
   admite $\eta = 0.1$ y llega a **RMSE 1.658** ($r^2 = 0.51$). El óptimo exacto de un modelo lineal
   con estas 17 features es 1.65: más allá hace falta otro modelo u otra representación.
5. **El learning rate $\eta$ es el hiperparámetro crítico**: si es demasiado grande diverge, y si es
   demasiado pequeño no llega. **SGD** (batches de 32) da el mismo resultado con mucho menos cálculo,
   pero un batch pequeño exige un $\eta$ más pequeño.
6. **Predecir no es explicar.** Con multicolinealidad, los pesos contradicen la química y aun así el
   modelo predice.
7. **Un número no basta.** El parity plot mostró que el modelo comprime el rango, tiene sesgo en las
   colas (+2.61 en los insolubles extremos) y hace extrapolaciones absurdas (llega a predecir −29.8).
8. **No supervisado:** k-means ($k = 4$) da grupos con identidad química, pero **siempre** devuelve $k$
   grupos, y no hay codo: el espacio químico es continuo. En PCA, **PC1 = tamaño (53.9 %)** y
   **PC2 = polaridad (13.7 %)**. PC2 explica menos varianza pero correlaciona más con la solubilidad
   (−0.49 frente a −0.29), porque PCA no mira la label.
9. **Todo lo anterior se midió con los mismos datos del entrenamiento**, así que no es una evaluación
   honesta. De eso va el capítulo 3.

**De los ejercicios:**
- "Lineal" significa lineal **en los parámetros**: la curva la ponen las features (ej. 3).
- Escalar las **labels** no cambia el learning rate válido, porque el Hessiano del MSE solo depende de
  $X$ (ej. 5).
- MSE y MAE dan modelos distintos: la pérdida define qué significa equivocarse (ej. 6).
- Los grupos de `Group` tienen sesgos propios (ej. 10). *Corregido en 3.8*: `Group` no es la fuente
  experimental sino el grupo de fiabilidad de AqSolDB (nº de medidas y si discrepan).

---

## Capítulo 3 — Regresión y evaluación de modelos (`03_regresion.ipynb`, `03_ejercicios_resueltos.ipynb`)

**La ecuación del capítulo:** error esperado en un dato nuevo = **sesgo² + varianza + ruido**. Sesgo =
underfitting (el modelo no puede representar $f$); varianza = overfitting (el modelo cambia según los datos
de train); ruido = el suelo que nadie baja.

**Ideas core:**

1. **Solo cuenta el error de test.** Con 25 moléculas y 18 parámetros: RMSE 0.59 en train, 2.60 en test. El
   test no entra en nada del entrenamiento, ni siquiera en la estandarización.
2. **El overfitting necesita ruido y flexibilidad a la vez.** Polinomio de 3.2.1: test 1.96 con ruido y las
   features justas, **3537** con ruido y features de más, 0 sin ruido.
3. **Los peores fallos son extrapolaciones** (la rafinosa: 11 dadores de H frente a un máximo de 4 en train).
   La varianza se concentra donde el modelo extrapola.
4. **Lo que importa es la proporción parámetros / datos.** El error de test tiene un pico cuando el número de
   features se acerca al de datos. **Más datos reducen la varianza como 1/N, pero no el sesgo** (ej. 2: con la
   solubilidad todo converge a MSE ≈ 2.7).
5. **Regularización**: L2 encoge los pesos, L1 los anula (selección de features), L∞ los iguala (ej. 5).
   **Estandarizar antes de regularizar es obligatorio**: con 7 features, λ = 0.1 baja el test de 1385 a 8.67.
   La selección del lasso es **inestable** con features correlacionadas (187 conjuntos distintos en 200
   muestras de 35 moléculas; solo `MolLogP` sobrevive siempre).
6. **Todo lo que se decide con datos (λ, features, early stopping) es entrenamiento** y no puede tocar el
   test. Para eso, **validación cruzada**, anidada si hay hiperparámetros.
7. **Barajar antes del k-fold.** AqSolDB viene ordenado por fuente: sin barajar, 2.97 ± 2.10; barajado,
   **2.80 ± 0.33**. Con el dataset entero no hay overfitting (train 2.724).
8. **Con pocos datos, la CV es necesaria pero no suficiente**: con 25 moléculas, su estimación casi no se
   correlaciona con el error real (Spearman ≈ 0.1).
9. **Intervalos por predicción**: el bootstrap solo mide la varianza del modelo (cubre el 25.6 % prometiendo
   95 %); el **jackknife+** cubre el 96.0 %, pero de media: fuera del rango del train, el 62.3 %.
10. **El error depende de a qué moléculas se aplique el modelo.** 10-fold 2.80, LOCOCV por fuente 5.02,
    *scaffold split* 5.48, predecir la media 5.61. La zona fiable es el **dominio de aplicabilidad**, y
    depende del modelo: el lineal no mejora por tener en train una molécula muy parecida.

**Hallazgos propios y correcciones al libro:** rango 16 de los 17 descriptores (3.3), train sorteado con
reemplazo (3.4), ridge con Adam sin converger (3.5), k-fold sin barajar (3.6), jackknife+ con residuos de
train (3.7), LOCOCV sin reiniciar `k_error` y `Group` ≠ fuente (3.8). *Corrección al cap. 2*: el óptimo
lineal es MSE 2.724, no 2.708 (artefacto de float32).

**De los ejercicios:**
- Sin ruido no se aprende ruido, pero si los datos no determinan los pesos (más features que datos, o una
  muestra que no varía en algún descriptor) hay train 0 y test > 0: **indeterminación** (ej. 1).
- En L1 sobrevive la feature más correlacionada con $y$ y caen primero las que repiten información de otras;
  qué descriptor de un grupo correlacionado sobrevive es azar (ej. 3 y 4).
- **El mejor modelo lineal** (ej. 6): ridge con 204 descriptores RDKit **recortados al rango del train** +
  huella de Morgan con cuentas. **MSE 1.873** en 10-fold anidado (frente a 2.792) y **4.132** en scaffold
  split (frente a 5.485). Sin recortar, los 204 descriptores dan 7.46, peor que la media, por unas pocas
  extrapolaciones. Lo que ayuda es la representación y el recorte, no la regularización. Aun así queda
  ~1.5 de MSE por encima del ruido experimental (≈ 0.27): sesgo que un modelo lineal no quita.
