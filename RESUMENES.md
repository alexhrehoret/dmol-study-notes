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
