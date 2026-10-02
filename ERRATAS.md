# Erratas y puntualizaciones del libro

Registro de los fallos y de las cosas mejorables que encontramos en *Deep Learning for Molecules and
Materials* (https://dmol.pub) al reproducirlo. Se amplía en cada sección. Versión en inglés: `ERRATA.md`.

El libro es muy bueno, y ninguna de estas entradas invalida las ideas que enseña. Sí cambian cifras y,
en algunos casos, la conclusión que se saca de una figura. El código del libro sale de su repositorio
fuente (`github.com/whitead/dmol-book`, carpeta `ml/`), que incluye las celdas ocultas en la web.

**Tipos**

| Tipo | Qué es |
|---|---|
| `bug` | el código no hace lo que dice el texto, y el resultado sale mal |
| `método` | el código funciona, pero el procedimiento no es el correcto (fuga de información, no barajar, no converger...) |
| `texto` | el texto no coincide con el código o con la matemática |
| `datos` | un problema del dataset que el libro no menciona y que afecta a los resultados |
| `API` | código que ya no funciona con las versiones actuales de las librerías |

**Impacto**: **alto** = cambia la conclusión; **medio** = cambia las cifras, no la idea; **bajo** = menor.

## Índice

| ID | Sección | Tipo | Impacto | Resumen |
|---|---|---|---|---|
| [G-1](#g-1) | todo el libro | `método` | medio | no fija semillas aleatorias |
| [2-1](#2-1) | 2.3 | `API` | bajo | `sns.distplot` ya no existe |
| [2-2](#2-2) | 2.3, 3.8 | `datos` | alto | `Group` no es la fuente de los datos |
| [2-3](#2-3) | 2.3 | `datos` | medio | AqSolDB tiene 573 mezclas con descriptores sumados y 400 `BalabanJ = 0` falsos |
| [3.3-a](#33-a) | 3.3 | `texto` | medio | el texto no describe el experimento que hace el código |
| [3.3-b](#33-b) | 3.3 | `método` | alto | los 17 descriptores tienen rango 16: las curvas no pueden cruzar f = N |
| [3.3-c](#33-c) | 3.3 | `método` | bajo | train y test se sortean por separado y comparten moléculas |
| [3.4-a](#34-a) | 3.4 | `texto` | bajo | escribe $f - \epsilon$ en vez de $f + \epsilon$ |
| [3.4-b](#34-b) | 3.4 | `texto` | bajo | los datos sintéticos cambian de σ = 5 a σ = 1 sin decirlo |
| [3.4-c](#34-c) | 3.4 | `bug` | medio | el train se sortea con reemplazo |
| [3.4-d](#34-d) | 3.4 | `método` | bajo | el "sesgo²" de 7 features es error de estimación |
| [3.5-a](#35-a) | 3.5 | `método` | alto | ridge sin converger, sin estandarizar y penalizando $w_0$ |
| [3.6-a](#36-a) | 3.6 | `método` | medio | el k-fold no baraja, y el CSV está ordenado por fuente |
| [3.6-b](#36-b) | 3.6 | `bug` | medio | segmentos de `N // k`: datos que nunca se examinan |
| [3.6-c](#36-c) | 3.6 | `método` | bajo | el término constante se calcula aparte |
| [3.7-a](#37-a) | 3.7 | `texto` | bajo | $2^N$ remuestreos en vez de $\binom{2N-1}{N}$ |
| [3.7-b](#37-b) | 3.7 | `bug` | alto | el jackknife+ usa residuos de train |
| [3.8-a](#38-a) | 3.8 | `bug` | alto | el LOCOCV no reinicia `k_error` |
| [4.1-a](#41-a) | 4.1 | `datos` | medio | la label es "fracasó por toxicidad", no "no aprobado" |
| [4.2-a](#42-a) | 4.2 | `datos` | bajo | descarta 4 SMILES sin decir cuáles |
| [4.2-b](#42-b) | 4.2 | `datos` | alto | las dos clases están escritas de forma distinta (atajo) |
| [4.3-a](#43-a) | 4.3 | `texto` | medio | los `NaN` no vienen de std = 0 |
| [4.3-b](#43-b) | 4.3 | `método` | medio | estandariza antes de repartir (fuga de información) |
| [4.4-a](#44-a) | 4.4 | `método` | alto | reparto 80/20 sin barajar sobre un CSV en orden alfabético |
| [4.4-b](#44-b) | 4.4 | `texto` | alto | da el modelo por bien entrenado sin compararlo con un listón |
| [4.4-c](#44-c) | 4.4 | `bug` | bajo | el bucle se salta el último batch del train |
| [4.4-d](#44-d) | 4.4 | `texto` | bajo | $\vec w\cdot\vec x + b$ es proporcional a la distancia a la frontera, no la distancia |
| [P-1](#p-1) | 4.5 | `bug` | ? | `accuracy` usa `yhat` en vez de `hard_yhat` |

---

## General

<a id="g-1"></a>
### G-1 · No fija semillas aleatorias · `método` · medio

- **Dónde**: todos los capítulos (`soldata.sample(...)`, `np.random.normal(...)`, `np.random.choice(...)`).
- **Qué pasa**: cada ejecución sortea datos y pesos distintos, así que las cifras del texto no se pueden
  reproducir. Con pocos datos cambian mucho: en 3.2, el RMSE de test con 25 moléculas va de 1.17 a 3.93 según
  el sorteo.
- **Cómo hacerlo**: fijar la semilla (`sample(..., random_state=N)`, `rng = np.random.default_rng(N)`). Si la
  conclusión depende de la semilla, enseñar varias.
- **En nuestros notebooks**: todas las celdas llevan semilla; en 3.2 y 3.2.1, comparación con 8 y 10 semillas.
  En 4.4, 3 semillas de los pesos iniciales en cada reparto: cambian la pérdida de test como mucho 0.055, algo
  menos que el reparto (0.170–0.230 entre 5 repartos estratificados).

## Capítulo 2 — Introducción al ML

<a id="2-1"></a>
### 2-1 · `sns.distplot` ya no existe · `API` · bajo

- **Dónde**: 2.3, histograma de la solubilidad (`sns.distplot(soldata.Solubility)`).
- **Qué pasa**: seaborn retiró `distplot`; la celda da error.
- **Cómo hacerlo**: `sns.histplot(soldata.Solubility, kde=True)`.
- **En nuestros notebooks**: `02_introduccion.ipynb`, 2.3.2.

<a id="2-2"></a>
### 2-2 · `Group` no es la fuente de los datos · `datos` · alto

- **Dónde**: 2.3 ("`Group` (where the data came from)") y 3.8, *Leave One Class Out*: "our solubility data
  actually is a combination of five other datasets so our data is already pre-classified based on who
  measured the solubility".
- **Qué pasa**: en AqSolDB, `Group` (G1–G5) es el **grupo de fiabilidad** de cada valor: G1 = una sola medida;
  G2/G3 = dos medidas con SD > 0.5 o ≤ 0.5; G4/G5 = tres o más medidas con SD > 0.5 o ≤ 0.5. La fuente es el
  prefijo A–I del `ID`, y son **9** datasets, no 5. Las 9 fuentes aportan moléculas a los 5 grupos.
- **Efecto**: el LOCOCV de 3.8 no mide lo que dice medir (cómo se generaliza a otro laboratorio). Hecho por
  fuente real, el error agregado es **5.02** (frente a 3.20 por `Group`). La fuente A sube de 4.75 (10-fold) a
  **10.40**.
- **Cómo hacerlo**: agrupar por `ID.str[0]` para el LOCOCV por fuente.
- **En nuestros notebooks**: `03_regresion.ipynb`, 3.8; nota añadida en `02_ejercicios_resueltos.ipynb`
  (ejercicio 10).

<a id="2-3"></a>
### 2-3 · Mezclas y `BalabanJ = 0` falsos en AqSolDB · `datos` · medio

- **Dónde**: 2.3, el dataset con sus 17 descriptores precalculados.
- **Qué pasa**: hay **573 mezclas** (nombre con `;`) cuyos descriptores son la suma de los componentes, y
  **400 `BalabanJ = 0`** que no son reales (363 son sales: estructuras de varios fragmentos).
- **Efecto**: las mezclas tienen RMSE 2.31 frente a 1.61 del resto con el modelo lineal del capítulo 2. Las
  peores moléculas en 3.6 y 3.8 son casi siempre mezclas o metales.
- **Cómo hacerlo**: marcar o quitar mezclas y sales antes de modelar, o quedarse con el fragmento principal
  (`rdMolStandardize.LargestFragmentChooser`).
- **En nuestros notebooks**: `02_introduccion.ipynb`, 2.3.4 y 2.3.9.

## Capítulo 3 — Regresión y evaluación de modelos

<a id="33-a"></a>
### 3.3-a · El texto no describe el experimento del código · `texto` · medio

- **Dónde**: 3.3, *Exploring Effect of Feature Number* (código oculto).
- **Qué pasa**: (1) las $f$ features son **combinaciones lineales aleatorias** de los 17 descriptores
  ($X \cdot F$, con $F$ normal), no descriptores distintos; (2) el ajuste es `lstsq` (el mínimo exacto), y el
  texto habla de que el modelo "deja de converger"; hay una función `adam_fit` definida pero sin usar; (3) la
  segunda figura usa **500** moléculas, no 250.
- **Cómo hacerlo**: describir el experimento real, o hacer el que describe el texto.
- **En nuestros notebooks**: `03_regresion.ipynb`, 3.3.

<a id="33-b"></a>
### 3.3-b · Rango 16: las curvas no pueden cruzar f = N · `método` · alto

- **Dónde**: 3.3, las dos figuras de error frente al número de features.
- **Qué pasa**: `RingCount = NumAromaticRings + NumAliphaticRings`, así que los 17 descriptores tienen
  **rango 16**. Las combinaciones lineales nunca contienen más de 16 features independientes: a partir de
  f = 16–17 las curvas son exactamente planas.
- **Efecto**: el experimento no puede ver lo que pasa al cruzar f = N = 25 (el pico de interpolación y el
  doble descenso).
- **Cómo hacerlo**: usar features realmente distintas. Con 204 descriptores de RDKit y N = 25, el test tiene
  un pico en f = 24–25 (mediana 118–138) y luego **doble descenso** hasta 5.72 con 150 features.
- **En nuestros notebooks**: `03_regresion.ipynb`, 3.3 (variante con descriptores de RDKit).

<a id="33-c"></a>
### 3.3-c · Train y test se sortean por separado · `método` · bajo

- **Dónde**: 3.3, código oculto (`soldata.sample(N)` dos veces).
- **Qué pasa**: la misma molécula puede caer en train y en test. Con N = 500, entre 17 y 32 moléculas
  repetidas por reparto.
- **Cómo hacerlo**: sortear 2N moléculas una vez y partirlas, como hace el propio libro en 3.2.

<a id="34-a"></a>
### 3.4-a · $f - \epsilon$ en vez de $f + \epsilon$ · `texto` · bajo

- **Dónde**: 3.4, derivación de la descomposición sesgo–varianza.
- **Qué pasa**: la label es $y = f(\vec x) + \epsilon$. El resultado no cambia porque el ruido es simétrico
  (media 0).

<a id="34-b"></a>
### 3.4-b · σ = 1 en vez de σ = 5, sin decirlo · `texto` · bajo

- **Dónde**: 3.4, código oculto (`syn_labels = ... + np.random.normal(size=N)`).
- **Qué pasa**: 3.2.1 usaba ruido con σ = 5; 3.4 rehace los datos con σ = 1 sin mencionarlo. Las cifras de
  ambas secciones no son comparables.

<a id="34-c"></a>
### 3.4-c · El train se sortea con reemplazo · `bug` · medio

- **Dónde**: 3.4, código oculto: `np.random.choice(range(N), size=N // 2)`.
- **Qué pasa**: `np.random.choice` sortea **con reemplazo** por defecto. De los 10 puntos de train solo hay
  8.02 distintos de media, y los puntos de test se definen como "los que no están en train".
- **Efecto**: infla la varianza. Con 4 features, var = 14.21 (test 16.11) con reemplazo, y **1.87** (test
  **3.77**) sin él: 7.6 veces menos.
- **Cómo hacerlo**: `np.random.choice(range(N), size=N // 2, replace=False)` (el propio libro lo hace así en
  3.5).
- **En nuestros notebooks**: `03_regresion.ipynb`, 3.4 (variante sin reemplazo).

<a id="34-d"></a>
### 3.4-d · El "sesgo²" de 7 features es error de estimación · `método` · bajo

- **Dónde**: 3.4, el ejemplo de 7 features.
- **Qué pasa**: el modelo de 7 features contiene la función verdadera, así que su sesgo real es ~0. El 34.10
  que sale es el error al promediar 1000 predicciones que valen ±1000 (varianza 20 572.59): la media no
  converge con tan pocas repeticiones.
- **Cómo hacerlo**: dar el sesgo² con su incertidumbre, o usar muchas más repeticiones. Con N = 100, la
  varianza de 7 features baja a 0.30 y el problema desaparece.

<a id="35-a"></a>
### 3.5-a · Ridge sin converger, sin estandarizar y penalizando $w_0$ · `método` · alto

- **Dónde**: 3.5, *Regularization*, código oculto:
  `reg_loss = mean((y - x·w)**2) + alpha * mean(w**2)`, ajustado con `adam_fit` (100 pasos de Adam desde la
  solución de `lstsq`).
- **Qué pasa**: (1) 100 pasos **no llegan al mínimo**: con λ = 1, pérdida 3.366 frente a 1.416 del mínimo
  exacto; (2) las 7 features polinómicas ($x^0 \dots x^6$) **no se estandarizan**, así que la penalización
  castiga mucho más a unos pesos que a otros; (3) `mean(w**2)` **incluye $w_0$**, el término constante, que
  no se debe penalizar. Además, L1 no tiene código: solo texto.
- **Efecto**: no aparece la U del compromiso sesgo–varianza. La varianza vuelve a subir (175.06 con
  λ = 31.62) y el mejor test es 64.61 con λ = 1000. Con ridge exacto, features estandarizadas y $w_0$ sin
  penalizar, sale una U limpia: mínimo en λ = 0.1 con **test 8.67** (160 veces menos que 1385.20 sin
  regularizar).
- **Cómo hacerlo**: fórmula cerrada $\vec w = (X^\top X + \lambda I')^{-1} X^\top \vec y$, con $I'$ = identidad
  con un 0 en la posición del término constante, o `sklearn.linear_model.Ridge`, que no penaliza el
  intercepto. Estandarizar antes con las estadísticas del train.
- **En nuestros notebooks**: `03_regresion.ipynb`, 3.5 (`ridge_exact` y variante estandarizada).

<a id="36-a"></a>
### 3.6-a · El k-fold no baraja, y el CSV está ordenado por fuente · `método` · medio

- **Dónde**: 3.6, *k-fold cross-validation*: `test = soldata[splits[i] : splits[i + 1]]`.
- **Qué pasa**: los segmentos son bloques consecutivos del CSV, y AqSolDB viene **ordenado por fuente** (prefijo
  A–I del `ID`; la A son 3656 filas). Cada segmento es casi una fuente distinta.
- **Efecto**: 10-fold da **2.97 ± 2.10**, y barajado **2.80 ± 0.33**. Sin barajar, k = 2 da 4.92; barajando
  sale 2.80–2.84 para todo k. Los segmentos malos son los de la fuente A (mezclas y metales).
- **Cómo hacerlo**: `sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=N)`. Si se quiere
  medir la generalización a otra fuente, hacerlo a propósito, agrupando por fuente ([2-2](#2-2)).
- **En nuestros notebooks**: `03_regresion.ipynb`, 3.6.

<a id="36-b"></a>
### 3.6-b · Segmentos de `N // k`: datos que nunca se examinan · `bug` · medio

- **Dónde**: 3.6: `splits = list(range(0, N + N // k, N // k))`.
- **Qué pasa**: con división entera, si N no es múltiplo de k el resto de datos no cae en ningún segmento de
  test. Con 9982 moléculas y k = 10 se pierden 2. Con N = 25 es grave: con k ≥ 13 los segmentos son de una
  molécula y **solo se examinan las k primeras**. Además LOOCV se define pero no tiene código.
- **Efecto**: con N = 25, la bajada del error de 16.52 a 11.42 entre k = 13 y k = 24 es un artefacto: los
  modelos son los mismos y solo cambia qué moléculas se examinan.
- **Cómo hacerlo**: `KFold` (reparte el resto en segmentos de ⌊N/k⌋ y ⌊N/k⌋ + 1) o `np.array_split`. Para
  LOOCV, `sklearn.model_selection.LeaveOneOut`.
- **En nuestros notebooks**: `03_regresion.ipynb`, 3.6 (con 25 moléculas).

<a id="36-c"></a>
### 3.6-c · El término constante se calcula aparte · `método` · bajo

- **Dónde**: 3.6–3.8: `w = lstsq(x, y)` sin columna de unos, y luego `b = np.mean(y - np.dot(x, w))`.
- **Qué pasa**: es una aproximación a ajustar $\vec w$ y $b$ a la vez. Con todos los datos, MSE de train
  2.738 frente a 2.724 del ajuste conjunto.
- **Cómo hacerlo**: añadir una columna de unos a `x` (o centrar `x` e `y` antes de `lstsq`).

<a id="37-a"></a>
### 3.7-a · $2^N$ remuestreos en vez de $\binom{2N-1}{N}$ · `texto` · bajo

- **Dónde**: 3.7, *Bootstrap resampling*: "we can generate $2^N$ new datasets".
- **Qué pasa**: el número de multiconjuntos distintos de tamaño N es $\binom{2N-1}{N}$. Con N = 5, 126 y no
  32. La idea (son muchísimos) no cambia.

<a id="37-b"></a>
### 3.7-b · El jackknife+ usa residuos de train · `bug` · alto

- **Dónde**: 3.7, *Jacknife+*:

  ```python
  yhat = np.dot(small_soldata.iloc[idx][feature_names].values, w) + b
  residuals.append(np.abs(yhat - small_soldata.iloc[idx]["Solubility"]))
  ```

- **Qué pasa**: `idx` son los N − 1 datos **de train** de esa ronda. El residuo que pide el método es el del
  dato **excluido**, `iloc[i]`. Cada `append` guarda 999 residuos de train en vez de 1 residuo LOO. Además:
  imprime "± −3.27" (`(qlow - qhigh) / 2` con los términos al revés), y "Average test error" es la mediana de
  residuos de train.
- **Efecto**: con N = 1000 casi no se nota (residuos LOO 0.977 frente a 0.968 de train). Con N = 25 el
  modelo sobreajusta, los residuos de train son pequeños y el intervalo, que debería cubrir al menos el
  90 %, cubre el **61.9 %**. Corregido: **93.3 %**.
- **Cómo hacerlo**:

  ```python
  yhat_i = np.dot(small_soldata.iloc[i][feature_names].values, w) + b
  residuals.append(np.abs(yhat_i - small_soldata.iloc[i]["Solubility"]))
  ...
  print(f"... +/- {(qhigh - qlow) / 2:.2f}")
  ```

- **En nuestros notebooks**: `03_regresion.ipynb`, 3.7 (y cobertura medida con N = 1000 y N = 25).

<a id="38-a"></a>
### 3.8-a · El LOCOCV no reinicia `k_error` · `bug` · alto

- **Dónde**: 3.8, *Leave One Class Out Cross-Validation*: el bucle `for c in unique_classes:` hace
  `k_error.append(...)` sin `k_error = []` previo.
- **Qué pasa**: `k_error` sigue lleno con los 24 errores del barrido de 25 moléculas de 3.6, y `error` guarda
  medias acumuladas. El valor impreso mezcla los dos experimentos.
- **Efecto**: la web imprime **50.33** y el texto lo llama "similar" al 5-fold (3.16). El valor correcto
  agregado es **3.20**; por clase va de 2.08 (G3) a 6.67 (G4). Además las clases no son fuentes
  ([2-2](#2-2)).
- **Cómo hacerlo**: `k_error = []` antes del bucle; dar el error de cada clase y el agregado, no medias de
  medias acumuladas.
- **En nuestros notebooks**: `03_regresion.ipynb`, 3.8.

## Capítulo 4 — Clasificación

<a id="41-a"></a>
### 4.1-a · La label es "fracasó por toxicidad" · `datos` · medio

- **Dónde**: 4.1, *Data*: el texto dice que los fármacos fracasan sobre todo por toxicidad, aunque algunos por
  falta de eficacia.
- **Qué pasa**: las **94** moléculas con `FDA_APPROVED = 0` tienen todas `CT_TOX = 1`. En ClinTox la clase
  negativa es "fracasó en ensayos clínicos **por toxicidad**" (MoleculeNet lo plantea como dos tareas). La
  lenalidomida, aprobada en 2005, está como no aprobada.
- **Cómo hacerlo**: describir la label como lo que es, y tener en cuenta que la clase negativa tiene ruido.
- **En nuestros notebooks**: `04_clasificacion.ipynb`, 4.1–4.2.

<a id="42-a"></a>
### 4.2-a · Descarta 4 SMILES sin decir cuáles · `datos` · bajo

- **Dónde**: 4.3 del libro (`valid_mol_idx = [bool(m) for m in molecules]`).
- **Qué pasa**: los 4 SMILES ilegibles son fármacos aprobados con errores químicos: cisplatino escrito con
  `[NH4]`, y sulfinpirazona, oxifenbutazona y fenilbutazona con la pirazolidinadiona escrita como aromática.
- **Cómo hacerlo**: listar lo que se descarta y, si se puede, corregir los SMILES.

<a id="42-b"></a>
### 4.2-b · Las dos clases están escritas de forma distinta · `datos` · alto

- **Dónde**: ClinTox, tal como lo usa el libro desde 4.3.
- **Qué pasa**: las 14 sales son todas no aprobadas; el 64.1 % de las aprobadas lleva cargas formales, frente
  al 3.2 % de las no aprobadas; las aromáticas aprobadas están en minúscula en el 99 % de los casos, las no
  aprobadas en ninguno. Una regla que solo lee el texto del SMILES acierta el **86.0 %** y detecta el
  96.8 % de los fracasos. La fuente coincide con la label: un **atajo**.
- **Efecto**: en los descriptores Mordred, normalizar las moléculas cambia 323 de los 483 descriptores en el
  58.7 % de las aprobadas. `SLogP` pasa de 0.630 a 0.542 de separación, y `BalabanJ` ≈ 0 marca las 14 sales.
  En el clasificador de 4.4 (5 repartos estratificados), la pérdida de test es 0.199 con las moléculas
  originales y **0.246** con las normalizadas, frente a un listón de 0.238: **sin el atajo, el modelo no supera
  al modelo constante**. Las 6 columnas de aminas protonadas (`NsNH3`, `SsssNH`...) aportan solas 0.014 de la
  ventaja. El AUC se medirá en 4.5.
- **Cómo hacerlo**: normalizar las moléculas antes de calcular descriptores (`rdMolStandardize`:
  `LargestFragmentChooser` + `Uncharger`) y comprobar que el modelo no aprende la fuente.
- **En nuestros notebooks**: `04_clasificacion.ipynb`, 4.2 y 4.3.

<a id="43-a"></a>
### 4.3-a · Los `NaN` no vienen de std = 0 · `texto` · medio

- **Dónde**: 4.3, *Molecular Descriptors*: `# we have some nans in features, likely because std was 0`.
- **Qué pasa**: de las 1130 columnas descartadas solo **113** son constantes. Las otras **1017** tienen huecos
  porque Mordred no pudo calcularlas en alguna molécula. Una sola molécula incompleta (con el comodín `*`)
  borra **147** descriptores para todo el dataset.
- **Cómo hacerlo**: mirar qué falta y por qué. Quitar primero las moléculas rotas (sin la del `*` quedan 630
  features en vez de 483), y luego tirar o rellenar las columnas con muchos huecos.
- **En nuestros notebooks**: `04_clasificacion.ipynb`, 4.3.

<a id="43-b"></a>
### 4.3-b · Estandariza antes de repartir · `método` · medio

- **Dónde**: 4.3: `features -= features.mean(); features /= features.std()` sobre las 1480 moléculas, antes
  del reparto train/test de 4.4.
- **Qué pasa**: **fuga de información**: el test contribuye a la media y la std con las que se transforma el
  train. Aquí mueve poco las cifras (la media, 0.016 std en mediana), pero esconde **10 descriptores
  constantes en el train** que en una molécula de test valen **38.44** (fuera del rango del train).
- **Efecto en 4.4**: con el reparto del libro, estandarizar solo con el train deja la pérdida de test casi igual
  (0.500 frente a 0.503, media de 3 semillas): el problema de ese test es otro ([4.4-a](#44-a)).
- **Cómo hacerlo**: repartir primero; calcular media y std solo con el train
  (`StandardScaler().fit(X_train)`), y quitar las columnas constantes en el train.
- **En nuestros notebooks**: `04_clasificacion.ipynb`, 4.3.

<a id="44-a"></a>
### 4.4-a · Reparto 80/20 sin barajar sobre un CSV en orden alfabético · `método` · alto

- **Dónde**: 4.4: `train_N = int(len(labels) * 0.8)`; `test_x = features[train_N:]`.
- **Qué pasa**: el CSV de ClinTox está casi en orden alfabético por el SMILES (el 93.0 % de las filas
  consecutivas), así que el test son los SMILES de `CCCC...` a `S=[Se]=S`. Tiene 27 no aprobadas (9.1 %, frente
  al 5.7 % del train) y **15 de las 22 moléculas sin carbono** (cloruros y óxidos metálicos, As₂O₃, ²⁰¹TlCl, I₂,
  SeS₂). Los pesos iniciales (`np.random.normal`) tampoco tienen semilla ([G-1](#g-1)).
- **Efecto**: el modelo no supera al listón de predecir la proporción de clases (pérdida de test 0.479 frente a
  0.315). Cuatro inorgánicos aprobados, a los que el modelo da p ≤ 0.001, aportan el **34.7 %** de la pérdida.
  Con un reparto estratificado al azar (5 repartos), el mismo modelo sí supera al listón en los 5: 0.170–0.230
  frente a 0.238.
- **Cómo hacerlo**: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=N)`, que baraja y conserva
  la proporción de clases. Si lo que se quiere es medir la extrapolación a química distinta, hacerlo a propósito
  (por ejemplo, con un scaffold split, como en 3.8).
- **En nuestros notebooks**: `04_clasificacion.ipynb`, 4.4.

<a id="44-b"></a>
### 4.4-b · Da el modelo por bien entrenado sin un listón · `texto` · alto

- **Dónde**: 4.4, tras la curva de entrenamiento: "We are making good progress with our classifier, as judged
  from testing loss. [...] We have a reasonably well-trained model."
- **Qué pasa**: la curva no se compara con nada. La referencia mínima es el modelo constante que predice la
  proporción de clases del train: entropía cruzada 0.315 en ese test.
- **Efecto**: la pérdida de test no baja nunca de esa línea (mínima 0.369, final 0.479) y termina peor que con
  los pesos iniciales, antes de entrenar (0.394). En el train sí la supera (0.126 frente a 0.217): sobreajuste.
  La "buena progresión" es la recuperación tras los primeros pasos (1.076 después del primero).
- **Cómo hacerlo**: dibujar siempre la pérdida del modelo constante junto a la curva (en regresión, la de
  predecir la media).
- **En nuestros notebooks**: `04_clasificacion.ipynb`, 4.4.

<a id="44-c"></a>
### 4.4-c · El bucle se salta el último batch · `bug` · bajo

- **Dónde**: 4.4: `batch_idx = range(0, train_N, batch_size)` y `for i in range(len(batch_idx) - 1):`.
- **Qué pasa**: `batch_idx` tiene 37 inicios (de 0 a 1152) y el bucle hace 36 batches, así que el que empieza en
  1152 no se hace. Las 32 moléculas de las filas 1152–1183 no se usan nunca. Es el mismo tipo de fallo que
  [3.6-b](#36-b).
- **Cómo hacerlo**: `for start in range(0, train_N, batch_size): x = X[start:start + batch_size]`, que incluye
  el último batch aunque esté incompleto.
- **En nuestros notebooks**: `04_clasificacion.ipynb`, 4.4 (la función `entrenar` usa todos los batches).

<a id="44-d"></a>
### 4.4-d · $\vec w\cdot\vec x + b$ no es la distancia a la frontera · `texto` · bajo

- **Dónde**: 4.4, *Linear Perceptron*: "The term $\vec{w}\cdot \vec{x} + b$ is called distance from the
  decision boundary".
- **Qué pasa**: es proporcional a la distancia (con signo); la distancia geométrica es
  $(\vec w\cdot\vec x + b)/\lVert\vec w\rVert$. Con la misma frontera y los pesos el doble de grandes, la
  "distancia" sale el doble. La idea de "confianza" no cambia.
- **En nuestros notebooks**: `04_clasificacion.ipynb`, 4.4.

## Pendientes de comprobar

Vistos en el código del libro, pero aún sin medir su efecto en nuestros notebooks.

<a id="p-1"></a>
### P-1 · `accuracy` usa `yhat` en vez de `hard_yhat` · `bug` · por medir (4.5)

- **Dónde**: `def accuracy(y, yhat)`: calcula `hard_yhat = np.where(yhat > 0.5, ...)` y luego usa
  `np.sum(np.abs(y - yhat))`.
- **Qué pasa**: `hard_yhat` no se usa. La función devuelve $1 - \text{media}|y - p|$, una "exactitud blanda" que
  depende de las probabilidades, no la fracción de aciertos.
- **Cómo hacerlo**: `np.mean(hard_yhat == y)`, o `sklearn.metrics.accuracy_score`.
