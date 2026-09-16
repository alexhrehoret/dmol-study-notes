# Progreso — Deep Learning for Molecules and Materials (dmol.pub)

Estado: **Capítulos 1 y 2 completos. Capítulo 3 en curso (siguiente: 3.4)** · Actualizado: 2026-09-16

Marca `[x]` cuando una sección esté hecha y entendida.

## A. Repaso de matemáticas
- [x] 1. Tensors and Shapes — `notebooks/01_tensores.ipynb`

## B. Machine Learning
- [x] 2. Introduction to Machine Learning — `notebooks/02_introduccion.ipynb`
  - [x] 2.1 The Ingredients (ampliada: los 6 ingredientes y la notación)
  - [x] **2.2 Supervised Learning**
  - [x] 2.3 Running This Notebook
    - [x] 2.3.1 RDKit: de texto a molécula
    - [x] 2.3.2 Distribución de la solubilidad
    - [x] 2.3.3 Moléculas extremas
    - [x] 2.3.4 Correlación feature–label
    - [x] 2.3.5 Modelo lineal + función de pérdida (MSE)
    - [x] 2.3.6 Descenso de gradiente con JAX
    - [x] 2.3.7 Batching / SGD
    - [x] 2.3.8 Estandarizar features
    - [x] 2.3.9 Parity plot y rendimiento

  - [x] 2.4 Unsupervised Learning (clustering + PCA)
  - [x] 2.5 Chapter Summary
  - [x] 2.6 Exercises — enunciados en `02_introduccion.ipynb`, soluciones en `02_ejercicios_resueltos.ipynb`
- [ ] 3. Regression & Model Assessment — `notebooks/03_regresion.ipynb`
  - [x] 3.1 Running This Notebook (qué es regresión, esperanza/varianza, imports)
  - [x] 3.2 Overfitting — introducción (25 train / 25 test con datos reales)
  - [x] 3.2.1 Overfitting with Synthetic Data (4 escenarios + controles sin ruido + 10 semillas)
  - [x] 3.2.2 Overfitting Conclusion
  - [x] 3.3 Exploring Effect of Feature Number (libro + variante con descriptores RDKit)
  - [ ] 3.4 Bias Variance Decomposition ← siguiente
  - [ ] 3.5 Regularization (L2, L1)
  - [ ] 3.6 Strategies to Assess Models (k-fold, LOOCV)
  - [ ] 3.7 Computing Other Measures (bootstrap, jackknife+)
  - [ ] 3.8 Training Data Distribution (leave-one-class-out, scaffold splits)
  - [ ] 3.9 Chapter Summary
  - [ ] 3.10 Exercises
- [ ] 4. Classification
- [ ] 5. Kernel Learning

## C. Deep Learning
- [ ] 6. Deep Learning Overview
- [ ] 7. Standard Layers
- [ ] 8. Graph Neural Networks
- [ ] 9. Input Data & Equivariances
- [ ] 10. Equivariant Neural Networks
- [ ] 11. Explaining Predictions
- [ ] 12. Attention Layers
- [ ] 13. Deep Learning on Sequences
- [ ] 14. Variational Autoencoder
- [ ] 15. Normalizing Flows
- [ ] 16. Modern Molecular NNs

## D. Aplicaciones
- [ ] 17. Predicting DFT Energies with GNNs
- [ ] 18. Generative RNN in Browser

## E. Capítulos contribuidos
- [ ] 19. Equivariant NN for Predicting Trajectories
- [ ] 20. Pretraining

## F. Apéndice
- [ ] 21. Style Guide
- [ ] 22. Changelog

---

## Notas de cada sesión

### 2026-09-11 — Montaje y capítulos 1 y 2
- Entorno conda `dmol` (Python 3.11) y estructura del proyecto. Repo privado:
  https://github.com/alexhrehoret/ML_Projects
- AqSolDB cacheado en `data/curated-solubility-dataset.csv` (9982 compuestos, 26 columnas).
- **2.1 y 2.2**: los 6 ingredientes, la notación ($f$ real vs $\hat{f}$ modelo), features
  (17 descriptores RDKit) vs label (`Solubility`).
- **2.3.1–2.3.3**: RDKit (InChI/SMILES → `Mol`), histograma de solubilidad
  (**std = 2.37 → el listón a batir**), moléculas extremas.
- **2.3.4**: correlaciones. `MolLogP` es la mejor feature ($r = -0.61$, $r^2 = 0.37$).
  Detectadas **multicolinealidad** entre 5 descriptores de tamaño y **400 `BalabanJ = 0`**
  falsos (363 son sales).
- **2.3.5**: modelo lineal y MSE. Baselines: predecir 0 → RMSE 3.74; predecir la media → 2.37.
- **2.3.6**: descenso de gradiente. $\eta = 10^{-6}$ converge lentísimo (RMSE 2.06) y el bias
  no se mueve; $\eta = 10^{-5}$ diverge. Causa: escalas incompatibles entre features.
- **2.3.7 y 2.3.8**: SGD con batch 32 (311 pasos igualan a 2000 de full-batch) y estandarización
  (permite $\eta = 0.1$). **RMSE 2.06 → 1.658**, casi el óptimo exacto de mínimos cuadrados
  (2.708). Los pesos resultantes contradicen la química: multicolinealidad.
- **2.3.9**: parity plot. $r^2 = 0.51$. Tres patologías: compresión del rango, sesgo sistemático
  en las colas (+2.61 en los insolubles) y extrapolación absurda (predice hasta −29.8).
  Descubiertas **573 mezclas** (nombre con `;`) con descriptores sumados: RMSE 2.31 vs 1.61.
- **2.4**: k-means ($k = 4$) da 4 grupos con identidad química clara. PCA: **PC1 = tamaño (53.9 %)**,
  **PC2 = polaridad (13.7 %)**, confirma la multicolinealidad. PC2 explica 4× menos varianza pero
  correlaciona mejor con la solubilidad (−0.49 vs −0.29): PCA es no supervisado. **No hay codo** —
  el espacio químico es continuo.
- **2.5 y 2.6**: resumen + glosario, y los 10 ejercicios planteados sin resolver.
- **Capítulo 1 completo** (`01_tensores.ipynb`): rank/shape, einsum, broadcasting, matriz de
  distancias interatómicas, newaxis/squeeze/reshape/ellipsis, vista vs copia.

### 2026-09-13 — Ejercicios del capítulo 2 resueltos
En `notebooks/02_ejercicios_resueltos.ipynb` (autocontenido). Hallazgos:
- **Ej 5**: estandarizar las *labels* **NO** cambia el learning rate — el Hessiano del MSE
  ($\frac{2}{N}X^\top X$) solo depende de $X$. Umbral teórico $2/\lambda_{max} = 0.1091$;
  empírico entre 0.108 y 0.110.
- **Ej 6**: MAE vs MSE. Cada uno gana en su métrica. MAE mejora la mediana (0.835 vs 0.983) y
  empeora los 100 peores errores (929 vs 680).
- **Ej 7**: batch pequeño exige $\eta$ más pequeño (bs = 1 diverge con $\eta = 0.01$).
- **Ej 8**: las etiquetas de cluster son nombres, no cantidades (demostrado permutando semillas).
- **Ej 10**: las fuentes experimentales (`Group`) se solapan pero tienen sesgos (G2: −4.05 vs
  G5: −2.68; G1 es el 78 % del dataset). Implica que el split aleatorio train/test puede mentir.
- **CORRECCIÓN a 2.3.2**: el "hombro" del histograma **no** se explica por mezcla de fuentes;
  todas las `Group` tienen distribuciones de solubilidad similares (std 2.26–2.38).

### 2026-09-14 — Herramientas y ampliación
- `Abrir Jupyter.command` ahora arranca con un **workspace** de Jupyter Lab (`dmol`) que abre
  todos los notebooks en pestañas. Se regenera con `python .jupyter/crear_workspace.py`.
- **Ampliado el ejercicio 3** a petición del usuario: qué significa "lineal" (respecto a los
  parámetros, no a $x$), la analogía del ecualizador con la curva descompuesta en sus 3 canales,
  y los paisajes de pérdida comparados — parábola con 1 mínimo ($w$ fuera del seno) frente a
  26 mínimos locales ($w$ dentro). Enlaza con caps. 5 y 6.
- Arreglado: la ampliación había quedado tras el ejercicio 4 porque la celda de cierre del 3
  llevaba pegado el enunciado del 4.

### 2026-09-14 — Capítulo 3: 3.1 y la introducción de 3.2
- Notebook nuevo `03_regresion.ipynb` (añadido al workspace). Semillas fijadas (`random_state=4`,
  `default_rng(4)`); el libro no las fija.
- **Experimento 25/25** (estandarizado con estadísticas solo del train): el test tiene su mínimo en
  el **paso 28** y luego sube; al final la loss de test es **19×** la de train.
  RMSE train 0.59 / test 2.60, frente a 2.85 de predecir la media en test (solo un 9 % mejor).
- Predicciones imposibles en test: **rafinosa** +6.92 (real +0.30; 11 dadores de H frente a un máximo
  de 4 en train) y un pigmento bis-azoico −15.20 (real −7.26).
- *Variante propia*: las 10 moléculas de test con alguna feature fuera del rango del train dan
  **RMSE 3.55 frente a 1.70** → gancho al dominio de aplicabilidad (3.8).
- *Variante propia*: 8 semillas. Hay brecha en todas; 7/8 con mínimo temprano (la 1 no sobreajusta en
  2000 pasos); en 3/8 el modelo final es peor que la media. RMSE de test entre 1.17 y 3.93 según el
  sorteo → gancho a la validación cruzada (3.6).
- Nota: `jax.example_libraries.optimizers` lo importa el libro pero no lo usa; omitido.

### 2026-09-16 — Capítulo 3: 3.2.1 y 3.2.2 (overfitting con datos sintéticos)
- $f(x) = x^3 - x^2 + x - 1$, 20 puntos; train = 10 de los extremos, test = 10 del centro (sin ruido).
  Ajuste con `np.linalg.lstsq` (mínimo exacto, sin pasos). Ruido `default_rng(4).normal(scale=5)`.
  Eje Y en (−50, 30) en vez de (−40, 40) para que no se corte el punto de train de −43.3.
- Loss train / test (semilla 4): sin ruido 0 / 0 · ruido 21.49 / **1.96** · ruido + $x^0..x^6$
  4.56 / **3537** · ruido + $[x^2, x, e^{-x^2}, \cos x, 1]$ 44.67 / **50 206**.
- Escenario 4: peso **−413** en $e^{-x^2}$, que en train vale ≤ 0.049 y en test hasta 0.98. Se usa
  para aprender ruido → misma historia que la rafinosa (feature fuera del rango del train).
- *Variante propia (controles sin ruido)*: features de más sin ruido → 0 / 0 y pesos exactos
  (overfitting = ruido + flexibilidad). Features malas sin ruido → 24.5 / 21.9: **underfitting / sesgo**
  (gancho a 3.4). Responde a la duda del libro: en el escenario 4 hay las dos cosas.
- *Variante propia (10 semillas)*: features de más peor que las exactas en 10/10 (×6.7 a ×10 850).
  Escenario 2 test entre 0.24 y 118.8; test < train solo en 5/10. Train medio 16.8 (4 params) y
  7.6 (7 params), bajo la varianza del ruido (25).
- Nota: este bloque reutiliza `w`, `test_x`, `test_y` (como el libro); reejecutar de arriba abajo.

### 2026-09-16 — Capítulo 3: 3.3 (efecto del número de features)
- El libro **no muestra código**; está en celdas `remove-cell` de `ml/regression.ipynb` en
  github.com/whitead/dmol-book (bajado con `gh api`). Diferencias con su texto: las $f$ features son
  combinaciones lineales aleatorias de los 17 descriptores ($X \cdot F$), el ajuste es `lstsq`
  (`adam_fit` definida pero sin usar) y la 2.ª figura es con **500** moléculas, no 250.
- **Rango de los 17 descriptores = 16**: `RingCount = NumAromaticRings + NumAliphaticRings`. Las curvas
  del libro son exactamente planas desde f = 16–17 → su experimento no puede cruzar f = N.
- Libro N = 25: mejor test 3.69 (f = 3), meseta test 15.11 / train 0.57. N = 500: meseta 3.10 / 2.43.
  Train y test se sortean por separado → 17–32 moléculas repetidas por reparto con N = 500.
- *Variante propia*: `Descriptors.CalcMolDescriptors` (217 en RDKit 2026.03.6), caché en
  `data/rdkit_descriptores.csv`. Fuera `Ipc` (hasta 1e158) y 12 descriptores de cargas Gasteiger
  (BCUT2D ×8 fallan en 886 moléculas, casi todas sales/metales) → 9980 moléculas × 204 descriptores.
  Mediana de 100 repartos (la media llega a 2e7 por extrapolaciones).
  - N = 25: pico del test en f = 24–25 (mediana 118–138), train = 0, **doble descenso** hasta 5.72 con
    150 features; nunca bate a predecir la media (5.61).
  - N = 250: óptimo 3.71 con f = 30; después overfitting (75.4 con 200).
  - Norma de los pesos: pico 29.3 en f = 24; 1.05 con 150 (lstsq da la solución de norma mínima) →
    gancho a L2 (3.5).

---

## Punto de partida de la próxima sesión

**3.4 Bias Variance Decomposition** en `notebooks/03_regresion.ipynb` (añadir celdas tras el resumen
de 3.3, que es la última celda). Mirar primero si el libro tiene celdas `remove-cell` en GitHub.

Ganchos que siguen abiertos para el resto del capítulo:
- **El RMSE = 1.658 del capítulo 2 sigue sin medirse en test** → 3.6 (k-fold).
- **Ejercicio 10**: sesgos por `Group` → leave-one-class-out (3.8.1).
- **Extrapolación / rafinosa** → dominio de aplicabilidad y scaffold splits (3.8).
- **Multicolinealidad** del capítulo 2 y **norma de los pesos** de 3.3 → regularización L2/L1 (3.5).
- **Underfitting / sesgo** visto en los controles de 3.2.1 → 3.4.
- Early stopping mencionado; el conjunto de validación aún no se ha tratado en detalle.
