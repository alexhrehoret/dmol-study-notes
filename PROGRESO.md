# Progreso — Deep Learning for Molecules and Materials (dmol.pub)

Estado: **Capítulos 1 y 2 completos. Siguiente: capítulo 3** · Actualizado: 2026-09-14

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
- [ ] 3. Regression & Model Assessment ← siguiente
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

---

## Punto de partida de la próxima sesión

**Capítulo 3 — Regression & Model Assessment** (https://dmol.pub/ml/regression.html),
en un `notebooks/03_regresion.ipynb` nuevo.

Es el capítulo que cierra el agujero que dejamos abierto: **todo el RMSE = 1.658 del capítulo 2
está medido sobre los mismos datos con los que se entrenó**. Ahí se ven train/test split,
overfitting y validación cruzada.

Ganchos ya preparados en el capítulo 2 que conviene retomar:
- El aviso de **data leakage** de 2.3.8: hay que calcular media y desviación **solo con train**.
- El **ejercicio 10**: las fuentes `Group` tienen sesgos → el split por grupos es más honesto
  que el aleatorio.
- El **error irreducible**: las 573 mezclas y las medidas dudosas ponen un suelo al RMSE.
- El **techo del modelo lineal**: 1.65. Para bajar de ahí hay que cambiar de modelo o de features.

Al crear el notebook, ejecutar `python .jupyter/crear_workspace.py` para que entre en las pestañas.
