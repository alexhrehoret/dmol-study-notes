# Progreso — Deep Learning for Molecules and Materials (dmol.pub)

Estado: **Cap. 2, sección 2.3** · Actualizado: 2026-09-11

Marca `[x]` cuando una sección esté hecha y entendida.

## A. Repaso de matemáticas
- [ ] 1. Tensors and Shapes — `notebooks/01_tensores.ipynb`

## B. Machine Learning
- [ ] 2. Introduction to Machine Learning — `notebooks/02_introduccion.ipynb`
  - [x] 2.1 The Ingredients
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
  - [ ] 2.5 Chapter Summary ← siguiente
  - [ ] 2.6 Exercises
- [ ] 3. Regression & Model Assessment
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
### 2026-09-11
- Montado el entorno conda `dmol` y la estructura del proyecto.
- Descargado AqSolDB (`data/curated-solubility-dataset.csv`, 9982 compuestos).
- Hechas 2.1 y 2.2 en `notebooks/02_introduccion.ipynb`: features (17 descriptores RDKit) vs label (Solubility).
- Repo privado en GitHub: https://github.com/alexhrehoret/ML_Projects
- Hecha la exploración de 2.3: RDKit, histograma de solubilidad (std=2.37, la vara de medir) y moléculas extremas.
- 2.4: k-means (k=4) da 4 grupos con identidad quimica clara. PCA: PC1=tamano (53.9%), PC2=polaridad (13.7%), confirma la multicolinealidad. PC2 explica 4x menos varianza pero correlaciona mejor con la solubilidad (-0.49 vs -0.29). No hay codo: el espacio quimico es continuo.
- 2.3.9: parity plot. r2=0.51, RMSE 1.658. El modelo comprime el rango (regresion a la media) y falla sistematicamente en las colas. Descubiertas 573 mezclas (nombre con ';') con descriptores sumados: RMSE 2.31 vs 1.61 de los compuestos puros.
- 2.3.7 y 2.3.8: SGD (batch 32) y estandarizacion. Con features estandarizadas eta=0.1 llega a loss 2.747 / RMSE 1.658, casi el optimo exacto (2.708). Los pesos salen sin sentido quimico por multicolinealidad.
- 2.3.6: descenso de gradiente. eta=1e-6 converge lento (loss 4.24, RMSE 2.06) y el bias no se mueve; eta=1e-5 diverge. Causa: escalas incompatibles entre features.
- 2.3.5: modelo lineal y MSE definidos (sin entrenar). Listón a batir: RMSE 2.37 (predecir la media).
- 2.3.4: correlaciones. MolLogP r=-0.61 es la mejor feature (r2=0.37). Detectada multicolinealidad entre los 5 descriptores de tamaño y 400 BalabanJ=0 falsos (363 son sales).
