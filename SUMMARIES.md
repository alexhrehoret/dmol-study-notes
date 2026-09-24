# Chapter summaries — what to remember

One page per chapter, core ideas only. Details are in `notebooks/` (Spanish). Numbers come from
my notebooks (fixed seeds), not from the book. Spanish version: [RESUMENES.md](RESUMENES.md).

---

## Chapter 1 — Tensors and Shapes (`01_tensores.ipynb`)

1. **A tensor = rank + shape.** `ndim` says how many axes; `shape`, how long each one is. The first
   axis is usually the **batch**: `(32, 50, 3)` = 32 molecules × 50 atoms × 3 coordinates.
2. **Reductions:** `axis=k` means "axis k disappears". `keepdims=True` keeps it with size 1.
3. **Broadcasting** aligns shapes **from the end**. Each axis must be equal, be 1, or not exist.
   `np.newaxis` (or `None`) adds a size-1 axis to force the alignment you want.
4. **`einsum`**: indices missing from the output are summed over. It is the notation of the papers.
5. **`*` is element-wise; `@` is matrix multiplication.**
6. **View vs copy:** slices and reshapes share memory. Use `.copy()` if you will modify.
7. **Habit:** for any bug, the first thing to do is `print(a.shape, b.shape)`.

Pattern that comes back throughout the book (interatomic distance matrix):

```python
diffs = coords[:, None, :] - coords[None, :, :]   # (N, N, 3)
D = np.sqrt(np.sum(diffs ** 2, axis=-1))          # (N, N)
```

---

## Chapter 2 — Introduction to ML (`02_introduccion.ipynb`, `02_ejercicios_resueltos.ipynb`)

**The scheme that serves for the whole book:**
data $(\vec{x}_i, y_i)$ → model $\hat{f}(\vec{x}; \vec{w})$ → loss $L$ → optimizer → better $\vec{w}$.
What changes later is the model and the molecular representation; the loop is always the same.

**Supervised** (labels exist: approximate $f$) vs **unsupervised** (no labels: find structure, with
no objective criterion of success).

**Core ideas:**

1. **Look at the data before modelling.** The multicollinearity, the 400 spurious `BalabanJ = 0` and
   the 573 mixtures with summed descriptors came from looking, not from training.
2. **Always have a dumb baseline.** Predicting the mean gives RMSE **2.37** (the standard deviation of
   solubility). Without that reference, no other RMSE means anything.
3. **No single feature is enough.** The best one, `MolLogP`, has $r = -0.61$ ($r^2 = 0.37$).
4. **Standardize the features** (mean 0, standard deviation 1). With incompatible scales, gradient
   descent chokes: with $\eta = 10^{-6}$ it stalls at RMSE 2.06, and with $10^{-5}$ it diverges.
   Standardized, it accepts $\eta = 0.1$ and reaches **RMSE 1.658** ($r^2 = 0.51$). The exact optimum
   of a linear model on these 17 features is 1.65: going further needs another model or another
   representation.
5. **The learning rate $\eta$ is the critical hyperparameter**: too large diverges, too small never
   arrives. **SGD** (batches of 32) gives the same result with much less computation, but a small
   batch needs a smaller $\eta$.
6. **Predicting is not explaining.** With multicollinearity the weights contradict chemistry, and the
   model still predicts.
7. **One number is not enough.** The parity plot showed that the model compresses the range, is
   biased in the tails (+2.61 for the extremely insoluble) and extrapolates absurdly (it predicts
   down to −29.8).
8. **Unsupervised:** k-means ($k = 4$) gives groups with a chemical identity, but it **always** returns
   $k$ groups, and there is no elbow: chemical space is continuous. In PCA, **PC1 = size (53.9 %)** and
   **PC2 = polarity (13.7 %)**. PC2 explains less variance but correlates more with solubility (−0.49
   vs −0.29), because PCA never looks at the label.
9. **All of the above was measured on the training data**, so it is not an honest evaluation. That is
   what chapter 3 is about.

**From the exercises:**
- "Linear" means linear **in the parameters**: the curve comes from the features (ex. 3).
- Scaling the **labels** does not change the valid learning rate, because the Hessian of the MSE only
  depends on $X$ (ex. 5).
- MSE and MAE give different models: the loss defines what "being wrong" means (ex. 6).
- The `Group` classes have their own biases (ex. 10). *Corrected in 3.8*: `Group` is not the
  experimental source but AqSolDB's reliability group (number of measurements and whether they agree).

---

## Chapter 3 — Regression and model assessment (`03_regresion.ipynb`, `03_ejercicios_resueltos.ipynb`)

**The equation of the chapter:** expected error on a new point = **bias² + variance + noise**. Bias =
underfitting (the model cannot represent $f$); variance = overfitting (the model changes with the training
data); noise = the floor nobody can go below.

**Core ideas:**

1. **Only the test error counts.** With 25 molecules and 18 parameters: RMSE 0.59 on train, 2.60 on test.
   The test set takes no part in training, not even in the standardization.
2. **Overfitting needs noise and flexibility together.** Polynomial of 3.2.1: test 1.96 with noise and the
   right features, **3537** with noise and extra features, 0 without noise.
3. **The worst failures are extrapolations** (raffinose: 11 H-bond donors vs a training maximum of 4).
   Variance concentrates where the model extrapolates.
4. **What matters is the parameters / data ratio.** Test error peaks when the number of features approaches
   the number of data points. **More data reduce variance as 1/N, but not bias** (ex. 2: on solubility
   everything converges to MSE ≈ 2.7).
5. **Regularization**: L2 shrinks weights, L1 zeroes them (feature selection), L∞ equalizes them (ex. 5).
   **Standardize before regularizing**: with 7 features, λ = 0.1 cuts test error from 1385 to 8.67. Lasso
   selection is **unstable** with correlated features (187 different sets in 200 samples of 35 molecules;
   only `MolLogP` always survives).
6. **Everything decided from data (λ, features, early stopping) is training** and must not touch the test
   set. Use **cross-validation**, nested if there are hyperparameters.
7. **Shuffle before k-fold.** AqSolDB is sorted by source: unshuffled 2.97 ± 2.10; shuffled **2.80 ± 0.33**.
   On the full dataset there is no overfitting (train 2.724).
8. **With little data, CV is necessary but not sufficient**: with 25 molecules its estimate barely correlates
   with the real error (Spearman ≈ 0.1).
9. **Per-prediction intervals**: bootstrap only measures model variance (25.6 % coverage for a promised
   95 %); **jackknife+** covers 96.0 %, but on average: outside the training range, 62.3 %.
10. **The error depends on which molecules the model is applied to.** 10-fold 2.80, leave-one-source-out
    5.02, scaffold split 5.48, predicting the mean 5.61. The reliable zone is the **applicability domain**,
    and it depends on the model: the linear model gains nothing from having a near-identical molecule in
    the training set.

**Own findings and corrections to the book:** the 17 descriptors have rank 16 (3.3), training set sampled
with replacement (3.4), ridge fitted with non-converged Adam (3.5), unshuffled k-fold (3.6), jackknife+ with
training residuals (3.7), LOCOCV without resetting `k_error`, and `Group` ≠ source (3.8). *Correction to ch.
2*: the linear optimum is MSE 2.724, not 2.708 (a float32 artifact).

**From the exercises:**
- Without noise there is no noise to learn, but if the data do not determine the weights (more features than
  data, or a sample that does not vary in some descriptor) you get train 0 and test > 0: **indeterminacy**
  (ex. 1).
- In L1 the feature most correlated with $y$ survives and those that repeat others' information go first;
  which member of a correlated group survives is chance (ex. 3 and 4).
- **The best linear model** (ex. 6): ridge on 204 RDKit descriptors **clipped to the training range** + Morgan
  count fingerprint. **MSE 1.873** in nested 10-fold (vs 2.792) and **4.132** on a scaffold split (vs 5.485).
  Unclipped, the 204 descriptors give 7.46, worse than the mean, because of a few extrapolations. What helps
  is the representation and the clipping, not the regularization. Still ~1.5 MSE above the experimental
  noise (≈ 0.27): bias a linear model cannot remove.
