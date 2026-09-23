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
