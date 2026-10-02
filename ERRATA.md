# Errata and notes on the book

A log of the bugs and questionable choices found in *Deep Learning for Molecules and Materials*
(https://dmol.pub) while reproducing it. It grows with every section. Spanish version: `ERRATAS.md`.

The book is very good, and none of these entries invalidates the ideas it teaches. Some do change the
numbers and, in a few cases, the conclusion drawn from a figure. The book's code comes from its source
repository (`github.com/whitead/dmol-book`, folder `ml/`), which includes the cells hidden on the website.

**Types**

| Type | Meaning |
|---|---|
| `bug` | the code does not do what the text says, and the result is wrong |
| `method` | the code runs, but the procedure is not the right one (data leakage, no shuffling, no convergence...) |
| `text` | the text does not match the code or the math |
| `data` | a dataset problem the book does not mention that affects the results |
| `API` | code that no longer runs with current library versions |

**Impact**: **high** = changes the conclusion; **medium** = changes the numbers, not the idea; **low** = minor.

## Index

| ID | Section | Type | Impact | Summary |
|---|---|---|---|---|
| [G-1](#g-1) | whole book | `method` | medium | random seeds are never fixed |
| [2-1](#2-1) | 2.3 | `API` | low | `sns.distplot` no longer exists |
| [2-2](#2-2) | 2.3, 3.8 | `data` | high | `Group` is not the data source |
| [2-3](#2-3) | 2.3 | `data` | medium | AqSolDB has 573 mixtures with summed descriptors and 400 spurious `BalabanJ = 0` |
| [3.3-a](#33-a) | 3.3 | `text` | medium | the text does not describe the experiment the code runs |
| [3.3-b](#33-b) | 3.3 | `method` | high | the 17 descriptors have rank 16: the curves cannot cross f = N |
| [3.3-c](#33-c) | 3.3 | `method` | low | train and test are drawn separately and share molecules |
| [3.4-a](#34-a) | 3.4 | `text` | low | writes $f - \epsilon$ instead of $f + \epsilon$ |
| [3.4-b](#34-b) | 3.4 | `text` | low | the synthetic data silently switch from σ = 5 to σ = 1 |
| [3.4-c](#34-c) | 3.4 | `bug` | medium | the training set is sampled with replacement |
| [3.4-d](#34-d) | 3.4 | `method` | low | the 7-feature "bias²" is estimation error |
| [3.5-a](#35-a) | 3.5 | `method` | high | ridge not converged, not standardized, and penalizing $w_0$ |
| [3.6-a](#36-a) | 3.6 | `method` | medium | k-fold does not shuffle, and the CSV is sorted by source |
| [3.6-b](#36-b) | 3.6 | `bug` | medium | `N // k` segments: some data are never tested |
| [3.6-c](#36-c) | 3.6 | `method` | low | the intercept is computed separately |
| [3.7-a](#37-a) | 3.7 | `text` | low | $2^N$ resamples instead of $\binom{2N-1}{N}$ |
| [3.7-b](#37-b) | 3.7 | `bug` | high | jackknife+ uses training residuals |
| [3.8-a](#38-a) | 3.8 | `bug` | high | LOCOCV never resets `k_error` |
| [4.1-a](#41-a) | 4.1 | `data` | medium | the label is "failed for toxicity", not "not approved" |
| [4.2-a](#42-a) | 4.2 | `data` | low | drops 4 SMILES without saying which |
| [4.2-b](#42-b) | 4.2 | `data` | high | the two classes are written differently (shortcut) |
| [4.3-a](#43-a) | 4.3 | `text` | medium | the `NaN`s do not come from std = 0 |
| [4.3-b](#43-b) | 4.3 | `method` | medium | standardizes before splitting (data leakage) |
| [4.4-a](#44-a) | 4.4 | `method` | high | unshuffled 80/20 split of an alphabetically sorted CSV |
| [4.4-b](#44-b) | 4.4 | `text` | high | calls the model well trained without comparing it to a baseline |
| [4.4-c](#44-c) | 4.4 | `bug` | low | the loop skips the last training batch |
| [4.4-d](#44-d) | 4.4 | `text` | low | $\vec w\cdot\vec x + b$ is proportional to the distance to the boundary, not the distance |
| [P-1](#p-1) | 4.5 | `bug` | ? | `accuracy` uses `yhat` instead of `hard_yhat` |

---

## General

<a id="g-1"></a>
### G-1 · Random seeds are never fixed · `method` · medium

- **Where**: every chapter (`soldata.sample(...)`, `np.random.normal(...)`, `np.random.choice(...)`).
- **What happens**: every run draws different data and weights, so the numbers in the text cannot be
  reproduced. With little data they change a lot: in 3.2, the test RMSE with 25 molecules ranges from 1.17 to
  3.93 depending on the draw.
- **How to do it**: fix the seed (`sample(..., random_state=N)`, `rng = np.random.default_rng(N)`). If the
  conclusion depends on the seed, show several.
- **In our notebooks**: every cell is seeded; 3.2 and 3.2.1 compare 8 and 10 seeds. In 4.4, 3 seeds for the
  initial weights in each split: they change the test loss by at most 0.055, slightly less than the split does
  (0.170–0.230 across 5 stratified splits).

## Chapter 2 — Introduction to ML

<a id="2-1"></a>
### 2-1 · `sns.distplot` no longer exists · `API` · low

- **Where**: 2.3, solubility histogram (`sns.distplot(soldata.Solubility)`).
- **What happens**: seaborn removed `distplot`; the cell fails.
- **How to do it**: `sns.histplot(soldata.Solubility, kde=True)`.
- **In our notebooks**: `02_introduccion.ipynb`, 2.3.2.

<a id="2-2"></a>
### 2-2 · `Group` is not the data source · `data` · high

- **Where**: 2.3 ("`Group` (where the data came from)") and 3.8, *Leave One Class Out*: "our solubility data
  actually is a combination of five other datasets so our data is already pre-classified based on who
  measured the solubility".
- **What happens**: in AqSolDB, `Group` (G1–G5) is the **reliability group** of each value: G1 = a single
  measurement; G2/G3 = two measurements with SD > 0.5 or ≤ 0.5; G4/G5 = three or more with SD > 0.5 or ≤ 0.5.
  The source is the A–I prefix of the `ID`, and there are **9** datasets, not 5. All 9 sources contribute
  molecules to all 5 groups.
- **Effect**: the LOCOCV in 3.8 does not measure what it claims to (generalization to another lab). Done by
  actual source, the pooled error is **5.02** (vs 3.20 by `Group`). Source A rises from 4.75 (10-fold) to
  **10.40**.
- **How to do it**: group by `ID.str[0]` for a by-source LOCOCV.
- **In our notebooks**: `03_regresion.ipynb`, 3.8; note added to `02_ejercicios_resueltos.ipynb` (exercise 10).

<a id="2-3"></a>
### 2-3 · Mixtures and spurious `BalabanJ = 0` in AqSolDB · `data` · medium

- **Where**: 2.3, the dataset with its 17 precomputed descriptors.
- **What happens**: there are **573 mixtures** (name containing `;`) whose descriptors are the sum of their
  components, and **400 `BalabanJ = 0`** values that are not real (363 are salts: multi-fragment structures).
- **Effect**: mixtures have RMSE 2.31 vs 1.61 for the rest with the chapter 2 linear model. The worst molecules
  in 3.6 and 3.8 are almost always mixtures or metals.
- **How to do it**: flag or remove mixtures and salts before modeling, or keep the main fragment
  (`rdMolStandardize.LargestFragmentChooser`).
- **In our notebooks**: `02_introduccion.ipynb`, 2.3.4 and 2.3.9.

## Chapter 3 — Regression and model assessment

<a id="33-a"></a>
### 3.3-a · The text does not describe the experiment the code runs · `text` · medium

- **Where**: 3.3, *Exploring Effect of Feature Number* (hidden code).
- **What happens**: (1) the $f$ features are **random linear combinations** of the 17 descriptors
  ($X \cdot F$, with normal $F$), not distinct descriptors; (2) the fit is `lstsq` (the exact minimum), while the
  text talks about the model "no longer converging"; an `adam_fit` function is defined but never used; (3) the
  second figure uses **500** molecules, not 250.
- **How to do it**: describe the actual experiment, or run the one the text describes.
- **In our notebooks**: `03_regresion.ipynb`, 3.3.

<a id="33-b"></a>
### 3.3-b · Rank 16: the curves cannot cross f = N · `method` · high

- **Where**: 3.3, both figures of error vs number of features.
- **What happens**: `RingCount = NumAromaticRings + NumAliphaticRings`, so the 17 descriptors have **rank 16**.
  The linear combinations never contain more than 16 independent features: from f = 16–17 on, the curves are
  exactly flat.
- **Effect**: the experiment cannot show what happens when crossing f = N = 25 (the interpolation peak and
  double descent).
- **How to do it**: use genuinely different features. With 204 RDKit descriptors and N = 25, the test error
  peaks at f = 24–25 (median 118–138) and then shows **double descent** down to 5.72 with 150 features.
- **In our notebooks**: `03_regresion.ipynb`, 3.3 (variant with RDKit descriptors).

<a id="33-c"></a>
### 3.3-c · Train and test are drawn separately · `method` · low

- **Where**: 3.3, hidden code (`soldata.sample(N)` twice).
- **What happens**: the same molecule can land in both train and test. With N = 500, 17 to 32 shared molecules
  per split.
- **How to do it**: draw 2N molecules once and split them, as the book itself does in 3.2.

<a id="34-a"></a>
### 3.4-a · $f - \epsilon$ instead of $f + \epsilon$ · `text` · low

- **Where**: 3.4, derivation of the bias–variance decomposition.
- **What happens**: the label is $y = f(\vec x) + \epsilon$. The result does not change because the noise is
  symmetric (zero mean).

<a id="34-b"></a>
### 3.4-b · σ = 1 instead of σ = 5, unannounced · `text` · low

- **Where**: 3.4, hidden code (`syn_labels = ... + np.random.normal(size=N)`).
- **What happens**: 3.2.1 used noise with σ = 5; 3.4 regenerates the data with σ = 1 without saying so. The
  numbers of the two sections are not comparable.

<a id="34-c"></a>
### 3.4-c · The training set is sampled with replacement · `bug` · medium

- **Where**: 3.4, hidden code: `np.random.choice(range(N), size=N // 2)`.
- **What happens**: `np.random.choice` samples **with replacement** by default. Of the 10 training points only
  8.02 are distinct on average, and the test points are defined as "those not in train".
- **Effect**: inflates the variance. With 4 features, var = 14.21 (test 16.11) with replacement and **1.87**
  (test **3.77**) without: 7.6 times less.
- **How to do it**: `np.random.choice(range(N), size=N // 2, replace=False)` (the book itself does this in 3.5).
- **In our notebooks**: `03_regresion.ipynb`, 3.4 (variant without replacement).

<a id="34-d"></a>
### 3.4-d · The 7-feature "bias²" is estimation error · `method` · low

- **Where**: 3.4, the 7-feature example.
- **What happens**: the 7-feature model contains the true function, so its real bias is ~0. The 34.10 it
  reports is the error of averaging 1000 predictions of ±1000 (variance 20 572.59): the mean does not converge
  with so few repetitions.
- **How to do it**: report bias² with its uncertainty, or use many more repetitions. With N = 100 the
  7-feature variance drops to 0.30 and the problem disappears.

<a id="35-a"></a>
### 3.5-a · Ridge not converged, not standardized, and penalizing $w_0$ · `method` · high

- **Where**: 3.5, *Regularization*, hidden code:
  `reg_loss = mean((y - x·w)**2) + alpha * mean(w**2)`, fitted with `adam_fit` (100 Adam steps starting from
  the `lstsq` solution).
- **What happens**: (1) 100 steps **do not reach the minimum**: at λ = 1, loss 3.366 vs 1.416 at the exact
  minimum; (2) the 7 polynomial features ($x^0 \dots x^6$) are **not standardized**, so the penalty hits some
  weights far harder than others; (3) `mean(w**2)` **includes $w_0$**, the intercept, which should not be
  penalized. Also, L1 has no code, only text.
- **Effect**: the bias–variance "U" does not appear. The variance rises again (175.06 at λ = 31.62) and the
  best test error is 64.61 at λ = 1000. With exact ridge, standardized features and unpenalized $w_0$, a clean U
  appears: minimum at λ = 0.1 with **test 8.67** (160 times less than 1385.20 unregularized).
- **How to do it**: closed form $\vec w = (X^\top X + \lambda I')^{-1} X^\top \vec y$, with $I'$ = identity with a 0
  at the intercept position, or `sklearn.linear_model.Ridge`, which does not penalize the intercept.
  Standardize first, with training-set statistics.
- **In our notebooks**: `03_regresion.ipynb`, 3.5 (`ridge_exact` and standardized variant).

<a id="36-a"></a>
### 3.6-a · k-fold does not shuffle, and the CSV is sorted by source · `method` · medium

- **Where**: 3.6, *k-fold cross-validation*: `test = soldata[splits[i] : splits[i + 1]]`.
- **What happens**: the segments are consecutive blocks of the CSV, and AqSolDB comes **sorted by source** (A–I
  prefix of the `ID`; A is 3656 rows). Each segment is nearly a different source.
- **Effect**: 10-fold gives **2.97 ± 2.10**; shuffled, **2.80 ± 0.33**. Without shuffling, k = 2 gives 4.92;
  shuffled, 2.80–2.84 for every k. The bad segments are those from source A (mixtures and metals).
- **How to do it**: `sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=N)`. To measure
  generalization to another source, do it on purpose by grouping by source ([2-2](#2-2)).
- **In our notebooks**: `03_regresion.ipynb`, 3.6.

<a id="36-b"></a>
### 3.6-b · `N // k` segments: some data are never tested · `bug` · medium

- **Where**: 3.6: `splits = list(range(0, N + N // k, N // k))`.
- **What happens**: with integer division, if N is not a multiple of k the remainder never falls in a test
  segment. With 9982 molecules and k = 10, 2 are lost. With N = 25 it is serious: for k ≥ 13 the segments hold
  one molecule and **only the first k are ever tested**. LOOCV is also defined but has no code.
- **Effect**: with N = 25, the drop in error from 16.52 to 11.42 between k = 13 and k = 24 is an artifact: the
  models are the same and only the tested molecules change.
- **How to do it**: `KFold` (spreads the remainder over segments of ⌊N/k⌋ and ⌊N/k⌋ + 1) or `np.array_split`.
  For LOOCV, `sklearn.model_selection.LeaveOneOut`.
- **In our notebooks**: `03_regresion.ipynb`, 3.6 (with 25 molecules).

<a id="36-c"></a>
### 3.6-c · The intercept is computed separately · `method` · low

- **Where**: 3.6–3.8: `w = lstsq(x, y)` without a column of ones, then `b = np.mean(y - np.dot(x, w))`.
- **What happens**: an approximation to fitting $\vec w$ and $b$ jointly. On the full data, training MSE 2.738
  vs 2.724 for the joint fit.
- **How to do it**: add a column of ones to `x` (or center `x` and `y` before `lstsq`).

<a id="37-a"></a>
### 3.7-a · $2^N$ resamples instead of $\binom{2N-1}{N}$ · `text` · low

- **Where**: 3.7, *Bootstrap resampling*: "we can generate $2^N$ new datasets".
- **What happens**: the number of distinct multisets of size N is $\binom{2N-1}{N}$. For N = 5 it is 126, not
  32. The point (there are a huge number of them) stands.

<a id="37-b"></a>
### 3.7-b · Jackknife+ uses training residuals · `bug` · high

- **Where**: 3.7, *Jacknife+*:

  ```python
  yhat = np.dot(small_soldata.iloc[idx][feature_names].values, w) + b
  residuals.append(np.abs(yhat - small_soldata.iloc[idx]["Solubility"]))
  ```

- **What happens**: `idx` holds the N − 1 **training** points of that round. The method needs the residual of
  the **left-out** point, `iloc[i]`. Each `append` stores 999 training residuals instead of 1 LOO residual. It
  also prints "± −3.27" (`(qlow - qhigh) / 2` with the terms reversed), and "Average test error" is the median
  training residual.
- **Effect**: with N = 1000 it barely shows (LOO residuals 0.977 vs 0.968 training). With N = 25 the model
  overfits, the training residuals are small, and the interval, which should cover at least 90 %, covers
  **61.9 %**. Fixed: **93.3 %**.
- **How to do it**:

  ```python
  yhat_i = np.dot(small_soldata.iloc[i][feature_names].values, w) + b
  residuals.append(np.abs(yhat_i - small_soldata.iloc[i]["Solubility"]))
  ...
  print(f"... +/- {(qhigh - qlow) / 2:.2f}")
  ```

- **In our notebooks**: `03_regresion.ipynb`, 3.7 (with coverage measured for N = 1000 and N = 25).

<a id="38-a"></a>
### 3.8-a · LOCOCV never resets `k_error` · `bug` · high

- **Where**: 3.8, *Leave One Class Out Cross-Validation*: the `for c in unique_classes:` loop calls
  `k_error.append(...)` with no `k_error = []` before it.
- **What happens**: `k_error` is still full of the 24 errors from the 25-molecule sweep in 3.6, and `error`
  stores running means. The printed value mixes the two experiments.
- **Effect**: the website prints **50.33** and the text calls it "similar" to 5-fold (3.16). The correct pooled
  value is **3.20**; per class it ranges from 2.08 (G3) to 6.67 (G4). Also, the classes are not sources
  ([2-2](#2-2)).
- **How to do it**: `k_error = []` before the loop; report each class's error and the pooled error, not means
  of running means.
- **In our notebooks**: `03_regresion.ipynb`, 3.8.

## Chapter 4 — Classification

<a id="41-a"></a>
### 4.1-a · The label is "failed for toxicity" · `data` · medium

- **Where**: 4.1, *Data*: the text says drugs fail mostly because of toxicity, though some for lack of efficacy.
- **What happens**: all **94** molecules with `FDA_APPROVED = 0` have `CT_TOX = 1`. In ClinTox the negative class
  is "failed clinical trials **for toxicity**" (MoleculeNet frames it as two tasks). Lenalidomide, approved in
  2005, is labeled not approved.
- **How to do it**: describe the label as what it is, and keep in mind that the negative class is noisy.
- **In our notebooks**: `04_clasificacion.ipynb`, 4.1–4.2.

<a id="42-a"></a>
### 4.2-a · Drops 4 SMILES without saying which · `data` · low

- **Where**: section 4.3 of the book (`valid_mol_idx = [bool(m) for m in molecules]`).
- **What happens**: the 4 unreadable SMILES are approved drugs with chemistry errors: cisplatin written with
  `[NH4]`, and sulfinpyrazone, oxyphenbutazone and phenylbutazone with the pyrazolidinedione written as
  aromatic.
- **How to do it**: list what is dropped and, where possible, fix the SMILES.

<a id="42-b"></a>
### 4.2-b · The two classes are written differently · `data` · high

- **Where**: ClinTox, as the book uses it from 4.3 on.
- **What happens**: all 14 salts are not approved; 64.1 % of approved drugs carry formal charges vs 3.2 % of
  the rest; aromatic approved drugs are written in lowercase in 99 % of cases, not approved ones never. A rule
  that only reads the SMILES text gets **86.0 %** accuracy and catches 96.8 % of the failures. The source
  matches the label: a **shortcut**.
- **Effect**: on the Mordred descriptors, normalizing the molecules changes 323 of the 483 descriptors in
  58.7 % of approved drugs. `SLogP` drops from 0.630 to 0.542 class separation, and `BalabanJ` ≈ 0 flags the
  14 salts. In the 4.4 classifier (5 stratified splits), the test loss is 0.199 with the original molecules
  and **0.246** with the normalized ones, against a baseline of 0.238: **without the shortcut, the model does
  not beat the constant model**. The 6 protonated-amine columns (`NsNH3`, `SsssNH`...) alone account for 0.014
  of the advantage. AUC will be measured in 4.5.
- **How to do it**: normalize molecules before computing descriptors (`rdMolStandardize`:
  `LargestFragmentChooser` + `Uncharger`) and check that the model does not learn the source.
- **In our notebooks**: `04_clasificacion.ipynb`, 4.2 and 4.3.

<a id="43-a"></a>
### 4.3-a · The `NaN`s do not come from std = 0 · `text` · medium

- **Where**: 4.3, *Molecular Descriptors*: `# we have some nans in features, likely because std was 0`.
- **What happens**: of the 1130 dropped columns only **113** are constant. The other **1017** have gaps because
  Mordred could not compute them for some molecule. A single incomplete molecule (with the `*` wildcard) wipes
  out **147** descriptors for the whole dataset.
- **How to do it**: look at what is missing and why. Remove broken molecules first (without the `*` one, 630
  features remain instead of 483), then drop or impute columns with many gaps.
- **In our notebooks**: `04_clasificacion.ipynb`, 4.3.

<a id="43-b"></a>
### 4.3-b · Standardizes before splitting · `method` · medium

- **Where**: 4.3: `features -= features.mean(); features /= features.std()` over all 1480 molecules, before
  the train/test split in 4.4.
- **What happens**: **data leakage**: the test set contributes to the mean and std used to transform the
  training set. Here it barely moves the numbers (the mean, 0.016 std at the median), but it hides **10
  descriptors that are constant in the training set** and take the value **38.44** in one test molecule
  (outside the training range).
- **Effect in 4.4**: with the book's split, standardizing with the training set only leaves the test loss
  almost unchanged (0.500 vs 0.503, mean of 3 seeds): that test set has a different problem ([4.4-a](#44-a)).
- **How to do it**: split first; compute mean and std on the training set only (`StandardScaler().fit(X_train)`),
  and drop columns that are constant in the training set.
- **In our notebooks**: `04_clasificacion.ipynb`, 4.3.

<a id="44-a"></a>
### 4.4-a · Unshuffled 80/20 split of an alphabetically sorted CSV · `method` · high

- **Where**: 4.4: `train_N = int(len(labels) * 0.8)`; `test_x = features[train_N:]`.
- **What happens**: the ClinTox CSV is almost in alphabetical order by SMILES (93.0 % of consecutive rows), so
  the test set is the SMILES from `CCCC...` to `S=[Se]=S`. It has 27 not approved (9.1 %, vs 5.7 % in the
  training set) and **15 of the 22 carbon-free molecules** (metal chlorides and oxides, As₂O₃, ²⁰¹TlCl, I₂,
  SeS₂). The initial weights (`np.random.normal`) are not seeded either ([G-1](#g-1)).
- **Effect**: the model does not beat the baseline of predicting the class ratio (test loss 0.479 vs 0.315).
  Four approved inorganic compounds, given p ≤ 0.001 by the model, account for **34.7 %** of the loss. With a
  stratified random split (5 splits), the same model beats the baseline in all 5: 0.170–0.230 vs 0.238.
- **How to do it**: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=N)`, which shuffles and
  preserves the class ratio. If the goal is to measure extrapolation to different chemistry, do it on purpose
  (e.g. with a scaffold split, as in 3.8).
- **In our notebooks**: `04_clasificacion.ipynb`, 4.4.

<a id="44-b"></a>
### 4.4-b · Calls the model well trained without a baseline · `text` · high

- **Where**: 4.4, after the training curve: "We are making good progress with our classifier, as judged from
  testing loss. [...] We have a reasonably well-trained model."
- **What happens**: the curve is not compared with anything. The minimum reference is the constant model that
  predicts the training class ratio: cross-entropy 0.315 on that test set.
- **Effect**: the test loss never goes below that line (minimum 0.369, final 0.479) and ends worse than with the
  initial weights, before training (0.394). On the training set it does beat it (0.126 vs 0.217): overfitting.
  The "good progress" is the recovery after the first steps (1.076 after the first one).
- **How to do it**: always plot the constant model's loss next to the curve (in regression, predicting the
  mean).
- **In our notebooks**: `04_clasificacion.ipynb`, 4.4.

<a id="44-c"></a>
### 4.4-c · The loop skips the last batch · `bug` · low

- **Where**: 4.4: `batch_idx = range(0, train_N, batch_size)` and `for i in range(len(batch_idx) - 1):`.
- **What happens**: `batch_idx` has 37 start points (0 to 1152) and the loop runs 36 batches, so the one
  starting at 1152 is never run. The 32 molecules in rows 1152–1183 are never used. Same kind of bug as
  [3.6-b](#36-b).
- **How to do it**: `for start in range(0, train_N, batch_size): x = X[start:start + batch_size]`, which
  includes the last batch even if it is incomplete.
- **In our notebooks**: `04_clasificacion.ipynb`, 4.4 (the `entrenar` function uses every batch).

<a id="44-d"></a>
### 4.4-d · $\vec w\cdot\vec x + b$ is not the distance to the boundary · `text` · low

- **Where**: 4.4, *Linear Perceptron*: "The term $\vec{w}\cdot \vec{x} + b$ is called distance from the
  decision boundary".
- **What happens**: it is proportional to the (signed) distance; the geometric distance is
  $(\vec w\cdot\vec x + b)/\lVert\vec w\rVert$. Same boundary with weights twice as large gives twice the
  "distance". The "confidence" idea still holds.
- **In our notebooks**: `04_clasificacion.ipynb`, 4.4.

## To be checked

Seen in the book's code, but their effect has not yet been measured in our notebooks.

<a id="p-1"></a>
### P-1 · `accuracy` uses `yhat` instead of `hard_yhat` · `bug` · to be measured (4.5)

- **Where**: `def accuracy(y, yhat)`: computes `hard_yhat = np.where(yhat > 0.5, ...)` and then uses
  `np.sum(np.abs(y - yhat))`.
- **What happens**: `hard_yhat` is never used. The function returns $1 - \text{mean}|y - p|$, a "soft accuracy"
  that depends on the probabilities, not the fraction of correct predictions.
- **How to do it**: `np.mean(hard_yhat == y)`, or `sklearn.metrics.accuracy_score`.
