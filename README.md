# dmol study notes — *Deep Learning for Molecules and Materials*, worked through step by step

*[Versión en español más abajo](#en-español)*

These are my study notebooks for the book
**[Deep Learning for Molecules and Materials](https://dmol.pub)** by **Andrew D. White**.
I am a chemist learning machine learning, and this is the way I chose to work through the book:
one section at a time, running every piece of code, and stopping to explain every tool, every
decision and every number in the output.

The book is excellent but dense if you start with no ML or Python-tooling background. I needed a
more detailed version, so I wrote one. I am making it public in case it helps someone else who
needs the same.

> **All credit for the original material goes to Andrew D. White.** This is not an official
> companion and is not affiliated with the author. Any errors here are mine.

## The original book — please cite it

- Book: <https://dmol.pub> · Source: <https://github.com/whitead/dmol-book>
- License of the book: [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/)
- Citation requested by the author:

```bibtex
@article{white2021deep,
  title   = {Deep Learning for Molecules and Materials},
  journal = {Living Journal of Computational Molecular Science},
  author  = {White, Andrew D},
  url     = {https://dmol.pub},
  year    = {2021},
  volume  = {3},
  number  = {1},
  pages   = {1499},
  doi     = {10.33011/livecoms.3.1.1499}
}
```

The solubility data used throughout is **AqSolDB**: Sorkun, M. C.; Khetan, A.; Er, S.
*AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of
compounds.* Scientific Data **2019**, 6, 143. <https://doi.org/10.1038/s41597-019-0151-1>

## What is different from the book

- **Explanations assume no prior ML knowledge** (chemistry is assumed). Every library call, every
  hyperparameter and every printed number is explained.
- **Reproduce first, then extend.** Each section first reproduces what the book does; my own
  experiments come after and are explicitly marked as *variante propia* (own variant).
- **Fixed random seeds**, so the numbers quoted in the text match the cells when re-run. The book
  does not fix them.
- **Updated to current library versions** (the book dates from 2021; e.g. `sns.distplot` →
  `sns.histplot`).
- **Hidden code checked.** Several figures in the book come from cells hidden on the website. I
  pulled them from the source repository and compared them with the text. Some things I found:
  - **3.3** — the 17 descriptors have rank 16 (`RingCount = NumAromaticRings + NumAliphaticRings`),
    which is why the book's curves go flat at 16–17 features.
  - **3.4** — the training set is sampled *with replacement* (about 8 distinct points out of 10),
    which inflates the measured variance about 7.6× for the 4-feature model.
  - **3.5** — the ridge fit (100 Adam steps) does not reach the minimum. Without standardizing the
    features, the clean bias–variance "U" does not appear. With standardization, the test error
    drops from 1385 to 8.67.
  - **3.6** — the k-fold code does not shuffle, and the CSV is sorted by data source: 2.97 ± 2.10
    instead of 2.80 ± 0.33.
  - **3.7** — the jackknife+ code uses training residuals instead of leave-one-out residuals. With
    25 molecules its intervals, which should cover at least 90 %, cover only 61.9 % of the true
    values (93.3 % once fixed).
  - **3.8** — the leave-one-class-out loop never resets its error list (the site prints 50.33; the
    real value is 3.20), and the `Group` column is AqSolDB's reliability group, not the data source.
- **Exercises solved with extra analysis.** For example, chapter 3's "best linear model": ridge on
  204 RDKit descriptors clipped to the training range plus Morgan count fingerprints reaches MSE
  1.87 in nested cross-validation (vs 2.79 for the book's model) and 4.13 on a scaffold split (vs
  5.49). Without clipping, the same descriptors do worse than predicting the mean.
- **Looking at the data.** For example, AqSolDB contains 573 mixtures whose descriptors are summed,
  and 400 molecules (mostly salts) with a spurious `BalabanJ = 0`. In ClinTox the two classes were
  written differently: all 14 salts are "not approved", 64 % of approved drugs carry formal charges
  vs 3 % of the rest, and aromatic rings are lowercase in 99 % of approved drugs and in none of the
  others. A rule that only reads the SMILES text gets 86 % accuracy.

## Contents

| Chapter | Notebook | Status |
|---|---|---|
| 1. Tensors and Shapes | [`01_tensores.ipynb`](notebooks/01_tensores.ipynb) | ✅ |
| 2. Introduction to Machine Learning | [`02_introduccion.ipynb`](notebooks/02_introduccion.ipynb) · [exercises solved](notebooks/02_ejercicios_resueltos.ipynb) | ✅ |
| 3. Regression & Model Assessment | [`03_regresion.ipynb`](notebooks/03_regresion.ipynb) · [exercises solved](notebooks/03_ejercicios_resueltos.ipynb) | ✅ |
| 4. Classification | [`04_clasificacion.ipynb`](notebooks/04_clasificacion.ipynb) | in progress (4.1–4.2) |
| 5–22 | — | to do |

- **[SUMMARIES.md](SUMMARIES.md)** — the core lessons of each finished chapter, in English, on one
  page each.
- [RESUMENES.md](RESUMENES.md) — the same summaries in Spanish.
- [PROGRESO.md](PROGRESO.md) — my study log, session by session (Spanish).

The notebooks themselves are written in **Spanish**. Code, variable names and figure labels are in
English, as in the book.

## How this was made

I used **Claude (Anthropic) as a tutor** throughout. I chose the pace, the questions and the
decisions, and checked every result; Claude drafted most of the explanatory text and code under
that direction. [`CLAUDE.md`](CLAUDE.md) contains the working rules we followed. The most
important one: run the code first and write the interpretation from the real output, never from
memory.

## Running the notebooks

```bash
conda env create -f environment.yml
conda activate dmol
# download the dataset once (see data/README.md)
curl -L -o data/curated-solubility-dataset.csv \
  https://raw.githubusercontent.com/whitead/dmol-book/main/data/curated-solubility-dataset.csv
jupyter lab notebooks/
```

## License

My notebooks and text are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see [LICENSE](LICENSE)). They are
a derivative of the book, which is licensed CC BY-NC 3.0. Code and ideas taken from the book remain
© Andrew D. White under its original license.

---

## En español

Estos son mis cuadernos de estudio del libro **[Deep Learning for Molecules and Materials](https://dmol.pub)**,
de **Andrew D. White**. Soy químico y estoy aprendiendo machine learning. Esta es la forma que elegí
para desgranar el libro: una sección cada vez, ejecutando todo el código y parando a explicar cada
herramienta, cada decisión y cada número del resultado.

El libro es excelente, pero denso si se empieza sin base de ML ni de herramientas de Python. Yo
necesitaba una versión más detallada y la escribí. La publico por si le sirve a alguien que necesite
lo mismo.

**Todo el mérito del material original es de Andrew D. White** (cita arriba). Esto no es material
oficial ni tiene relación con el autor, y los errores son míos. Los notebooks están en español; el
código, en inglés. Los resúmenes de cada capítulo están en [RESUMENES.md](RESUMENES.md), y la
bitácora en [PROGRESO.md](PROGRESO.md).

He usado **Claude (Anthropic) como tutor**. El ritmo, las preguntas, las decisiones y la
comprobación de los resultados son míos; Claude ha redactado buena parte del texto y del código
siguiendo esas indicaciones.
