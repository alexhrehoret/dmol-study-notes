# Data

The CSV files are not tracked in git. Download once:

| File | Source | Used in |
|---|---|---|
| `curated-solubility-dataset.csv` | AqSolDB as distributed with the book: <https://raw.githubusercontent.com/whitead/dmol-book/main/data/curated-solubility-dataset.csv> (9982 compounds, 26 columns) | chapters 2, 3 |
| `rdkit_descriptores.csv` | Generated automatically by `03_regresion.ipynb` (section 3.3) the first time it runs: ~200 RDKit descriptors per molecule, cached because it is slow to compute | chapter 3 |
| `clintox.csv.gz` | ClinTox (MoleculeNet) as distributed with the book: <https://github.com/whitead/dmol-book/raw/main/data/clintox.csv.gz> (1484 drugs; columns `smiles`, `FDA_APPROVED`, `CT_TOX`) | chapter 4 |

AqSolDB: Sorkun, M. C.; Khetan, A.; Er, S. *Scientific Data* **2019**, 6, 143.
<https://doi.org/10.1038/s41597-019-0151-1>
