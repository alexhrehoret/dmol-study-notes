# ML_Projects — curso autoguiado de dmol.pub

Seguimos el libro **Deep Learning for Molecules and Materials** (https://dmol.pub) capítulo a capítulo.

## Cómo trabajamos
- Una sección del libro a la vez. Al terminar cada una: enseñar resultado, explicar, y parar.
- Explicaciones en español; código y variables en inglés, como en el libro.
- **Nivel de detalle alto**: explicar qué es cada herramienta, por qué se usa y qué significa cada número del resultado. Cero conocimiento previo de ML o de tooling de Python; la química sí se da por sabida.
- Reproducir primero lo que hace el libro; las variantes propias van después y se marcan como tal.
- Actualizar `PROGRESO.md` al cerrar cada sección.

## Estructura
- `notebooks/` — un notebook por capítulo (`02_introduccion.ipynb`, ...).
- `data/` — datasets descargados una sola vez y cacheados en local (no re-descargar desde el notebook).
- `src/` — funciones reutilizables entre capítulos (featurizers, métricas, plots).
- `figuras/` — figuras exportadas.

## Entorno
Env conda `dmol` (Python 3.11): numpy, pandas, matplotlib, seaborn, scikit-learn, rdkit, jax, jupyterlab.

```bash
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate dmol
```

Kernel de Jupyter: `Python (dmol)`.

Arranque: doble clic en `Abrir Jupyter.command`. Usa un *workspace* de Jupyter Lab llamado `dmol`
que abre todos los notebooks en pestañas. Si añades un notebook nuevo y quieres que se abra solo:

```bash
python .jupyter/crear_workspace.py
```

## Detalles que difieren del libro
- El libro hace `import dmol` solo para aplicar su estilo de gráficas; no lo usamos.
- El libro descarga el CSV por URL en cada celda; nosotros leemos `data/curated-solubility-dataset.csv`.
