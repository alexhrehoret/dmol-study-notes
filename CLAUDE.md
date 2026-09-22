# ML_Projects — curso autoguiado de dmol.pub

Seguimos el libro **Deep Learning for Molecules and Materials** (https://dmol.pub) capítulo a capítulo.

## Cómo trabajamos
- Una sección del libro a la vez. Al terminar cada una: enseñar resultado, explicar, y parar.
- Explicaciones en español; código y variables en inglés, como en el libro.
- **Nivel de detalle alto**: explicar qué es cada herramienta, por qué se usa y qué significa cada número del resultado. Cero conocimiento previo de ML o de tooling de Python; la química sí se da por sabida.
- Reproducir primero lo que hace el libro; las variantes propias van después y se marcan como tal.
- Actualizar `PROGRESO.md` al cerrar cada sección, y commit + push al repo **público**
  https://github.com/alexhrehoret/dmol-study-notes (antes `ML_Projects`, privado).
- **Calcular antes de redactar**: ejecutar el código y mirar la salida real antes de escribir la
  celda de markdown que la interpreta. Nunca poner cifras de memoria.
- **Resumen al cerrar cada capítulo**, antes de pasar al siguiente: breve, solo las enseñanzas
  core (incluidos los hallazgos de nuestras variantes, con sus cifras). Va en `RESUMENES.md`
  (español) **y** en `SUMMARIES.md` (inglés), un apartado por capítulo, no en el notebook.
- **El repo es público** y sirve de credencial ante un público que no habla español. Al cerrar un
  capítulo: actualizar la tabla *Contents* del `README.md` (y *What is different from the book* si
  hay hallazgos nuevos). Los notebooks siguen en español; traducir uno a `notebooks_en/` solo si se
  pide, y solo con el capítulo cerrado.
- **Una idea por celda de markdown.** No juntar el cierre de una sección con el encabezado de la
  siguiente: rompe la posibilidad de insertar contenido después sin descolocarlo.

## Estructura
- `notebooks/` — un notebook por capítulo. Nomenclatura: `NN_nombre.ipynb` para el capítulo y
  `NN_ejercicios_resueltos.ipynb` para sus soluciones (los enunciados sin resolver se quedan en
  el notebook del capítulo).
- `data/` — datasets descargados una sola vez y cacheados en local (no re-descargar desde el notebook).
- `src/` — funciones reutilizables entre capítulos (featurizers, métricas, plots).
- `figuras/` — figuras exportadas.

## Entorno
Env conda `dmol` (Python 3.11): numpy, pandas, matplotlib, seaborn, scikit-learn, rdkit, jax, jupyterlab.

```bash
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate dmol
```

Kernel de Jupyter: `Python (dmol)`.

Arranque: doble clic en `Abrir Jupyter.command` (solo local: está en `.gitignore`, igual que `.jupyter/`). Usa un *workspace* de Jupyter Lab llamado `dmol`
que abre todos los notebooks en pestañas. Si añades un notebook nuevo y quieres que se abra solo:

```bash
python .jupyter/crear_workspace.py
```

## Figuras
Paleta fija para que todas las gráficas del libro se lean igual (validada para daltonismo):

| Uso | Color |
|---|---|
| serie 1 / una sola serie | `#2a78d6` azul |
| serie 2 / referencia | `#eb6834` naranja |
| serie 3 | `#1baf7a` verde |
| serie 4 | `#4a3aa7` violeta |
| serie 5 | `#eda100` amarillo |

En scatter con muchos puntos: `s=3..5`, `alpha=0.12..0.3`, `rasterized=True`. Quitar siempre los
spines `top` y `right`, `grid(alpha=0.2)`. Guardar con `fig.savefig("../figuras/NN_nombre.png", dpi=150)`
además de mostrarla, para poder revisarla sin abrir Jupyter. En scatter con más de 3 categorías,
usar *small multiples* en vez de más colores.

## Detalles que difieren del libro
- El libro hace `import dmol` solo para aplicar su estilo de gráficas; no lo usamos.
- El libro descarga el CSV por URL en cada celda; nosotros leemos `data/curated-solubility-dataset.csv`.
- El libro es de 2021 y usa APIs retiradas. Ya encontrado: `sns.distplot` → `sns.histplot`.
- `rdkit.Chem.rdMolDescriptors` hay que importarlo explícitamente (`from rdkit.Chem import rdMolDescriptors`).
- El libro no fija semillas aleatorias. Nosotros **sí** (`sample(..., random_state=N)`,
  `np.random.default_rng(N)`), para que las cifras del texto coincidan con las celdas al reejecutar.
  Decirlo en el notebook, y si la semilla cambia la conclusión, enseñar varias.
