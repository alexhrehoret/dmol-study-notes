#!/bin/zsh
# Doble clic en este archivo abre Jupyter Lab con TODOS los notebooks del
# proyecto en pestañas. Para cerrarlo: pulsa Ctrl+C en esta ventana.

cd "$(dirname "$0")"

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate dmol

# La primera vez (o si se borró) genera el workspace con todos los notebooks.
# Después, Jupyter guarda solo las pestañas que dejes abiertas.
if ! ls ~/.jupyter/lab/workspaces/dmol-*.jupyterlab-workspace > /dev/null 2>&1; then
  echo "  Primera vez: preparando las pestañas..."
  python .jupyter/crear_workspace.py
fi

echo ""
echo "  Entorno: $CONDA_DEFAULT_ENV  ($(python --version))"
echo "  Carpeta: $(pwd)"
echo "  Abriendo Jupyter Lab en Chrome..."
echo "  (para cerrar: Ctrl+C aqui)"
echo ""

jupyter lab --LabApp.default_url=/lab/workspaces/dmol
