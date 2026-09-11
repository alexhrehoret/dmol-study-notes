#!/bin/zsh
# Doble clic en este archivo abre Jupyter Lab en el proyecto.
# Para cerrarlo: pulsa Ctrl+C en esta ventana (o ciérrala).

cd "$(dirname "$0")"

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate dmol

echo ""
echo "  Entorno: $CONDA_DEFAULT_ENV  ($(python --version))"
echo "  Carpeta: $(pwd)"
echo "  Abriendo Jupyter Lab en Chrome..."
echo "  (para cerrar: Ctrl+C aqui)"
echo ""

jupyter lab
