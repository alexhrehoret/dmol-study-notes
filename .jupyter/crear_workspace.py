"""Genera el 'workspace' de Jupyter Lab con todos los notebooks del proyecto abiertos en pestañas.

Ejecútalo cuando añadas un notebook nuevo y quieras que se abra solo al arrancar:

    conda activate dmol
    python .jupyter/crear_workspace.py
"""
import json
import pathlib
import subprocess
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DESTINO = RAIZ / ".jupyter" / "workspace-dmol.json"

def orden(ruta):
    """Ordena por capítulo, y dentro de cada uno el capítulo antes que sus ejercicios."""
    nombre = pathlib.Path(ruta).stem
    capitulo = nombre.split("_")[0]
    es_ejercicio = 1 if "ejercicio" in nombre else 0
    return (capitulo, es_ejercicio, nombre)


notebooks = sorted(
    (p.relative_to(RAIZ).as_posix() for p in (RAIZ / "notebooks").glob("*.ipynb")),
    key=orden,
)
if not notebooks:
    sys.exit("No hay notebooks en notebooks/")

data = {}
widgets = []
for ruta in notebooks:
    clave = f"notebook:{ruta}"
    data[clave] = {"data": {"path": ruta, "factory": "Notebook"}}
    widgets.append(clave)

# la pestaña activa al abrir: el último capítulo (no sus ejercicios)
capitulos = [i for i, r in enumerate(notebooks) if "ejercicio" not in r]
activo = capitulos[-1] if capitulos else len(widgets) - 1

data["layout-restorer:data"] = {
    "main": {
        "dock": {"type": "tab-area", "currentIndex": activo, "widgets": widgets},
        "current": widgets[activo],
    },
    "down": {"size": 0, "widgets": []},
    "left": {
        "collapsed": False,
        "visible": True,
        "current": "filebrowser",
        "widgets": ["filebrowser", "running-sessions", "@jupyterlab/toc:plugin"],
    },
    "right": {"collapsed": True, "visible": True, "widgets": ["jp-property-inspector", "debugger-sidebar"]},
    "relativeSizes": [0.22, 0.78, 0],
}

DESTINO.write_text(json.dumps({"data": data, "metadata": {"id": "dmol"}}, indent=1))
subprocess.run(["jupyter", "lab", "workspaces", "import", str(DESTINO)], check=True)

print(f"\nWorkspace 'dmol' con {len(notebooks)} notebooks:")
for n in notebooks:
    print("  -", n)
