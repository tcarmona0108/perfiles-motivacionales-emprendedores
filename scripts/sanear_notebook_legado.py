"""Genera la copia pública del notebook original sin outputs identificables.

Qué hace
--------
Lee el notebook original (guardado solo en local, en ``data/raw/legado_privado``)
y escribe ``legacy/Proyecto_Profesora_Angela.ipynb`` con el **código y el texto
intactos**, pero borrando los outputs de las celdas que imprimen datos a nivel de
persona.

Por qué así
-----------
La regla del proyecto es conservar el trabajo original sin modificarlo para
mostrar el "antes vs después". Pero dos celdas guardaron en sus outputs datos
crudos con cuasi-identificadores:

* celda 5: ``data.head(5)`` (ciudad, objeto social, demografía de 5 personas);
* celda 41: ``ciudad.value_counts()`` (24 ciudades con una sola persona).

Se borran solo esos outputs, y no todos, para que las figuras y métricas
originales (silueta, cargas, matrices de confusión) sigan visibles en la
comparación. El resto de outputs son agregados o puntajes sin columnas que
permitan enlazarlos con una persona.

Uso::

    uv run python scripts/sanear_notebook_legado.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ORIGEN = RAIZ / "data" / "raw" / "legado_privado" / "Proyecto_Profesora_Angela.ipynb"
DESTINO = RAIZ / "legacy" / "Proyecto_Profesora_Angela.ipynb"

# Índices (0-based) de las celdas cuyos outputs exponen datos a nivel de persona.
CELDAS_CON_DATOS_PERSONALES = (5, 41)

AVISO = (
    "[Output eliminado por privacidad: mostraba datos crudos con "
    "cuasi-identificadores. Ver docs/diagnostico.md §2.5]\n"
)


def sanear(origen: Path = ORIGEN, destino: Path = DESTINO) -> list[int]:
    """Copia el notebook borrando los outputs sensibles.

    Parámetros
    ----------
    origen:
        Ruta del notebook original con outputs.
    destino:
        Ruta de la copia pública.

    Retorna
    -------
    list[int]
        Índices de las celdas cuyos outputs se reemplazaron.
    """
    nb = json.loads(origen.read_text(encoding="utf-8"))
    tocadas: list[int] = []
    for i in CELDAS_CON_DATOS_PERSONALES:
        celda = nb["cells"][i]
        if celda["cell_type"] != "code":
            raise ValueError(f"La celda {i} no es de código; revisar índices.")
        celda["outputs"] = [{"name": "stdout", "output_type": "stream", "text": [AVISO]}]
        tocadas.append(i)

    destino.parent.mkdir(parents=True, exist_ok=True)
    # indent=1 + ensure_ascii=False reproduce el formato de Jupyter/Colab.
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return tocadas


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


if __name__ == "__main__":
    celdas = sanear()
    print(f"Copia saneada escrita en {DESTINO.relative_to(RAIZ)}")
    print(f"Outputs eliminados en celdas: {celdas}")
    print(f"SHA-256 del original (local, no versionado): {_sha256(ORIGEN)}")
