"""Auditoría de sesgo: ¿los segmentos quedan desbalanceados por atributos sensibles?

Los atributos sensibles (género, edad, estado civil, educación) **nunca** entran
al clustering ni al clasificador. Aquí solo se describen después, para detectar
si un segmento termina concentrando a un grupo demográfico. Si así fuera, usar el
segmento para decidir sobre personas reproduciría esa desigualdad aunque el
modelo no "vea" el atributo.

Privacidad: en las tablas publicables, las celdas con menos de ``n_minimo``
personas se suprimen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estadistica import chi2_permutacion


def auditar(
    contexto: pd.DataFrame,
    etiquetas: np.ndarray,
    atributos: list[str],
    n_minimo: int,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Asociación segmento × atributo sensible.

    Retorna
    -------
    tuple[pandas.DataFrame, dict]
        (resumen por atributo con χ², p por permutación y V de Cramér;
        tabla publicable por atributo: % por segmento con celdas pequeñas suprimidas).
    """
    seg = pd.Series(etiquetas, index=contexto.index, name="segmento")
    filas, tablas = [], {}
    for at in atributos:
        if at not in contexto:
            continue
        r = chi2_permutacion(contexto[at], seg, n_perm, rng)
        filas.append({"atributo": at, **r})
        conteo = pd.crosstab(contexto[at], seg)
        pct = (100 * conteo / conteo.sum()).round(0)
        publicable = pct.astype(int).astype(str) + "%"
        publicable[conteo < n_minimo] = f"<{n_minimo} pers."
        tablas[at] = publicable
    resumen = pd.DataFrame(filas)
    resumen["lectura"] = np.where(
        resumen["p_permutacion"] < 0.05,
        "Desbalance detectable: revisar antes de usar el segmento para decidir",
        "Sin desbalance detectable con esta muestra",
    )
    return resumen, tablas
