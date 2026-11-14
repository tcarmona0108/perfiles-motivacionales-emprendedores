"""Reducción del cuestionario: ¿cuántas preguntas bastan?

Regla de selección
------------------
Por factor se eligen los ítems con mayor carga **entre los "limpios"** (carga
≥ 0.40 y sin carga cruzada). Así cada pregunta del formulario corto mide un solo
factor, lo que hace interpretables los puntajes de la versión corta. Si un
factor tiene menos ítems limpios que los pedidos, se usan los que haya y se
marca: completar con ítems cruzados contaminaría la subescala.

Cómo se juzga la versión corta
------------------------------
* ω de la subescala corta vs la larga (confiabilidad).
* r entre el puntaje corto (media 1–5 de sus ítems) y el puntaje factorial del
  cuestionario completo (validez de reemplazo).
* r entre el puntaje corto y la media de los ítems del factor que **no** entraron
  al formulario corto. Es una prueba más estricta, porque la correlación
  corto-largo está inflada por el solapamiento de ítems (parte-todo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .psicometria import alfa_cronbach, omega_unifactorial


def seleccionar_items(
    asignacion: pd.DataFrame, k: int
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Elige hasta ``k`` ítems por factor.

    Parámetros
    ----------
    asignacion:
        Salida de :func:`emprendimiento.psicometria.asignar_items` con el factor
        ya renombrado.
    k:
        Ítems por factor (3 = mínimo para identificar un factor).

    Retorna
    -------
    tuple[dict, pandas.DataFrame]
        (factor -> ítems elegidos, tabla de ítems con columna ``en_corto``).
    """
    t = asignacion.copy()
    t["en_corto"] = False
    elegidos: dict[str, list[str]] = {}
    for factor, grupo in t[t["limpio"]].groupby("factor"):
        top = grupo["carga"].abs().sort_values(ascending=False).index[:k].tolist()
        elegidos[factor] = top
        t.loc[top, "en_corto"] = True
    return elegidos, t


def puntajes_media(x: pd.DataFrame, grupos: dict[str, list[str]]) -> pd.DataFrame:
    """Puntaje de cada subescala como media de sus ítems (escala 1–5, fácil de explicar)."""
    return pd.DataFrame({f: x[its].mean(axis=1) for f, its in grupos.items()})


def comparar_corto_largo(
    x: pd.DataFrame,
    r: pd.DataFrame,
    asignacion: pd.Series,
    corto: dict[str, list[str]],
    puntajes_largos: pd.DataFrame,
) -> pd.DataFrame:
    """Compara, factor por factor, la subescala corta con la larga.

    Parámetros
    ----------
    x:
        Ítems.
    r:
        Matriz policórica.
    asignacion:
        Factor asignado a cada ítem (versión larga).
    corto:
        Ítems del formulario corto por factor.
    puntajes_largos:
        Puntajes factoriales con el cuestionario completo (columnas = factores).
    """
    filas = []
    for f, items_corto in corto.items():
        items_largo = asignacion.index[asignacion == f].tolist()
        resto = [i for i in items_largo if i not in items_corto]
        p_corto = x[items_corto].mean(axis=1)
        filas.append(
            {
                "factor": f,
                "items_largo": len(items_largo),
                "items_corto": len(items_corto),
                "omega_largo": omega_unifactorial(r.loc[items_largo, items_largo].to_numpy()),
                "omega_corto": omega_unifactorial(r.loc[items_corto, items_corto].to_numpy()),
                "alfa_largo": alfa_cronbach(x[items_largo]),
                "alfa_corto": alfa_cronbach(x[items_corto]),
                "r_corto_puntaje_factorial": np.corrcoef(p_corto, puntajes_largos[f])[0, 1],
                "r_corto_resto": (
                    np.corrcoef(p_corto, x[resto].mean(axis=1))[0, 1] if resto else np.nan
                ),
                "items": ", ".join(items_corto),
            }
        )
    return pd.DataFrame(filas).set_index("factor")


def curva_items(
    x: pd.DataFrame, asignacion: pd.DataFrame, puntajes_largos: pd.DataFrame
) -> pd.DataFrame:
    """Correlación con el puntaje factorial completo al ir agregando ítems.

    Para cada factor se agregan sus ítems limpios en orden de carga y se mide
    cuánto del puntaje completo se recupera con 1, 2, 3… preguntas. Es la
    respuesta visual a "¿cuántas preguntas bastan?".
    """
    filas = []
    for f, grupo in asignacion[asignacion["limpio"]].groupby("factor"):
        orden = grupo["carga"].abs().sort_values(ascending=False).index.tolist()
        for k in range(1, len(orden) + 1):
            p = x[orden[:k]].mean(axis=1)
            filas.append({"factor": f, "n_items": k, "r": np.corrcoef(p, puntajes_largos[f])[0, 1]})
    return pd.DataFrame(filas)
