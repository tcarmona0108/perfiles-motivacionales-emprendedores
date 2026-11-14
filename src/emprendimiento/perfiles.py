"""Perfilado de segmentos y validación externa.

Advertencia que va en el reporte
--------------------------------
Las pruebas de Kruskal-Wallis comparan segmentos **en las mismas respuestas que
se usaron para formarlos**. El clustering garantiza que difieran, así que los
p-valores son descriptivos (circulares), no evidencia de que los grupos existan.
Lo informativo es el **tamaño de efecto** (ε²), que ordena qué ítems distinguen
más a los segmentos. La evidencia de que los grupos existen viene de la
estabilidad y de la prueba contra "sin grupos" (``segmentacion``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estadistica import ajustar_p, chi2_permutacion, kruskal_epsilon


def resumen_segmentos(
    z: pd.DataFrame, medias_1a5: pd.DataFrame, etiquetas: np.ndarray
) -> pd.DataFrame:
    """Tamaño y medias por segmento.

    Parámetros
    ----------
    z:
        Puntajes factoriales estandarizados (media 0, DE 1): permiten ver en qué
        factor un segmento está por encima o por debajo del promedio.
    medias_1a5:
        Media de los ítems de cada factor en la escala original 1–5, más fácil
        de comunicar.
    etiquetas:
        Segmento de cada persona (1..k).
    """
    g = pd.Series(etiquetas, name="segmento")
    t = pd.concat(
        [
            g.value_counts().sort_index().rename("n"),
            (100 * g.value_counts(normalize=True)).sort_index().rename("pct"),
            z.groupby(etiquetas).mean().add_prefix("z_"),
            medias_1a5.groupby(etiquetas).mean().add_prefix("media_"),
        ],
        axis=1,
    )
    t.index.name = "segmento"
    return t


def nombrar_segmentos(resumen: pd.DataFrame, factores: list[str]) -> dict[int, str]:
    """Nombres descriptivos generados con una regla fija (reproducible).

    * Si todos los factores de un segmento tienen el mismo signo, el segmento se
      distingue por **nivel**: "Motivación alta" o "Motivación moderada" en
      todos los factores. Se dice "moderada" y no "baja" porque la media de
      estos grupos suele seguir por encima del punto medio de la escala.
    * Si no, se distingue por **forma** y se nombra por el factor relativo más alto.
    """
    nombres = {}
    for seg, fila in resumen.iterrows():
        zf = fila[[f"z_{f}" for f in factores]]
        zf.index = factores
        if (zf > 0).all():
            nombres[seg] = "Motivación alta en todos los factores"
        elif (zf < 0).all():
            nombres[seg] = "Motivación moderada en todos los factores"
        else:
            nombres[seg] = f"Predomina: {(zf - zf.mean()).idxmax()}"
    # Desambiguar nombres repetidos agregando el factor relativo más bajo.
    vistos = pd.Series(nombres).value_counts()
    for seg, nombre in list(nombres.items()):
        if vistos[nombre] > 1:
            zf = resumen.loc[seg, [f"z_{f}" for f in factores]]
            zf.index = factores
            nombres[seg] = f"{nombre}; menor: {(zf - zf.mean()).idxmin()}"
    return nombres


def items_que_diferencian(
    x: pd.DataFrame, etiquetas: np.ndarray, etiquetas_items: dict[str, str], correccion: str
) -> pd.DataFrame:
    """Kruskal-Wallis por ítem con corrección de Holm y ε², ordenado por efecto."""
    filas = []
    for it in x.columns:
        r = kruskal_epsilon(x[it], etiquetas)
        medias = x[it].groupby(etiquetas).mean()
        filas.append(
            {"item": it, "etiqueta": etiquetas_items[it], **r}
            | {f"media_seg{s}": m for s, m in medias.items()}
        )
    t = pd.DataFrame(filas).set_index("item")
    t["p_ajustado"] = ajustar_p(t["p"], correccion)
    return t.sort_values("epsilon2", ascending=False)


def validacion_externa(
    contexto: pd.DataFrame, etiquetas: np.ndarray, n_perm: int, rng: np.random.Generator
) -> pd.DataFrame:
    """¿Difieren los segmentos en variables que NO se usaron para formarlos?

    Variables disponibles relacionadas con emprender:
    * ``ocupacion`` (solo emprendedor vs emprendedor y estudiante).
    * ``anios_funcionamiento`` (trayectoria del negocio).

    Ambas son validadores **débiles**: toda la muestra ya emprende, así que no
    hay contraste emprendedor/no emprendedor ni una medida de éxito.
    """
    filas = []
    seg = pd.Series(etiquetas, index=contexto.index)
    if "ocupacion" in contexto:
        r = chi2_permutacion(contexto["ocupacion"], seg, n_perm, rng)
        pct = pd.crosstab(seg, contexto["ocupacion"], normalize="index")
        col = "Ambos" if "Ambos" in pct else pct.columns[-1]
        filas.append(
            {
                "variable": "Ocupación (emprendedor / ambos)",
                "prueba": "χ² (p por permutación)",
                "estadistico": r["chi2"],
                "p": r["p_permutacion"],
                "efecto": r["v_cramer"],
                "medida_efecto": "V de Cramér",
                "detalle": "; ".join(f"seg{s}: {100 * v:.0f}% ambos" for s, v in pct[col].items()),
            }
        )
    if "anios_funcionamiento" in contexto:
        anios = contexto["anios_funcionamiento"]
        ok = anios.notna()
        r = kruskal_epsilon(anios[ok], seg[ok].to_numpy())
        medianas = anios[ok].groupby(seg[ok]).median()
        filas.append(
            {
                "variable": "Años de funcionamiento",
                "prueba": "Kruskal-Wallis",
                "estadistico": r["H"],
                "p": r["p"],
                "efecto": r["epsilon2"],
                "medida_efecto": "ε²",
                "detalle": "; ".join(f"seg{s}: mediana {m:g}" for s, m in medianas.items()),
            }
        )
    return pd.DataFrame(filas)
