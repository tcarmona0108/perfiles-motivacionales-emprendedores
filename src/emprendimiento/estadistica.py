"""Pruebas estadísticas compartidas por el perfilado, la validación externa y el sesgo.

Por qué permutación: con n ≈ 120 y varias categorías, muchas tablas de
contingencia tienen frecuencias esperadas < 5 y la aproximación χ² deja de ser
fiable. El p-valor por permutación no depende de esa aproximación.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from statsmodels.stats.multitest import multipletests


def chi2_permutacion(
    a: pd.Series, b: pd.Series, n_perm: int, rng: np.random.Generator
) -> dict[str, float]:
    """Asociación entre dos variables categóricas: χ², p por permutación y V de Cramér.

    La V de Cramér es el tamaño de efecto: 0.1 pequeño, 0.3 mediano, 0.5 grande
    (Cohen, 1988, para gl = 1; con más categorías los umbrales bajan).
    """
    tabla = pd.crosstab(a, b)
    if min(tabla.shape) < 2:
        return {"chi2": np.nan, "p_permutacion": np.nan, "v_cramer": np.nan}
    obs = chi2_contingency(tabla, correction=False)[0]
    av, bv = a.to_numpy(), b.to_numpy()
    nulos = np.array(
        [
            chi2_contingency(pd.crosstab(av, rng.permutation(bv)), correction=False)[0]
            for _ in range(n_perm)
        ]
    )
    n = tabla.to_numpy().sum()
    v = np.sqrt(obs / (n * (min(tabla.shape) - 1)))
    return {
        "chi2": float(obs),
        "p_permutacion": float((1 + (nulos >= obs).sum()) / (1 + n_perm)),
        "v_cramer": float(v),
    }


def kruskal_epsilon(valores: pd.Series, grupos: np.ndarray) -> dict[str, float]:
    """Kruskal-Wallis y su tamaño de efecto ε² = H / (n - 1) (Tomczak y Tomczak, 2014).

    ε² ≈ 0.01 pequeño, 0.08 mediano, 0.26 grande.
    """
    muestras = [valores[grupos == g] for g in np.unique(grupos)]
    h, p = kruskal(*muestras)
    return {"H": float(h), "p": float(p), "epsilon2": float(h / (len(valores) - 1))}


def ajustar_p(p: pd.Series, metodo: str = "holm") -> pd.Series:
    """Corrige p-valores por comparaciones múltiples (Holm controla el error familiar)."""
    return pd.Series(multipletests(p, method=metodo)[1], index=p.index)
