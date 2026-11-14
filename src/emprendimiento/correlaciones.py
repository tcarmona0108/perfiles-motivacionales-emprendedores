"""Correlaciones policóricas para ítems ordinales.

Por qué policóricas y no Pearson
--------------------------------
Los ítems son Likert de 5 puntos con fuerte efecto techo (hasta 66 % de
respuestas en 5). Pearson trata 1–5 como intervalos y, con distribuciones tan
asimétricas, **subestima** la asociación y puede crear "factores de dificultad"
espurios. La correlación policórica supone que detrás de cada ítem hay una
variable latente normal cortada en umbrales, y estima la correlación entre esas
latentes. Es el estándar para EFA con ítems ordinales (Holgado-Tello et al., 2010).

Implementación
--------------
Estimación en dos pasos (Olsson, 1979): (1) umbrales desde las proporciones
marginales; (2) máxima verosimilitud de ρ con los umbrales fijos. La CDF normal
bivariada usa el algoritmo de Genz (2004) vectorizado, precisión ~1e-15, mucho
más rápido que ``scipy.stats.multivariate_normal.cdf`` (importa para el bootstrap).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

# Nodos y pesos de Gauss-Legendre (mitad positiva) usados por Genz (2004).
_GL6 = (
    np.array([0.1713244923791705, 0.3607615730481384, 0.4679139345726904]),
    np.array([0.9324695142031522, 0.6612093864662647, 0.2386191860831970]),
)
_GL12 = (
    np.array([0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
              0.2031674267230659, 0.2334925365383547, 0.2491470458134029]),
    np.array([0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
              0.5873179542866171, 0.3678314989981802, 0.1252334085114692]),
)  # fmt: skip
_GL20 = (
    np.array([0.01761400713915212, 0.04060142980038694, 0.06267204833410906,
              0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
              0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
              0.1527533871307259]),
    np.array([0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
              0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
              0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
              0.07652652113349733]),
)  # fmt: skip
_DOS_PI = 2.0 * np.pi


def _bvn_superior(h: np.ndarray, k: np.ndarray, r: float) -> np.ndarray:
    """P(X > h, Y > k) para normales estándar con correlación ``r`` (Genz, 2004).

    ``h`` y ``k`` son arreglos finitos de la misma forma; ``r`` es escalar en (-1, 1).
    """
    h, k = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float))
    if r == 0.0:
        return norm.sf(h) * norm.sf(k)
    ar = abs(r)
    w, x = _GL6 if ar < 0.3 else _GL12 if ar < 0.75 else _GL20
    w = np.concatenate([w, w])
    x = np.concatenate([1.0 - x, 1.0 + x])
    hk = h * k

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if ar < 0.925:
            hs = (h * h + k * k) / 2.0
            asr = np.arcsin(r) / 2.0
            sn = np.sin(asr * x)
            expo = (sn * hk[..., None] - hs[..., None]) / (1.0 - sn**2)
            bvn = np.exp(expo) @ w
            bvn = bvn * asr / _DOS_PI + norm.sf(h) * norm.sf(k)
            return np.clip(bvn, 0.0, 1.0)

        if r < 0:
            k = -k
            hk = -hk
        as_ = 1.0 - r * r
        a = np.sqrt(as_)
        bs = (h - k) ** 2
        asr = -(bs / as_ + hk) / 2.0
        c = (4.0 - hk) / 8.0
        d = (12.0 - hk) / 80.0
        bvn = np.where(
            asr > -100,
            a * np.exp(asr) * (1 - c * (bs - as_) * (1 - d * bs) / 3 + c * d * as_**2),
            0.0,
        )
        b = np.sqrt(bs)
        sp = np.sqrt(_DOS_PI) * norm.cdf(-b / a)
        bvn = np.where(
            hk > -100, bvn - np.exp(-hk / 2) * sp * b * (1 - c * bs * (1 - d * bs) / 3), bvn
        )
        a2 = a / 2.0
        xs = (a2 * x) ** 2
        asr2 = -(bs[..., None] / xs + hk[..., None]) / 2.0
        sp2 = 1.0 + c[..., None] * xs * (1.0 + 5.0 * d[..., None] * xs)
        rs = np.sqrt(1.0 - xs)
        ep = np.exp(-(hk[..., None] / 2.0) * xs / (1.0 + rs) ** 2) / rs
        termino = np.where(asr2 > -100, np.exp(asr2) * (sp2 - ep), 0.0)
        bvn = (a2 * (termino @ w) - bvn) / _DOS_PI
        if r > 0:
            bvn = bvn + norm.cdf(-np.maximum(h, k))
        else:
            ancho = np.where(h < 0, norm.cdf(k) - norm.cdf(h), norm.cdf(-h) - norm.cdf(-k))
            bvn = np.where(h >= k, -bvn, ancho - bvn)
    return np.clip(bvn, 0.0, 1.0)


def cdf_normal_bivariada(h, k, r: float) -> np.ndarray:
    """Φ₂(h, k; r) = P(X ≤ h, Y ≤ k) para normales estándar con correlación ``r``."""
    return _bvn_superior(-np.asarray(h, float), -np.asarray(k, float), r)


def _umbrales(codigos: np.ndarray, n_cat: int) -> np.ndarray:
    """Umbrales latentes τ a partir de las proporciones acumuladas observadas."""
    acum = np.cumsum(np.bincount(codigos, minlength=n_cat))[:-1] / len(codigos)
    return norm.ppf(acum)


def _codificar(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Recodifica a 0..m-1 solo con las categorías observadas (evita umbrales ±∞ repetidos)."""
    cats, codigos = np.unique(x, return_inverse=True)
    return codigos, len(cats)


def policorica_par(x: np.ndarray, y: np.ndarray, correccion: float = 0.5) -> float:
    """Correlación policórica entre dos variables ordinales (ML en dos pasos).

    Parámetros
    ----------
    x, y:
        Vectores de respuestas ordinales de la misma longitud.
    correccion:
        Corrección de continuidad que se suma a las celdas vacías de la tabla de
        contingencia (0.5, como ``psych::polychoric``). Con n ≈ 120 y efecto
        techo, ~22 % de las celdas están vacías; sin corrección, la máxima
        verosimilitud empuja ρ hacia ±1 (casos Heywood en la EFA).

    Retorna
    -------
    float
        ρ estimada en [-0.999, 0.999]; ``nan`` si alguna variable es constante.
    """
    cx, mx = _codificar(np.asarray(x))
    cy, my = _codificar(np.asarray(y))
    if mx < 2 or my < 2:
        return float("nan")
    tx, ty = _umbrales(cx, mx), _umbrales(cy, my)
    tabla = np.zeros((mx, my))
    np.add.at(tabla, (cx, cy), 1)
    tabla[tabla == 0] = correccion

    # Φ(τ) marginales: bordes de la grilla de la CDF bivariada.
    fx = np.concatenate([[0.0], norm.cdf(tx), [1.0]])
    fy = np.concatenate([[0.0], norm.cdf(ty), [1.0]])
    hh, kk = np.meshgrid(tx, ty, indexing="ij")

    def menos_logver(r: float) -> float:
        grilla = np.zeros((mx + 1, my + 1))
        grilla[1:-1, 1:-1] = cdf_normal_bivariada(hh, kk, r)
        grilla[-1, :] = fy
        grilla[:, -1] = fx
        p = grilla[1:, 1:] - grilla[:-1, 1:] - grilla[1:, :-1] + grilla[:-1, :-1]
        return -float((tabla * np.log(np.clip(p, 1e-300, None))).sum())

    res = minimize_scalar(
        menos_logver, bounds=(-0.999, 0.999), method="bounded", options={"xatol": 1e-6}
    )
    return float(res.x)


def suavizar_definida_positiva(r: np.ndarray, minimo: float = 1e-6) -> tuple[np.ndarray, bool]:
    """Fuerza que una matriz de correlación sea definida positiva.

    Las policóricas se estiman par a par y la matriz resultante puede no ser
    definida positiva, lo que rompe la EFA. Se recortan los autovalores
    negativos y se reescala a diagonal 1 (método de autovalores, como
    ``psych::cor.smooth``).

    Retorna
    -------
    tuple[numpy.ndarray, bool]
        La matriz suavizada y si fue necesario suavizar.
    """
    vals, vecs = np.linalg.eigh(r)
    if vals.min() > minimo:
        return r, False
    vals = np.clip(vals, minimo, None)
    r2 = vecs @ np.diag(vals) @ vecs.T
    dd = np.sqrt(np.diag(r2))
    r2 = r2 / np.outer(dd, dd)
    np.fill_diagonal(r2, 1.0)
    return r2, True


def matriz_policorica(df: pd.DataFrame, correccion: float = 0.5) -> pd.DataFrame:
    """Matriz de correlaciones policóricas de todas las columnas.

    Parámetros
    ----------
    df:
        Ítems ordinales (una columna por ítem).

    Retorna
    -------
    pandas.DataFrame
        Matriz simétrica, definida positiva, con diagonal 1. El atributo
        ``attrs["suavizada"]`` indica si hubo que suavizarla.
    """
    x = df.to_numpy()
    p = x.shape[1]
    r = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            r[i, j] = r[j, i] = policorica_par(x[:, i], x[:, j], correccion)
    r, suavizada = suavizar_definida_positiva(r)
    salida = pd.DataFrame(r, index=df.columns, columns=df.columns)
    salida.attrs["suavizada"] = suavizada
    return salida
