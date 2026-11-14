"""Psicometría: descriptivos por ítem, idoneidad, número de factores, EFA y confiabilidad.

Decisiones metodológicas (resumen para defenderlas)
---------------------------------------------------
* **Matriz de entrada:** policórica (ítems ordinales con efecto techo).
* **Número de factores:** análisis paralelo con policóricas y permutación de
  columnas (conserva las marginales reales, incluido el techo), criterio
  percentil 95. Se compara con MAP de Velicer, Kaiser y el PA clásico de Pearson.
  Garrido, Abad y Ponsoda (2013) muestran que el PA con policóricas es el
  criterio más preciso para ítems ordinales; Kaiser sobreestima.
* **Extracción:** ``minres`` (factores comunes). El PCA modela la varianza total,
  incluido el error, e infla las cargas; por eso no es análisis factorial.
* **Rotación:** oblimin (oblicua). Se espera que los factores motivacionales
  correlacionen; si no lo hicieran, oblimin daría una solución casi ortogonal.
  Varimax impone incorrelación de antemano.
* **Confiabilidad:** α de Cronbach (comparable con el trabajo original), α
  ordinal y ω de McDonald por factor (no asume tau-equivalencia), y ω jerárquico
  (Schmid-Leiman) para medir cuánto del puntaje total refleja un factor general.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2, kurtosis, skew

from .correlaciones import matriz_policorica

# factor_analyzer 0.5.1 emite un FutureWarning de scikit-learn en cada ajuste (ver pyproject).
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
from factor_analyzer import FactorAnalyzer  # noqa: E402


# ═══════════════════════════ Paso 1: descriptivos ════════════════════════════
def descriptivos_items(
    df: pd.DataFrame, items: list[str], etiquetas: dict[str, str]
) -> pd.DataFrame:
    """Distribución, techo/piso, forma y discriminación de cada ítem.

    Retorna
    -------
    pandas.DataFrame
        Una fila por ítem: media, DE, mediana, % de cada categoría, % piso (1),
        % techo (5), asimetría, curtosis y correlación ítem-resto (discriminación:
        cuánto se alinea el ítem con la suma de los demás).
    """
    x = df[items]
    total = x.sum(axis=1)
    filas = []
    for it in items:
        v = x[it]
        fila = {
            "item": it,
            "etiqueta": etiquetas[it],
            "media": v.mean(),
            "de": v.std(),
            "mediana": v.median(),
            **{f"pct_{c}": 100 * (v == c).mean() for c in range(1, 6)},
            "asimetria": skew(v),
            "curtosis": kurtosis(v),
            "r_item_resto": np.corrcoef(v, total - v)[0, 1],
        }
        fila["pct_piso"], fila["pct_techo"] = fila["pct_1"], fila["pct_5"]
        filas.append(fila)
    return pd.DataFrame(filas).set_index("item")


# ═══════════════════════════ Paso 2: idoneidad ═══════════════════════════════
def kmo(r: np.ndarray) -> tuple[np.ndarray, float]:
    """Medida de Kaiser-Meyer-Olkin a partir de una matriz de correlación.

    Se calcula desde la matriz (y no desde los datos) para poder usar la
    policórica. KMO compara correlaciones con correlaciones parciales: si las
    parciales son pequeñas, la varianza es compartida y hay factores comunes.
    """
    inv = np.linalg.inv(r)
    parcial = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    np.fill_diagonal(parcial, 0)
    r0 = r.copy()
    np.fill_diagonal(r0, 0)
    r2, p2 = r0**2, parcial**2
    por_item = r2.sum(0) / (r2.sum(0) + p2.sum(0))
    total = r2.sum() / (r2.sum() + p2.sum())
    return por_item, float(total)


def bartlett(r: np.ndarray, n: int) -> tuple[float, int, float]:
    """Prueba de esfericidad de Bartlett: H0 = la matriz de correlación es la identidad."""
    p = r.shape[0]
    estad = -(n - 1 - (2 * p + 5) / 6) * np.linalg.slogdet(r)[1]
    gl = p * (p - 1) // 2
    return float(estad), gl, float(chi2.sf(estad, gl))


# ═══════════════════════════ Paso 3: número de factores ══════════════════════
def _autovalores_reducidos(r: np.ndarray) -> np.ndarray:
    """Autovalores de la matriz reducida (diagonal = SMC), la versión 'factorial' del scree."""
    smc = 1 - 1 / np.diag(np.linalg.inv(r))
    rr = r.copy()
    np.fill_diagonal(rr, smc)
    return np.sort(np.linalg.eigvalsh(rr))[::-1]


def _replica_nula(x: np.ndarray, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    """Permuta cada columna por separado (rompe correlaciones, conserva marginales)."""
    rng = np.random.default_rng(semilla)
    xp = np.column_stack([rng.permutation(col) for col in x.T])
    r = matriz_policorica(pd.DataFrame(xp)).to_numpy()
    return np.sort(np.linalg.eigvalsh(r))[::-1], _autovalores_reducidos(r)


def _n_por_paralelo(observado: np.ndarray, umbral: np.ndarray) -> int:
    """Cuenta autovalores observados consecutivos que superan al azar (para en el primer fallo)."""
    supera = observado > umbral
    return int(np.argmin(supera)) if not supera.all() else len(supera)


def analisis_paralelo(
    x: pd.DataFrame, r_obs: np.ndarray, n_iter: int, percentil: float, semilla: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Análisis paralelo de Horn con policóricas y datos permutados.

    Parámetros
    ----------
    x:
        Ítems (muestra analítica).
    r_obs:
        Matriz policórica observada.
    n_iter:
        Número de réplicas nulas.
    percentil:
        Percentil del autovalor aleatorio usado como umbral (95 = conservador).
    semilla:
        Semilla base; la réplica i usa ``semilla + i`` (reproducible y paralelizable).

    Retorna
    -------
    tuple[pandas.DataFrame, dict]
        Tabla de autovalores (observados, media y percentil nulos; versiones de
        componentes y de factores) y el número de factores por criterio.
    """
    xv = x.to_numpy()
    res = Parallel(n_jobs=-1)(delayed(_replica_nula)(xv, semilla + i) for i in range(n_iter))
    nulos_pc = np.array([a for a, _ in res])
    nulos_fa = np.array([b for _, b in res])

    obs_pc = np.sort(np.linalg.eigvalsh(r_obs))[::-1]
    obs_fa = _autovalores_reducidos(r_obs)
    tabla = pd.DataFrame(
        {
            "observado": obs_pc,
            "nulo_media": nulos_pc.mean(0),
            f"nulo_p{percentil:g}": np.percentile(nulos_pc, percentil, axis=0),
            "observado_reducido": obs_fa,
            f"nulo_reducido_p{percentil:g}": np.percentile(nulos_fa, percentil, axis=0),
        },
        index=pd.RangeIndex(1, len(obs_pc) + 1, name="factor"),
    )
    n = {
        "paralelo_policorico": _n_por_paralelo(obs_pc, tabla[f"nulo_p{percentil:g}"].to_numpy()),
        "paralelo_policorico_reducido": _n_por_paralelo(
            obs_fa, tabla[f"nulo_reducido_p{percentil:g}"].to_numpy()
        ),
        "kaiser": int((obs_pc > 1).sum()),
    }
    return tabla, n


def paralelo_pearson_clasico(x: pd.DataFrame, n_iter: int, percentil: float, semilla: int) -> int:
    """PA clásico (Pearson vs normales independientes). Solo como referencia frente al original."""
    rng = np.random.default_rng(semilla)
    n, p = x.shape
    obs = np.sort(np.linalg.eigvalsh(x.corr().to_numpy()))[::-1]
    nulos = np.array(
        [
            np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)))[
                ::-1
            ]
            for _ in range(n_iter)
        ]
    )
    return _n_por_paralelo(obs, np.percentile(nulos, percentil, axis=0))


def map_velicer(r: np.ndarray) -> tuple[int, np.ndarray]:
    """Prueba MAP de Velicer (versión 2000, parciales al cuadrado).

    Se van extrayendo componentes y se mide la correlación parcial media que
    queda. El mínimo indica cuándo se extrajo la varianza sistemática y solo
    queda ruido. Tiende a subestimar; por eso se usa como contraste del PA.
    """
    p = r.shape[0]
    vals, vecs = np.linalg.eigh(r)
    orden = np.argsort(vals)[::-1]
    cargas = vecs[:, orden] * np.sqrt(np.clip(vals[orden], 0, None))
    promedios = []
    for m in range(p - 1):
        c = r - cargas[:, :m] @ cargas[:, :m].T
        d = np.sqrt(np.diag(c))
        parcial = c / np.outer(d, d)
        promedios.append((parcial[~np.eye(p, dtype=bool)] ** 2).mean())
    promedios = np.array(promedios)
    return int(np.argmin(promedios)), promedios


# ═══════════════════════════ Paso 3: EFA ═════════════════════════════════════
@dataclass
class ResultadoEFA:
    """Solución factorial rotada.

    Atributos
    ---------
    cargas:
        Matriz de patrón (ítems × factores). Con rotación oblicua es el efecto
        único de cada factor sobre el ítem.
    phi:
        Correlaciones entre factores.
    comunalidades:
        Proporción de la varianza de cada ítem explicada por los factores.
    """

    cargas: pd.DataFrame
    phi: pd.DataFrame
    comunalidades: pd.Series

    @property
    def estructura(self) -> pd.DataFrame:
        """Matriz de estructura (correlación ítem-factor) = patrón × Φ."""
        return self.cargas @ self.phi

    @property
    def n_factores(self) -> int:
        return self.cargas.shape[1]


def efa(
    r: pd.DataFrame, n_factores: int, metodo: str = "minres", rotacion: str = "oblimin"
) -> ResultadoEFA:
    """Análisis factorial exploratorio sobre una matriz de correlación.

    Los factores se ordenan por varianza explicada y se orientan para que su
    carga media sea positiva. Sin esto, el orden y el signo cambian entre
    corridas o bootstraps y no se pueden comparar.
    """
    fa = FactorAnalyzer(
        n_factors=n_factores,
        rotation=rotacion if n_factores > 1 else None,
        method=metodo,
        is_corr_matrix=True,
        svd_method="lapack",  # 'randomized' (el valor por defecto) no es determinista
    )
    fa.fit(r.to_numpy())
    cargas = np.asarray(fa.loadings_)
    phi = np.asarray(fa.phi_) if getattr(fa, "phi_", None) is not None else np.eye(n_factores)

    signo = np.sign(cargas.sum(0))
    signo[signo == 0] = 1
    cargas = cargas * signo
    phi = phi * np.outer(signo, signo)
    orden = np.argsort(-(cargas**2).sum(0))
    cargas, phi = cargas[:, orden], phi[np.ix_(orden, orden)]

    nombres = [f"F{i + 1}" for i in range(n_factores)]
    estructura = cargas @ phi
    comunalidad = (cargas * estructura).sum(1)  # h² con factores oblicuos
    return ResultadoEFA(
        cargas=pd.DataFrame(cargas, index=r.index, columns=nombres),
        phi=pd.DataFrame(phi, index=nombres, columns=nombres),
        comunalidades=pd.Series(comunalidad, index=r.index, name="h2"),
    )


def asignar_items(
    cargas: pd.DataFrame, carga_minima: float, diferencia_cruzada: float
) -> pd.DataFrame:
    """Asigna cada ítem a su factor dominante y marca problemas.

    * ``baja``: la carga principal es menor que ``carga_minima``.
    * ``cruzada``: la diferencia entre la carga principal y la segunda es menor
      que ``diferencia_cruzada`` (el ítem no discrimina entre factores).
    """
    a = cargas.abs()
    orden = np.argsort(-a.to_numpy(), axis=1)
    principal = a.to_numpy()[np.arange(len(a)), orden[:, 0]]
    segunda = a.to_numpy()[np.arange(len(a)), orden[:, 1]] if a.shape[1] > 1 else np.zeros(len(a))
    t = pd.DataFrame(
        {
            "factor": a.columns.to_numpy()[orden[:, 0]],
            "carga": cargas.to_numpy()[np.arange(len(a)), orden[:, 0]],
            "segunda_carga": segunda,
        },
        index=cargas.index,
    )
    t["baja"] = principal < carga_minima
    t["cruzada"] = (principal - segunda) < diferencia_cruzada
    t["limpio"] = ~(t["baja"] | t["cruzada"])
    return t


def congruencia_tucker(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Coeficiente de congruencia de Tucker entre columnas de dos matrices de cargas.

    ≥ 0.95: factores prácticamente iguales; 0.85–0.94: similitud aceptable
    (Lorenzo-Seva y ten Berge, 2006).
    """
    num = a.T @ b
    den = np.sqrt(np.outer((a**2).sum(0), (b**2).sum(0)))
    return num / den


def alinear(referencia: np.ndarray, cargas: np.ndarray) -> np.ndarray:
    """Reordena y reorienta factores de un bootstrap para que coincidan con la referencia."""
    from scipy.optimize import linear_sum_assignment

    c = congruencia_tucker(referencia, cargas)
    fila, col = linear_sum_assignment(-np.abs(c))
    alineadas = cargas[:, col] * np.sign(c[fila, col])
    return alineadas


# ═══════════════════════════ Paso 2: confiabilidad ═══════════════════════════
def alfa_cronbach(x: pd.DataFrame) -> float:
    """α de Cronbach sobre puntajes brutos (comparable con el trabajo original)."""
    k = x.shape[1]
    return float(k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1)))


def alfa_desde_matriz(r: np.ndarray) -> float:
    """α estandarizado desde una matriz de correlación (α ordinal si es policórica)."""
    k = r.shape[0]
    rbar = r[np.triu_indices(k, 1)].mean()
    return float(k * rbar / (1 + (k - 1) * rbar))


def omega_unifactorial(r: np.ndarray) -> float:
    """ω de McDonald de una subescala: ajusta un factor y calcula (Σλ)² / ((Σλ)² + Σψ).

    A diferencia de α, no supone que todos los ítems carguen igual.
    """
    if r.shape[0] < 3:
        # Con 2 ítems un factor no está identificado; ω se reduce a α estandarizado.
        return alfa_desde_matriz(r)
    fa = FactorAnalyzer(
        n_factors=1, rotation=None, method="minres", is_corr_matrix=True, svd_method="lapack"
    )
    fa.fit(r)
    lam = np.abs(np.asarray(fa.loadings_).ravel())
    psi = 1 - lam**2
    return float(lam.sum() ** 2 / (lam.sum() ** 2 + psi.sum()))


def omega_jerarquico(r: np.ndarray, sol: ResultadoEFA) -> dict[str, float]:
    """ω total y ω jerárquico por transformación de Schmid-Leiman.

    ω_h es la proporción de varianza del puntaje total debida a un factor
    general común a todos los ítems. Si es alto (> 0.70), el total es
    esencialmente unidimensional y los "perfiles" tenderán a ordenarse por
    nivel general más que a diferir en forma. Es la pregunta clave para este
    cuestionario, cuyo primer autovalor explica el 40 % de la varianza.
    """
    lam = sol.cargas.to_numpy()
    phi = sol.phi.to_numpy()
    m = lam.shape[1]
    if m == 1:
        g = np.array([1.0])
    elif m == 2:
        g = np.full(2, np.sqrt(max(phi[0, 1], 0)))  # 2 factores: segundo orden no identificado
    else:
        fa2 = FactorAnalyzer(
            n_factors=1, rotation=None, method="minres", is_corr_matrix=True, svd_method="lapack"
        )
        fa2.fit(phi)
        g = np.abs(np.asarray(fa2.loadings_).ravel())
    general = lam @ g
    grupo = lam * np.sqrt(np.clip(1 - g**2, 0, None))
    var_total = r.sum()
    omega_h = general.sum() ** 2 / var_total
    omega_t = (general.sum() ** 2 + sum(grupo[:, j].sum() ** 2 for j in range(m))) / var_total
    return {"omega_total": float(min(omega_t, 1.0)), "omega_jerarquico": float(omega_h)}


def confiabilidad(x: pd.DataFrame, r: pd.DataFrame, asignacion: pd.Series) -> pd.DataFrame:
    """α, α ordinal y ω por subescala y para la escala completa.

    Parámetros
    ----------
    x:
        Ítems brutos.
    r:
        Matriz policórica.
    asignacion:
        Factor asignado a cada ítem (índice = ítem).
    """
    filas = []
    for f, items in asignacion.groupby(asignacion).groups.items():
        items = list(items)
        sub = r.loc[items, items].to_numpy()
        filas.append(
            {
                "escala": f,
                "n_items": len(items),
                "alfa": alfa_cronbach(x[items]),
                "alfa_ordinal": alfa_desde_matriz(sub),
                "omega": omega_unifactorial(sub),
            }
        )
    filas.append(
        {
            "escala": "Total",
            "n_items": x.shape[1],
            "alfa": alfa_cronbach(x),
            "alfa_ordinal": alfa_desde_matriz(r.to_numpy()),
            "omega": omega_unifactorial(r.to_numpy()),
        }
    )
    return pd.DataFrame(filas).set_index("escala")


# ═══════════════════════════ Puntajes factoriales ════════════════════════════
@dataclass
class Puntuador:
    """Convierte respuestas brutas en puntajes factoriales (método de regresión de Thurstone).

    Es una transformación lineal: z = (x - media) / de;  puntaje = z · W, con
    W = R⁻¹ · Λ · Φ. Guardar media, de y W permite puntuar personas nuevas sin
    necesitar los datos de entrenamiento.
    """

    items: list[str]
    media: np.ndarray
    de: np.ndarray
    pesos: pd.DataFrame

    @classmethod
    def ajustar(cls, x: pd.DataFrame, r: pd.DataFrame, sol: ResultadoEFA) -> Puntuador:
        w = np.linalg.solve(r.to_numpy(), sol.estructura.to_numpy())
        return cls(
            items=list(x.columns),
            media=x.mean().to_numpy(),
            de=x.std(ddof=1).to_numpy(),
            pesos=pd.DataFrame(w, index=x.columns, columns=sol.cargas.columns),
        )

    def transformar(self, x: pd.DataFrame) -> pd.DataFrame:
        z = (x[self.items].to_numpy() - self.media) / self.de
        return pd.DataFrame(z @ self.pesos.to_numpy(), index=x.index, columns=self.pesos.columns)


# ═══════════════════════════ Estabilidad por bootstrap ═══════════════════════
def _replica_bootstrap(
    x: np.ndarray, columnas, n_factores, metodo, rotacion, ref, asignacion, semilla
):
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(x), len(x))
    xb = pd.DataFrame(x[idx], columns=columnas)
    rb = matriz_policorica(xb)
    sol = efa(rb, n_factores, metodo, rotacion)
    alineadas = alinear(ref, sol.cargas.to_numpy())
    congr = np.diag(congruencia_tucker(ref, alineadas))
    omegas = {
        f: omega_unifactorial(rb.loc[list(its), list(its)].to_numpy())
        for f, its in asignacion.groupby(asignacion).groups.items()
    }
    return congr, omegas, alfa_cronbach(xb)


def bootstrap_efa(
    x: pd.DataFrame,
    sol: ResultadoEFA,
    asignacion: pd.Series,
    n_boot: int,
    semilla: int,
    metodo: str = "minres",
    rotacion: str = "oblimin",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estabilidad de la solución factorial e intervalos de confianza de ω.

    En cada réplica se re-estima la policórica y la EFA, se alinean los factores
    con la solución original y se mide la congruencia de Tucker.

    Retorna
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        (congruencia por factor: media, p5 y % de réplicas ≥ 0.85;
        IC 95 % percentil de ω por factor y de α total).
    """
    res = Parallel(n_jobs=-1)(
        delayed(_replica_bootstrap)(
            x.to_numpy(),
            list(x.columns),
            sol.n_factores,
            metodo,
            rotacion,
            sol.cargas.to_numpy(),
            asignacion,
            semilla + i,
        )
        for i in range(n_boot)
    )
    congr = pd.DataFrame([c for c, _, _ in res], columns=sol.cargas.columns)
    estab = pd.DataFrame(
        {
            "congruencia_media": congr.mean(),
            "congruencia_p5": congr.quantile(0.05),
            "pct_replicas_ge_085": 100 * (congr >= 0.85).mean(),
        }
    )
    omegas = pd.DataFrame([o for _, o, _ in res])
    omegas["Total_alfa"] = [a for _, _, a in res]
    ic = pd.DataFrame({"ic_inf": omegas.quantile(0.025), "ic_sup": omegas.quantile(0.975)})
    return estab, ic
