"""Segmentación: comparación de algoritmos y selección del número de grupos.

Algoritmos comparados
---------------------
* **K-means:** grupos esféricos de tamaño parecido; referencia clásica.
* **Ward:** jerárquico; minimiza la varianza intra-grupo como K-means, pero sin
  depender de una inicialización aleatoria.
* **GMM (covarianza completa):** grupos elípticos y probabilidades de pertenencia.
* **LPA (Latent Profile Analysis):** mezcla gaussiana con covarianza
  **diagonal** (independencia local dentro de cada perfil y varianzas
  distintas por perfil). Es la especificación estándar de LPA en psicología
  (modelo "varying variances" de Mplus/tidyLPA), implementada con
  ``sklearn.mixture.GaussianMixture(covariance_type="diag")``. Así se evita
  depender de paquetes con mantenimiento incierto.

Cómo se elige k (ningún índice decide solo)
-------------------------------------------
1. Se descartan soluciones con un segmento < ``tamano_minimo`` de la muestra:
   un "perfil" de 5 personas no es accionable ni reportable con privacidad.
2. Entre las restantes, se exige **estabilidad**: ARI medio ≥ 0.75 entre la
   solución completa y réplicas sobre submuestras del 80 % (Hennig, 2007).
3. Entre las estables, se elige la de mejor silueta (desempate: menor k).
4. Si ninguna es estable, se reporta la más estable y se declara que la
   estructura de grupos es débil.

Además, la silueta de la solución elegida se compara con la de datos simulados
**sin grupos** (una normal multivariada con la misma media y covarianza). Si no
la supera, los segmentos son cortes de un continuo, no tipos naturales. Es
legítimo usarlos para describir, pero hay que decirlo.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

NOMBRES = {
    "kmeans": "K-means",
    "ward": "Ward",
    "gmm_completa": "GMM (cov. completa)",
    "lpa": "LPA (GMM diagonal)",
}


def ajustar(algoritmo: str, k: int, x: np.ndarray, semilla: int, n_init: int = 50):
    """Ajusta un algoritmo y devuelve (etiquetas, modelo)."""
    if algoritmo == "kmeans":
        m = KMeans(n_clusters=k, n_init=n_init, random_state=semilla)
    elif algoritmo == "ward":
        m = AgglomerativeClustering(n_clusters=k, linkage="ward")
    elif algoritmo in ("gmm_completa", "lpa"):
        m = GaussianMixture(
            n_components=k,
            covariance_type="full" if algoritmo == "gmm_completa" else "diag",
            n_init=max(5, n_init // 5),
            reg_covar=1e-4,
            random_state=semilla,
        )
    else:
        raise ValueError(f"Algoritmo desconocido: {algoritmo}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        etiquetas = m.fit_predict(x)
    return etiquetas, m


def _ari_replica(algoritmo, k, x, ref, frac, semilla, n_init):
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(x), int(frac * len(x)), replace=False)
    etq, _ = ajustar(algoritmo, k, x[idx], semilla, n_init)
    return adjusted_rand_score(ref[idx], etq)


def estabilidad(algoritmo, k, x, ref, n_boot, frac, semilla, n_init) -> np.ndarray:
    """ARI entre la solución completa y soluciones re-ajustadas en submuestras.

    Se usa submuestreo sin reemplazo (no bootstrap clásico) porque los duplicados
    del bootstrap distorsionan Ward y las mezclas gaussianas.
    """
    return np.array(
        Parallel(n_jobs=-1)(
            delayed(_ari_replica)(algoritmo, k, x, ref, frac, semilla + i, n_init)
            for i in range(n_boot)
        )
    )


def evaluar(x: np.ndarray, cfg_seg: dict, semilla: int) -> pd.DataFrame:
    """Tabla de criterios para cada algoritmo y cada k candidato.

    Retorna
    -------
    pandas.DataFrame
        Columnas: algoritmo, k, silueta, davies_bouldin, calinski_harabasz,
        bic (solo mezclas), tamano_min (proporción del segmento más chico),
        ari_medio y ari_p5.
    """
    filas = []
    for alg in cfg_seg["algoritmos"]:
        for k in cfg_seg["k_candidatos"]:
            etq, m = ajustar(alg, k, x, semilla, cfg_seg["n_init"])
            if len(np.unique(etq)) < k:
                continue  # la mezcla colapsó componentes: solución degenerada
            ari = estabilidad(
                alg,
                k,
                x,
                etq,
                cfg_seg["n_bootstrap"],
                cfg_seg["fraccion_bootstrap"],
                semilla,
                cfg_seg["n_init"],
            )
            filas.append(
                {
                    "algoritmo": alg,
                    "k": k,
                    "silueta": silhouette_score(x, etq),
                    "davies_bouldin": davies_bouldin_score(x, etq),
                    "calinski_harabasz": calinski_harabasz_score(x, etq),
                    "bic": m.bic(x) if hasattr(m, "bic") else np.nan,
                    "tamano_min": np.bincount(etq).min() / len(etq),
                    "ari_medio": ari.mean(),
                    "ari_p5": np.percentile(ari, 5),
                }
            )
    return pd.DataFrame(filas)


def elegir(tabla: pd.DataFrame, cfg_seg: dict) -> tuple[pd.Series, str]:
    """Aplica la regla de selección descrita en el docstring del módulo.

    Retorna
    -------
    tuple[pandas.Series, str]
        La fila elegida y una explicación en texto para el reporte.
    """
    t = tabla[tabla["tamano_min"] >= cfg_seg["tamano_minimo_segmento"]]
    if t.empty:
        t = tabla
    estables = t[t["ari_medio"] >= cfg_seg["ari_estable"]]
    if not estables.empty:
        fila = estables.sort_values(["silueta", "k"], ascending=[False, True]).iloc[0]
        motivo = (
            f"{len(estables)} de {len(tabla)} configuraciones son estables "
            f"(ARI medio ≥ {cfg_seg['ari_estable']}); entre ellas, {NOMBRES[fila.algoritmo]} "
            f"con k={int(fila.k)} tiene la mejor silueta ({fila.silueta:.2f})."
        )
        return fila, motivo
    fila = t.sort_values(["ari_medio", "k"], ascending=[False, True]).iloc[0]
    motivo = (
        f"Ninguna configuración alcanza ARI medio ≥ {cfg_seg['ari_estable']}: la estructura "
        f"de grupos es débil. Se usa la más estable, {NOMBRES[fila.algoritmo]} con "
        f"k={int(fila.k)} (ARI medio {fila.ari_medio:.2f})."
    )
    return fila, motivo


def _silueta_sin_grupos(algoritmo, k, media, cov, n, semilla, n_init):
    rng = np.random.default_rng(semilla)
    z = rng.multivariate_normal(media, cov, size=n)
    etq, _ = ajustar(algoritmo, k, z, semilla, n_init)
    return silhouette_score(z, etq) if len(np.unique(etq)) > 1 else np.nan


def prueba_sin_grupos(
    x: np.ndarray, algoritmo: str, k: int, silueta_obs: float, n_sim: int, semilla: int, n_init: int
) -> dict[str, float]:
    """¿Supera la silueta observada a la de datos sin estructura de grupos?

    Retorna la silueta nula media, su percentil 95 y un p-valor de Monte Carlo
    (proporción de simulaciones con silueta ≥ observada).
    """
    nulas = np.array(
        Parallel(n_jobs=-1)(
            delayed(_silueta_sin_grupos)(
                algoritmo, k, x.mean(0), np.cov(x, rowvar=False), len(x), semilla + i, n_init
            )
            for i in range(n_sim)
        )
    )
    nulas = nulas[~np.isnan(nulas)]
    return {
        "silueta_observada": float(silueta_obs),
        "silueta_nula_media": float(nulas.mean()),
        "silueta_nula_p95": float(np.percentile(nulas, 95)),
        "p_valor": float((1 + (nulas >= silueta_obs).sum()) / (1 + len(nulas))),
    }


def ordenar_por_nivel(etiquetas: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Re-etiqueta los segmentos 1..k de menor a mayor nivel medio en los factores.

    Hace que la numeración sea estable entre corridas. NO implica que un nivel
    más alto signifique "más potencial": eso se discute aparte.
    """
    niveles = pd.Series(x.mean(axis=1)).groupby(etiquetas).mean().sort_values()
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(niveles.index, start=1)}
    return np.vectorize(mapa.get)(etiquetas)


def estructura_hdbscan(x: np.ndarray, min_cluster: int) -> dict[str, float]:
    """Prueba de estructura por densidad con HDBSCAN.

    HDBSCAN solo forma grupos donde hay núcleos densos separados por zonas
    vacías. Si no encuentra ninguno (todo "ruido"), no hay grupos naturales por
    densidad y cualquier partición es un corte de un continuo.
    """
    etq = HDBSCAN(min_cluster_size=min_cluster).fit_predict(x)
    return {
        "n_grupos_densos": len(set(etq) - {-1}),
        "pct_ruido": float(100 * (etq == -1).mean()),
    }


def representacion_forma(z: np.ndarray) -> np.ndarray:
    """Quita el nivel general de cada persona y conserva solo su "forma".

    Se resta a cada persona su media en los factores (estandarizados): queda qué
    factor pesa más **para ella**, sin importar si responde alto o bajo en todo.
    Los datos centrados viven en un subespacio de dimensión m-1; se proyectan a
    una base ortonormal (contrastes de Helmert) para que las distancias no
    cambien y las mezclas gaussianas no tengan covarianzas singulares.
    """
    from scipy.linalg import helmert

    centrado = z - z.mean(axis=1, keepdims=True)
    return centrado @ helmert(z.shape[1]).T
