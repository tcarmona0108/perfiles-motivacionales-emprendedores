"""Generación del dataset sintético público.

Por qué un sintético
--------------------
Los datos reales tienen cuasi-identificadores y no pueden publicarse. El repo
necesita, sin embargo, datos con la **misma estructura** para que cualquiera
pueda correr el pipeline, los tests y la CI.

Método: cópula gaussiana ordinal
--------------------------------
1. Se separan las respuestas planas (todo igual) y se modelan como una mezcla:
   con probabilidad p_plana la fila es plana. Así el sintético reproduce el
   fenómeno de aquiescencia que el pipeline debe manejar.
2. Con las filas no planas se estiman los umbrales de cada ítem (proporciones
   marginales) y la matriz de correlaciones **policóricas**.
3. Se muestrea Z ~ N(0, R_policórica) y cada columna se corta en sus umbrales:
   esto reproduce las marginales de cada ítem y la estructura de correlación
   latente, que es lo que usan la EFA y el clustering.
4. La demografía se muestrea **de forma independiente** de sus marginales.

Qué NO conserva (y se declara en la ficha)
------------------------------------------
* Relaciones demografía–ítems y entre variables demográficas: son
  independientes por diseño. La validación externa y la auditoría de sesgo con
  el sintético son solo demostrativas; los resultados reales se reportan
  agregados.
* Estructura no lineal o de mezcla latente en los ítems: la cópula es una sola
  normal multivariada, así que el sintético **no tiene clusters "verdaderos"**.
  Si el pipeline encuentra perfiles en el sintético, son particiones de un
  continuo. Es una prueba de honestidad útil para el método.
* Ciudad y objeto social: no se generan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import Config
from .correlaciones import matriz_policorica

DEMOGRAFIA_CATEGORICA = ["ocupacion", "genero", "rango_edad", "estado_civil", "nivel_educativo"]


@dataclass
class ModeloCopula:
    """Parámetros ajustados de la cópula ordinal.

    Atributos
    ---------
    items:
        Códigos de ítems en orden.
    categorias:
        Valores observados de cada ítem (p. ej. ``[1, 2, 3, 4, 5]``).
    umbrales:
        Umbrales latentes τ de cada ítem (longitud = n categorías - 1).
    r:
        Matriz policórica (p × p).
    p_plana:
        Proporción de respuestas planas.
    valores_planos:
        Distribución del valor repetido en las filas planas.
    """

    items: list[str]
    categorias: dict[str, np.ndarray]
    umbrales: dict[str, np.ndarray]
    r: np.ndarray
    p_plana: float
    valores_planos: pd.Series
    marginales_demografia: dict[str, pd.Series] = field(default_factory=dict)
    anios: np.ndarray | None = None


def ajustar(df: pd.DataFrame, cfg: Config) -> ModeloCopula:
    """Ajusta la cópula ordinal y las marginales demográficas sobre datos limpios.

    Parámetros
    ----------
    df:
        Salida de :func:`emprendimiento.datos.limpiar` con los datos reales.
    cfg:
        Configuración.
    """
    items = cfg.items
    planas = df["respuesta_plana"]
    x = df.loc[~planas, items]
    categorias, umbrales = {}, {}
    for it in items:
        cats = np.sort(x[it].unique())
        acum = np.cumsum([(x[it] == c).mean() for c in cats])[:-1]
        categorias[it] = cats
        umbrales[it] = norm.ppf(acum)
    r = matriz_policorica(x).to_numpy()

    marginales = {
        c: df[c].value_counts(normalize=True) for c in DEMOGRAFIA_CATEGORICA if c in df.columns
    }
    anios = df["anios_funcionamiento"].dropna().round().to_numpy()
    return ModeloCopula(
        items=items,
        categorias=categorias,
        umbrales=umbrales,
        r=r,
        p_plana=float(planas.mean()),
        valores_planos=df.loc[planas, items[0]].value_counts(normalize=True),
        marginales_demografia=marginales,
        anios=anios,
    )


def muestrear(modelo: ModeloCopula, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Genera ``n`` filas sintéticas a partir de la cópula ajustada.

    Retorna
    -------
    pandas.DataFrame
        Mismas columnas que el CSV limpio (sin identificadores), listo para
        :func:`emprendimiento.datos.limpiar`.
    """
    n_planas = int(rng.binomial(n, modelo.p_plana))
    n_normales = n - n_planas

    z = rng.multivariate_normal(np.zeros(len(modelo.items)), modelo.r, size=n_normales)
    items = {}
    for j, it in enumerate(modelo.items):
        idx = np.searchsorted(modelo.umbrales[it], z[:, j])
        items[it] = modelo.categorias[it][idx]
    x = pd.DataFrame(items)

    if n_planas:
        valores = rng.choice(
            modelo.valores_planos.index.to_numpy(), size=n_planas, p=modelo.valores_planos.values
        )
        planas = pd.DataFrame({it: valores for it in modelo.items})
        x = pd.concat([x, planas], ignore_index=True)
    # Mezclar el orden para que las filas planas no queden al final.
    x = x.iloc[rng.permutation(n)].reset_index(drop=True)

    demo = {
        c: rng.choice(m.index.to_numpy(), size=n, p=m.values)
        for c, m in modelo.marginales_demografia.items()
    }
    if modelo.anios is not None:
        demo["anios_funcionamiento"] = rng.choice(modelo.anios, size=n)
    columnas = [
        "ocupacion",
        "anios_funcionamiento",
        "genero",
        "rango_edad",
        "estado_civil",
        "nivel_educativo",
    ]
    demo_df = pd.DataFrame(demo)[[c for c in columnas if c in demo]]
    return pd.concat([demo_df, x.astype(int)], axis=1)


def evaluar_fidelidad(real: pd.DataFrame, sint: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """Métricas agregadas de fidelidad y privacidad del sintético.

    * Diferencia máxima en proporciones marginales de los ítems.
    * Diferencia media y máxima entre matrices policóricas (sin respuestas planas).
    * Cuántas filas sintéticas no planas coinciden exactamente con una real no
      plana (riesgo de copia), y la distancia de Hamming mínima mediana al
      registro real más cercano (DCR).
    """
    items = cfg.items
    dif_marg = 0.0
    for it in items:
        pr = real[it].value_counts(normalize=True)
        ps = sint[it].value_counts(normalize=True)
        dif_marg = max(dif_marg, float(pr.sub(ps, fill_value=0).abs().max()))

    r_real = matriz_policorica(real.loc[~real["respuesta_plana"], items]).to_numpy()
    r_sint = matriz_policorica(sint.loc[~sint["respuesta_plana"], items]).to_numpy()
    tri = np.triu_indices_from(r_real, 1)
    dif_r = np.abs(r_real[tri] - r_sint[tri])

    xr = real.loc[~real["respuesta_plana"], items].to_numpy()
    xs = sint.loc[~sint["respuesta_plana"], items].to_numpy()
    hamming = (xs[:, None, :] != xr[None, :, :]).sum(axis=2).min(axis=1)
    return {
        "dif_max_marginal": round(dif_marg, 3),
        "dif_media_policorica": round(float(dif_r.mean()), 3),
        "dif_max_policorica": round(float(dif_r.max()), 3),
        "filas_copiadas": int((hamming == 0).sum()),
        "dcr_hamming_mediana": float(np.median(hamming)),
        "dcr_hamming_minima": int(hamming.min()),
    }


def escribir_ficha(ruta, metricas: dict[str, float], modelo: ModeloCopula, n: int, semilla: int):
    """Escribe ``data/synthetic/FICHA.md``, la documentación del sintético."""
    texto = f"""# Ficha del dataset sintético

> Archivo generado automáticamente por `uv run emprende sintetico`. No editar a mano.

## Qué es
`respuestas_sinteticas.csv` imita la **estructura** de la encuesta real
({n} filas, 19 ítems Likert 1–5 y 6 variables de contexto), pero **no contiene
ninguna respuesta real**. Se publica para que el pipeline, los tests y la CI
puedan correr sin exponer datos personales.

## Cómo se generó
- **Ítems:** cópula gaussiana ordinal. Umbrales por ítem a partir de las
  proporciones marginales reales y matriz de correlaciones **policóricas** reales
  (estimadas sin las respuestas planas). Se muestrea Z ~ N(0, R) y se corta en
  los umbrales.
- **Respuestas planas** (los 19 ítems con el mismo valor): mezcla con
  probabilidad {modelo.p_plana:.3f}, igual a la observada.
- **Contexto** (ocupación, años, género, edad, estado civil, educación):
  muestreo **independiente** de cada marginal.
- **No se generan** ciudad ni objeto social.
- Semilla: `{semilla}` (en `config.yaml`).

## Fidelidad (agregados)
| Métrica | Valor |
|---|---|
| Diferencia máxima en proporciones de respuesta por ítem | {metricas["dif_max_marginal"]} |
| Diferencia media entre policóricas (real vs sintético) | {metricas["dif_media_policorica"]} |
| Diferencia máxima entre correlaciones policóricas | {metricas["dif_max_policorica"]} |

## Privacidad
| Métrica | Valor |
|---|---|
| Filas sintéticas (no planas) idénticas a una real | {metricas["filas_copiadas"]} |
| Hamming mediana al registro real más cercano (de 19) | {metricas["dcr_hamming_mediana"]} |
| Distancia de Hamming mínima | {metricas["dcr_hamming_minima"]} |

Las variables de contexto se muestrean de forma independiente de los ítems, así
que ni siquiera una fila sintética parecida a una real conserva su demografía.

## Limitaciones (léase antes de interpretar resultados con el sintético)
1. **Sin relación demografía–ítems:** la validación externa y la auditoría de
   sesgo sobre el sintético dan, por construcción, diferencias nulas. Son solo
   una demostración del código.
2. **Sin clusters verdaderos:** una sola normal multivariada no tiene grupos. Los
   segmentos que aparezcan son particiones de un continuo; su estabilidad debería
   ser baja, y eso es una prueba útil del método.
3. Los resultados publicados en el reporte provienen de la corrida local con los
   datos reales, reportados de forma agregada.
"""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": mismo archivo byte a byte en Windows y Linux.
    ruta.write_text(texto, encoding="utf-8", newline="\n")
