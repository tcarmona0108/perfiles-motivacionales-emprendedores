"""Orquestación del pipeline: un paso = una función que lee y escribe en disco.

Cada paso lee lo que dejó el anterior en ``outputs/<fuente>/`` y escribe sus
tablas, figuras y modelos. Así cualquier paso se puede re-ejecutar solo, y el
reporte Quarto lee archivos en lugar de recalcular.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from . import datos, sintetico
from .config import Config, Fuente


def _tablas(cfg: Config, fuente: Fuente):
    return cfg.dir_salidas(fuente) / "tablas"


def paso_datos(cfg: Config, fuente: Fuente) -> pd.DataFrame:
    """Carga, limpia y guarda la tabla analítica (todas las filas, con marca de planas)."""
    df = datos.cargar(cfg, fuente)
    df.to_csv(_tablas(cfg, fuente) / "respuestas_limpias.csv", index=False)
    resumen = pd.Series(
        {
            "n_total": len(df),
            "n_respuesta_plana": int(df["respuesta_plana"].sum()),
            "n_analisis": len(datos.para_analisis(df, cfg)),
            "n_items": len(cfg.items),
        },
        name="valor",
    )
    resumen.to_csv(_tablas(cfg, fuente) / "resumen_muestra.csv")
    return df


def leer_limpias(cfg: Config, fuente: Fuente) -> pd.DataFrame:
    """Lee la tabla limpia que dejó :func:`paso_datos`."""
    return pd.read_csv(_tablas(cfg, fuente) / "respuestas_limpias.csv")


def generar_sintetico(cfg: Config) -> None:
    """Ajusta la cópula sobre los datos reales y escribe el CSV sintético y su ficha."""
    real = datos.cargar(cfg, "real")
    modelo = sintetico.ajustar(real, cfg)
    n = cfg["sintetico"]["n"] or len(real)
    sint = sintetico.muestrear(modelo, n, cfg.rng(desplazamiento=1))
    ruta = cfg.ruta(cfg["datos"]["sintetico"])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    sint.to_csv(ruta, index=False)
    metricas = sintetico.evaluar_fidelidad(real, datos.limpiar(sint, cfg), cfg)
    sintetico.escribir_ficha(
        cfg.ruta(cfg["datos"]["ficha_sintetico"]), metricas, modelo, n, cfg.semilla
    )
    print(f"Sintético escrito en {ruta.relative_to(cfg.raiz)} ({n} filas)")
    for k, v in metricas.items():
        print(f"  {k}: {v}")


# Orden de ejecución de `emprende all`. Los pasos de la Fase 2 se agregan aquí.
PASOS: dict[str, Callable[[Config, Fuente], object]] = {
    "datos": paso_datos,
}
