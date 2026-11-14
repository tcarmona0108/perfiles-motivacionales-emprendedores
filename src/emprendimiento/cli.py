"""Interfaz de línea de comandos: ``emprende <paso> [--fuente real|sintetico]``.

Por qué una CLI en Python y no solo un Makefile: ``make`` no viene en Windows. La
CLI corre igual en todos los sistemas; el ``Makefile`` solo la envuelve para
Linux/macOS y la CI.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import pipeline
from .config import cargar_config


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del comando ``emprende``."""
    # La consola de Windows usa cp1252 por defecto; forzar UTF-8 evita errores con tildes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="emprende",
        description="Pipeline reproducible de perfiles motivacionales de emprendedores.",
    )
    parser.add_argument(
        "paso",
        choices=[*pipeline.PASOS.keys(), "all", "sintetico"],
        help="Paso a ejecutar. 'all' corre todos en orden; "
        "'sintetico' regenera el dataset público (requiere los datos reales).",
    )
    parser.add_argument("--fuente", choices=["real", "sintetico"], default=None)
    parser.add_argument("--config", default=None, help="Ruta alternativa a config.yaml")
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="~10 %% de las iteraciones (solo CI/pruebas; no son los números oficiales).",
    )
    args = parser.parse_args(argv)

    cfg = cargar_config(args.config)
    if args.rapido:
        cfg = cfg.rapido()
    fuente = args.fuente or cfg["datos"]["fuente"]

    if args.paso == "sintetico":
        pipeline.generar_sintetico(cfg)
        return 0

    pasos = list(pipeline.PASOS) if args.paso == "all" else [args.paso]
    for nombre in pasos:
        t0 = time.perf_counter()
        print(f"▶ {nombre} (fuente={fuente})", flush=True)
        pipeline.PASOS[nombre](cfg, fuente)
        print(f"  ✓ {nombre} en {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"Salidas en {cfg.dir_salidas(fuente).relative_to(cfg.raiz)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
