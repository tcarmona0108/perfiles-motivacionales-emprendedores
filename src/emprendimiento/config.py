"""Lectura de ``config.yaml`` y resolución de rutas.

Por qué existe: centralizar parámetros (semillas, umbrales, k candidatos) evita
números mágicos dispersos en el código y permite regenerar cualquier resultado
cambiando un solo archivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[2]
Fuente = Literal["real", "sintetico"]


@dataclass(frozen=True)
class Config:
    """Configuración del proyecto.

    Atributos
    ---------
    datos:
        Diccionario completo leído del YAML.
    raiz:
        Carpeta raíz del repositorio; todas las rutas relativas se resuelven
        contra ella para que el pipeline no dependa del directorio de trabajo.
    """

    datos: dict[str, Any]
    raiz: Path = RAIZ

    def __getitem__(self, clave: str) -> Any:
        return self.datos[clave]

    @property
    def semilla(self) -> int:
        """Semilla maestra de la que se derivan todas las demás."""
        return int(self.datos["semilla"])

    @property
    def items(self) -> list[str]:
        """Códigos de los ítems en orden (``p01`` … ``p19``)."""
        return list(self.datos["items"].keys())

    @property
    def etiquetas(self) -> dict[str, str]:
        """Etiqueta legible de cada ítem."""
        return dict(self.datos["items"])

    def ruta(self, relativa: str | Path) -> Path:
        """Resuelve una ruta del YAML contra la raíz del repositorio."""
        p = Path(relativa)
        return p if p.is_absolute() else self.raiz / p

    def dir_salidas(self, fuente: Fuente) -> Path:
        """Carpeta de salidas para una fuente: ``outputs/<fuente>/``.

        Separar por fuente impide que resultados con datos reales se mezclen
        (o se publiquen) con los sintéticos.
        """
        d = self.ruta(self.datos["salidas"]) / fuente
        for sub in ("tablas", "figuras", "modelos"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        return d

    def rapido(self) -> Config:
        """Copia con ~10 % de las iteraciones (bootstrap, permutaciones, CV).

        Solo para la CI y pruebas de humo: verifica que todo corre, pero sus
        números NO son los oficiales del reporte.
        """
        import copy

        d = copy.deepcopy(self.datos)
        d["psicometria"].update(n_iter_paralelo=100, n_bootstrap=40, n_bootstrap_alternativos=20)
        d["segmentacion"].update(n_bootstrap=20, n_sim_sin_grupos=20, n_init=10)
        d["modelo"].update(cv_repeticiones=2)
        d["perfiles"].update(n_permutaciones=200)
        return Config(datos=d, raiz=self.raiz)

    def rng(self, desplazamiento: int = 0) -> np.random.Generator:
        """Generador aleatorio derivado de la semilla maestra.

        Cada paso usa un desplazamiento distinto: así los pasos son
        independientes entre sí pero todos reproducibles.
        """
        return np.random.default_rng(self.semilla + desplazamiento)


def cargar_config(ruta: str | Path | None = None) -> Config:
    """Lee el YAML de configuración.

    Parámetros
    ----------
    ruta:
        Ruta al YAML. Por defecto, ``config.yaml`` en la raíz del repositorio.

    Retorna
    -------
    Config
    """
    ruta = Path(ruta) if ruta else RAIZ / "config.yaml"
    with ruta.open(encoding="utf-8") as f:
        return Config(datos=yaml.safe_load(f))
