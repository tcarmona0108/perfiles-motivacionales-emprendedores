"""Carga, validación y limpieza de las respuestas.

Decisiones clave
----------------
* **Minimización de datos:** ``ciudad`` y ``objeto_social`` se eliminan en cuanto
  se leen. Son cuasi-identificadores (84 % de combinaciones únicas) y ningún paso
  del análisis los necesita.
* **Faltantes:** el crudo no tiene. Si apareciera alguno, la carga falla en vez
  de imputar en silencio: imputar respuestas Likert cambia correlaciones y
  segmentos, y debe ser una decisión explícita.
* **Respuestas planas:** se marcan (no se borran) y el filtrado se decide en
  ``config.yaml`` para poder hacer análisis de sensibilidad.
"""

from __future__ import annotations

import pandas as pd

from .config import Config, Fuente

COLUMNAS_IDENTIFICABLES = ("ciudad", "objeto_social")

# Prefijo de la columna original -> nombre corto. Se usa prefijo porque los
# encabezados originales traen espacios dobles y finales irregulares.
_PREFIJOS = {
    "Ciudad": "ciudad",
    "Actualmente soy": "ocupacion",
    "Número de años": "anios_funcionamiento",
    "Objeto social": "objeto_social",
    "Género": "genero",
    "Rango de edad": "rango_edad",
    "Estado civil": "estado_civil",
    "Máximo nivel educativo": "nivel_educativo",
}

COLUMNAS_CONTEXTO = [
    "ocupacion",
    "anios_funcionamiento",
    "genero",
    "rango_edad",
    "estado_civil",
    "nivel_educativo",
]


def _nombre_corto(col: str) -> str:
    col = " ".join(col.split())
    if col.startswith("Pregunta "):
        return f"p{int(col.split()[1]):02d}"
    for prefijo, corto in _PREFIJOS.items():
        if col.startswith(prefijo):
            return corto
    raise ValueError(f"Columna no reconocida en el crudo: {col!r}")


def leer_crudo(ruta) -> pd.DataFrame:
    """Lee el CSV original y normaliza los nombres de columna.

    Parámetros
    ----------
    ruta:
        Ruta al CSV crudo (``;`` como separador, UTF-8 con BOM).

    Retorna
    -------
    pandas.DataFrame
        Con nombres cortos (``p01`` … ``p19``, ``genero``…). Aún incluye las
        columnas identificables; :func:`limpiar` las elimina.
    """
    df = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
    return df.rename(columns=_nombre_corto)


def validar_items(df: pd.DataFrame, cfg: Config) -> None:
    """Falla con un mensaje claro si los ítems no son Likert válidos.

    Comprueba que existan todos los ítems, que no haya faltantes y que todos
    los valores sean enteros dentro de la escala.
    """
    faltan = [c for c in cfg.items if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan ítems en los datos: {faltan}")
    x = df[cfg.items]
    if x.isna().any().any():
        n = int(x.isna().sum().sum())
        raise ValueError(f"Hay {n} respuestas faltantes; definir política de imputación.")
    lo, hi = cfg["escala"]["minimo"], cfg["escala"]["maximo"]
    fuera = ~x.isin(range(lo, hi + 1))
    if fuera.any().any():
        raise ValueError(f"Hay {int(fuera.sum().sum())} valores fuera de la escala {lo}-{hi}.")


def limpiar(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Devuelve la tabla analítica limpia y sin identificadores.

    Pasos: (1) elimina columnas identificables; (2) valida ítems; (3) invierte
    los ítems configurados; (4) normaliza textos de contexto; (5) agrupa el
    nivel educativo; (6) marca respuestas planas.

    Parámetros
    ----------
    df:
        Salida de :func:`leer_crudo` o el CSV sintético.
    cfg:
        Configuración del proyecto.

    Retorna
    -------
    pandas.DataFrame
        Columnas de contexto, ``nivel_educativo_grupo``, ``respuesta_plana`` e
        ítems ``p01`` … ``p19`` como enteros.
    """
    df = df.drop(columns=[c for c in COLUMNAS_IDENTIFICABLES if c in df.columns]).copy()
    validar_items(df, cfg)

    df[cfg.items] = df[cfg.items].astype(int)
    lo, hi = cfg["escala"]["minimo"], cfg["escala"]["maximo"]
    for item in cfg["items_invertidos"]:
        df[item] = lo + hi - df[item]

    for col in ("ocupacion", "genero", "rango_edad", "estado_civil", "nivel_educativo"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "anios_funcionamiento" in df.columns:
        df["anios_funcionamiento"] = pd.to_numeric(df["anios_funcionamiento"], errors="coerce")

    if "nivel_educativo" in df.columns:
        grupos = cfg["nivel_educativo_grupos"]
        df["nivel_educativo_grupo"] = df["nivel_educativo"].map(grupos)
        sin_grupo = df.loc[df["nivel_educativo_grupo"].isna(), "nivel_educativo"].unique()
        if len(sin_grupo):
            raise ValueError(f"Niveles educativos sin grupo en config.yaml: {list(sin_grupo)}")

    # Respuesta plana = varianza intrapersona cero en los ítems.
    df["respuesta_plana"] = df[cfg.items].nunique(axis=1).eq(1)

    contexto = [c for c in [*COLUMNAS_CONTEXTO, "nivel_educativo_grupo"] if c in df.columns]
    return df[[*contexto, "respuesta_plana", *cfg.items]].reset_index(drop=True)


def cargar(cfg: Config, fuente: Fuente) -> pd.DataFrame:
    """Carga y limpia los datos de la fuente pedida.

    Parámetros
    ----------
    cfg:
        Configuración.
    fuente:
        ``"real"`` (crudo local, no versionado) o ``"sintetico"`` (versionado).

    Retorna
    -------
    pandas.DataFrame
        Tabla limpia de :func:`limpiar`, con todas las filas (incluidas las
        planas, marcadas).
    """
    if fuente == "real":
        ruta = cfg.ruta(cfg["datos"]["crudo"])
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe {ruta}. Los datos reales no se versionan; "
                "usa --fuente sintetico o copia el CSV a data/raw/."
            )
        crudo = leer_crudo(ruta)
    elif fuente == "sintetico":
        crudo = pd.read_csv(cfg.ruta(cfg["datos"]["sintetico"]))
    else:
        raise ValueError(f"Fuente desconocida: {fuente!r}")
    return limpiar(crudo, cfg)


def para_analisis(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aplica los filtros de la muestra analítica (p. ej. excluir respuestas planas)."""
    if cfg["limpieza"]["excluir_respuesta_plana"]:
        df = df.loc[~df["respuesta_plana"]]
    return df.reset_index(drop=True)
