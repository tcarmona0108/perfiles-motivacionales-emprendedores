"""Figuras con estilo único.

Reglas de estilo (para que todas las figuras se lean igual)
-----------------------------------------------------------
* El **título dice la conclusión** y se calcula a partir de los resultados; así
  nunca contradice los datos si estos cambian. El subtítulo dice qué se grafica.
* Paleta validada para daltonismo (paleta de referencia del skill de
  visualización): categórica azul / naranja / aqua en orden fijo; divergente
  azul ↔ rojo con gris neutro en el centro; secuencial en un solo tono de azul.
* Marcas delgadas, grilla tenue, sin doble eje, texto en tinta neutra (nunca del
  color de la serie).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: el pipeline corre en CI y en servidores

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRILLA = "#e4e3df"
CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NEUTRO = "#c9c8c2"
DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_rojo", ["#d03b3b", "#ec9a8e", "#f0efec", "#86b6ef", "#1c5cab"]
)
SECUENCIAL = LinearSegmentedColormap.from_list("azul", ["#f0efec", "#86b6ef", "#2a78d6", "#0d366b"])


def estilo() -> None:
    """Aplica el estilo común a matplotlib."""
    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TINTA_2,
            "axes.edgecolor": GRILLA,
            "axes.grid": True,
            "grid.color": GRILLA,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TINTA_2,
            "ytick.color": TINTA_2,
            "text.color": TINTA,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def _titulo(fig, titulo: str, subtitulo: str) -> None:
    """Título (conclusión) y subtítulo (qué se grafica) con posiciones en pulgadas.

    Se ubican en unidades absolutas y se ajusta el margen superior según cuántas
    líneas ocupan, para que nunca se superpongan con el gráfico ni entre sí.
    """
    import textwrap

    ancho = int(fig.get_figwidth() * 10.5)
    t = textwrap.wrap(titulo, ancho)
    s = textwrap.wrap(subtitulo, int(ancho * 1.35))
    h = fig.get_figheight()
    y = 1 - 0.05 / h
    salto = "\n"
    fig.text(0.01, y, salto.join(t), ha="left", va="top", fontsize=12.5, fontweight="bold")
    y -= (0.23 * len(t) + 0.05) / h
    fig.text(0.01, y, salto.join(s), ha="left", va="top", fontsize=9.5, color=TINTA_2)
    y -= (0.18 * len(s) + 0.30) / h
    fig.subplots_adjust(top=y)


def guardar(fig, ruta: Path) -> None:
    """Guarda en PNG a 150 ppp y cierra la figura (evita fugas de memoria en el pipeline)."""
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────── Paso 1 ──────────────────────────────────────
def distribucion_items(desc: pd.DataFrame) -> plt.Figure:
    """Barras apiladas divergentes de las respuestas 1–5 por ítem, ordenadas por media."""
    estilo()
    d = desc.sort_values("media")
    n_techo = int((d["pct_5"] > 50).sum())
    colores = ["#d03b3b", "#ec9a8e", NEUTRO, "#86b6ef", "#1c5cab"]
    fig, ax = plt.subplots(figsize=(9, 7))
    izquierda = -(d["pct_1"] + d["pct_2"] + d["pct_3"] / 2)
    etiquetas = [f"{i} · {e}" for i, e in zip(d.index, d["etiqueta"], strict=True)]
    for c, color in zip(range(1, 6), colores, strict=True):
        ax.barh(
            etiquetas,
            d[f"pct_{c}"],
            left=izquierda,
            color=color,
            height=0.7,
            edgecolor=SUPERFICIE,
            linewidth=1,
            label=str(c),
        )
        izquierda = izquierda + d[f"pct_{c}"]
    ax.axvline(0, color=TINTA_2, linewidth=0.8)
    ax.set_xlabel("% de respuestas (izquierda: 1–2, centro: 3, derecha: 4–5)")
    ax.grid(axis="y", visible=False)
    ax.legend(title="Respuesta", ncols=5, loc="lower right", bbox_to_anchor=(1, 1.0))
    _titulo(
        fig,
        f"{n_techo} de {len(d)} motivos tienen más de la mitad de respuestas en 5 (efecto techo)",
        "Distribución de respuestas por ítem, ordenada por media",
    )
    return fig


# ─────────────────────────────── Paso 3 ──────────────────────────────────────
def analisis_paralelo(tabla: pd.DataFrame, n_factores: int, col_umbral: str) -> plt.Figure:
    """Autovalores observados frente al umbral del azar (percentil de las réplicas nulas)."""
    estilo()
    t = tabla.iloc[:10]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(
        t.index, t["observado"], marker="o", ms=6, color=CATEGORICA[0], label="Datos observados"
    )
    ax.plot(
        t.index,
        t[col_umbral],
        marker="o",
        ms=5,
        color=NEUTRO,
        linestyle="--",
        label="Azar (percentil 95, datos permutados)",
    )
    ax.axhline(1, color=GRILLA, linewidth=1)
    ax.text(10, 1.05, "Kaiser (autovalor = 1)", ha="right", va="bottom", fontsize=8, color=TINTA_2)
    ax.axvspan(0.5, n_factores + 0.5, color=CATEGORICA[0], alpha=0.06)
    ax.set_xticks(t.index)
    ax.set_xlabel("Factor")
    ax.set_ylabel("Autovalor")
    ax.legend(loc="upper right")
    _titulo(
        fig,
        f"Solo {n_factores} factores superan lo que produciría el azar",
        "Análisis paralelo con correlaciones policóricas",
    )
    return fig


def cargas(
    cargas_df: pd.DataFrame, etiquetas: dict[str, str], asignacion: pd.DataFrame
) -> plt.Figure:
    """Mapa de calor de cargas (patrón) con los valores ≥ 0.30 rotulados."""
    estilo()
    orden = asignacion.sort_values(["factor", "carga"], ascending=[True, False]).index
    c = cargas_df.loc[orden]
    n_limpios = int(asignacion["limpio"].sum())
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.imshow(c.to_numpy(), cmap=DIVERGENTE, vmin=-1, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(c.shape[1]), [n.replace(" y ", " y\n") for n in c.columns], fontsize=9)
    ax.xaxis.tick_top()
    marcas = ["" if asignacion.loc[i, "limpio"] else "  ⚠" for i in c.index]
    ax.set_yticks(
        range(len(c)), [f"{i} · {etiquetas[i]}{m}" for i, m in zip(c.index, marcas, strict=True)]
    )
    for (fi, co), v in np.ndenumerate(c.to_numpy()):
        if abs(v) >= 0.30:
            ax.text(
                co,
                fi,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if abs(v) > 0.6 else TINTA,
            )
    _titulo(
        fig,
        f"{n_limpios} de {len(c)} preguntas miden un solo factor sin ambigüedad",
        "Cargas factoriales (EFA, minres + oblimin, policóricas). ⚠ = carga baja o cruzada",
    )
    return fig


# ─────────────────────────────── Paso 4 ──────────────────────────────────────
def curva_items(curva: pd.DataFrame, k: int) -> plt.Figure:
    """r entre el puntaje con n preguntas y el puntaje factorial completo."""
    estilo()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for color, (f, g) in zip(CATEGORICA, curva.groupby("factor", sort=False), strict=False):
        ax.plot(g["n_items"], g["r"], marker="o", ms=6, color=color, label=f)
    ax.axhline(0.90, color=TINTA_2, linewidth=0.8, linestyle=":")
    ax.axvline(k, color=GRILLA, linewidth=1.2)
    ax.set_xlabel("Número de preguntas del factor en el formulario")
    ax.set_ylabel("r con el puntaje completo")
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="lower right")
    # El título se calcula solo con los factores que tienen al menos k preguntas limpias;
    # los que tienen menos se declaran en el subtítulo para no inflar ni ocultar nada.
    maximo = curva.groupby("factor")["n_items"].max()
    completos = maximo.index[maximo >= k]
    r_k = curva[(curva["n_items"] == k) & curva["factor"].isin(completos)]["r"]
    r_final = curva.groupby("factor")["r"].last()
    cortos = [
        f"{f} solo tiene {n} preguntas limpias (r = {r_final[f]:.2f})"
        for f, n in maximo[maximo < k].items()
    ]
    _titulo(
        fig,
        f"Con {k} preguntas por factor se recupera r ≥ {r_k.min():.2f} del puntaje completo",
        "; ".join(["Preguntas agregadas en orden de carga factorial", *cortos]),
    )
    return fig


# ─────────────────────────────── Paso 6 ──────────────────────────────────────
def seleccion_k(
    tabla: pd.DataFrame,
    nombres: dict[str, str],
    umbral_ari: float,
    tamano_minimo: float,
    titulo: str,
) -> plt.Figure:
    """Estabilidad (ARI) y silueta por k y algoritmo, en dos paneles con el mismo eje x.

    Los marcadores huecos señalan soluciones descartadas por tener un segmento
    menor que ``tamano_minimo`` (aunque sean estables, aíslan a unas pocas personas).
    """
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True)
    for color, (alg, g) in zip(CATEGORICA, tabla.groupby("algoritmo", sort=False), strict=False):
        g = g.sort_values("k")
        chico = g["tamano_min"] < tamano_minimo
        for ax, col in ((a1, "ari_medio"), (a2, "silueta")):
            ax.plot(g["k"], g[col], color=color, label=nombres[alg] if ax is a1 else None)
            ax.scatter(g.loc[~chico, "k"], g.loc[~chico, col], s=40, color=color, zorder=3)
            ax.scatter(g.loc[chico, "k"], g.loc[chico, col], s=40, facecolor=SUPERFICIE,
                       edgecolor=color, linewidth=1.6, zorder=3)  # fmt: skip
    a1.scatter([], [], s=40, facecolor=SUPERFICIE, edgecolor=TINTA_2, linewidth=1.6,
               label=f"Hueco: algún segmento < {tamano_minimo:.0%} de la muestra")  # fmt: skip
    a1.axhline(umbral_ari, color=TINTA_2, linewidth=0.8, linestyle=":")
    a1.text(
        tabla["k"].max(),
        umbral_ari + 0.01,
        "umbral de estabilidad",
        ha="right",
        fontsize=8,
        color=TINTA_2,
    )
    a1.set_ylabel("ARI medio (submuestras del 80 %)")
    a2.set_ylabel("Silueta")
    for a in (a1, a2):
        a.set_xlabel("Número de segmentos (k)")
        a.set_xticks(sorted(tabla["k"].unique()))
    a1.set_ylim(0, 1)
    fig.legend(loc="lower center", ncols=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    _titulo(
        fig,
        titulo,
        "Más alto es mejor en ambos paneles. Estabilidad: acuerdo con la solución completa",
    )
    fig.subplots_adjust(wspace=0.25, bottom=0.24)
    return fig


# ─────────────────────────────── Paso 7 ──────────────────────────────────────
def perfiles(
    resumen: pd.DataFrame, factores: list[str], nombres: dict[int, str], titulo: str
) -> plt.Figure:
    """Puntaje factorial medio (z) de cada segmento: un panel por segmento."""
    estilo()
    segs = list(resumen.index)
    import textwrap

    fig, ejes = plt.subplots(1, len(segs), figsize=(3.6 * len(segs), 4.4), sharey=True)
    ejes = np.atleast_1d(ejes)
    # Margen amplio para que las etiquetas de valor no choquen con el eje ni con el panel vecino.
    zmax = max(1.0, resumen[[f"z_{f}" for f in factores]].abs().to_numpy().max() * 1.45)
    for ax, s, color in zip(ejes, segs, CATEGORICA, strict=False):
        valores = [resumen.loc[s, f"z_{f}"] for f in factores]
        ax.barh(range(len(factores)), valores, color=color, height=0.6)
        ax.axvline(0, color=TINTA_2, linewidth=0.8)
        for i, v in enumerate(valores):
            ax.text(
                v + (0.04 if v >= 0 else -0.04),
                i,
                f"{v:+.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8.5,
                color=TINTA_2,
            )
        ax.set_xlim(-zmax, zmax)
        ax.set_yticks(range(len(factores)), factores)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        n, pct = int(resumen.loc[s, "n"]), resumen.loc[s, "pct"]
        nombre = "\n".join(textwrap.wrap(f"{s}. {nombres[s]}", 30))
        ax.set_title(
            f"{nombre}\n{n} personas ({pct:.0f} %)",
            fontsize=9.5,
            fontweight="normal",
            color=TINTA,
        )
        ax.set_xlabel("Desviaciones respecto al promedio")
    _titulo(fig, titulo, "Puntaje factorial medio por segmento (0 = promedio de la muestra)")
    # Espacio extra bajo el subtítulo para los títulos de panel (hasta 3 líneas).
    fig.subplots_adjust(top=fig.subplotpars.top - 0.55 / fig.get_figheight(), wspace=0.12)
    return fig


# ─────────────────────────────── Paso 9 ──────────────────────────────────────
def confusion(cm: np.ndarray, clases: list, nombre_modelo: str, titulo: str) -> plt.Figure:
    """Matriz de confusión normalizada por fila (acumulada en la validación cruzada)."""
    estilo()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap=SECUENCIAL, vmin=0, vmax=1)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            f"{100 * v:.0f}%",
            ha="center",
            va="center",
            fontsize=10,
            color="white" if v > 0.55 else TINTA,
        )
    ax.set_xticks(range(len(clases)), clases)
    ax.set_yticks(range(len(clases)), clases)
    ax.set_xlabel("Segmento predicho (formulario corto)")
    ax.set_ylabel("Segmento real (cuestionario completo)")
    _titulo(fig, titulo, f"{nombre_modelo}; validación cruzada 5×20, % por fila")
    return fig
