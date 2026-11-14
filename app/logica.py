"""Lógica de la demo, sin interfaz: cargar el modelo, predecir y explicar.

Está separada de ``streamlit_app.py`` para poder probarla con pytest y para que
la interfaz solo se ocupe de mostrar. Depende únicamente de numpy: el modelo
llega como parámetros en JSON (``results/real/tablas/modelo_app.json``), así
que la demo no necesita scikit-learn ni los datos de entrenamiento.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
RUTA_MODELO = RAIZ / "results" / "real" / "tablas" / "modelo_app.json"
ESCALA = (1, 5)


@dataclass(frozen=True)
class Resultado:
    """Lo que la demo le muestra a una persona.

    Atributos
    ---------
    segmento:
        Perfil asignado (1..k).
    nombre:
        Nombre del perfil.
    probabilidades:
        Probabilidad estimada de cada perfil (suma 1).
    puntajes:
        Media 1–5 de la persona en cada motivación (con sus preguntas del formulario).
    """

    segmento: int
    nombre: str
    probabilidades: dict[int, float]
    puntajes: dict[str, float]

    @property
    def confianza(self) -> float:
        return self.probabilidades[self.segmento]


def cargar_modelo(ruta: Path = RUTA_MODELO) -> dict:
    """Lee el JSON del modelo y valida que tenga lo necesario."""
    m = json.loads(Path(ruta).read_text(encoding="utf-8"))
    p = m["parametros"]
    n = len(m["items"])
    if not (len(p["media"]) == len(p["escala"]) == n):
        raise ValueError("El JSON del modelo es inconsistente con su lista de preguntas.")
    return m


def _validar(modelo: dict, respuestas: dict[str, int]) -> np.ndarray:
    faltan = [i for i in modelo["items"] if i not in respuestas]
    if faltan:
        raise ValueError(f"Faltan respuestas: {faltan}")
    x = np.array([respuestas[i] for i in modelo["items"]], dtype=float)
    if ((x < ESCALA[0]) | (x > ESCALA[1])).any():
        raise ValueError("Las respuestas deben estar entre 1 y 5.")
    return x


def probabilidades(modelo: dict, x: np.ndarray) -> np.ndarray:
    """Probabilidad de cada clase con la regresión logística exportada.

    Reproduce ``StandardScaler`` + ``LogisticRegression.predict_proba`` de
    scikit-learn: binaria con sigmoide (un vector de coeficientes para la clase
    positiva) o multinomial con softmax.
    """
    p = modelo["parametros"]
    z = (x - np.array(p["media"])) / np.array(p["escala"])
    w = np.array(p["coeficientes"])
    b = np.array(p["interceptos"])
    logits = w @ z + b
    if w.shape[0] == 1:  # binaria
        p1 = 1.0 / (1.0 + np.exp(-logits[0]))
        return np.array([1.0 - p1, p1])
    e = np.exp(logits - logits.max())
    return e / e.sum()


def evaluar(modelo: dict, respuestas: dict[str, int]) -> Resultado:
    """Asigna perfil y calcula los puntajes por motivación de una persona."""
    x = _validar(modelo, respuestas)
    proba = probabilidades(modelo, x)
    clases = modelo["parametros"]["clases"]
    seg = int(clases[int(np.argmax(proba))])
    puntajes = {
        f: float(np.mean([respuestas[i] for i in items]))
        for f, items in modelo["items_por_factor"].items()
    }
    return Resultado(
        segmento=seg,
        nombre=modelo["nombres_segmentos"][str(seg)],
        probabilidades={int(c): float(v) for c, v in zip(clases, proba, strict=True)},
        puntajes=puntajes,
    )


def explicar(modelo: dict, r: Resultado) -> list[str]:
    """Frases en lenguaje claro sobre qué define el perfil de la persona.

    Se compara a la persona con el promedio de la muestra, no con un ideal: decir
    "más que la mayoría" es honesto; decir "mejor" no lo sería.
    """
    glob = modelo["media_corta_global"]
    dif = {f: r.puntajes[f] - glob[f] for f in r.puntajes}
    alto = max(dif, key=dif.get)
    bajo = min(dif, key=dif.get)
    frases = []
    if dif[alto] > 0.25:
        frases.append(
            f"Lo que más te distingue es **{alto}**: lo valoras con {r.puntajes[alto]:.1f} de 5, "
            f"por encima del promedio de las personas encuestadas ({glob[alto]:.1f})."
        )
    else:
        frases.append(
            "Ninguna motivación sobresale con claridad frente al promedio de las personas "
            "encuestadas: tu perfil es parejo."
        )
    if dif[bajo] < -0.25:
        frases.append(
            f"La que menos pesa para ti es **{bajo}** ({r.puntajes[bajo]:.1f} de 5, frente a "
            f"{glob[bajo]:.1f} en promedio)."
        )
    if "alta" in r.nombre.lower():
        frases.append(
            "Tu perfil reúne a quienes declaran que **todas** las motivaciones pesan mucho en "
            "su decisión de emprender."
        )
    elif "moderada" in r.nombre.lower():
        frases.append(
            "Tu perfil reúne a quienes declaran motivaciones algo menos intensas que el resto. "
            "Moderada no significa baja: en este grupo la autonomía suele seguir siendo muy "
            "importante."
        )
    if r.confianza < 0.65:
        frases.append(
            "Estás **cerca del límite entre los dos perfiles**. Recuerda que no son tipos "
            "cerrados, sino puntos de un mismo continuo."
        )
    return frases
