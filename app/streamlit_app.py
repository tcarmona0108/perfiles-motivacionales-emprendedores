"""Demo "Descubre tu perfil emprendedor".

Qué hace: la persona responde las 8 preguntas del formulario corto y ve su perfil
motivacional, un gráfico radar de sus motivaciones frente al promedio de su
perfil y de la muestra, y una explicación en lenguaje claro.

Garantías de privacidad (también se muestran en pantalla):
* No pide datos personales.
* No guarda respuestas: no hay base de datos ni registros; las respuestas viven
  solo en la sesión del navegador y desaparecen al cerrarla.
* No usa los datos crudos: solo los parámetros agregados del modelo
  (``results/real/tablas/modelo_app.json``).

Ejecutar en local::

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import logica

# Paleta validada (la misma del reporte): azul = tú, naranja = tu perfil, gris = muestra.
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#8c8b85"

# Orden de las preguntas: se intercalan las motivaciones para que no se respondan
# en bloque (responder en bloque infla la consistencia de forma artificial).
ORDEN = ["p03", "p10", "p16", "p02", "p12", "p15", "p07", "p11"]

st.set_page_config(page_title="Descubre tu perfil emprendedor", page_icon="🧭", layout="centered")


@st.cache_data
def _modelo() -> dict:
    return logica.cargar_modelo()


def _radar(modelo: dict, r: logica.Resultado) -> go.Figure:
    factores = list(r.puntajes)
    perfil = modelo["media_corta_por_segmento"][str(r.segmento)]
    glob = modelo["media_corta_global"]
    cerrar = factores + factores[:1]  # el polígono se cierra repitiendo el primer eje

    def traza(valores, nombre, color, discontinua=False, relleno=False):
        v = [valores[f] for f in factores]
        return go.Scatterpolar(
            r=v + v[:1],
            theta=cerrar,
            name=nombre,
            line={"color": color, "width": 2, "dash": "dot" if discontinua else "solid"},
            fill="toself" if relleno else None,
            fillcolor="rgba(42,120,214,0.15)" if relleno else None,
            hovertemplate="%{theta}: %{r:.1f} de 5<extra>" + nombre + "</extra>",
        )

    fig = go.Figure(
        [
            traza(glob, "Promedio de todas las personas", GRIS, discontinua=True),
            traza(perfil, f"Promedio de tu perfil ({r.nombre.lower()})", NARANJA),
            traza(r.puntajes, "Tú", AZUL, relleno=True),
        ]
    )
    fig.update_layout(
        polar={"radialaxis": {"range": [1, 5], "tickvals": [1, 2, 3, 4, 5]}},
        legend={"orientation": "h", "y": -0.15},
        margin={"l": 40, "r": 40, "t": 30, "b": 30},
        height=420,
    )
    return fig


def pestana_perfil(modelo: dict) -> None:
    st.warning(
        "**Herramienta orientativa.** Se basa en una investigación con 123 emprendedores y "
        "describe **motivaciones declaradas**. No es una evaluación definitiva, no mide tu "
        "capacidad ni predice el éxito de un negocio, y no debe usarse por sí sola para "
        "decidir sobre una persona.",
        icon="⚠️",
    )
    st.caption(
        "No pedimos datos personales y no guardamos tus respuestas: desaparecen al cerrar la "
        "página."
    )

    with st.form("formulario"):
        st.markdown(
            "**¿Qué tan importante es (o fue) cada uno de estos motivos en tu decisión de "
            "emprender?**  \n1 = menor importancia · 5 = mayor importancia"
        )
        respuestas = {}
        for i, item in enumerate(ORDEN, start=1):
            respuestas[item] = st.radio(
                f"{i}. {modelo['etiquetas_items'][item]}",
                options=[1, 2, 3, 4, 5],
                index=None,
                horizontal=True,
                key=item,
            )
        enviado = st.form_submit_button("Ver mi perfil", type="primary")

    if not enviado:
        return
    faltan = [k for k, v in respuestas.items() if v is None]
    if faltan:
        st.error(f"Te faltan {len(faltan)} pregunta(s) por responder.")
        return

    r = logica.evaluar(modelo, respuestas)
    st.subheader(f"Tu perfil: {r.nombre}")
    st.markdown(
        f"Según tus respuestas, el formulario te ubica en este perfil con una **probabilidad "
        f"estimada del {100 * r.confianza:.0f} %**."
    )
    st.plotly_chart(_radar(modelo, r), width="stretch", config={"displayModeBar": False})
    for frase in logica.explicar(modelo, r):
        st.markdown(f"- {frase}")
    with st.expander("¿Cómo se calcula esto?"):
        st.markdown(
            "- Cada motivación es el **promedio de sus preguntas** (escala 1–5).\n"
            "- El perfil lo asigna una **regresión logística** entrenada con las respuestas de "
            "la investigación: es un modelo simple cuyos pesos se pueden inspeccionar.\n"
            "- La probabilidad indica qué tan clara es la asignación, **no** qué tan buena "
            "emprendedora es una persona."
        )


def pestana_sobre(modelo: dict) -> None:
    sitio = logica.cargar_sitio()
    c, enlaces_cfg = sitio["creditos"], sitio["enlaces"]
    st.markdown(
        f"""
### Sobre el proyecto

**Investigación de origen:** {c["investigacion_origen"]}

**Análisis reproducible y demo:** {c["autor_v2"]}

Un cuestionario de 19 motivos para emprender se analizó con psicometría (análisis
factorial exploratorio con correlaciones policóricas) y segmentación validada por
estabilidad. Se encontraron **3 motivaciones** (*{"*, *".join(modelo["items_por_factor"])}*) y se
redujo el cuestionario a **{len(modelo["items"])} preguntas**.

**Qué tan bien funciona:** con validación cruzada, el formulario corto reproduce el perfil
obtenido con el cuestionario completo en un **{100 * modelo["accuracy_cv"]:.0f} %** de los
casos (F1-macro {modelo["f1_macro_cv"]:.2f}; una regla trivial obtiene
{modelo["f1_macro_linea_base"]:.2f}).

**Limitaciones importantes**
- Los perfiles son **cortes de un continuo** de intensidad motivacional, no "tipos" de personas.
- La muestra incluye solo personas que ya emprenden ({modelo["n_entrenamiento"]} respuestas
  válidas) y **no hay datos de éxito**: ningún perfil es "mejor" ni predice resultados.
- No usa ni guarda datos personales. Los atributos como género o edad nunca se usaron
  en el modelo; se revisó que los perfiles no queden desbalanceados por ellos.
"""
    )
    enlaces = [
        (t, u)
        for t, u in (
            ("Reporte técnico", enlaces_cfg.get("reporte")),
            ("Repositorio (código)", enlaces_cfg.get("repositorio")),
        )
        if u
    ]
    if enlaces:
        st.markdown(" · ".join(f"[{t}]({u})" for t, u in enlaces))


def main() -> None:
    st.title("🧭 Descubre tu perfil emprendedor")
    st.caption("¿Qué te motiva a emprender? 8 preguntas, menos de 2 minutos.")
    modelo = _modelo()
    perfil, sobre = st.tabs(["Descubre tu perfil", "Sobre el proyecto"])
    with perfil:
        pestana_perfil(modelo)
    with sobre:
        pestana_sobre(modelo)


main()
