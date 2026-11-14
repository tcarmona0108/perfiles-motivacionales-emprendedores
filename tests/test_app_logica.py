"""La demo predice con numpy a partir de un JSON: debe coincidir con scikit-learn."""

import sys
from pathlib import Path

import numpy as np
import pytest

from emprendimiento import modelo as mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import logica


@pytest.fixture(scope="module")
def entrenado(df_sint, cfg):
    """Regresión logística con 8 ítems del sintético: etiqueta binaria y multiclase."""
    items = ["p03", "p02", "p07", "p16", "p15", "p10", "p12", "p11"]
    x = df_sint[items]
    nivel = x.mean(axis=1)
    y2 = np.where(nivel > nivel.median(), 2, 1)
    y3 = np.digitize(nivel, np.quantile(nivel, [1 / 3, 2 / 3])) + 1
    salida = {}
    for nombre, y in (("binaria", y2), ("multiclase", y3)):
        m = mod.candidatos(0)["Regresión logística"].fit(x, y)
        clases = sorted(set(y.tolist()))
        json_modelo = {
            "items": items,
            "items_por_factor": {"A": items[:3], "B": items[3:5], "C": items[5:]},
            "nombres_segmentos": {str(c): f"Perfil {c}" for c in clases},
            "media_corta_global": {"A": 4.0, "B": 3.5, "C": 3.5},
            "parametros": mod.exportar_logistica(m),
        }
        salida[nombre] = (m, json_modelo, x)
    return salida


@pytest.mark.parametrize("tipo", ["binaria", "multiclase"])
def test_numpy_reproduce_predict_proba(entrenado, tipo):
    m, js, x = entrenado[tipo]
    esperado = m.predict_proba(x)
    obtenido = np.vstack([logica.probabilidades(js, fila) for fila in x.to_numpy(float)])
    np.testing.assert_allclose(obtenido, esperado, atol=1e-5)


def test_evaluar_y_explicar(entrenado):
    _, js, _ = entrenado["binaria"]
    respuestas = dict.fromkeys(js["items"], 5)
    r = logica.evaluar(js, respuestas)
    assert r.segmento == 2
    assert abs(sum(r.probabilidades.values()) - 1) < 1e-9
    assert r.puntajes == {"A": 5.0, "B": 5.0, "C": 5.0}
    assert all(isinstance(f, str) and f for f in logica.explicar(js, r))


def test_respuestas_invalidas(entrenado):
    _, js, _ = entrenado["binaria"]
    with pytest.raises(ValueError, match="Faltan"):
        logica.evaluar(js, {"p03": 3})
    with pytest.raises(ValueError, match="entre 1 y 5"):
        logica.evaluar(js, dict.fromkeys(js["items"], 9))


def test_modelo_publicado_es_valido():
    """El JSON versionado (results/real) debe cargarse y no contener datos por persona."""
    if not logica.RUTA_MODELO.exists():
        pytest.skip("Aún no se ha publicado el modelo de la demo.")
    m = logica.cargar_modelo()
    texto = logica.RUTA_MODELO.read_text(encoding="utf-8")
    assert len(texto) < 20_000  # solo parámetros y agregados
    r = logica.evaluar(m, dict.fromkeys(m["items"], 3))
    assert r.segmento in m["parametros"]["clases"]
