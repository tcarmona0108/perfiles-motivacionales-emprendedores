"""Tests de los métodos de la Fase 2 con datos simulados de estructura CONOCIDA.

La lógica: si simulamos 2 factores verdaderos o 3 grupos bien separados, cada
método debe recuperar esa verdad. Si no lo hace con datos fáciles, no se puede
confiar en él con los datos reales.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from emprendimiento import modelo, perfiles, psicometria, reduccion, segmentacion, sesgo
from emprendimiento.correlaciones import matriz_policorica
from emprendimiento.estadistica import chi2_permutacion

warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="module")
def dos_factores():
    """6 ítems Likert: p1–p3 miden el factor A, p4–p6 el factor B (corr. 0.3)."""
    rng = np.random.default_rng(0)
    n = 600
    f = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], size=n)
    lam = 0.8
    z = np.column_stack(
        [lam * f[:, 0] + np.sqrt(1 - lam**2) * rng.standard_normal(n) for _ in range(3)]
        + [lam * f[:, 1] + np.sqrt(1 - lam**2) * rng.standard_normal(n) for _ in range(3)]
    )
    x = np.searchsorted([-1.5, -0.7, 0.0, 0.8], z) + 1
    return pd.DataFrame(x, columns=[f"p{i}" for i in range(1, 7)])


# ─────────────────────────── psicometría ─────────────────────────────────────
def test_kmo_y_bartlett_coinciden_con_factor_analyzer(dos_factores):
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

    r = dos_factores.corr().to_numpy()
    items, total = psicometria.kmo(r)
    fa_items, fa_total = calculate_kmo(dos_factores)
    np.testing.assert_allclose(items, fa_items, atol=1e-8)
    assert total == pytest.approx(fa_total)
    chi, _, _ = psicometria.bartlett(r, len(dos_factores))
    assert chi == pytest.approx(calculate_bartlett_sphericity(dos_factores)[0], rel=1e-6)


def test_paralelo_recupera_dos_factores(dos_factores):
    r = matriz_policorica(dos_factores)
    _, n = psicometria.analisis_paralelo(dos_factores, r.to_numpy(), 30, 95, semilla=1)
    assert n["paralelo_policorico"] == 2


def test_efa_asigna_items_al_factor_correcto(dos_factores):
    r = matriz_policorica(dos_factores)
    sol = psicometria.efa(r, 2)
    asig = psicometria.asignar_items(sol.cargas, 0.4, 0.2)
    grupos = asig.groupby("factor").groups
    assert sorted(sorted(g) for g in grupos.values()) == [["p1", "p2", "p3"], ["p4", "p5", "p6"]]
    assert asig["limpio"].all()
    assert 0.15 < abs(sol.phi.iloc[0, 1]) < 0.5  # recupera la correlación entre factores (0.3)


def test_efa_es_determinista(dos_factores):
    r = matriz_policorica(dos_factores)
    a, b = psicometria.efa(r, 2), psicometria.efa(r, 2)
    pd.testing.assert_frame_equal(a.cargas, b.cargas)


def test_alfa_cronbach_valor_conocido():
    x = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5], "c": [1, 2, 3, 4, 5]})
    assert psicometria.alfa_cronbach(x) == pytest.approx(1.0)


def test_omega_igual_a_alfa_si_cargas_iguales():
    r = np.full((4, 4), 0.5)
    np.fill_diagonal(r, 1)
    assert psicometria.omega_unifactorial(r) == pytest.approx(
        psicometria.alfa_desde_matriz(r), abs=1e-3
    )


def test_puntuador_es_lineal_y_reproducible(dos_factores):
    r = matriz_policorica(dos_factores)
    sol = psicometria.efa(r, 2)
    pt = psicometria.Puntuador.ajustar(dos_factores, r, sol)
    p = pt.transformar(dos_factores)
    assert p.shape == (len(dos_factores), 2)
    assert abs(p.mean()).max() < 1e-10  # centrados
    np.testing.assert_allclose(pt.transformar(dos_factores.iloc[:5]), p.iloc[:5])


def test_congruencia_y_alineacion():
    a = np.array([[0.8, 0.0], [0.7, 0.1], [0.0, 0.9], [0.1, 0.6]])
    b = -a[:, ::-1]  # mismo patrón, factores invertidos de orden y de signo
    alineada = psicometria.alinear(a, b)
    np.testing.assert_allclose(np.diag(psicometria.congruencia_tucker(a, alineada)), 1.0)


# ─────────────────────────── reducción ───────────────────────────────────────
def test_seleccion_toma_items_limpios_de_mayor_carga():
    asig = pd.DataFrame(
        {
            "factor": ["A", "A", "A", "A", "B", "B"],
            "carga": [0.9, 0.5, 0.7, 0.8, 0.6, 0.65],
            "limpio": [True, True, True, False, True, True],
        },
        index=[f"p{i}" for i in range(1, 7)],
    )
    corto, tabla = reduccion.seleccionar_items(asig, 2)
    assert corto["A"] == ["p1", "p3"]  # p4 tiene más carga pero no es limpio
    assert corto["B"] == ["p6", "p5"]
    assert tabla["en_corto"].sum() == 4


# ─────────────────────────── segmentación ────────────────────────────────────
@pytest.fixture(scope="module")
def tres_grupos():
    rng = np.random.default_rng(3)
    centros = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0]])
    return np.vstack([c + rng.standard_normal((60, 3)) * 0.6 for c in centros])


def test_segmentacion_recupera_tres_grupos_estables(tres_grupos):
    cfg_seg = {
        "algoritmos": ["kmeans", "lpa"],
        "k_candidatos": [2, 3, 4],
        "n_bootstrap": 15,
        "fraccion_bootstrap": 0.8,
        "n_init": 10,
        "ari_estable": 0.75,
        "tamano_minimo_segmento": 0.10,
    }
    tabla = segmentacion.evaluar(tres_grupos, cfg_seg, semilla=0)
    fila, _ = segmentacion.elegir(tabla, cfg_seg)
    assert int(fila.k) == 3
    assert fila.ari_medio > 0.95


def test_prueba_sin_grupos_distingue_estructura(tres_grupos):
    etq, _ = segmentacion.ajustar("kmeans", 3, tres_grupos, 0, 10)
    from sklearn.metrics import silhouette_score

    con = segmentacion.prueba_sin_grupos(
        tres_grupos, "kmeans", 3, silhouette_score(tres_grupos, etq), 30, 0, 10
    )
    assert con["p_valor"] < 0.05

    continuo = np.random.default_rng(5).standard_normal((180, 3))
    etq2, _ = segmentacion.ajustar("kmeans", 3, continuo, 0, 10)
    sin = segmentacion.prueba_sin_grupos(
        continuo, "kmeans", 3, silhouette_score(continuo, etq2), 30, 0, 10
    )
    assert sin["p_valor"] > 0.05


def test_forma_ignora_el_nivel_general():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((50, 3))
    desplazado = z + rng.normal(0, 3, size=(50, 1))  # cada persona responde más alto o bajo en todo
    np.testing.assert_allclose(
        segmentacion.representacion_forma(z), segmentacion.representacion_forma(desplazado)
    )
    assert segmentacion.representacion_forma(z).shape == (50, 2)


def test_ordenar_por_nivel():
    x = np.array([[5.0], [5.1], [-5.0], [-4.9]])
    etq = np.array([0, 0, 1, 1])
    assert segmentacion.ordenar_por_nivel(etq, x).tolist() == [2, 2, 1, 1]


# ─────────────────────────── perfiles / modelo / sesgo ───────────────────────
def test_nombres_por_nivel_y_por_forma():
    res = pd.DataFrame(
        {"z_A": [-0.8, 0.6, 0.5], "z_B": [-0.5, 0.4, -0.6], "z_C": [-0.6, 0.3, 0.2]},
        index=[1, 2, 3],
    )
    n = perfiles.nombrar_segmentos(res, ["A", "B", "C"])
    assert n[1] == "Motivación moderada en todos los factores"
    assert n[2] == "Motivación alta en todos los factores"
    assert n[3] == "Predomina: A"


def test_elige_interpretable_si_la_diferencia_es_pequena():
    res = pd.DataFrame(
        {"f1_macro": [0.4, 0.80, 0.70, 0.82]},
        index=[
            "Línea base (clase mayoritaria)",
            "Regresión logística",
            "Árbol (profundidad 3)",
            "Random forest",
        ],
    )
    assert modelo.elegir(res, 0.03)[0] == "Regresión logística"
    assert modelo.elegir(res, 0.01)[0] == "Random forest"


def test_chi2_permutacion_detecta_asociacion():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.integers(0, 2, 200))
    assert chi2_permutacion(a, a.copy(), 200, rng)["p_permutacion"] < 0.01
    b = pd.Series(rng.integers(0, 2, 200))
    assert chi2_permutacion(a, b, 200, rng)["p_permutacion"] > 0.01


def test_sesgo_suprime_celdas_pequenas():
    ctx = pd.DataFrame({"genero": ["F"] * 40 + ["M"] * 38 + ["X"] * 2})
    etq = np.array([1, 2] * 40)
    _, tablas = sesgo.auditar(ctx, etq, ["genero"], 5, 50, np.random.default_rng(0))
    assert tablas["genero"].loc["X"].str.startswith("<5").all()
