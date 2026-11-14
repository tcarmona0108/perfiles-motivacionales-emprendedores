import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from emprendimiento.correlaciones import (
    cdf_normal_bivariada,
    matriz_policorica,
    policorica_par,
    suavizar_definida_positiva,
)


@pytest.mark.parametrize("r", [-0.97, -0.8, -0.5, -0.1, 0.0, 0.2, 0.6, 0.8, 0.93, 0.99])
def test_cdf_bivariada_coincide_con_scipy(r):
    """La implementación de Genz debe coincidir con scipy en todo el rango de ρ,
    incluidas las tres ramas del algoritmo (|ρ|<0.3, <0.75, <0.925 y ≥0.925)."""
    rng = np.random.default_rng(0)
    h, k = rng.uniform(-3, 3, 40), rng.uniform(-3, 3, 40)
    esperado = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf(np.c_[h, k])
    np.testing.assert_allclose(cdf_normal_bivariada(h, k, r), esperado, atol=1e-6)


def _discretizar(z, cortes):
    return np.searchsorted(cortes, z) + 1


@pytest.mark.parametrize("rho", [-0.5, 0.3, 0.6, 0.9])
def test_policorica_recupera_rho_latente(rho):
    """Con latentes normales cortadas en umbrales asimétricos (como el efecto
    techo real), la policórica recupera ρ y Pearson la subestima."""
    rng = np.random.default_rng(1)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=5000)
    cortes = np.array([-2.0, -1.4, -0.7, 0.1])  # mayoría de respuestas en 5
    x, y = _discretizar(z[:, 0], cortes), _discretizar(z[:, 1], cortes)
    est = policorica_par(x, y)
    assert abs(est - rho) < 0.04
    assert abs(np.corrcoef(x, y)[0, 1]) < abs(est)


def test_policorica_constante_es_nan():
    assert np.isnan(policorica_par(np.ones(50), np.arange(50) % 5))


def test_matriz_simetrica_definida_positiva(df_sint, cfg):
    r = matriz_policorica(df_sint[cfg.items[:6]])
    np.testing.assert_allclose(r.values, r.values.T)
    np.testing.assert_allclose(np.diag(r.values), 1.0)
    assert np.linalg.eigvalsh(r.values).min() > 0


def test_suavizado_repara_matriz_no_definida():
    r = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1.0]])
    r2, suavizada = suavizar_definida_positiva(r)
    assert suavizada
    assert np.linalg.eigvalsh(r2).min() > 0
    np.testing.assert_allclose(np.diag(r2), 1.0)


def test_matriz_conserva_nombres(df_sint, cfg):
    r = matriz_policorica(df_sint[cfg.items[:3]])
    assert isinstance(r, pd.DataFrame)
    assert list(r.columns) == cfg.items[:3]
