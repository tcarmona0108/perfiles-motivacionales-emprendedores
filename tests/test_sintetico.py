import numpy as np
import pandas as pd

from emprendimiento import datos, sintetico


def test_muestreo_reproducible_y_en_escala(cfg, df_sint):
    """Ajustar sobre el sintético (en los tests no hay datos reales) y volver a muestrear."""
    modelo = sintetico.ajustar(df_sint, cfg)
    a = sintetico.muestrear(modelo, 200, np.random.default_rng(7))
    b = sintetico.muestrear(modelo, 200, np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)
    assert a[cfg.items].isin(range(1, 6)).all().all()
    assert len(a) == 200
    # El resultado debe pasar la misma limpieza que los datos reales.
    limpio = datos.limpiar(a, cfg)
    assert not set(datos.COLUMNAS_IDENTIFICABLES) & set(limpio.columns)


def test_muestreo_conserva_marginales(cfg, df_sint):
    modelo = sintetico.ajustar(df_sint, cfg)
    grande = datos.limpiar(sintetico.muestrear(modelo, 20000, np.random.default_rng(3)), cfg)
    base = df_sint.loc[~df_sint["respuesta_plana"], "p13"].value_counts(normalize=True)
    nuevo = grande.loc[~grande["respuesta_plana"], "p13"].value_counts(normalize=True)
    assert base.sub(nuevo, fill_value=0).abs().max() < 0.02
    assert abs(grande["respuesta_plana"].mean() - df_sint["respuesta_plana"].mean()) < 0.01
