"""La publicación de resultados es la frontera de privacidad: se prueba aparte."""

import pandas as pd
import pytest

from emprendimiento import pipeline
from emprendimiento.config import Config


def _cfg_temporal(cfg, tmp_path) -> Config:
    return Config(datos=cfg.datos, raiz=tmp_path)


def test_publicar_excluye_tablas_por_persona(cfg, tmp_path):
    c = _cfg_temporal(cfg, tmp_path)
    tablas = c.dir_salidas("sintetico") / "tablas"
    pd.DataFrame({"a": [1, 2]}).to_csv(tablas / "confiabilidad.csv", index=False)
    pd.DataFrame({"p01": range(100)}).to_csv(tablas / "respuestas_limpias.csv", index=False)
    pd.DataFrame({"F1": range(100)}).to_csv(tablas / "puntajes_factoriales.csv", index=False)
    pipeline.paso_publicar(c, "sintetico")
    publicadas = {p.name for p in (tmp_path / "results" / "sintetico" / "tablas").iterdir()}
    assert publicadas == {"confiabilidad.csv"}


def test_publicar_frena_tablas_con_demasiadas_filas(cfg, tmp_path):
    c = _cfg_temporal(cfg, tmp_path)
    tablas = c.dir_salidas("sintetico") / "tablas"
    pd.DataFrame({"x": range(pipeline.MAX_FILAS_PUBLICABLES + 1)}).to_csv(
        tablas / "tabla_nueva_sospechosa.csv", index=False
    )
    with pytest.raises(ValueError, match="por persona"):
        pipeline.paso_publicar(c, "sintetico")
