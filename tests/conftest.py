"""Fixtures compartidas. Los tests usan SOLO datos sintéticos o simulados."""

from __future__ import annotations

import pandas as pd
import pytest

from emprendimiento import datos
from emprendimiento.config import Config, cargar_config


@pytest.fixture(scope="session")
def cfg() -> Config:
    return cargar_config()


@pytest.fixture(scope="session")
def df_sint(cfg: Config) -> pd.DataFrame:
    """Datos sintéticos versionados, ya limpios."""
    return datos.cargar(cfg, "sintetico")
