import pandas as pd
import pytest

from emprendimiento import datos

# Encabezados con la forma exacta del CSV original (espacios dobles y finales).
ENCABEZADOS = [
    "Ciudad",
    "Actualmente soy",
    "Número de años que lleva funcionando  (Si es estudiante y no tiene empresa, "
    "por favor escribir 0)",
    "Objeto social (negocio principal de la empresa)  (Si es estudiante y no tiene "
    "empresa, por favor escribir NA)",
    "Género",
    "Rango de edad",
    "Estado civil",
    "Máximo nivel educativo terminado ",
    "Pregunta 1 ",
    *[f"Pregunta {i}" for i in range(2, 20)],
]


def _crudo_falso(tmp_path, filas):
    ruta = tmp_path / "crudo.csv"
    lineas = [";".join(ENCABEZADOS)] + [";".join(map(str, f)) for f in filas]
    ruta.write_text("\n".join(lineas), encoding="utf-8-sig")
    return ruta


def _fila(items, edu="Universitaria"):
    return [
        "CiudadX",
        "Emprendedor (a)",
        3,
        "Negocio X",
        "Femenino",
        "Entre 25 y 35 años",
        "Casada (o)",
        edu,
        *items,
    ]


def test_leer_y_limpiar_elimina_identificadores(tmp_path, cfg):
    ruta = _crudo_falso(tmp_path, [_fila([4] * 18 + [3]), _fila([5] * 19, "Maestría")])
    df = datos.limpiar(datos.leer_crudo(ruta), cfg)
    assert "ciudad" not in df.columns
    assert "objeto_social" not in df.columns
    assert list(df.columns[-19:]) == cfg.items
    assert df["respuesta_plana"].tolist() == [False, True]
    assert df["nivel_educativo_grupo"].tolist() == ["Universitaria", "Posgrado"]


def test_falla_con_faltantes(cfg, df_sint):
    roto = df_sint.copy()
    roto.loc[0, "p03"] = None
    with pytest.raises(ValueError, match="faltantes"):
        datos.limpiar(roto, cfg)


def test_falla_fuera_de_escala(cfg, df_sint):
    roto = df_sint.copy()
    roto.loc[0, "p03"] = 7
    with pytest.raises(ValueError, match="fuera de la escala"):
        datos.limpiar(roto, cfg)


def test_invierte_items_configurados(cfg, df_sint):
    from emprendimiento.config import Config

    cfg_inv = Config(datos={**cfg.datos, "items_invertidos": ["p01"]})
    original = datos.limpiar(df_sint, cfg)
    invertido = datos.limpiar(df_sint, cfg_inv)
    assert (invertido["p01"] == 6 - original["p01"]).all()


def test_para_analisis_excluye_planas(cfg, df_sint):
    analisis = datos.para_analisis(df_sint, cfg)
    assert not analisis["respuesta_plana"].any()
    assert len(analisis) == len(df_sint) - df_sint["respuesta_plana"].sum()


def test_fuente_real_ausente_da_mensaje_claro(cfg, tmp_path):
    from emprendimiento.config import Config

    cfg_falso = Config(
        datos={**cfg.datos, "datos": {**cfg["datos"], "crudo": str(tmp_path / "no.csv")}}
    )
    with pytest.raises(FileNotFoundError, match="--fuente sintetico"):
        datos.cargar(cfg_falso, "real")


def test_sintetico_publico_no_tiene_identificadores(cfg):
    crudo = pd.read_csv(cfg.ruta(cfg["datos"]["sintetico"]))
    assert not set(datos.COLUMNAS_IDENTIFICABLES) & set(crudo.columns)
