"""Orquestación del pipeline: un paso = una función que lee y escribe en disco.

Cada paso lee lo que dejó el anterior en ``outputs/<fuente>/`` y escribe sus
tablas, figuras y modelos. Así cualquier paso se puede re-ejecutar solo, y el
reporte Quarto lee archivos en lugar de recalcular.

Segmentaciones
--------------
* ``factores`` (principal, fijada antes de ver los datos): puntajes factoriales
  estandarizados. Captura nivel + forma.
* ``forma`` (sensibilidad planificada en el diagnóstico): puntajes centrados por
  persona; captura solo qué motiva más a cada quien.
* ``pca``: componentes principales, solo como referencia frente al original
  (se evalúa, pero no se perfila).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import datos, modelo, perfiles, psicometria, reduccion, segmentacion, sesgo, sintetico
from . import visualizacion as viz
from .config import Config, Fuente
from .correlaciones import matriz_policorica

SEGMENTACIONES = ("factores", "forma")


# ═══════════════════════════ utilidades de E/S ═══════════════════════════════
def _dir(cfg: Config, fuente: Fuente, sub: str) -> Path:
    return cfg.dir_salidas(fuente) / sub


def _escribir(
    df: pd.DataFrame, cfg: Config, fuente: Fuente, nombre: str, index: bool = True
) -> None:
    df.to_csv(_dir(cfg, fuente, "tablas") / f"{nombre}.csv", index=index, lineterminator="\n")


def _leer(cfg: Config, fuente: Fuente, nombre: str, **kw) -> pd.DataFrame:
    return pd.read_csv(_dir(cfg, fuente, "tablas") / f"{nombre}.csv", **kw)


def _json(cfg: Config, fuente: Fuente, nombre: str, obj: dict | None = None) -> dict:
    ruta = _dir(cfg, fuente, "tablas") / f"{nombre}.json"
    if obj is not None:
        ruta.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8",
            newline="\n",
        )
        return obj
    return json.loads(ruta.read_text(encoding="utf-8"))


def _figura(fig, cfg: Config, fuente: Fuente, nombre: str) -> None:
    viz.guardar(fig, _dir(cfg, fuente, "figuras") / f"{nombre}.png")


def _muestra(cfg: Config, fuente: Fuente) -> pd.DataFrame:
    return datos.para_analisis(_leer(cfg, fuente, "respuestas_limpias"), cfg)


# ═══════════════════════════ Paso 0: datos ═══════════════════════════════════
def paso_datos(cfg: Config, fuente: Fuente) -> None:
    """Carga, limpia y guarda la tabla analítica (todas las filas, con marca de planas)."""
    df = datos.cargar(cfg, fuente)
    _escribir(df, cfg, fuente, "respuestas_limpias", index=False)
    resumen = pd.Series(
        {
            "n_total": len(df),
            "n_respuesta_plana": int(df["respuesta_plana"].sum()),
            "n_analisis": len(datos.para_analisis(df, cfg)),
            "n_items": len(cfg.items),
        },
        name="valor",
    )
    _escribir(resumen.to_frame(), cfg, fuente, "resumen_muestra")


# ═══════════════════════════ Paso 1: descriptivos ════════════════════════════
def paso_descriptivos(cfg: Config, fuente: Fuente) -> None:
    """Distribución, techo/piso y discriminación por ítem."""
    df = _muestra(cfg, fuente)
    desc = psicometria.descriptivos_items(df, cfg.items, cfg.etiquetas)
    _escribir(desc.round(3), cfg, fuente, "descriptivos_items")


# ═══════════════════════════ Pasos 2–3: psicometría ══════════════════════════
def nombrar_factores(sol: psicometria.ResultadoEFA, anclas: dict[str, str]) -> dict[str, str]:
    """Asigna nombres a F1..Fm según el factor en que más carga cada ítem ancla.

    Si dos nombres apuntan al mismo factor (p. ej. otra estructura en el
    sintético), solo se nombra el primero y el resto queda como "Factor k".
    """
    mapa: dict[str, str] = {}
    for nombre, ancla in anclas.items():
        if ancla in sol.cargas.index:
            f = sol.cargas.loc[ancla].abs().idxmax()
            mapa.setdefault(f, nombre)
    return {f: mapa.get(f, f"Factor {f[1:]}") for f in sol.cargas.columns}


def _renombrar(sol: psicometria.ResultadoEFA, nombres: dict[str, str]) -> psicometria.ResultadoEFA:
    return psicometria.ResultadoEFA(
        cargas=sol.cargas.rename(columns=nombres),
        phi=sol.phi.rename(index=nombres, columns=nombres),
        comunalidades=sol.comunalidades,
    )


def paso_psicometria(cfg: Config, fuente: Fuente) -> None:
    """Idoneidad, número de factores, EFA, confiabilidad, estabilidad y puntajes."""
    p = cfg["psicometria"]
    df = _muestra(cfg, fuente)
    x = df[cfg.items]
    r = matriz_policorica(x, p["correccion_continuidad"])
    _escribir(r.round(4), cfg, fuente, "correlacion_policorica")

    # Idoneidad
    kmo_items, kmo_total = psicometria.kmo(r.to_numpy())
    chi, gl, pval = psicometria.bartlett(r.to_numpy(), len(x))
    _escribir(pd.DataFrame({"kmo": kmo_items}, index=cfg.items).round(3), cfg, fuente, "kmo_items")

    # Número de factores
    paralelo, n_crit = psicometria.analisis_paralelo(
        x, r.to_numpy(), p["n_iter_paralelo"], p["percentil_paralelo"], cfg.semilla + 10
    )
    n_crit["map_velicer"] = psicometria.map_velicer(r.to_numpy())[0]
    n_crit["paralelo_pearson_clasico"] = psicometria.paralelo_pearson_clasico(
        x, p["n_iter_paralelo"], p["percentil_paralelo"], cfg.semilla + 11
    )
    m = n_crit["paralelo_policorico"] if p["n_factores"] == "paralelo" else int(p["n_factores"])
    m = max(m, 1)
    _escribir(paralelo.round(4), cfg, fuente, "analisis_paralelo")

    # EFA principal
    sol = psicometria.efa(r, m, p["metodo_extraccion"], p["rotacion"])
    nombres = nombrar_factores(sol, cfg["factores"])
    sol = _renombrar(sol, nombres)
    asig = psicometria.asignar_items(sol.cargas, p["carga_minima"], p["diferencia_cruzada"])
    asig.insert(0, "etiqueta", [cfg.etiquetas[i] for i in asig.index])
    _escribir(pd.concat([sol.cargas, sol.comunalidades], axis=1).round(3), cfg, fuente, "cargas")
    _escribir(sol.phi.round(3), cfg, fuente, "correlacion_factores")
    _escribir(asig.round(3), cfg, fuente, "asignacion_items")

    # Confiabilidad + ω jerárquico
    conf = psicometria.confiabilidad(x, r, asig["factor"])
    omega_h = psicometria.omega_jerarquico(r.to_numpy(), sol)
    estab, ic = psicometria.bootstrap_efa(
        x,
        sol,
        asig["factor"],
        p["n_bootstrap"],
        cfg.semilla + 12,
        p["metodo_extraccion"],
        p["rotacion"],
    )
    # IC 95 % bootstrap: de ω para cada factor y de α para la escala total.
    conf["ic95_inf"] = ic["ic_inf"]
    conf["ic95_sup"] = ic["ic_sup"]
    conf["ic95_de"] = "omega"
    conf.loc["Total", ["ic95_inf", "ic95_sup", "ic95_de"]] = [
        ic.loc["Total_alfa", "ic_inf"],
        ic.loc["Total_alfa", "ic_sup"],
        "alfa",
    ]
    _escribir(conf.round(3), cfg, fuente, "confiabilidad")
    _escribir(estab.round(3), cfg, fuente, "estabilidad_factores")

    # Robustez: misma EFA con Pearson
    sol_p = psicometria.efa(x.corr(), m, p["metodo_extraccion"], p["rotacion"])
    alin = psicometria.alinear(sol.cargas.to_numpy(), sol_p.cargas.to_numpy())
    congr = np.diag(psicometria.congruencia_tucker(sol.cargas.to_numpy(), alin))

    # Soluciones alternativas (tabla de sensibilidad del número de factores)
    filas = []
    for m_alt in p["m_alternativos"]:
        s_alt = psicometria.efa(r, m_alt, p["metodo_extraccion"], p["rotacion"])
        a_alt = psicometria.asignar_items(s_alt.cargas, p["carga_minima"], p["diferencia_cruzada"])
        e_alt, _ = psicometria.bootstrap_efa(
            x,
            s_alt,
            a_alt["factor"],
            p["n_bootstrap_alternativos"],
            cfg.semilla + 13,
            p["metodo_extraccion"],
            p["rotacion"],
        )
        por_factor = (
            a_alt[a_alt["limpio"]]
            .groupby("factor")
            .size()
            .reindex(s_alt.cargas.columns, fill_value=0)
        )
        filas.append(
            {
                "n_factores": m_alt,
                "items_limpios": int(a_alt["limpio"].sum()),
                "min_items_limpios_por_factor": int(por_factor.min()),
                "heywood": bool((s_alt.comunalidades >= 0.995).any()),
                "congruencia_min": float(e_alt["congruencia_media"].min()),
                "pct_replicas_estables_min": float(e_alt["pct_replicas_ge_085"].min()),
                "elegida": m_alt == m,
            }
        )
    _escribir(
        pd.DataFrame(filas).set_index("n_factores").round(3), cfg, fuente, "alternativas_n_factores"
    )

    # Puntajes factoriales (por persona: SOLO en outputs/, nunca versionados)
    puntuador = psicometria.Puntuador.ajustar(x, r, sol)
    joblib.dump(puntuador, _dir(cfg, fuente, "modelos") / "puntuador.joblib")
    puntajes = puntuador.transformar(x)
    _escribir(puntajes.round(5), cfg, fuente, "puntajes_factoriales", index=False)

    _json(
        cfg,
        fuente,
        "psicometria",
        {
            "n": len(x),
            "kmo_total": kmo_total,
            "kmo_min_item": float(kmo_items.min()),
            "bartlett_chi2": chi,
            "bartlett_gl": gl,
            "bartlett_p": pval,
            "matriz_suavizada": bool(r.attrs["suavizada"]),
            "n_factores_por_criterio": n_crit,
            "n_factores_elegido": m,
            "nombres_factores": list(sol.cargas.columns),
            "congruencia_vs_pearson": dict(zip(sol.cargas.columns, map(float, congr), strict=True)),
            **omega_h,
        },
    )


def _cargar_psico(cfg: Config, fuente: Fuente):
    asig = _leer(cfg, fuente, "asignacion_items", index_col=0)
    puntajes = _leer(cfg, fuente, "puntajes_factoriales")
    return asig, puntajes


# ═══════════════════════════ Paso 4: reducción ═══════════════════════════════
def paso_reduccion(cfg: Config, fuente: Fuente) -> None:
    """Formulario corto y su comparación con el cuestionario completo."""
    k = cfg["reduccion"]["items_por_factor"]
    x = _muestra(cfg, fuente)[cfg.items]
    asig, puntajes = _cargar_psico(cfg, fuente)
    r = _leer(cfg, fuente, "correlacion_policorica", index_col=0)
    corto, _ = reduccion.seleccionar_items(asig, k)
    comp = reduccion.comparar_corto_largo(x, r, asig["factor"], corto, puntajes)
    curva = reduccion.curva_items(x, asig, puntajes)

    # Puntaje total corto vs largo (la escala completa como un solo índice)
    items_corto = [i for its in corto.values() for i in its]
    r_total = float(np.corrcoef(x[items_corto].mean(axis=1), x.mean(axis=1))[0, 1])

    _escribir(comp.round(3), cfg, fuente, "corto_vs_largo")
    _escribir(curva.round(3), cfg, fuente, "curva_items", index=False)
    _json(
        cfg,
        fuente,
        "formulario_corto",
        {
            "items_por_factor": corto,
            "n_items": len(items_corto),
            "factores_incompletos": [f for f, its in corto.items() if len(its) < k],
            "r_total_corto_largo": r_total,
        },
    )


# ═══════════════════════════ Paso 6: segmentación ════════════════════════════
def _representaciones(cfg: Config, fuente: Fuente) -> dict[str, np.ndarray]:
    x = _muestra(cfg, fuente)[cfg.items]
    _, puntajes = _cargar_psico(cfg, fuente)
    z = StandardScaler().fit_transform(puntajes)
    pca = PCA(n_components=puntajes.shape[1]).fit_transform(StandardScaler().fit_transform(x))
    return {
        "factores": z,
        "forma": segmentacion.representacion_forma(z),
        "pca": StandardScaler().fit_transform(pca),
    }


def paso_segmentacion(cfg: Config, fuente: Fuente) -> None:
    """Compara algoritmos y k en cada representación y elige con la regla del módulo."""
    cs = cfg["segmentacion"]
    reps = _representaciones(cfg, fuente)
    elecciones, etiquetas = {}, {}
    for nombre, z in reps.items():
        tabla = segmentacion.evaluar(z, cs, cfg.semilla + 20)
        _escribir(tabla.round(4), cfg, fuente, f"evaluacion_clusters_{nombre}", index=False)
        fila, motivo = segmentacion.elegir(tabla, cs)
        etq, _ = segmentacion.ajustar(
            fila.algoritmo, int(fila.k), z, cfg.semilla + 20, cs["n_init"]
        )
        etq = segmentacion.ordenar_por_nivel(etq, z if nombre != "forma" else reps["factores"])
        nula = segmentacion.prueba_sin_grupos(
            z,
            fila.algoritmo,
            int(fila.k),
            fila.silueta,
            cs["n_sim_sin_grupos"],
            cfg.semilla + 21,
            cs["n_init"],
        )
        elecciones[nombre] = {
            "algoritmo": fila.algoritmo,
            "k": int(fila.k),
            "silueta": float(fila.silueta),
            "ari_medio": float(fila.ari_medio),
            "estable": bool(fila.ari_medio >= cs["ari_estable"]),
            "motivo": motivo,
            "tamanos": np.bincount(etq)[1:].tolist(),
            "prueba_sin_grupos": nula,
            "hdbscan": segmentacion.estructura_hdbscan(z, cs["min_cluster_hdbscan"]),
        }
        etiquetas[nombre] = etq
    _json(cfg, fuente, "segmentacion", elecciones)
    _escribir(
        pd.DataFrame({k: v for k, v in etiquetas.items() if k in SEGMENTACIONES}),
        cfg,
        fuente,
        "segmentos",
        index=False,
    )


# ═══════════════════════════ Pasos 7–8: perfiles ═════════════════════════════
def paso_perfiles(cfg: Config, fuente: Fuente) -> None:
    """Perfil de cada segmento, ítems que más diferencian y validación externa."""
    df = _muestra(cfg, fuente)
    x = df[cfg.items]
    asig, puntajes = _cargar_psico(cfg, fuente)
    factores = list(puntajes.columns)
    z = pd.DataFrame(StandardScaler().fit_transform(puntajes), columns=factores)
    grupos = {f: asig.index[asig["factor"] == f].tolist() for f in factores}
    medias = reduccion.puntajes_media(x, grupos)
    segs = _leer(cfg, fuente, "segmentos")
    rng = cfg.rng(30)
    n_perm = cfg["perfiles"]["n_permutaciones"]
    nombres_todos = {}
    for s in SEGMENTACIONES:
        etq = segs[s].to_numpy()
        res = perfiles.resumen_segmentos(z, medias, etq)
        nombres = perfiles.nombrar_segmentos(res, factores)
        res.insert(0, "nombre", pd.Series(nombres))
        nombres_todos[s] = nombres
        _escribir(res.round(3), cfg, fuente, f"perfiles_{s}")
        dif = perfiles.items_que_diferencian(x, etq, cfg.etiquetas, cfg["perfiles"]["correccion"])
        _escribir(dif.round(4), cfg, fuente, f"items_diferencian_{s}")
        val = perfiles.validacion_externa(df, etq, n_perm, rng)
        _escribir(val.round(4), cfg, fuente, f"validacion_externa_{s}", index=False)
        # Nivel general (media de los 19 ítems): ¿los segmentos se ordenan solo por intensidad?
        nivel = x.mean(axis=1)
        res_nivel = nivel.groupby(etq).agg(["mean", "std"]).round(3)
        _escribir(res_nivel, cfg, fuente, f"nivel_general_{s}")
    _json(
        cfg,
        fuente,
        "nombres_segmentos",
        {s: {str(k): v for k, v in n.items()} for s, n in nombres_todos.items()},
    )


# ═══════════════════════════ Paso 9: modelo ══════════════════════════════════
def paso_modelo(cfg: Config, fuente: Fuente) -> None:
    """Clasificador formulario corto → segmento, para cada segmentación."""
    cm = cfg["modelo"]
    x = _muestra(cfg, fuente)[cfg.items]
    corto = _json(cfg, fuente, "formulario_corto")
    items = [i for its in corto["items_por_factor"].values() for i in its]
    segs = _leer(cfg, fuente, "segmentos")
    nombres = _json(cfg, fuente, "nombres_segmentos")
    resumen_global = {}
    for s in SEGMENTACIONES:
        y = segs[s].to_numpy()
        res, matrices = modelo.evaluar(
            x[items], y, cm["cv_splits"], cm["cv_repeticiones"], cfg.semilla + 40
        )
        elegido, motivo = modelo.elegir(res, cm["tolerancia_f1"])
        _escribir(res.round(4), cfg, fuente, f"modelos_{s}")
        final = modelo.candidatos(cfg.semilla + 40)[elegido].fit(x[items], y)
        expl = modelo.explicar(final, x[items], y, cfg.semilla + 41)
        expl.insert(0, "etiqueta", [cfg.etiquetas[i] for i in expl.index])
        _escribir(expl.round(4), cfg, fuente, f"explicacion_modelo_{s}")
        clases = sorted(np.unique(y).tolist())
        f1 = res.loc[elegido, "f1_macro"]
        base = res.loc["Línea base (clase mayoritaria)", "f1_macro"]
        _escribir(
            pd.DataFrame(matrices[elegido], index=clases, columns=clases).round(4),
            cfg,
            fuente,
            f"confusion_{s}",
        )
        # Metadatos para la demo: nada a nivel de persona, solo agregados.
        perf = _leer(cfg, fuente, f"perfiles_{s}", index_col=0)
        grupos_corto = corto["items_por_factor"]
        medias_corto = reduccion.puntajes_media(x, grupos_corto)
        modelo.guardar(
            final,
            _dir(cfg, fuente, "modelos") / f"clasificador_{s}.joblib",
            {
                "segmentacion": s,
                "modelo": elegido,
                "motivo": motivo,
                "items": items,
                "etiquetas_items": {i: cfg.etiquetas[i] for i in items},
                "items_por_factor": grupos_corto,
                "nombres_segmentos": nombres[s],
                "tamanos_segmentos": perf["n"].astype(int).to_dict(),
                "media_corta_por_segmento": medias_corto.groupby(y)
                .mean()
                .round(3)
                .to_dict(orient="index"),
                "media_corta_global": medias_corto.mean().round(3).to_dict(),
                "f1_macro_cv": float(f1),
                "accuracy_cv": float(res.loc[elegido, "accuracy"]),
                "f1_macro_linea_base": float(base),
            },
        )
        resumen_global[s] = {
            "modelo": elegido,
            "motivo": motivo,
            "f1_macro": float(f1),
            "accuracy": float(res.loc[elegido, "accuracy"]),
            "f1_base": float(base),
        }
    _json(cfg, fuente, "modelo", resumen_global)


# ═══════════════════════════ Paso 10: sesgo ══════════════════════════════════
def paso_sesgo(cfg: Config, fuente: Fuente) -> None:
    """Distribución de segmentos por atributos sensibles (no usados en ningún modelo)."""
    df = _muestra(cfg, fuente)
    segs = _leer(cfg, fuente, "segmentos")
    for s in SEGMENTACIONES:
        res, tablas = sesgo.auditar(
            df,
            segs[s].to_numpy(),
            cfg["atributos_sensibles"],
            cfg["sesgo"]["n_minimo_celda"],
            cfg["perfiles"]["n_permutaciones"],
            cfg.rng(50),
        )
        _escribir(res.round(4), cfg, fuente, f"sesgo_{s}", index=False)
        for at, t in tablas.items():
            _escribir(t, cfg, fuente, f"sesgo_{s}_{at}")


# ═══════════════════════════ Figuras ═════════════════════════════════════════
def paso_figuras(cfg: Config, fuente: Fuente) -> None:
    """Dibuja todas las figuras a partir de las tablas guardadas.

    Separado del cálculo a propósito: cambiar un estilo no obliga a repetir
    bootstraps ni validaciones cruzadas, y garantiza que cada figura muestra
    exactamente lo que está en las tablas del reporte.
    """
    p, cs = cfg["psicometria"], cfg["segmentacion"]
    ps = _json(cfg, fuente, "psicometria")
    desc = _leer(cfg, fuente, "descriptivos_items", index_col=0)
    _figura(viz.distribucion_items(desc), cfg, fuente, "01_distribucion_items")

    paralelo = _leer(cfg, fuente, "analisis_paralelo", index_col=0)
    col_umbral = f"nulo_p{p['percentil_paralelo']:g}"
    m = ps["n_factores_elegido"]
    _figura(viz.analisis_paralelo(paralelo, m, col_umbral), cfg, fuente, "02_analisis_paralelo")

    cargas = _leer(cfg, fuente, "cargas", index_col=0)[ps["nombres_factores"]]
    asig, _ = _cargar_psico(cfg, fuente)
    _figura(viz.cargas(cargas, cfg.etiquetas, asig), cfg, fuente, "03_cargas")

    curva = _leer(cfg, fuente, "curva_items")
    k_corto = cfg["reduccion"]["items_por_factor"]
    _figura(viz.curva_items(curva, k_corto), cfg, fuente, "04_curva_items")

    sg = _json(cfg, fuente, "segmentacion")
    for nombre, el in sg.items():
        tabla = _leer(cfg, fuente, f"evaluacion_clusters_{nombre}")
        validas = tabla[tabla["tamano_min"] >= cs["tamano_minimo_segmento"]]
        n_estables = int((validas["ari_medio"] >= cs["ari_estable"]).sum())
        alg = segmentacion.NOMBRES[el["algoritmo"]]
        if not el["estable"]:
            titulo = "Ninguna segmentación es estable: la estructura de grupos es débil"
        elif n_estables == 1:
            titulo = f"Solo {alg} con k={el['k']} es estable y sin segmentos diminutos"
        else:
            titulo = f"{n_estables} opciones estables; {alg} con k={el['k']} separa mejor"
        _figura(
            viz.seleccion_k(
                tabla, segmentacion.NOMBRES, cs["ari_estable"], cs["tamano_minimo_segmento"], titulo
            ),
            cfg,
            fuente,
            f"05_seleccion_k_{nombre}",
        )

    nombres = _json(cfg, fuente, "nombres_segmentos")
    md = _json(cfg, fuente, "modelo")
    fc = _json(cfg, fuente, "formulario_corto")
    factores = ps["nombres_factores"]
    for s in SEGMENTACIONES:
        res = _leer(cfg, fuente, f"perfiles_{s}", index_col=0)
        zs = res[[f"z_{f}" for f in factores]]
        solo_nivel = all(len(set(np.sign(fila))) == 1 for _, fila in zs.iterrows())
        titulo = (
            "Los segmentos difieren en intensidad, no en tipo de motivación"
            if solo_nivel
            else "Los segmentos difieren en qué motiva más, no solo en cuánto"
        )
        nom = {int(k): v for k, v in nombres[s].items()}
        _figura(viz.perfiles(res, factores, nom, titulo), cfg, fuente, f"06_perfiles_{s}")

        cm_df = _leer(cfg, fuente, f"confusion_{s}", index_col=0)
        f1, base = md[s]["f1_macro"], md[s]["f1_base"]
        _figura(
            viz.confusion(
                cm_df.to_numpy(),
                list(cm_df.index),
                md[s]["modelo"],
                f"{fc['n_items']} preguntas reproducen el segmento: F1-macro {f1:.2f} "
                f"(base {base:.2f})",
            ),
            cfg,
            fuente,
            f"07_confusion_{s}",
        )


# ═══════════════════════════ Resumen para el punto de control ════════════════
ORIGINAL = {
    "Matriz de correlación": "Pearson",
    "Extracción / rotación": "Componentes principales + varimax",
    "N.º de factores": "5 (Kaiser)",
    "Ítems del cuestionario": "19 (sin versión corta; la CFA usó 10)",
    "Confiabilidad": "α = 0.911 (escala completa)",
    "Segmentación": "K-means k=5 sin criterio",
    "Silueta": "0.30 (calculada sobre datos distintos a los del modelo)",
    "Estabilidad": "No evaluada",
    "Clasificador": "LogReg 97 % accuracy (etiqueta circular)",
}


def paso_resumen(cfg: Config, fuente: Fuente) -> None:
    """Tabla resumen 'antes vs después' de los resultados clave."""
    ps = _json(cfg, fuente, "psicometria")
    fc = _json(cfg, fuente, "formulario_corto")
    sg = _json(cfg, fuente, "segmentacion")
    md = _json(cfg, fuente, "modelo")
    conf = _leer(cfg, fuente, "confiabilidad", index_col=0)
    comp = _leer(cfg, fuente, "corto_vs_largo", index_col=0)
    nombres = _json(cfg, fuente, "nombres_segmentos")
    crit = ps["n_factores_por_criterio"]
    f = sg["factores"]
    fo = sg["forma"]
    nuevo = {
        "Matriz de correlación": "Policórica (ítems ordinales con efecto techo)",
        "Extracción / rotación": "Factores comunes (minres) + oblimin",
        "N.º de factores": (
            f"{ps['n_factores_elegido']} (PA policórico); Kaiser={crit['kaiser']}, "
            f"PA reducido={crit['paralelo_policorico_reducido']}, MAP={crit['map_velicer']}"
        ),
        "Ítems del cuestionario": (
            f"{fc['n_items']} (r total corto–largo = {fc['r_total_corto_largo']:.2f})"
        ),
        "Confiabilidad": "; ".join(f"{i}: ω={conf.loc[i, 'omega']:.2f}" for i in conf.index)
        + f"; ω_h={ps['omega_jerarquico']:.2f}",
        "Segmentación": (
            f"Principal: {segmentacion.NOMBRES[f['algoritmo']]} k={f['k']} {f['tamanos']}; "
            f"forma: {segmentacion.NOMBRES[fo['algoritmo']]} k={fo['k']} {fo['tamanos']}"
        ),
        "Silueta": (
            f"{f['silueta']:.2f} (sin grupos: {f['prueba_sin_grupos']['silueta_nula_media']:.2f}, "
            f"p={f['prueba_sin_grupos']['p_valor']:.2f})"
        ),
        "Estabilidad": f"ARI principal {f['ari_medio']:.2f}; forma {fo['ari_medio']:.2f}",
        "Clasificador": "; ".join(
            f"{s}: {v['modelo']} F1-macro={v['f1_macro']:.2f} (base {v['f1_base']:.2f})"
            for s, v in md.items()
        ),
    }
    t = pd.DataFrame({"Original": ORIGINAL, "v2": nuevo})
    t.index.name = "Aspecto"
    _escribir(t, cfg, fuente, "resumen_antes_despues")
    _json(
        cfg,
        fuente,
        "resumen",
        {"nombres_segmentos": nombres, "corto_vs_largo": comp.round(3).to_dict(orient="index")},
    )


# ═══════════════════════════ Publicación de resultados agregados ═════════════
# Tablas a nivel de persona: nunca salen de outputs/ (ignorado por git).
NO_PUBLICABLES = {"respuestas_limpias.csv", "puntajes_factoriales.csv", "segmentos.csv"}
# Red de seguridad: ninguna tabla agregada tiene tantas filas (la mayor tiene 20).
# Si una la supera, probablemente es una tabla por persona que se coló.
MAX_FILAS_PUBLICABLES = 40


def paso_publicar(cfg: Config, fuente: Fuente) -> None:
    """Copia a ``results/<fuente>/`` solo resultados agregados (tablas y figuras).

    ``results/real/`` se versiona: es lo que alimenta el reporte público y la
    demo. Por eso se excluyen explícitamente las tablas por persona y se
    verifica el tamaño de cada tabla antes de copiarla.
    """
    import shutil

    origen = cfg.dir_salidas(fuente)
    destino = cfg.ruta("results") / fuente
    if destino.exists():
        shutil.rmtree(destino)
    (destino / "tablas").mkdir(parents=True)
    (destino / "figuras").mkdir(parents=True)
    for archivo in sorted((origen / "tablas").iterdir()):
        if archivo.name in NO_PUBLICABLES:
            continue
        if archivo.suffix == ".csv":
            filas = len(pd.read_csv(archivo))
            if filas > MAX_FILAS_PUBLICABLES:
                raise ValueError(
                    f"{archivo.name} tiene {filas} filas: ¿datos por persona? No se publica."
                )
        shutil.copy2(archivo, destino / "tablas" / archivo.name)
    for archivo in sorted((origen / "figuras").glob("*.png")):
        shutil.copy2(archivo, destino / "figuras" / archivo.name)


def paso_reporte(cfg: Config, fuente: Fuente) -> None:
    """Renderiza el reporte Quarto (``reports/``) a HTML con los resultados de la fuente.

    Quarto se busca en la variable de entorno ``QUARTO_BIN`` o en el PATH.
    ``QUARTO_PYTHON`` apunta al intérprete actual: en Windows, Quarto usaría
    por defecto el lanzador ``py`` (otro Python, sin las dependencias del proyecto).
    """
    import os
    import shutil
    import subprocess
    import sys

    quarto = os.environ.get("QUARTO_BIN") or shutil.which("quarto")
    if not quarto:
        raise RuntimeError(
            "No se encontró Quarto. Instálalo desde https://quarto.org o define QUARTO_BIN "
            "con la ruta al ejecutable."
        )
    # La fuente se pasa por variable de entorno: `quarto render -P` exigiría papermill.
    entorno = {**os.environ, "QUARTO_PYTHON": sys.executable, "EMPRENDE_FUENTE": fuente}
    subprocess.run([quarto, "render", str(cfg.ruta("reports"))], check=True, env=entorno)


# ═══════════════════════════ Sintético ═══════════════════════════════════════
def generar_sintetico(cfg: Config) -> None:
    """Ajusta la cópula sobre los datos reales y escribe el CSV sintético y su ficha."""
    real = datos.cargar(cfg, "real")
    mod = sintetico.ajustar(real, cfg)
    n = cfg["sintetico"]["n"] or len(real)
    sint = sintetico.muestrear(mod, n, cfg.rng(desplazamiento=1))
    ruta = cfg.ruta(cfg["datos"]["sintetico"])
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # LF explícito: el CSV versionado debe tener el mismo hash en Windows y Linux.
    sint.to_csv(ruta, index=False, lineterminator="\n")
    metricas = sintetico.evaluar_fidelidad(real, datos.limpiar(sint, cfg), cfg)
    sintetico.escribir_ficha(
        cfg.ruta(cfg["datos"]["ficha_sintetico"]), metricas, mod, n, cfg.semilla
    )
    print(f"Sintético escrito en {ruta.relative_to(cfg.raiz)} ({n} filas)")
    for k, v in metricas.items():
        print(f"  {k}: {v}")


# Orden de ejecución de `emprende all`.
PASOS: dict[str, Callable[[Config, Fuente], object]] = {
    "datos": paso_datos,
    "descriptivos": paso_descriptivos,
    "psicometria": paso_psicometria,
    "reduccion": paso_reduccion,
    "segmentacion": paso_segmentacion,
    "perfiles": paso_perfiles,
    "modelo": paso_modelo,
    "sesgo": paso_sesgo,
    "figuras": paso_figuras,
    "resumen": paso_resumen,
    "publicar": paso_publicar,
    "reporte": paso_reporte,
}
