"""Modelo del formulario: predecir el segmento de una persona nueva con el cuestionario corto.

Qué mide y qué no
-----------------
Mide **qué tan bien el cuestionario corto reproduce la asignación de segmento**
que se obtuvo con el cuestionario completo. No mide si alguien "es" o será un
buen emprendedor: no hay ninguna variable de resultado en los datos.

Diseño
------
* Predictores: solo los ítems del formulario corto (respuestas 1–5).
* Validación cruzada estratificada repetida (5 pliegues × 20 repeticiones): con
  n ≈ 120, una sola partición entrenamiento/prueba (como en el original) da
  métricas muy ruidosas.
* Línea base: clase mayoritaria. Un modelo útil tiene que superarla con claridad.
* Se prefiere el modelo interpretable si el potente no lo supera en F1-macro por
  más de ``tolerancia_f1``: un formulario que se explica con coeficientes es más
  defendible ante las personas evaluadas.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

INTERPRETABLES = ("Regresión logística", "Árbol (profundidad 3)")


def candidatos(semilla: int) -> dict[str, object]:
    """Modelos comparados. Hiperparámetros fijos y conservadores (n pequeño: sin búsqueda)."""
    return {
        "Línea base (clase mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "Regresión logística": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)
        ),
        "Árbol (profundidad 3)": DecisionTreeClassifier(max_depth=3, random_state=semilla),
        "Random forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, random_state=semilla, n_jobs=-1
        ),
    }


def evaluar(
    x: pd.DataFrame, y: np.ndarray, splits: int, repeticiones: int, semilla: int
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Validación cruzada estratificada repetida de todos los candidatos.

    Retorna
    -------
    tuple[pandas.DataFrame, dict]
        (métricas por modelo: media y DE de accuracy, F1-macro y accuracy
        balanceada; matriz de confusión acumulada por modelo, normalizada por fila).
    """
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeticiones, random_state=semilla)
    clases = np.unique(y)
    filas, matrices = [], {}
    for nombre, modelo in candidatos(semilla).items():
        acc, f1, bal = [], [], []
        cm = np.zeros((len(clases), len(clases)))
        for tr, te in cv.split(x, y):
            m = clone(modelo).fit(x.iloc[tr], y[tr])
            pred = m.predict(x.iloc[te])
            acc.append(accuracy_score(y[te], pred))
            f1.append(f1_score(y[te], pred, average="macro", zero_division=0))
            bal.append(balanced_accuracy_score(y[te], pred))
            cm += confusion_matrix(y[te], pred, labels=clases)
        filas.append(
            {
                "modelo": nombre,
                "accuracy": np.mean(acc),
                "accuracy_de": np.std(acc),
                "f1_macro": np.mean(f1),
                "f1_macro_de": np.std(f1),
                "accuracy_balanceada": np.mean(bal),
            }
        )
        matrices[nombre] = cm / cm.sum(axis=1, keepdims=True)
    return pd.DataFrame(filas).set_index("modelo"), matrices


def elegir(resumen: pd.DataFrame, tolerancia: float) -> tuple[str, str]:
    """Elige el modelo final y explica por qué."""
    sin_base = resumen.drop(index="Línea base (clase mayoritaria)")
    mejor = sin_base["f1_macro"].idxmax()
    mejor_interp = sin_base.loc[list(INTERPRETABLES), "f1_macro"].idxmax()
    dif = sin_base.loc[mejor, "f1_macro"] - sin_base.loc[mejor_interp, "f1_macro"]
    if mejor == mejor_interp:
        segundo = sin_base.drop(index=mejor)["f1_macro"]
        motivo = (
            f"{mejor} es el modelo con mayor F1-macro ({sin_base.loc[mejor, 'f1_macro']:.2f}) "
            f"y además es interpretable; el siguiente es {segundo.idxmax()} "
            f"({segundo.max():.2f})."
        )
        return mejor, motivo
    if dif <= tolerancia:
        motivo = (
            f"{mejor_interp} (F1-macro {sin_base.loc[mejor_interp, 'f1_macro']:.2f}) queda a "
            f"{dif:.3f} del mejor ({mejor}); la diferencia es menor que la tolerancia "
            f"({tolerancia}), así que se prefiere el modelo interpretable."
        )
        return mejor_interp, motivo
    motivo = (
        f"{mejor} supera al mejor interpretable por {dif:.3f} en F1-macro (> {tolerancia}); "
        "se usa el potente y se explica con importancia por permutación."
    )
    return mejor, motivo


def explicar(modelo, x: pd.DataFrame, y: np.ndarray, semilla: int) -> pd.DataFrame:
    """Explicación global del modelo.

    * Regresión logística: coeficientes sobre ítems estandarizados (cuánto sube
      el log-odds de cada segmento por 1 DE más en el ítem).
    * Otros: importancia por permutación (caída de F1-macro al desordenar el ítem).
    """
    final = modelo[-1] if hasattr(modelo, "steps") else modelo
    if isinstance(final, LogisticRegression):
        coef = final.coef_
        if coef.shape[0] == 1:  # binario: sklearn devuelve un solo vector (clase positiva)
            coef = np.vstack([-coef[0], coef[0]])
        return pd.DataFrame(coef.T, index=x.columns, columns=[f"seg{c}" for c in final.classes_])
    imp = permutation_importance(
        modelo, x, y, scoring="f1_macro", n_repeats=30, random_state=semilla
    )
    return pd.DataFrame(
        {"importancia_media": imp.importances_mean, "importancia_de": imp.importances_std},
        index=x.columns,
    ).sort_values("importancia_media", ascending=False)


def exportar_logistica(modelo) -> dict | None:
    """Parámetros de una regresión logística (con escalado) como diccionario JSON.

    Por qué JSON y no pickle: la demo solo necesita numpy para predecir (no
    scikit-learn), no depende de la versión con la que se entrenó y no carga
    código serializado. Solo contiene parámetros agregados: medias y escalas
    de los ítems, coeficientes e interceptos.

    Retorna ``None`` si el modelo no es una regresión logística con escalado.
    """
    if not (hasattr(modelo, "steps") and isinstance(modelo[-1], LogisticRegression)):
        return None
    escalador, logit = modelo[0], modelo[-1]
    return {
        "media": escalador.mean_.round(6).tolist(),
        "escala": escalador.scale_.round(6).tolist(),
        "coeficientes": logit.coef_.round(6).tolist(),
        "interceptos": logit.intercept_.round(6).tolist(),
        "clases": [int(c) for c in logit.classes_],
    }


def guardar(modelo, ruta: Path, metadatos: dict) -> None:
    """Guarda el modelo y un JSON con lo necesario para usarlo en la demo.

    El JSON registra la versión de scikit-learn: un modelo serializado solo se
    garantiza con la misma versión con la que se entrenó.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)
    meta = {**metadatos, "sklearn": sklearn.__version__}
    ruta.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
