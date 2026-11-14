"""Verifica que ningún archivo versionable contenga valores identificables del crudo.

Qué hace: toma los valores de las columnas de cuasi-identificadores del CSV crudo
(ciudad y objeto social) y busca cada uno, como texto, en todos los archivos que
git versionaría (los no ignorados). Si encuentra alguno, falla con código 1.

Por qué: es la red de seguridad de la regla de privacidad. Se corre antes de cada
commit y en el checklist final. Solo puede correr en local, porque necesita el crudo.

Uso::

    uv run python scripts/auditar_privacidad.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CRUDO = RAIZ / "data" / "raw" / "CSV Datos test.csv"
# Valores muy cortos o genéricos ("NA", "Turismo") darían falsos positivos.
LONGITUD_MINIMA = 12
GENERICOS = {"na", "n/a", "estudiante", "ninguno", "no aplica"}


def valores_sensibles() -> set[str]:
    """Devuelve los textos de ciudad y objeto social que no deben aparecer en el repo."""
    d = pd.read_csv(CRUDO, sep=";", encoding="utf-8-sig")
    d.columns = d.columns.str.strip()
    cols = [c for c in d.columns if c.startswith(("Ciudad", "Objeto social"))]
    vals: set[str] = set()
    for c in cols:
        for v in d[c].dropna().astype(str):
            v = " ".join(v.split())
            if v.lower() in GENERICOS:
                continue
            # Las ciudades se revisan siempre, aunque sean cortas: son la llave principal.
            if c.startswith("Ciudad") or len(v) >= LONGITUD_MINIMA:
                vals.add(v)
    return vals


def archivos_versionables() -> list[Path]:
    """Lista los archivos rastreados o no ignorados por git."""
    salida = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=RAIZ, capture_output=True, text=True, check=True,
    ).stdout
    return [RAIZ / p for p in salida.splitlines() if p]


def main() -> int:
    vals = valores_sensibles()
    hallazgos: list[tuple[str, int]] = []
    for ruta in archivos_versionables():
        if ruta.suffix.lower() in {".png", ".jpg", ".joblib", ".pkl"}:
            continue
        try:
            texto = " ".join(ruta.read_text(encoding="utf-8", errors="ignore").split())
        except OSError:
            continue
        texto_l = texto.lower()
        n = sum(1 for v in vals if f"{v.lower()}" in texto_l and _como_palabra(v.lower(), texto_l))
        if n:
            hallazgos.append((str(ruta.relative_to(RAIZ)), n))
    if hallazgos:
        print("FALLA: valores identificables encontrados (archivo, n valores):")
        for h in hallazgos:
            print("  ", h)
        return 1
    print(f"OK: {len(vals)} valores sensibles revisados en {len(archivos_versionables())} archivos.")
    return 0


def _como_palabra(v: str, texto: str) -> bool:
    """Evita falsos positivos de subcadenas (p. ej. una ciudad dentro de otra palabra)."""
    import re

    return re.search(rf"(?<![\w]){re.escape(v)}(?![\w])", texto) is not None


if __name__ == "__main__":
    sys.exit(main())
