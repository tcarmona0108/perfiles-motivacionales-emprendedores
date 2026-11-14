# Checklist final

Verificado el 2026-11-14 sobre la rama `mejora/v2`.

| | Ítem | Evidencia |
|---|---|---|
| ✅ | **`emprende all` corre desde cero en un entorno limpio** | (1) GitHub Actions, Ubuntu, modo rápido, con datos sintéticos y render del reporte. (2) Clon nuevo desde GitHub en Windows: `uv sync --locked`, pipeline **completo** y reporte en 17,5 min. El árbol de trabajo queda limpio después de correrlo. |
| ✅ | **Tests en verde** | 51 tests (local y CI). Incluyen recuperación de estructura conocida (2 factores, 3 grupos), equivalencia numpy ↔ scikit-learn del modelo de la demo y la frontera de privacidad de `publicar`. |
| ✅ | **Lint y formato** | `ruff check` y `ruff format --check` en la CI. |
| ✅ | **Reproducibilidad exacta** | Dos corridas completas con los datos reales producen las ~60 tablas idénticas byte a byte. El dataset sintético también se regenera byte a byte. |
| ✅ | **Ningún dato sensible en el repo** | `scripts/auditar_privacidad.py`: 140 valores identificables (ciudades, objetos sociales) buscados en los 108 archivos versionados, sin hallazgos. Control positivo: detecta 39 en el notebook original. |
| ✅ | **Ningún dato sensible en el historial** | Las 133 versiones de archivos de todos los commits auditadas antes del primer push, sin hallazgos. `data/raw`, `data/processed` y `outputs/` nunca se versionaron. |
| ✅ | **Repositorio público con CI** | https://github.com/tcarmona0108/perfiles-motivacionales-emprendedores |
| ✅ | **Reporte publicado** | https://tcarmona0108.github.io/perfiles-motivacionales-emprendedores/ (se republica en cada push a `main`; solo `main` puede desplegar). |
| ✅ | **Demo publicada** | https://perfiles-motivacionales-emprendedores-ghv2zpyfalwitmfntdbmhh.streamlit.app/ (Streamlit Community Cloud, rama `main`). |
| ✅ | **Créditos de la docente** | Profesora Flor Angela Marulanda, en `config.yaml` (reporte y demo) y en el README. Afiliación y título de la investigación, si se desean, se agregan en el mismo campo. |
| ⏳ | **Texto del instrumento** | Confirmar la redacción de los ítems (en especial p18) y las anclas de la escala. |
| ⏳ | **Licencia** | Sin licencia por ahora: los datos y el instrumento pertenecen a la investigación de origen; decidir con la docente. |
