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
| ⏳ | **Reporte publicado** | Workflow `reporte.yml` listo. Requiere: (1) aprobar el paso a `main`; (2) *Settings → Pages → Source: GitHub Actions*. |
| ⏳ | **Demo publicada** | App probada en local (escritorio y móvil). Requiere desplegarla en Streamlit Community Cloud con la cuenta del autor. |
| ⏳ | **Créditos de la docente** | Texto pendiente en `config.yaml` → `creditos.investigacion_origen` (se refleja en el reporte y la demo) y en el README. |
| ⏳ | **Texto del instrumento** | Confirmar la redacción de los ítems (en especial p18) y las anclas de la escala. |
| ⏳ | **Licencia** | Sin licencia por ahora: los datos y el instrumento pertenecen a la investigación de origen; decidir con la docente. |
