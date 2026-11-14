# Perfiles motivacionales de emprendedores (v2, en construcción)

> README provisional. La versión final llega en la Fase 5.
> Diagnóstico del trabajo original: [docs/diagnostico.md](docs/diagnostico.md).

## Cómo reproducir

```bash
uv sync --locked
uv run emprende all            # usa los datos sintéticos versionados
uv run pytest
```

Con los datos reales en `data/raw/` (no versionados):

```bash
uv run emprende all --fuente real
```
