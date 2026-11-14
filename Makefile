# Envoltorio de la CLI `emprende` para Linux/macOS/CI.
# En Windows (sin make) usa directamente: uv run emprende all [--fuente real]
#
#   make all                 # pipeline completo con datos sintéticos
#   make all FUENTE=real     # con los datos reales en data/raw (solo local)

FUENTE ?= sintetico

.PHONY: all instalar analisis reporte sintetico test lint formato privacidad limpiar

# Datos -> psicometría -> segmentación -> modelo -> figuras -> results/ -> reporte HTML
all: analisis

instalar:
	uv sync --locked

analisis:
	uv run emprende all --fuente $(FUENTE)

# Solo re-renderiza el reporte (requiere Quarto en el PATH o QUARTO_BIN).
reporte:
	uv run emprende reporte --fuente $(FUENTE)

# Regenera data/synthetic/ a partir de los datos reales (solo local).
sintetico:
	uv run emprende sintetico

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

formato:
	uv run ruff check --fix .
	uv run ruff format .

# Verifica que ningún archivo versionable contenga valores identificables (solo local).
privacidad:
	uv run python scripts/auditar_privacidad.py

limpiar:
	rm -rf outputs data/processed .pytest_cache .ruff_cache
