# Ficha del dataset sintético

> Archivo generado automáticamente por `uv run emprende sintetico`. No editar a mano.

## Qué es
`respuestas_sinteticas.csv` imita la **estructura** de la encuesta real
(123 filas, 19 ítems Likert 1–5 y 6 variables de contexto), pero **no contiene
ninguna respuesta real**. Se publica para que el pipeline, los tests y la CI
puedan correr sin exponer datos personales.

## Cómo se generó
- **Ítems:** cópula gaussiana ordinal. Umbrales por ítem a partir de las
  proporciones marginales reales y matriz de correlaciones **policóricas** reales
  (estimadas sin las respuestas planas). Se muestrea Z ~ N(0, R) y se corta en
  los umbrales.
- **Respuestas planas** (los 19 ítems con el mismo valor): mezcla con
  probabilidad 0.057, igual a la observada.
- **Contexto** (ocupación, años, género, edad, estado civil, educación):
  muestreo **independiente** de cada marginal.
- **No se generan** ciudad ni objeto social.
- Semilla: `20241021` (en `config.yaml`).

## Fidelidad (agregados)
| Métrica | Valor |
|---|---|
| Diferencia máxima en proporciones de respuesta por ítem | 0.073 |
| Diferencia media entre policóricas (real vs sintético) | 0.072 |
| Diferencia máxima entre correlaciones policóricas | 0.252 |

## Privacidad
| Métrica | Valor |
|---|---|
| Filas sintéticas (no planas) idénticas a una real | 0 |
| Hamming mediana al registro real más cercano (de 19) | 7.0 |
| Distancia de Hamming mínima | 2 |

Las variables de contexto se muestrean de forma independiente de los ítems, así
que ni siquiera una fila sintética parecida a una real conserva su demografía.

## Limitaciones (léase antes de interpretar resultados con el sintético)
1. **Sin relación demografía–ítems:** la validación externa y la auditoría de
   sesgo sobre el sintético dan, por construcción, diferencias nulas. Son solo
   una demostración del código.
2. **Sin clusters verdaderos:** una sola normal multivariada no tiene grupos. Los
   segmentos que aparezcan son particiones de un continuo; su estabilidad debería
   ser baja, y eso es una prueba útil del método.
3. Los resultados publicados en el reporte provienen de la corrida local con los
   datos reales, reportados de forma agregada.
