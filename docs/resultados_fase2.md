# Fase 2 — Resultados clave (punto de control)

> Corrida oficial: `uv run emprende all --fuente real` (semilla 20241021), datos
> reales en local, n = 116 (123 menos 7 respuestas planas).
> Solo resultados **agregados**; las celdas con menos de 5 personas se suprimen.
> Las figuras están en `outputs/real/figuras/` (no versionadas).

## 1. Resumen antes vs después

| Aspecto | Trabajo original | v2 |
|---|---|---|
| Correlaciones | Pearson | Policóricas con corrección de continuidad 0,5 (ítems ordinales con efecto techo) |
| Extracción / rotación | Componentes principales + varimax | Factores comunes (minres) + oblimin |
| N.º de factores | 5 (Kaiser) | **3** (análisis paralelo policórico, criterio fijado antes de ver datos). PA de Pearson = 3; Kaiser = 4; PA reducido = 5; MAP plano entre 2 y 5 |
| Cuestionario | 19 ítems | **8 ítems** (3 + 3 + 2); r puntaje total corto–largo = 0,95 |
| Confiabilidad | α = 0,911 total | ω: Recursos 0,87 · Autonomía 0,90 · Oportunidad 0,73 · Total 0,91; **ω jerárquico = 0,55** |
| Segmentación | K-means k = 5, sin criterio | Principal: **K-means k = 2** (50 / 66), la única configuración estable de 20 |
| Silueta | 0,30 (calculada sobre datos distintos a los del modelo) | 0,34; con datos **sin grupos** da 0,31 (p = 0,12) |
| Estabilidad | No evaluada | ARI medio 0,91 (submuestras del 80 %, 200 réplicas) |
| Clasificador | LogReg 97 % (etiqueta circular: suma de factores ≥ 85) | LogReg con 8 preguntas: **F1-macro 0,89** (línea base 0,36), CV 5×20 |
| Sesgo | No evaluado | Segmentación principal sin desbalance detectable por género, edad, estado civil ni educación |

## 2. Psicometría

- **Idoneidad:** KMO = 0,80 (mínimo por ítem 0,66); Bartlett χ²(171) = 1401, p < 0,001.
- **Corrección de continuidad:** sin ella la matriz policórica no era definida positiva y
  aparecían casos Heywood (cargas ≥ 1). Con 0,5 en celdas vacías (por defecto en
  `psych::polychoric`) la matriz es definida positiva sin suavizado.
- **Robustez:** la misma EFA con Pearson da factores casi idénticos (congruencia de Tucker 0,99 / 0,93 / 0,92).

| Factor | Ítems (carga principal) | ω [IC 95 %] | Estabilidad bootstrap* |
|---|---|---|---|
| **Recursos y entorno de apoyo** | p10 políticas públicas .84 · p12 recursos propios .77 · p11 apoyos públicos .72 · p09 financiación externa .71 · p13 tradición familiar .70 · p05 negocio familiar .51 · p18 .47 · (p17 cruzado) | 0,87 [0,82; 0,90] | 97 % |
| **Autonomía y autorrealización** | p03 autonomía .78 · p02 flexibilidad .77 · p07 aplicar experiencia .72 · p06 explotar conocimiento .67 · p08 habilidades .62 · p04 ingresos .59 · p14 bienestar comunidad .56 · p01 sueño .53 | 0,90 [0,85; 0,93] | 82 % |
| **Oportunidad e innovación** (tentativo) | p16 nuevo producto .73 · p15 necesidad insatisfecha .61 | 0,73 [0,59; 0,83] | **55 %** |

\* % de 500 réplicas bootstrap con congruencia de Tucker ≥ 0,85 con la solución original.

- Correlaciones entre factores: 0,16 a 0,38. Justifican la rotación oblicua.
- **p19 (red de contactos)** no carga en ningún factor (h² = 0,24) y **p17** carga cruzado. Son los 2 ítems no "limpios".

**Por qué esos nombres:** *Recursos y entorno de apoyo* agrupa los motivos que dependen de
condiciones externas (financiación, apoyos públicos, capital propio y familiar).
*Autonomía y autorrealización* agrupa motivos personales: independencia, flexibilidad y
aplicar lo que uno sabe. *Oportunidad e innovación* reúne detectar una necesidad y crear
un producto nuevo. Corresponde a la idea de "emprendimiento por oportunidad".

**Sensibilidad del número de factores:**

| Factores | Ítems limpios | Mín. ítems limpios por factor | Estabilidad mínima* |
|---|---|---|---|
| 2 | 16 | 8 | **92 %** |
| **3 (elegido)** | 17 | 2 | 51 % |
| 4 | 16 | 2 | 71 % |
| 5 (como el original) | 14 | 2 | 73 % |

**Lectura honesta:** los datos sostienen con claridad **dos** dimensiones amplias (recursos
externos vs motivación personal). La tercera (innovación) supera al azar en el análisis
paralelo, pero tiene solo 2 ítems y es inestable. Los 5 factores del original aparecen en
la EFA, pero 2 de ellos tienen solo 2 ítems y no son estables.

**ω jerárquico = 0,55:** algo más de la mitad de la varianza del puntaje total se debe a
un factor general ("cuánto me motiva emprender en general"). Esto explica lo que ocurre
en la segmentación.

## 3. Cuestionario corto: ¿cuántas preguntas bastan?

| Factor | Ítems cortos | ω largo → corto | r con el puntaje factorial completo | r con los ítems no incluidos |
|---|---|---|---|---|
| Recursos y entorno de apoyo | p10, p12, p11 | 0,87 → 0,87 | 0,94 | 0,70 |
| Autonomía y autorrealización | p03, p02, p07 | 0,89 → 0,79 | 0,95 | 0,78 |
| Oportunidad e innovación | p16, p15 | 0,73 → 0,73 | 0,89 | — |

**8 preguntas** reproducen el puntaje total con r = 0,95. Con 3 por factor se recupera
r ≥ 0,94 en los dos factores sólidos.

## 4. Segmentación

Se compararon K-means, Ward, GMM y LPA (GMM con covarianza diagonal) con k = 2…6, sobre
tres representaciones. Para cada configuración se midieron silueta, Davies-Bouldin,
Calinski-Harabasz, BIC y estabilidad (200 submuestras).

| Representación | Elegida | Tamaños | ARI | Silueta vs "sin grupos" | HDBSCAN |
|---|---|---|---|---|---|
| **Puntajes factoriales (principal)** | K-means k = 2 | 50 / 66 | **0,91** | 0,34 vs 0,31 (p = 0,12) | 0 núcleos densos |
| Forma (centrado por persona) | K-means k = 3 | 33 / 34 / 49 | 0,77 | 0,35 vs 0,35 (p = 0,45) | 0 núcleos densos |
| PCA (referencia del original) | ninguna estable | — | máx. 0,56 | — | 0 núcleos densos |

**Conclusión central: no hay grupos naturales de emprendedores en estos datos.** Las
segmentaciones estables son **cortes de un continuo**: ninguna separa mejor que datos
simulados sin estructura de grupos, y HDBSCAN no encuentra ningún núcleo denso. Se pueden
usar para describir y comunicar, pero no son "tipos" de personas.

### Perfiles — segmentación principal (nivel)

| Segmento | n (%) | Recursos | Autonomía | Oportunidad | Media 1–5 (Rec. / Aut. / Op.) |
|---|---|---|---|---|---|
| 1. Motivación moderada en todos los factores | 50 (43 %) | −0,88 | −0,52 | −0,60 | 2,6 / 3,9 / 3,2 |
| 2. Motivación alta en todos los factores | 66 (57 %) | +0,67 | +0,39 | +0,45 | 4,0 / 4,7 / 4,1 |

Difieren en **intensidad**, no en tipo. "Moderada" no significa baja: la autonomía sigue en 3,9 de 5.

### Perfiles — segmentación por forma (sensibilidad)

| Segmento | n (%) | Recursos | Autonomía | Oportunidad |
|---|---|---|---|---|
| 1. Predomina: Oportunidad e innovación | 33 (28 %) | −0,89 | −0,04 | +0,41 |
| 2. Predomina: Autonomía y autorrealización | 34 (29 %) | +0,12 | +0,54 | −0,91 |
| 3. Predomina: Recursos y entorno de apoyo | 49 (42 %) | +0,51 | −0,35 | +0,35 |

Ítems que más distinguen estos segmentos (ε² de Kruskal-Wallis): p11 apoyos públicos (0,33),
p10 políticas públicas (0,28) y p16 nuevo producto (0,25). Los p-valores son descriptivos:
los segmentos se formaron con estas mismas respuestas.

**Orden de "potencial":** no se ordena. En la segmentación principal el orden refleja
**intensidad motivacional**, no potencial ni éxito, porque no hay ninguna variable de
resultado. En la de forma, ningún perfil es "mejor" que otro.

## 5. Validación externa (variables no usadas para agrupar)

| Variable | Segmentación principal | Segmentación por forma |
|---|---|---|
| Ocupación (solo emprendedor vs emprendedor y estudiante) | p = 0,14, V = 0,16 | p = 0,03, V = 0,24 (seg. 1: 3 % "ambos" vs 22–24 %) |
| Años de funcionamiento | p = 0,23 | p = 0,39 |

Hay una sola diferencia nominalmente significativa entre 4 pruebas. Con corrección de
Holm deja de serlo (p ≈ 0,12). **No hay validación externa que respalde los segmentos.**
Además, la muestra no permite construirla: todas las personas ya emprenden y no hay
ninguna medida de resultado.

## 6. Modelo del formulario (8 preguntas → segmento)

Validación cruzada estratificada 5 × 20.

| Modelo | Seg. principal F1-macro | Seg. forma F1-macro |
|---|---|---|
| Línea base (clase mayoritaria) | 0,36 | 0,20 |
| **Regresión logística** | **0,89** (acc. 0,90) | **0,85** (acc. 0,85) |
| Árbol (profundidad 3) | 0,78 | 0,71 |
| Random forest | 0,85 | 0,80 |

La regresión logística es a la vez la mejor y la interpretable. Mide qué tan bien el
formulario corto **reproduce el segmento**, no si alguien es o será buen emprendedor.

## 7. Sesgo (atributos sensibles, nunca usados como variables)

| Atributo | Segmentación principal | Segmentación por forma |
|---|---|---|
| Género | p = 0,86, V = 0,03 | **p = 0,01, V = 0,29** |
| Rango de edad | p = 0,54 | p = 0,15 |
| Estado civil | p = 0,90 | p = 0,34 |
| Nivel educativo | p = 0,27 | **p = 0,02, V = 0,25** |

En la segmentación por forma, el perfil "Predomina: Oportunidad e innovación" es **70 %
hombres** (frente a 35–39 % en los otros) y **48 % con posgrado** (frente a 18–29 %). Si ese
perfil se usara como criterio de inversión, favorecería sistemáticamente a hombres con
más educación.

## 8. Decisiones pendientes (punto de control)

1. **Número de factores:** 3 (PA, con el tercero marcado como tentativo) o 2 (más estable).
   Recomendación: **3**. El criterio se fijó antes de ver los datos y la dimensión de
   innovación tiene sentido sustantivo; se reporta como tentativa.
2. **Segmentación para la app:** recomendación: **la principal (k = 2)**, porque es estable
   y no muestra desbalance demográfico. La "forma" se mostraría en la app como el gráfico
   de los 3 puntajes de la persona, sin una etiqueta de tipo, y en el reporte como análisis
   de sensibilidad con su hallazgo de sesgo.
3. **Instrumento:** sigue pendiente confirmar el texto de los ítems (en especial p18) antes
   de publicar los nombres de los factores.
