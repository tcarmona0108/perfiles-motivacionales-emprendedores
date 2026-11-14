# Fase 0 — Diagnóstico del proyecto

> Estado: diagnóstico de solo lectura. No se modificó ningún archivo original.
> Fecha: 2026-11-14. Las cifras de este documento se calcularon sobre `CSV Datos test.csv`
> y sobre los outputs que el notebook ya tenía guardados.
> Este documento **no reproduce ningún valor de las columnas identificables** (ciudad, objeto social).

---

## 0. Resumen en 6 líneas

1. **Privacidad (bloqueante):** no hay identificadores directos, pero hay cuasi-identificadores fuertes: **103 de 123 personas son únicas** al combinar ciudad + género + edad + estado civil + educación, y el texto libre "objeto social" es casi único por persona (117 valores distintos en 123). El notebook además **imprime esos datos en sus outputs**. Detengo aquí según la regla 2.
2. **El instrumento mide motivaciones para emprender, no rasgos de personalidad emprendedora.** Los 19 ítems son razones ("sueño de ser emprendedor", "altos ingresos", "acceder a apoyos públicos", "tradición familiar"…). Los factores no van a llamarse "Tolerancia al riesgo" o "Proactividad"; eso no está en el cuestionario.
3. **La muestra son solo emprendedores** (102 "Emprendedor(a)" + 21 "Ambos"; nadie es solo estudiante ni no emprendedor). No hay contraste posible "emprendedor vs. no emprendedor" y, por tanto, **"potencial emprendedor" no se puede validar con estos datos**.
4. **El modelo predictivo original es circular:** la etiqueta `Potencial_Emprendedor` se construyó como `suma de factores ≥ 85` y luego se predijo **con esos mismos factores**. El 97 % de exactitud solo mide que una regresión logística recupera un umbral lineal. Esa métrica no puede presentarse.
5. **Lo que sí está bien:** KMO (0,85) y Bartlett apropiados, alfa (0,911) correcta, la intuición de comparar varios algoritmos de clustering y de buscar una versión corta del cuestionario.
6. **Lo débil:** PCA presentado como análisis factorial, número de factores solo por Kaiser (el análisis paralelo sugiere 3, no 5), rotación ortogonal con factores que correlacionan, k = 5 sin criterio, sin estabilidad, CFA sobre la misma muestra que la EFA, y un notebook que no es re-ejecutable en orden.

---

## 1. Mapa del proyecto

| Archivo | Qué es | Observaciones |
|---|---|---|
| `CSV Datos test.csv` | Respuestas crudas, 123 filas × 27 columnas, `;` como separador, UTF-8 con BOM | Contiene cuasi-identificadores (ver §2.5) |
| `Proyecto_Profesora_Angela.ipynb` | Todo el análisis (116 celdas, hecho en Google Colab, 14 figuras embebidas) | Ruta absoluta `/content/…`, `%pip install` sin versiones, contadores de ejecución borrados |
| `README.md` | Un párrafo de descripción | Dice "tesis doctoral" y "predecir el potencial emprendedor" (afirmación no sostenible, ver §3.8) |

No hay: control de versiones (la carpeta **no es un repositorio git**), `requirements`, scripts, tests, licencia ni documentación de los ítems.

### Flujo actual de punta a punta (según el notebook)

```
CSV ─► descriptivos + Shapiro por ítem + "tamaño de muestra" (celdas 5–29)
     ─► boxplot, VIF, alfa de Cronbach, KMO, Bartlett (30–38)
     ─► descriptivos demográficos (39–46)
     ─► correlaciones de Pearson (48–51)
     ─► PCA estandarizado, 19 componentes (52–60)
     ─► FactorAnalyzer(method='principal', 5 factores, varimax) (61–66)
     ─► "CFA" con semopy: 5 factores × 2 ítems + caminos estructurales (70–78)
     ─► puntajes = datos crudos × cargas varimax ─► estandarizar
     ─► K-means k=5 · DBSCAN · Ward k=5 (79–99)
     ─► etiqueta "Potencial" = suma(puntajes) ≥ 85 ─► LogReg y RF (100–115)
```

---

## 2. Descripción de los datos

### 2.1 Tamaño y estructura
- **n = 123** respuestas, **19 ítems** (`Pregunta 1` … `Pregunta 19`).
- **Escala Likert de 5 puntos** (valores observados: 1, 2, 3, 4, 5). No conozco las anclas verbales; las necesito (ver §6).
- **Faltantes: 0** en todas las columnas.
- Ratio casos/ítems ≈ 6,5 : 1. Es aceptable para una EFA exploratoria con comunalidades moderadas, pero ajustado; con n = 123 los resultados deben acompañarse de estabilidad por bootstrap.

### 2.2 Contenido de los ítems (según las etiquetas del propio notebook, celda 60)

| Ítem | Motivación | Ítem | Motivación |
|---|---|---|---|
| 1 | Sueño de ser emprendedor | 11 | Acceso a apoyos públicos |
| 2 | Flexibilidad de tiempo | 12 | Invertir recursos financieros propios |
| 3 | Autonomía e independencia | 13 | Continuar tradición familiar |
| 4 | Altos ingresos | 14 | Contribuir al bienestar de la comunidad |
| 5 | Construir negocio familiar | 15 | Detectar necesidad insatisfecha |
| 6 | Explotar conocimiento | 16 | Desarrollar nuevo producto/servicio |
| 7 | Aplicar experiencia | 17 | Nuevo método de producción |
| 8 | Poner en práctica habilidades | 18 | *(rotulado igual que el 17: "Nuevo método de producción")* |
| 9 | Financiación externa | 19 | Red de contactos |
| 10 | Políticas públicas | | |

⚠️ Las etiquetas son **inconsistentes dentro del notebook**: la celda 65 dice que el ítem 11 es "invertir recursos propios" y que 18–19 son "tradición familiar", contradiciendo la celda 60. Necesito el cuestionario original para no nombrar factores sobre etiquetas erróneas.

### 2.3 Calidad de las respuestas
- **Efecto techo fuerte:** en los ítems 1, 2, 3, 4, 6, 7, 8 y 14 más del 50 % respondió 5; asimetrías entre −1,5 y −2,5. Solo el ítem 13 (tradición familiar) es simétrico (media 2,88, 27 % respondió 1).
- **7 personas respondieron 5 en los 19 ítems** (varianza cero intrapersona): posible aquiescencia o respuesta automática. Hay que decidir si se excluyen o se marcan.
- **6 patrones de respuesta duplicados exactos** en los 19 ítems (no son filas duplicadas: difieren en demografía).
- **Ítems invertidos:** ninguna correlación inter-ítem es negativa (mín. 0,03; máx. 0,86) y todas las correlaciones ítem-total corregidas son positivas (0,40–0,72). **No hay evidencia de ítems invertidos**; lo confirmaré con el texto del instrumento.
- **Factor general dominante:** el primer autovalor (7,61) explica el 40 % de la varianza y todos los ítems cargan positivo en él. Esto importa para el clustering (ver §3.6).

### 2.4 Variables no usadas en el modelo (demografía y contexto)

| Variable | Distribución | Uso propuesto |
|---|---|---|
| Actualmente soy | Emprendedor(a) 102 · Ambos 21 | Validación externa débil |
| Años de funcionamiento | 0–17, mediana 4; 12 personas con 0; 7 valores no enteros | Validación externa débil (proxy de trayectoria) |
| Género | Femenino 67 · Masculino 56 | **Sensible** → solo auditoría de sesgo |
| Rango de edad | 5 rangos (7 personas < 25 años) | **Sensible** → solo auditoría |
| Estado civil | Soltero 66 · Casado 38 · Otro 19 | **Sensible** → solo auditoría |
| Nivel educativo | 8 niveles (1 persona en "Primaria") | **Sensible** → solo auditoría; agrupar niveles |
| Ciudad | 35 ciudades, **24 con una sola persona** | Cuasi-identificador → no publicar |
| Objeto social | Texto libre, 117 valores distintos | Cuasi-identificador → no publicar |

### 2.5 Hallazgo de privacidad (regla 2) — **BLOQUEANTE**

- **Identificadores directos:** no encontré nombres de persona, documentos, correos (0 coincidencias de `@`), teléfonos (0 secuencias de ≥ 7 dígitos) ni URLs. Las fechas de nacimiento vienen en rangos.
- **Cuasi-identificadores (riesgo alto de reidentificación):**
  - Con ciudad + género + rango de edad + estado civil + educación, **103 de 123 personas (84 %) tienen una combinación única** (k-anonimato = 1).
  - Solo con ciudad + género + edad, 46 personas ya son únicas.
  - El **objeto social** es texto libre que describe el negocio concreto; algunos registros parecen incluir nombres comerciales. Junto con la ciudad (muchas son municipios pequeños con una sola persona en la muestra), basta para identificar al negocio y, por tanto, a su dueño.
- **El notebook también expone los datos:** los outputs de las celdas 5 (`data.head()`) y 41 (`value_counts()` de ciudad) quedaron guardados dentro del `.ipynb`.
- **Implicación:** ni el CSV crudo ni el notebook original con sus outputs deberían publicarse. Si este código ya estuvo en un repositorio público (el nombre de la carpeta `…-main` sugiere una descarga de GitHub), el dato ya está expuesto en ese historial; eso es algo que tú o la docente deben decidir cómo tratar.

---

## 3. Revisión metodológica

### 3.1 Lo que está bien
- **KMO = 0,85** (individuales 0,73–0,92) y **Bartlett p ≈ 0**: los datos son aptos para análisis factorial. Correcto y bien interpretado.
- **Alfa de Cronbach = 0,911** (lo verifiqué: 0,911). El cálculo es correcto.
- Estandarizar antes del PCA y antes del clustering.
- Comparar más de un algoritmo de clustering y reportar silueta.
- Reconocer la redundancia de los ítems 8–10–12: es justo la pregunta "¿cuántas preguntas bastan?".

### 3.2 Normalidad y "tamaño de muestra" (celdas 10–29)
- Shapiro-Wilk sobre ítems Likert de 5 puntos siempre rechaza; no aporta. Lo relevante es la **asimetría y el efecto techo**, que justifican correlaciones policóricas o un estimador robusto.
- La fórmula de tamaño de muestra `(z·s / (1 − precision))²` no corresponde a ninguna fórmula estándar (la usual es `(z·s / E)²`, con E en unidades de la escala). Con E = 0,5 el resultado coincide por casualidad. Además no responde la pregunta relevante (n suficiente para EFA).
- 19 celdas copiadas y pegadas, una por ítem.

### 3.3 VIF (celda 32)
- `variance_inflation_factor` se aplicó a datos crudos **sin constante**, lo que infla los VIF (50–104). Con constante serían mucho menores.
- Además, el VIF es un diagnóstico de regresión; en análisis factorial la colinealidad **es lo que se busca**. Se puede eliminar o reemplazar por el determinante de la matriz de correlaciones.

### 3.4 "Análisis factorial exploratorio" (celdas 52–66)
- **PCA ≠ análisis factorial.** `FactorAnalyzer(method='principal')` es extracción por componentes principales: modela la varianza total y no separa la varianza común de la única, así que infla las cargas. El README lo llama "análisis factorial exploratorio mediante Componentes Principales", mezclando los dos.
- **Número de factores solo por Kaiser (autovalor > 1) → 5.** Kaiser sobreestima de forma conocida. Un análisis paralelo rápido (500 réplicas, Pearson, percentil 95) **sugiere 3**; los autovalores 4.º (1,29) y 5.º (1,15) están por debajo del azar (1,47 y 1,38). Hay que confirmarlo con policóricas y extracción por factores comunes.
- **Rotación varimax (ortogonal)** cuando los factores comparten un factor general fuerte (40 % de varianza): con factores correlacionados lo adecuado es una rotación oblicua (oblimin/promax).
- **Cargas cruzadas ignoradas:** ítem 5 (0,44 / 0,42), ítem 11 (0,56 / 0,47), ítem 8 (0,40 / 0,65), ítem 1 (0,38 / 0,59).
- **Los nombres de los factores no coinciden con las cargas.** Ejemplo: se dice que el Factor 3 agrupa 15, 16, 17 y el Factor 4 agrupa 18 y 19 "tradición familiar", pero la tradición familiar es el ítem 13 y ese carga en el Factor 1.
- Correlaciones de Pearson sobre ítems ordinales con efecto techo: subestiman la asociación y pueden generar factores de "dificultad".

### 3.5 "Análisis factorial confirmatorio" (celdas 70–78)
- Se ajustó **sobre la misma muestra** que la EFA: no confirma nada, es circular.
- Usa **10 de los 19 ítems**, con **2 ítems por factor** (identificación débil; los factores de 2 ítems son inestables).
- Las **relaciones estructurales entre factores** (`Factor2 ~ Factor1`, etc.) no tienen justificación teórica escrita.
- Estimador ML para datos ordinales y asimétricos; con n = 123 los índices (CFI 0,963, TLI 0,943, RMSEA 0,078) son poco confiables y RMSEA 0,078 es solo "aceptable".
- "Un objetivo de 0,410 representa un buen ajuste" no es una interpretación válida: el valor de la función objetivo no tiene escala absoluta.
- **Propuesta:** con n = 123 no hay muestra para separar EFA/CFA. Lo honesto es hacer solo EFA y declarar la CFA en una muestra independiente como trabajo futuro.

### 3.6 Puntajes factoriales y clustering (celdas 81–99)
- **Puntajes calculados mal:** `datos_crudos @ cargas_varimax` no es un puntaje factorial: es una suma ponderada de ítems sin centrar. Por eso todos los puntajes son positivos (≈ 6–34) y están dominados por el nivel general de respuesta.
- **Los clusters ordenan por "cuánto respondes alto", no por tipo de motivación.** Las medias por grupo jerárquico son crecientes en los 5 factores a la vez (grupo 4 ≈ 6–12 → grupo 3 ≈ 19–34). Es el efecto del factor general + aquiescencia. Para obtener perfiles de forma (qué motiva más que qué), hay que considerar como análisis de sensibilidad centrar las respuestas por persona o separar el factor general.
- **k = 5 sin criterio:** coincide con el número de factores, pero no hay relación lógica entre ambos. No hay codo, BIC ni estabilidad.
- **Silueta inconsistente:** K-means se ajustó sobre los puntajes **estandarizados** y la silueta se calculó sobre los **no estandarizados** (celda 86). El 0,30 reportado no corresponde al modelo.
- **DBSCAN con `eps=0,5` sin ajustar:** 94 de 123 puntos (76 %) quedaron como ruido; la silueta de 0,59 "sin ruido" se calcula sobre 29 personas y no es comparable.
- **Las descripciones de los clusters (celda 69) aparecen antes de calcular los clusters** y no hay una tabla de medias por cluster de K-means que las respalde.
- No hay validación de estabilidad (bootstrap / ARI) ni comparación entre algoritmos.

### 3.7 Modelo predictivo de "potencial emprendedor" (celdas 100–115) — **error grave**
- La etiqueta es `Potencial = "Sí" si suma(Factor1..5) ≥ 85`. El umbral de 85 es arbitrario y no está justificado.
- Luego se predice esa etiqueta **usando los mismos Factor1..5 como predictores**. Es **fuga de información (leakage) por construcción**: la etiqueta es una función lineal determinística de los predictores. La exactitud del 97 % de la regresión logística es un artefacto.
- El conjunto de prueba tiene 37 casos con **solo 5 "No"**; la ROC y la matriz de confusión se basan en esos 5 casos.
- **"Potencial emprendedor" no existe en los datos** (todos ya son emprendedores) y el instrumento mide motivaciones, no potencial. El README dice que se "predice el potencial emprendedor"; esto debe retirarse.

### 3.8 Ética
- No se usaron atributos sensibles como variables del modelo (bien), pero tampoco se revisó si los grupos quedan desbalanceados por género, edad o educación.
- El objetivo declarado ("decidir si vale la pena invertir en una persona") **no se puede sostener con estos datos**: no hay ninguna variable de resultado (supervivencia, ventas, financiación obtenida). Lo defendible es: *perfiles motivacionales de emprendedores como apoyo a la conversación o a la mentoría*, no como filtro de inversión.

---

## 4. Revisión de código

| Problema | Dónde | Efecto |
|---|---|---|
| Ruta absoluta de Colab `/content/CSV Datos test.csv` | celda 5 | No corre fuera de Colab |
| `%pip install` sin versiones (y `semopy` instalado dos veces, `scikit-learn-extra` sin usar) | celda 2 | No reproducible |
| Sin `random_state` en algunos pasos; sin semillas globales | varias | Resultados no deterministas |
| Nombre de columna con espacio final (`'Pregunta 1 '`), luego se limpian los nombres a mitad del notebook (celda 50) | 8–33 vs 50+ | **Reejecutar las celdas 8–33 después de la 50 falla** (`KeyError`) |
| `data_preguntas.columns` se modifica en el sitio (celda 72) | 72 | Rompe las celdas anteriores que usan ese objeto |
| `alpha_de_cronbach(data)` ignora su argumento y usa la variable global | 33 | Funciona por casualidad |
| n hardcodeado (`/123`) en porcentajes | 42–45 | Frágil |
| Lógica de conteo con `else` que cuenta todo lo no "Emprendedor" como "Ambos" | 42 | Correcto hoy por casualidad |
| `SettingWithCopyWarning` (asignar sobre una vista) | 103–104 | Comportamiento no garantizado |
| Silueta sobre datos distintos a los del modelo | 86 | Métrica incorrecta |
| Imports duplicados o sin usar (`StandardScaler` ×2, `Axes3D`, `Optimizer`, `kstest`, `datetime`) | 3 | Ruido |
| 19 bloques copiados y pegados | 10–28 | Duplicación |
| Contadores de ejecución borrados y outputs con `ipython-input-63/64` en celdas 103–104 | — | **No se puede asegurar que el notebook corra en orden**; la celda 69 interpreta clusters calculados después |
| Outputs con datos crudos guardados en el `.ipynb` | 5, 41 | Privacidad |
| Entorno local: Python 3.14 sin `factor_analyzer` ni `semopy` | — | Hay que fijar una versión de Python compatible (probablemente 3.12) y verificarla en la Fase 1 |

---

## 5. Plan de mejora priorizado

Esfuerzo: **S** ≤ 2 h · **M** medio día · **L** 1–2 días de trabajo.

| # | Acción | Fase | Prioridad | Esfuerzo |
|---|---|---|---|---|
| 1 | **Resolver privacidad:** decidir dato público (sintético o anonimizado), mantener el crudo fuera de git, limpiar outputs del notebook legado antes de publicarlo | 0→1 | Bloqueante | S |
| 2 | Inicializar git (o clonar el repo original), rama `mejora/v2`, mover el original a `legacy/`, `.gitignore` de `data/raw` | 1 | Alta | S |
| 3 | Estructura del paquete, `pyproject.toml` + `uv.lock` (Python 3.12), `config.yaml`, `Makefile`, ruff, pytest, GitHub Actions | 1 | Alta | M |
| 4 | Generador de datos sintéticos que preserve las marginales por ítem y la matriz de correlaciones policóricas (cópula gaussiana ordinal), documentado y con semilla | 1 | Alta | M |
| 5 | Preparación: descriptivos por ítem, techo/piso, política explícita para los 7 "todo 5" (análisis con y sin ellos) | 2 | Alta | S |
| 6 | Psicometría: policóricas, KMO/Bartlett, análisis paralelo + MAP de Velicer + scree, EFA con factores comunes (minres/ULS) + oblimin, alfa y omega por factor | 2 | Alta | M |
| 7 | Reducción del cuestionario: selección de ítems por carga, ausencia de carga cruzada y aporte a omega; comparar corto vs largo (fiabilidad y correlación de puntajes, con bootstrap) | 2 | Alta | M |
| 8 | Representación: puntajes factoriales correctos (Bartlett/Thurstone) vs PCA; sensibilidad con centrado por persona para separar nivel de forma | 2 | Alta | S |
| 9 | Clustering: K-means, Ward, GMM y LPA (GMM con covarianza diagonal o StepMix, a verificar), k = 2…6, silueta, DB, CH, BIC y **estabilidad bootstrap (ARI)** | 2 | Alta | L |
| 10 | Perfilado: medias por factor, Kruskal-Wallis con Holm/BH y épsilon², nombres en lenguaje claro | 2 | Alta | M |
| 11 | Validación externa con "Actualmente soy" y "años de funcionamiento", declarada como débil | 2 | Media | S |
| 12 | Clasificador del formulario corto → segmento (LogReg multinomial vs RF/GB, CV estratificada repetida, F1-macro, SHAP/coeficientes). Eliminar el modelo circular | 2 | Alta | M |
| 13 | Auditoría de sesgo: distribución de segmentos por género, edad, estado civil y educación (chi² / V de Cramér) | 2 | Media | S |
| 14 | Reporte Quarto + GitHub Pages | 3 | Media | L |
| 15 | Demo Streamlit o Gradio con solo el modelo entrenado sobre el cuestionario corto | 4 | Media | M |
| 16 | README, créditos, párrafo de portafolio, checklist final | 5 | Media | S |

**Riesgo metodológico principal anticipado:** con un factor general del 40 %, efecto techo y n = 123, es probable que (a) aparezcan 2–3 factores en lugar de 5 y (b) la solución de clusters más estable sea de pocos grupos (2–3) ordenados por intensidad. Si es así, lo reportaré tal cual, con evidencia, en lugar de forzar 5 perfiles.

---

## 6. Lo que necesito de ti

1. **Privacidad (bloqueante).** ¿El repositorio original es o fue público? Para la versión pública propongo: **(a)** que `data/raw` nunca se versione; **(b)** que el repositorio público use **solo datos sintéticos**; **(c)** que el análisis real corra localmente sobre los datos crudos y que solo se publiquen resultados agregados (medias por grupo con n ≥ 5, sin ciudad ni objeto social). ¿Lo apruebas, o la docente prefiere otra cosa (por ejemplo, un CSV anonimizado sin ciudad ni texto libre)?
2. **Notebook legado.** La regla 1 pide moverlo a `legacy/` sin modificarlo, pero sus outputs contienen datos identificables. Propongo publicar en `legacy/` una copia **con los outputs borrados** (el código queda intacto) y conservar el original con outputs solo en local. ¿De acuerdo?
3. **Git.** La carpeta no es un repositorio. ¿Existe un repositorio en GitHub del que venga? Si existe, prefiero clonarlo para conservar su historial; si no, hago `git init` aquí.
4. **Instrumento.** ¿Tienes el cuestionario original (texto exacto de los 19 ítems, anclas de la escala, dimensiones teóricas que planteó la docente)? Las etiquetas del notebook se contradicen.
5. **Alcance.** Propongo reformular el objetivo como *"perfiles motivacionales de emprendedores"* en lugar de *"potencial emprendedor para decidir inversión"*, porque la muestra son solo emprendedores y no hay variable de resultado. ¿Lo validas con la docente?
6. **Créditos.** ¿Cómo quiere ser citada la docente (nombre, afiliación, título de la investigación o tesis, año)? El README actual habla de "tesis doctoral".
