# Referencia completa del grafo de conocimiento — Graph RAG BOE

> Documento de referencia técnica sobre el modelo de datos del grafo Neo4j generado a partir de la Ley 22/2014 (BOE-A-2014-11714).

---

## Índice

1. [Visión general del modelo](#1-visión-general-del-modelo)
2. [Nodos — tipos, propiedades y constraints](#2-nodos--tipos-propiedades-y-constraints)
   - 2.1 `Articulo`
   - 2.2 `Titulo`
   - 2.3 `Capitulo`
   - 2.4 `Seccion`
   - 2.5 `Disposicion`
   - 2.6 `Entidad`
   - 2.7 `Concepto`
   - 2.8 Nodos opcionales del `LLMGraphTransformer`
3. [Aristas — tipos, dirección y semántica](#3-aristas--tipos-dirección-y-semántica)
   - 3.1 `PERTENECE_A`
   - 3.2 `REFERENCIA`
   - 3.3 `MENCIONA`
   - 3.4 `TRATA_SOBRE`
   - 3.5 Aristas opcionales del `LLMGraphTransformer`
4. [Diagrama del esquema completo](#4-diagrama-del-esquema-completo)
5. [Índices y constraints en Neo4j](#5-índices-y-constraints-en-neo4j)
6. [Cómo se construye el grafo — pipeline detallado](#6-cómo-se-construye-el-grafo--pipeline-detallado)
   - 6.1 Paso 1 — Extracción del PDF (regex)
   - 6.2 Paso 2 — Carga en Neo4j
   - 6.3 Paso 3 (opcional) — Enriquecimiento con LLM
7. [Cómo se consulta — los dos modos RAG](#7-cómo-se-consulta--los-dos-modos-rag)
   - 7.1 Modo `cypher` — LangChain genera Cypher
   - 7.2 Modo `grafo` — recuperación manual + expansión
8. [Catálogo completo de queries Cypher](#8-catálogo-completo-de-queries-cypher)
9. [Volumetría esperada](#9-volumetría-esperada)
10. [Limitaciones conocidas y extensiones posibles](#10-limitaciones-conocidas-y-extensiones-posibles)

---

## 1. Visión general del modelo

El grafo modela un documento jurídico (una ley del BOE) como una red de nodos interconectados. La filosofía central es que **las relaciones entre conceptos jurídicos son tan importantes como el texto en sí**: un artículo cita a otro, una entidad es regulada por varios artículos dispersos, un capítulo agrupa artículos de una misma materia.

```
          Titulo
            │  PERTENECE_A
            ▼
          Capitulo ◄──────────────────────────────────────────────┐
            │  PERTENECE_A                                         │
            ▼                                                      │
          Articulo ──REFERENCIA──► Articulo                        │
            │  MENCIONA                │  PERTENECE_A              │
            ▼                         └──────────────────────────►┘
          Entidad
            │  TRATA_SOBRE
            ▼
          Concepto
```

El grafo tiene **7 tipos de nodo** y **4 tipos de arista** en su configuración base, más nodos/aristas opcionales cuando se activa el `LLMGraphTransformer`.

---

## 2. Nodos — tipos, propiedades y constraints

---

### 2.1 `Articulo`

El nodo fundamental del grafo. Representa cada artículo numerado de la ley.

| Propiedad | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `id` | `string` (UNIQUE) | Identificador estable, slug del número | `"art_42"`, `"art_42_bis"` |
| `numero` | `string` | Número tal como aparece en el PDF | `"42"`, `"42 bis"` |
| `titulo` | `string` | Encabezado del artículo (primera línea) | `"Requisitos de acceso a la actividad"` |
| `texto` | `string` | Texto completo del artículo (máx. 8000 chars) | `"Artículo 42. Requisitos...\n1. Las SGEIC..."` |
| `titulo_padre` | `string` | Texto del Título al que pertenece | `"TÍTULO IV"` |
| `capitulo_padre` | `string` | Texto del Capítulo al que pertenece | `"CAPÍTULO I"` |
| `seccion_padre` | `string` \| `null` | Texto de la Sección (si existe) | `"Sección 1ª"` o `""` |

**Constraint:** `a.id IS UNIQUE`  
**Índice fulltext:** sobre `[a.titulo, a.texto]` → permite búsqueda de texto libre con Lucene.

**Cómo se genera el `id`:**  
```
"42"      → "art_42"
"42 bis"  → "art_42_bis"
"1"       → "art_1"
```

**Nota sobre el `texto`:** el texto se trunca a 8000 caracteres en Neo4j para evitar problemas de almacenamiento en artículos muy largos. El JSON intermedio (`data/articulos.json`) tiene el texto completo sin límite.

---

### 2.2 `Titulo`

Representa los grandes bloques estructurales de la ley: `TÍTULO I`, `TÍTULO II`, etc.

| Propiedad | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `id` | `string` (UNIQUE) | Slug generado desde el texto | `"t_tulo_iv"` |
| `texto` | `string` | Texto completo de la cabecera | `"TÍTULO IV Régimen de autorización"` |

**Constraint:** `t.id IS UNIQUE`

**Cómo se detecta:** el parser busca el patrón regex:
```
\bTÍTULO\s+(I{1,3}V?|VI{0,3}|IX|X{1,3}(?:I{0,3})?)
```
en líneas de menos de 120 caracteres (evita falsos positivos en el cuerpo de un artículo).

---

### 2.3 `Capitulo`

Subdivísión dentro de un `Titulo`.

| Propiedad | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `id` | `string` (UNIQUE) | Slug generado desde el texto | `"cap_tulo_iv"` |
| `texto` | `string` | Texto completo de la cabecera | `"CAPÍTULO IV Autorización de SGEICs"` |

**Constraint:** `c.id IS UNIQUE`

**Relación con Titulo:** cuando un `Articulo` informa a qué `Titulo` y `Capitulo` pertenece, la relación `PERTENECE_A` conecta también implícitamente el `Capitulo` con su `Titulo` (a través del contexto de parseo).

---

### 2.4 `Seccion`

Nivel de subdivisión opcional dentro de un `Capitulo`. Aparece como `Sección 1ª`, `Sección 2ª`, etc.

| Propiedad | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `id` | `string` (UNIQUE) | Slug | `"secci_n_1_"` |
| `texto` | `string` | Texto de la cabecera | `"Sección 1ª Requisitos generales"` |

**Constraint:** `s.id IS UNIQUE`

Muchos artículos tienen `seccion_padre = ""` si no están dentro de ninguna sección explícita.

---

### 2.5 `Disposicion`

Nodo para las disposiciones adicionales, finales, transitorias y derogatorias. No tienen número de artículo sino nombre propio.

| Propiedad | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `id` | `string` (UNIQUE) | Auto-generado: `disp_0`, `disp_1`… | `"disp_3"` |
| `titulo` | `string` | Nombre completo de la disposición | `"DISPOSICIÓN ADICIONAL PRIMERA"` |
| `texto` | `string` | Texto completo (máx. 4000 chars) | `"Las sociedades gestoras..."` |

**Constraint:** `d.id IS UNIQUE`

**Tipos detectados por regex:**
```
DISPOSICIÓN ADICIONAL   PRIMERA / SEGUNDA / … / ÚNICA / 1 / 2 …
DISPOSICIÓN FINAL
DISPOSICIÓN TRANSITORIA
DISPOSICIÓN DEROGATORIA
```

**Nota:** las disposiciones no tienen relación `PERTENECE_A` hacia `Titulo`/`Capitulo` porque aparecen al final del documento, fuera de la estructura jerárquica principal. Tampoco generan relaciones `REFERENCIA` automáticas en la versión base (aunque el texto sí puede contener referencias a artículos).

---

### 2.6 `Entidad`

Nodo de entidad jurídica o institucional que actúa como **punto de agregación** de todos los artículos que la regulan.

| Propiedad | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `nombre` | `string` (UNIQUE) | Nombre de la entidad tal como aparece en la lista de ENTIDADES | `"SGEIC"`, `"CNMV"`, `"ECR"` |

**Constraint:** `e.nombre IS UNIQUE`  
**Índice fulltext:** sobre `[e.nombre]`

**Lista completa de entidades indexadas** (detectadas por coincidencia exacta de subcadena, insensible a mayúsculas):

```
SGEIC                                    → Sociedad Gestora de Entidades de Inversión Colectiva de tipo Cerrado
ECR                                      → Entidad de Capital-Riesgo
SICC                                     → Sociedad de Inversión Colectiva de tipo Cerrado
FICC                                     → Fondo de Inversión Colectiva de tipo Cerrado
CNMV                                     → Comisión Nacional del Mercado de Valores
sociedad gestora                         → figura genérica
entidad de capital-riesgo                → ECR en texto completo
fondo de capital-riesgo                  → FCR
sociedad de capital-riesgo               → SCR
entidad de inversión colectiva de tipo cerrado → EICC
AIFMD                                    → Directiva 2011/61/UE (Directiva de Gestores)
GFIA                                     → Gestor de Fondos de Inversión Alternativos
```

**Por qué es el nodo más valioso:** buscar `(:Entidad {nombre: 'SGEIC'})` y recorrer `[:MENCIONA]` hacia atrás da directamente los ~40 artículos que regulan las SGEICs, sin necesidad de saber sus números ni recorrer el texto.

---

### 2.7 `Concepto`

Similar a `Entidad` pero para materias abstractas reguladas por la ley.

| Propiedad | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `nombre` | `string` (UNIQUE) | Nombre del concepto | `"capital mínimo"`, `"autorización"` |

**Constraint:** `c.nombre IS UNIQUE`

**Lista completa de conceptos indexados:**

```
autorización
capital mínimo
requisitos
registro
supervisión
solvencia
honorabilidad
conflictos de interés
política de inversión
depositario
apalancamiento
partícipes
socios
inversores profesionales
capital riesgo
private equity
venture capital
```

La detección es por subcadena en el texto del artículo (insensible a mayúsculas). Si el artículo contiene la palabra "autorización" en cualquier parte de su texto, se crea la arista `TRATA_SOBRE` hacia el nodo `Concepto {nombre: 'autorización'}`.

---

### 2.8 Nodos opcionales del `LLMGraphTransformer`

Cuando se activa `USAR_LLM_TRANSFORMER = True` en `config.py`, el LLM extrae entidades y relaciones adicionales del texto de cada artículo. Se generan nodos de estos tipos adicionales:

| Tipo de nodo | Descripción |
|---|---|
| `Requisito` | Un requisito legal concreto extraído del texto |
| `Obligacion` | Una obligación impuesta por la ley |
| `Plazo` | Un plazo temporal mencionado en el artículo |

Estos nodos pueden tener la propiedad `descripcion` con el texto del fragmento relevante.

---

## 3. Aristas — tipos, dirección y semántica

---

### 3.1 `PERTENECE_A`

**Dirección:**
```
(Articulo)  -[:PERTENECE_A]→ (Titulo)
(Articulo)  -[:PERTENECE_A]→ (Capitulo)
(Capitulo)  -[:PERTENECE_A]→ (Titulo)    ← implícito a través del contexto
```

**Sin propiedades.**

**Semántica:** define la jerarquía estructural del documento. Un artículo puede tener hasta **dos** aristas `PERTENECE_A`: una hacia su `Titulo` padre y otra hacia su `Capitulo` padre. Si el artículo está dentro de una `Seccion`, la sección se almacena como propiedad del nodo `Articulo` (`seccion_padre`), **no** como arista adicional.

**Cómo se crea:** durante la carga en Neo4j, para cada artículo se extrae el `titulo_padre` y el `capitulo_padre` almacenados en el JSON. Se genera el slug (`_id_seguro()`) y se hace `MERGE` de la arista.

**Uso típico en queries:**
```cypher
// Todos los artículos del Capítulo IV
MATCH (a:Articulo)-[:PERTENECE_A]->(c:Capitulo)
WHERE c.texto CONTAINS 'IV'
RETURN a.numero, a.titulo ORDER BY a.numero

// Estructura completa de un Título
MATCH (a:Articulo)-[:PERTENECE_A]->(t:Titulo {id: 't_tulo_iii'})
RETURN t.texto AS titulo, a.numero, a.titulo ORDER BY a.numero
```

---

### 3.2 `REFERENCIA`

**Dirección:**
```
(Articulo) -[:REFERENCIA]→ (Articulo)
```

**Sin propiedades.**

**Semántica:** el artículo origen menciona explícitamente al artículo destino en su texto, con expresiones como:
- *"conforme al artículo 38"*
- *"véase el artículo 41"*
- *"lo dispuesto en el art. 25"*
- *"arts. 30 y 31"*

**Regex de detección:**
```python
r'(?:artículo|arts?)\.?\s+(\d+(?:\s+(?:bis|ter|quater|quinquies|sexies|septies))?)'
```

**Cómo se crea:** el parser extrae todos los números de artículo referenciados durante la fase `01_extraer_pdf.py`. En `02_construir_grafo.py` se crea la arista solo si el artículo destino **existe realmente** en el JSON (no se crean aristas a artículos fantasma). Las auto-referencias (`art_42 → art_42`) se descartan.

**Por qué es la arista más valiosa:**
- Permite **expansión automática del contexto**: al recuperar el artículo 42, automáticamente se traen los artículos 38 y 41 que cita.
- Revela qué artículos son "hubs" de la ley (muchos apuntan a ellos).
- Permite preguntas del tipo: *"¿qué artículos debo leer si quiero entender el artículo 42?"*

**Uso típico en queries:**
```cypher
// ¿Qué artículos cita el artículo 42?
MATCH (a:Articulo {id: 'art_42'})-[:REFERENCIA]->(b:Articulo)
RETURN b.numero, b.titulo

// ¿Qué artículos citan al artículo 42? (arista invertida)
MATCH (a:Articulo)-[:REFERENCIA]->(b:Articulo {id: 'art_42'})
RETURN a.numero, a.titulo

// Red de 2 saltos: artículos relacionados con el 42
MATCH (a:Articulo {id: 'art_42'})-[:REFERENCIA*1..2]->(b:Articulo)
RETURN DISTINCT b.numero, b.titulo

// Artículos con más referencias entrantes (los más citados)
MATCH (a:Articulo)-[:REFERENCIA]->(b:Articulo)
RETURN b.numero, b.titulo, COUNT(a) AS veces_citado
ORDER BY veces_citado DESC LIMIT 10
```

---

### 3.3 `MENCIONA`

**Dirección:**
```
(Articulo) -[:MENCIONA]→ (Entidad)
```

**Sin propiedades.**

**Semántica:** el artículo contiene en su texto el nombre de la entidad jurídica. La detección es por **subcadena exacta** (insensible a mayúsculas), no por LLM.

**Cómo se crea:** en el enriquecimiento de `01_extraer_pdf.py`, la función `enriquecer_con_entidades()` itera la lista de ENTIDADES contra el `texto` de cada artículo. Si hay coincidencia, se añade al campo `entidades_mencionadas` del JSON. En `02_construir_grafo.py`, `crear_entidades_y_conceptos()` materializa los nodos `Entidad` y las aristas.

**Uso típico en queries:**
```cypher
// Artículos que regulan a SGEIC
MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad {nombre: 'SGEIC'})
RETURN a.numero, a.titulo

// Qué entidades menciona el artículo 48
MATCH (a:Articulo {id: 'art_48'})-[:MENCIONA]->(e:Entidad)
RETURN e.nombre

// Entidades más mencionadas en la ley
MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad)
RETURN e.nombre, COUNT(a) AS num_articulos
ORDER BY num_articulos DESC

// Artículos que mencionan tanto SGEIC como CNMV
MATCH (a:Articulo)-[:MENCIONA]->(e1:Entidad {nombre: 'SGEIC'})
MATCH (a)-[:MENCIONA]->(e2:Entidad {nombre: 'CNMV'})
RETURN a.numero, a.titulo
```

---

### 3.4 `TRATA_SOBRE`

**Dirección:**
```
(Articulo) -[:TRATA_SOBRE]→ (Concepto)
```

**Sin propiedades.**

**Semántica:** el artículo trata de una materia abstracta (autorización, capital mínimo, supervisión…). La detección es igual que `MENCIONA`: subcadena en el texto.

**Cómo se crea:** misma función `enriquecer_con_entidades()` / `crear_entidades_y_conceptos()` que para `MENCIONA`, pero iterando la lista CONCEPTOS.

**Diferencia con `MENCIONA`:** `MENCIONA` apunta a **figuras jurídicas** (sujetos: SGEIC, CNMV…); `TRATA_SOBRE` apunta a **materias** (objetos: autorización, capital mínimo…). Un mismo artículo puede tener ambos tipos.

**Uso típico en queries:**
```cypher
// Artículos sobre capital mínimo
MATCH (a:Articulo)-[:TRATA_SOBRE]->(c:Concepto {nombre: 'capital mínimo'})
RETURN a.numero, a.titulo

// Todos los conceptos que trata el artículo 65
MATCH (a:Articulo {id: 'art_65'})-[:TRATA_SOBRE]->(c:Concepto)
RETURN c.nombre

// Concepto más regulado
MATCH (a:Articulo)-[:TRATA_SOBRE]->(c:Concepto)
RETURN c.nombre, COUNT(a) AS num_articulos ORDER BY num_articulos DESC
```

---

### 3.5 Aristas opcionales del `LLMGraphTransformer`

Con `USAR_LLM_TRANSFORMER = True`, el LLM extrae relaciones semánticas adicionales entre nodos. El extractor está configurado para generar estos tipos:

| Tipo de arista | Semántica | Ejemplo |
|---|---|---|
| `REQUIERE` | Un concepto o entidad requiere de otro | `(autorización)-[:REQUIERE]->(capital mínimo)` |
| `IMPLICA` | Una obligación implica otra | `(registro)-[:IMPLICA]->(supervisión)` |
| `DEFINE` | Un artículo define un término | `(art_3)-[:DEFINE]->(SGEIC)` |
| `COMPLEMENTA` | Dos artículos se complementan | `(art_42)-[:COMPLEMENTA]->(art_38)` |
| `EXCLUYE` | Una norma excluye a otra | `(art_5)-[:EXCLUYE]->(Entidad X)` |

**Coste:** 1-3 llamadas LLM por artículo. Para la ley completa (~110 artículos) son ~200-300 llamadas. Controlar con `MAX_ARTICULOS_LLM` en `config.py`.

---

## 4. Diagrama del esquema completo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ESQUEMA BASE DEL GRAFO                              │
│                                                                               │
│   ┌─────────┐   PERTENECE_A   ┌──────────┐   PERTENECE_A   ┌────────────┐  │
│   │ Seccion │◄───────────────│ Articulo  │────────────────►│  Capitulo  │  │
│   └─────────┘                 └──────────┘                  └────────────┘  │
│   (id, texto)                  │   ▲  │                      (id, texto)     │
│                         PETENECE_A │  │ PERTENECE_A              │           │
│                                │   │  │                          │PERTENECE_A│
│                                ▼   │  │                          ▼           │
│                             ┌──────────┐                   ┌────────────┐   │
│                             │  Titulo  │                   │   Titulo   │   │
│                             └──────────┘                   └────────────┘   │
│                              (id, texto)                    (id, texto)      │
│                                                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                    Relaciones del Articulo                            │  │
│   │                                                                       │  │
│   │  (Articulo) ──REFERENCIA──► (Articulo)                               │  │
│   │  (Articulo) ──MENCIONA────► (Entidad {nombre})                       │  │
│   │  (Articulo) ──TRATA_SOBRE─► (Concepto {nombre})                      │  │
│   │  (Articulo) ──PERTENECE_A─► (Titulo)                                 │  │
│   │  (Articulo) ──PERTENECE_A─► (Capitulo)                               │  │
│   │                                                                       │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│   Nodo independiente:  (Disposicion {id, titulo, texto})                      │
│   (sin relaciones en la versión base)                                         │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│              ESQUEMA ADICIONAL (con LLMGraphTransformer)                      │
│                                                                               │
│   (Articulo) ──DEFINE────► (Entidad | Concepto | Requisito | Obligacion)     │
│   (Articulo) ──COMPLEMENTA─► (Articulo)                                      │
│   (Concepto) ──REQUIERE───► (Concepto)                                       │
│   (Obligacion) ──IMPLICA──► (Obligacion)                                     │
│   (Articulo) ──EXCLUYE────► (Entidad)                                        │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Índices y constraints en Neo4j

### Constraints de unicidad

| Constraint | Nodo | Propiedad |
|---|---|---|
| `articulo_id` | `Articulo` | `id` |
| `titulo_id` | `Titulo` | `id` |
| `capitulo_id` | `Capitulo` | `id` |
| `seccion_id` | `Seccion` | `id` |
| `entidad_nom` | `Entidad` | `nombre` |
| `concepto_nom` | `Concepto` | `nombre` |
| `disposicion_id` | `Disposicion` | `id` |

Todos usan `MERGE` en la carga, por lo que son idempotentes: ejecutar el script dos veces no duplica nodos.

### Índices fulltext (Lucene)

| Índice | Tipo | Propiedades indexadas | Uso |
|---|---|---|---|
| `articulo_texto` | Fulltext | `Articulo.titulo`, `Articulo.texto` | Búsqueda de texto libre con sintaxis Lucene en el modo `grafo` |
| `entidad_nombre` | Fulltext | `Entidad.nombre` | Búsqueda aproximada de entidades |

**Cómo usar el índice fulltext:**
```cypher
CALL db.index.fulltext.queryNodes('articulo_texto', 'autorización OR SGEIC')
YIELD node, score
WHERE score > 0.3
RETURN node.numero, node.titulo, score
ORDER BY score DESC LIMIT 10
```

---

## 6. Cómo se construye el grafo — pipeline detallado

### 6.1 Paso 1 — Extracción del PDF (`01_extraer_pdf.py`)

```
PDF (fitz/pymupdf)
       │
       ▼  extraer_texto_pdf()
texto bruto con marcadores [PAGINA_N]
       │
       ▼  limpiar_texto()
  - Une líneas rotas dentro de párrafos
  - Elimina espacios múltiples
       │
       ▼  parsear_estructura()
  ┌────────────────────────────────────────────────────┐
  │  Máquina de estados que lee línea a línea:         │
  │                                                    │
  │  estado: [titulo_actual, capitulo_actual,          │
  │           seccion_actual, articulo_abierto]        │
  │                                                    │
  │  Al detectar "TÍTULO IV":                          │
  │    → guarda artículo abierto                       │
  │    → actualiza titulo_actual                       │
  │    → resetea capitulo y sección                    │
  │                                                    │
  │  Al detectar "CAPÍTULO IV":                        │
  │    → guarda artículo abierto                       │
  │    → actualiza capitulo_actual                     │
  │    → resetea sección                               │
  │                                                    │
  │  Al detectar "Artículo 42. ...":                   │
  │    → guarda artículo anterior                      │
  │    → abre nuevo artículo con contexto actual       │
  │                                                    │
  │  Línea sin cabecera:                               │
  │    → acumula en texto del artículo abierto         │
  └────────────────────────────────────────────────────┘
       │
       ▼  enriquecer_con_entidades()
  Para cada artículo:
    entidades_mencionadas = [e for e in ENTIDADES if e.lower() in texto.lower()]
    conceptos_mencionados = [c for c in CONCEPTOS if c.lower() in texto.lower()]
       │
       ▼
  data/articulos.json
  {
    "titulos": [...],
    "capitulos": [...],
    "secciones": [...],
    "articulos": [
      {
        "id": "art_42",
        "numero": "42",
        "titulo": "Requisitos de acceso a la actividad",
        "texto": "Artículo 42. Requisitos...\n1. Las SGEIC...",
        "titulo_padre": "TÍTULO IV",
        "capitulo_padre": "CAPÍTULO I",
        "seccion_padre": "",
        "referencias": ["38", "41", "25"],
        "entidades_mencionadas": ["SGEIC", "CNMV"],
        "conceptos_mencionados": ["autorización", "requisitos"],
        "tipo": "articulo"
      },
      ...
    ],
    "disposiciones": [...]
  }
```

**Patrones regex clave:**

| Elemento | Regex |
|---|---|
| Título | `\bTÍTULO\s+(I{1,3}V?\|VI{0,3}\|IX\|X…)` |
| Capítulo | `\bCAPÍTULO\s+(I{1,3}V?\|VI{0,3}\|IX\|X…)` |
| Sección | `\bSección\s+(\d+[.ª]?)` |
| Artículo | `Artículo\s+(\d+(?:\s+(?:bis\|ter\|…))?)\.\s*([^\n]{3,120})` |
| Disposición | `DISPOSICIÓN\s+(?:ADICIONAL\|FINAL\|TRANSITORIA\|DEROGATORIA)\s+(?:PRIMERA\|…\|\d+)` |
| Referencia | `(?:artículo\|arts?\.)\s+(\d+(?:\s+(?:bis\|ter\|…))?)` |

### 6.2 Paso 2 — Carga en Neo4j (`02_construir_grafo.py`)

Siete fases secuenciales dentro de una misma sesión:

```
[1/5] crear_titulos_capitulos()
      MERGE (t:Titulo  {id: slug, texto: texto})
      MERGE (c:Capitulo {id: slug, texto: texto})
      MERGE (s:Seccion  {id: slug, texto: texto})

[2/5] crear_articulos()
      Para cada artículo del JSON:
        MERGE (a:Articulo {id: ...}) SET a.numero=..., a.titulo=..., a.texto=...
        MERGE (a)-[:PERTENECE_A]->(t:Titulo   {id: slug_titulo_padre})
        MERGE (a)-[:PERTENECE_A]->(c:Capitulo {id: slug_cap_padre})

[3/5] crear_referencias_cruzadas()
      Para cada artículo, para cada ref en art["referencias"]:
        Si ref_id existe en el JSON:
          MERGE (a:Articulo {id: art_id})-[:REFERENCIA]->(b:Articulo {id: ref_id})

[4/5] crear_entidades_y_conceptos()
      Para cada artículo:
        Para cada entidad en art["entidades_mencionadas"]:
          MERGE (e:Entidad {nombre: entidad})
          MERGE (a)-[:MENCIONA]->(e)
        Para cada concepto en art["conceptos_mencionados"]:
          MERGE (c:Concepto {nombre: concepto})
          MERGE (a)-[:TRATA_SOBRE]->(c)

[5/5] crear_disposiciones()
      MERGE (d:Disposicion {id: ...}) SET d.titulo=..., d.texto=...
```

**Nota sobre idempotencia:** todos los comandos usan `MERGE`, no `CREATE`. Ejecutar el script varias veces no duplica nodos ni aristas. Para limpiar completamente y reconstruir usar `--limpiar`.

### 6.3 Paso 3 (opcional) — Enriquecimiento con LLM (`enriquecer_con_llm()`)

```
Para cada artículo (hasta MAX_ARTICULOS_LLM):
  1. Crear Document(page_content=art["texto"])
  2. LLMGraphTransformer.convert_to_graph_documents([doc])
     → El LLM extrae entidades (Requisito, Obligacion, Plazo…)
       y relaciones (REQUIERE, IMPLICA, DEFINE, COMPLEMENTA, EXCLUYE)
  3. Neo4jGraph.add_graph_documents(graph_docs, baseEntityLabel=True)
     → Añade nodos y aristas al grafo existente
```

---

## 7. Cómo se consulta — los dos modos RAG

### 7.1 Modo `cypher` — LangChain genera Cypher

```
python 03_consultar.py --query "¿Requisitos SGEIC?" --modo cypher
                                                    (defecto)
```

**Flujo:**
```
Pregunta en español
        │
        ▼  GraphCypherQAChain.invoke()
   LLM recibe el schema del grafo + few-shot examples + pregunta
        │
        ▼  genera Cypher
   MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad {nombre: 'SGEIC'})
   WHERE a.titulo CONTAINS 'requisito'
   RETURN a.numero, a.titulo, a.texto LIMIT 5
        │
        ▼  Neo4j ejecuta la query
   [{'a.numero': '42', 'a.titulo': '...', 'a.texto': '...'}, ...]
        │
        ▼  LLM genera respuesta
   "Según el artículo 42, los requisitos son..."
```

**Few-shot examples incluidos en el prompt** (guían al LLM en el estilo de Cypher esperado):

| Pregunta | Cypher generado |
|---|---|
| ¿Qué artículos regulan autorización de SGEIC? | `MATCH (a)-[:MENCIONA]→(e {nombre:'SGEIC'}) WHERE a.titulo CONTAINS 'autorización'` |
| ¿Artículos del Capítulo IV? | `MATCH (a)-[:PERTENECE_A]→(c:Capitulo) WHERE c.texto CONTAINS 'IV'` |
| ¿A qué artículos referencia el artículo 42? | `MATCH (a {id:'art_42'})-[:REFERENCIA]→(b) RETURN b.numero, b.titulo` |
| ¿Artículos sobre capital mínimo? | `MATCH (a)-[:TRATA_SOBRE]→(c {nombre:'capital mínimo'})` |
| ¿Qué entidades menciona el artículo 48? | `MATCH (a {id:'art_48'})-[:MENCIONA]→(e) RETURN e.nombre` |

**Cuándo usarlo:** preguntas con estructura clara, referencias a artículos concretos, búsqueda por entidad/concepto específico.

---

### 7.2 Modo `grafo` — recuperación manual + expansión

```
python 03_consultar.py --query "¿Requisitos SGEIC?" --modo grafo
```

**Flujo con 4 estrategias de recuperación en cascada:**

```
Pregunta: "¿Qué requisitos necesita una SGEIC para obtener autorización?"
        │
        ├─ Estrategia 1: Extracción de números de artículo
        │   regex: artículo\s+(\d+...)
        │   → No hay número directo en esta pregunta → vacío
        │
        ├─ Estrategia 2: Entidades en la query
        │   Detecta "SGEIC" en la lista de ENTIDADES conocidas
        │   → MATCH (a)-[:MENCIONA]→(e:Entidad {nombre:'SGEIC'}) LIMIT 10
        │   → Recupera artículos 4, 5, 42, 48, 65...
        │
        ├─ Estrategia 3: Búsqueda FULLTEXT
        │   Palabras > 4 caracteres: ['necesita', 'SGEIC', 'obtener', 'autorización']
        │   Lucene query: "necesita OR SGEIC OR obtener OR autorización OR requisi"
        │   → CALL db.index.fulltext.queryNodes('articulo_texto', ...) WHERE score > 0.3
        │   → Añade artículos 38, 41...
        │
        └─ Estrategia 4: Expansión por referencias
            Ids base: ['art_4', 'art_5', 'art_42', 'art_48', 'art_38', 'art_41']
            → MATCH (a)-[:REFERENCIA]→(b) WHERE a.id IN ids LIMIT 15
            → Añade artículos que los anteriores citan: art_25, art_30...
                    │
                    ▼
           5-15 artículos con texto completo (~12.000 chars de contexto)
                    │
                    ▼
           LLM genera respuesta final citando número de artículo
```

**Cuándo usarlo:** preguntas abiertas o exploratorias, cuando no se conoce el número del artículo, cuando se quiere que el grafo "expanda" el contexto automáticamente.

---

## 8. Catálogo completo de queries Cypher

### Exploración básica del grafo

```cypher
-- Estadísticas generales
MATCH (n) RETURN labels(n)[0] AS tipo, COUNT(n) AS total ORDER BY total DESC

-- Ver el schema
CALL db.schema.visualization()

-- Primeros 5 artículos
MATCH (a:Articulo) RETURN a.numero, a.titulo LIMIT 5

-- Todas las entidades del grafo
MATCH (e:Entidad) RETURN e.nombre ORDER BY e.nombre

-- Todos los conceptos
MATCH (c:Concepto) RETURN c.nombre ORDER BY c.nombre
```

### Navegación jerárquica

```cypher
-- Estructura completa: Títulos → Capítulos → número de artículos
MATCH (a:Articulo)-[:PERTENECE_A]->(c:Capitulo)-[:PERTENECE_A]->(t:Titulo)
RETURN t.texto AS titulo, c.texto AS capitulo, COUNT(a) AS num_articulos
ORDER BY t.texto, c.texto

-- Todos los artículos de un Título concreto
MATCH (a:Articulo)-[:PERTENECE_A]->(t:Titulo)
WHERE t.texto CONTAINS 'IV'
RETURN a.numero, a.titulo ORDER BY toInteger(a.numero)

-- Artículos de un Capítulo concreto
MATCH (a:Articulo)-[:PERTENECE_A]->(c:Capitulo)
WHERE c.texto CONTAINS 'CAPÍTULO I'
RETURN a.numero, a.titulo

-- Artículos de una Sección (búsqueda por propiedad del nodo Articulo)
MATCH (a:Articulo)
WHERE a.seccion_padre CONTAINS 'Sección 1'
RETURN a.numero, a.titulo
```

### Referencias cruzadas

```cypher
-- Artículos citados por el artículo 42
MATCH (a:Articulo {id: 'art_42'})-[:REFERENCIA]->(b:Articulo)
RETURN b.numero, b.titulo

-- Artículos que citan al artículo 42
MATCH (a:Articulo)-[:REFERENCIA]->(b:Articulo {id: 'art_42'})
RETURN a.numero, a.titulo

-- Red de 2 saltos desde artículo 42
MATCH (a:Articulo {id: 'art_42'})-[:REFERENCIA*1..2]->(b:Articulo)
RETURN DISTINCT b.numero, b.titulo

-- Artículos más citados (hubs de la ley)
MATCH (a:Articulo)-[:REFERENCIA]->(b:Articulo)
RETURN b.numero, b.titulo, COUNT(a) AS veces_citado
ORDER BY veces_citado DESC LIMIT 10

-- Artículos que citan y son citados (nodos bidireccionales)
MATCH (a:Articulo)-[:REFERENCIA]->(b:Articulo)
WHERE (b)-[:REFERENCIA]->(a)
RETURN a.numero, b.numero

-- Componentes de la red de referencias
CALL gds.wcc.stream({
  nodeProjection: 'Articulo',
  relationshipProjection: 'REFERENCIA'
})
YIELD nodeId, componentId
RETURN componentId, COUNT(nodeId) AS tam ORDER BY tam DESC
```

### Entidades y conceptos

```cypher
-- Artículos que mencionan SGEIC
MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad {nombre: 'SGEIC'})
RETURN a.numero, a.titulo

-- Entidades más mencionadas
MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad)
RETURN e.nombre, COUNT(a) AS num_articulos ORDER BY num_articulos DESC

-- Artículos que mencionan a la vez SGEIC y CNMV
MATCH (a:Articulo)-[:MENCIONA]->(e1:Entidad {nombre: 'SGEIC'})
MATCH (a)-[:MENCIONA]->(e2:Entidad {nombre: 'CNMV'})
RETURN a.numero, a.titulo

-- Artículos sobre 'autorización' que mencionan 'SGEIC'
MATCH (a:Articulo)-[:TRATA_SOBRE]->(c:Concepto {nombre: 'autorización'})
MATCH (a)-[:MENCIONA]->(e:Entidad {nombre: 'SGEIC'})
RETURN a.numero, a.titulo, a.texto

-- Concepto más regulado
MATCH (a:Articulo)-[:TRATA_SOBRE]->(c:Concepto)
RETURN c.nombre, COUNT(a) AS freq ORDER BY freq DESC
```

### Búsqueda fulltext

```cypher
-- Artículos con 'capital mínimo' en el texto (fulltext Lucene)
CALL db.index.fulltext.queryNodes('articulo_texto', 'capital mínimo')
YIELD node, score
RETURN node.numero, node.titulo, score ORDER BY score DESC LIMIT 10

-- Búsqueda OR
CALL db.index.fulltext.queryNodes('articulo_texto', 'autorización OR registro')
YIELD node, score
WHERE score > 0.5
RETURN node.numero, node.titulo, score ORDER BY score DESC

-- Búsqueda exacta (entre comillas en Lucene)
CALL db.index.fulltext.queryNodes('articulo_texto', '"política de inversión"')
YIELD node, score
RETURN node.numero, node.titulo
```

### Visualización del grafo

```cypher
-- Mapa del grafo (muestra 50 relaciones)
MATCH (a:Articulo)-[r]->(b)
RETURN a, r, b LIMIT 50

-- Subgrafo del artículo 42 y sus vecinos
MATCH path=(a:Articulo {id: 'art_42'})-[*1..2]-(b)
RETURN path LIMIT 30

-- Red de referencias cruzadas (solo Articulo→Articulo)
MATCH (a:Articulo)-[r:REFERENCIA]->(b:Articulo)
RETURN a, r, b LIMIT 100
```

### Disposiciones

```cypher
-- Todas las disposiciones
MATCH (d:Disposicion) RETURN d.id, d.titulo ORDER BY d.titulo

-- Buscar en disposiciones
MATCH (d:Disposicion)
WHERE toLower(d.texto) CONTAINS 'sgeic'
RETURN d.titulo, d.texto
```

---

## 9. Volumetría esperada

Para la Ley 22/2014 completa (BOE-A-2014-11714-consolidado.pdf, ~109 páginas):

| Elemento | Cantidad aproximada |
|---|---|
| Páginas del PDF | ~109 |
| Títulos | 6-8 |
| Capítulos | 15-25 |
| Secciones | 10-20 |
| Artículos | 90-120 |
| Disposiciones | 15-25 |
| Nodos `Entidad` | 10-15 |
| Nodos `Concepto` | 15-20 |
| **Total nodos** | **~170-220** |
| Aristas `PERTENECE_A` | ~200 (2 por artículo promedio) |
| Aristas `REFERENCIA` | ~300-500 |
| Aristas `MENCIONA` | ~200-400 |
| Aristas `TRATA_SOBRE` | ~300-600 |
| **Total aristas (base)** | **~1.000-1.800** |

Con `LLMGraphTransformer` activo, los nodos y aristas pueden multiplicarse por 3-5x.

---

## 10. Limitaciones conocidas y extensiones posibles

### Limitaciones actuales

| Limitación | Causa | Impacto |
|---|---|---|
| Texto truncado a 8.000 chars por artículo en Neo4j | Límite de propiedad Neo4j | Artículos muy largos pueden perder texto en el grafo |
| Las disposiciones no tienen `PERTENECE_A` | No tienen estructura jerárquica en el PDF | No se pueden filtrar por título/capítulo |
| Las disposiciones no generan `REFERENCIA` automáticamente | El enriquecimiento de referencias solo se hace sobre artículos | Se pierden las referencias desde disposiciones |
| Detección de entidades por subcadena exacta | Sin NLP/NER | Variantes morfológicas ("las SGEICs", "SGEIC/ECR") pueden no detectarse |
| IDs de nodo `Titulo`/`Capitulo` son slugs del texto | La función `_id_seguro()` trunca a 60 chars y elimina acentos | Puede haber colisiones si dos cabeceras tienen el mismo inicio |
| `seccion_padre` se guarda como propiedad, no como arista | Decisión de diseño para simplificar | No se puede hacer `MATCH (a)-[:PERTENECE_A]->(s:Seccion)` |
| Sin embeddings vectoriales | El proyecto es puramente simbólico | Preguntas con paráfrasis pueden no encontrar artículos relevantes |

### Extensiones posibles

| Extensión | Descripción |
|---|---|
| Añadir embeddings a los nodos `Articulo` | Combinar búsqueda vectorial + grafo (hybrid search) |
| Arista `PERTENECE_A` desde `Disposicion` | Contextualizar las disposiciones en su sección del documento |
| NER con spaCy para entidades | Detectar organismos, plazos, importes con NLP real |
| Paginación en el nodo `Articulo` | Añadir propiedad `pagina` para enlazar con el PDF original |
| Grafo temporal | Añadir propiedad `vigente_desde`/`derogado_por` para versiones históricas de artículos |
| Múltiples documentos | Añadir propiedad `fuente` a los nodos para indexar múltiples leyes relacionadas con relaciones `DEROGA`, `MODIFICA`, `DESARROLLA` entre documentos |
| Índice vectorial en Neo4j | Usar `db.index.vector` de Neo4j ≥ 5.11 para búsqueda semántica nativa |
