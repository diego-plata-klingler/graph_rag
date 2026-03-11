# Graph RAG — Cómo funciona el grafo de conocimiento

## El problema que resuelve

Un RAG clásico divide el documento en trozos de texto (*chunks*) y busca por similitud vectorial. Si la respuesta a una pregunta está en 3 artículos distintos que se referencian entre sí, un chunk nunca contiene esa información completa. El grafo lo resuelve modelando explícitamente las **relaciones** entre esos artículos.

---

## Los nodos — unidades de conocimiento

Cada nodo es una entidad del mundo jurídico. En este proyecto hay 7 tipos:

| Nodo | Qué representa | Propiedades clave |
|---|---|---|
| `Articulo` | Cada artículo de la ley | `id`, `numero`, `titulo`, `texto` (completo) |
| `Titulo` | "TÍTULO I — Disposiciones generales" | `id`, `texto` |
| `Capitulo` | "CAPÍTULO IV — Autorización de SGEICs" | `id`, `texto` |
| `Seccion` | "Sección 1ª — Requisitos" | `id`, `texto` |
| `Entidad` | Una figura jurídica o institucional (SGEIC, CNMV, ECR…) | `nombre` |
| `Concepto` | Materia regulada (autorización, capital mínimo…) | `nombre` |
| `Disposicion` | Disposiciones adicionales, finales, transitorias | `id`, `titulo`, `texto` |

> El nodo `Entidad` es especialmente útil: el mismo nodo "SGEIC" se conecta a los ~40 artículos que la mencionan. Basta con encontrar ese nodo para tener todos los artículos relevantes de golpe.

---

## Las relaciones — conexiones entre nodos

### `PERTENECE_A`
```
(Articulo) -[:PERTENECE_A]→ (Capitulo)
(Articulo) -[:PERTENECE_A]→ (Titulo)
```
Define la jerarquía del documento. Permite preguntar: *"dame todos los artículos del Capítulo IV"*.

---

### `REFERENCIA`
```
(Articulo) -[:REFERENCIA]→ (Articulo)
```
Se crea cuando un artículo dice *"conforme al artículo 38"* o *"véase artículo 41"*.  
Es la relación más valiosa del grafo. Permite:
- Preguntar: *"¿qué artículos complementan al artículo 42?"*
- **Expansión automática**: recuperado el art. 42, se traen el 38 y el 41 automáticamente.

---

### `MENCIONA`
```
(Articulo) -[:MENCIONA]→ (Entidad)
```
Agrupa todos los artículos que regulan una misma figura jurídica.  
Si preguntas por SGEIC, encuentras sus ~40 artículos sin necesidad de conocer sus números.

---

### `TRATA_SOBRE`
```
(Articulo) -[:TRATA_SOBRE]→ (Concepto)
```
Similar a `MENCIONA` pero para materias abstractas.  
`"capital mínimo"` → encuentra art. 48, art. 65, art. 72…

---

## Cómo se construye — el pipeline

```
PDF (109 páginas de texto)
        │
        ▼
   PASO 1 — Parser regex   (01_extraer_pdf.py)

   Lee el texto línea a línea y detecta cabeceras por patrón:
   ┌─────────────────────────────────────────────────────┐
   │  "TÍTULO III"         → nuevo nodo Titulo           │
   │  "CAPÍTULO IV"        → nuevo nodo Capitulo         │
   │  "Artículo 42. ..."   → nuevo nodo Articulo         │
   │  "DISPOSICIÓN FINAL"  → nuevo nodo Disposicion      │
   │  "conforme al art. 38"→ arista REFERENCIA           │
   │  texto que sigue      → acumulado en el nodo actual │
   └─────────────────────────────────────────────────────┘

   Resultado: data/articulos.json
   → artículos con texto completo + jerarquía + referencias
        │
        ▼
   PASO 2 — Carga en Neo4j   (02_construir_grafo.py)

   Para cada artículo:
     MERGE (:Articulo {id: 'art_42', numero: '42', titulo: '...', texto: '...'})
     MERGE (art_42)-[:PERTENECE_A]->(CAPÍTULO_IV)
     MERGE (art_42)-[:REFERENCIA]->(art_38)
     MERGE (art_42)-[:MENCIONA]->(:Entidad {nombre: 'SGEIC'})
        │
        ▼
   Neo4j — base de datos de grafos
   Accesible en http://localhost:7474
```

---

## Cómo se consulta — los dos modos

### Modo `--modo cypher` (defecto)

```
Pregunta en español
        │
        ▼
   LLM convierte la pregunta en Cypher
   (lenguaje de consulta de grafos, como SQL pero para nodos y aristas)

   Ejemplo:
   "¿Qué artículos regulan la autorización de SGEIC?"
         ↓ genera
   MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad {nombre: 'SGEIC'})
   WHERE a.titulo CONTAINS 'autorización'
   RETURN a.numero, a.titulo, a.texto LIMIT 5
        │
        ▼
   Neo4j ejecuta → devuelve filas de resultados
        │
        ▼
   LLM genera respuesta con esos resultados como contexto
```

Mejor para: preguntas estructuradas, búsqueda por artículo concreto, relaciones explícitas.

---

### Modo `--modo grafo` (expansión manual)

```
Pregunta: "¿Qué requisitos necesita una SGEIC para obtener autorización?"
        │
        ├──[1] Detecta "SGEIC"          → busca Articulo -[:MENCIONA]→ SGEIC
        ├──[2] Detecta palabras clave    → fulltext search en titulo + texto
        ├──[3] Detecta "artículo 42"    → búsqueda directa por id
        └──[4] EXPANSIÓN de grafo       → para cada artículo encontrado,
                                          trae los que referencia (REFERENCIA)
                        │
                        ▼
               5–15 artículos reales con texto completo
                        │
                        ▼
               LLM genera respuesta
```

Mejor para: preguntas abiertas, cuando no se conoce el número de artículo.

---

## Comparativa con PageIndex

| | PageIndex (LLM detecta estructura) | Este grafo (regex detecta estructura) |
|---|---|---|
| Localización del texto | LLM adivina → `start_index: 12` (incorrecto) | Texto completo en el nodo, exacto |
| Pregunta sobre SGEIC | Busca en árbol por similitud | `MATCH -[:MENCIONA]->(SGEIC)` — exacto |
| Referencias cruzadas | No las modela | Arista `REFERENCIA` explícita |
| Artículos distribuidos en la ley | Solo los chunks más similares | Nodo `Entidad` agrupa todos |
| Coste de construcción | Muchas llamadas LLM | **0 llamadas LLM** |

---

## Queries de exploración en Neo4j Browser

```cypher
// Artículos que mencionan SGEIC
MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad {nombre: 'SGEIC'})
RETURN a.numero, a.titulo

// Referencias cruzadas del artículo 42
MATCH (a:Articulo {id: 'art_42'})-[:REFERENCIA]->(b:Articulo)
RETURN b.numero, b.titulo

// Todos los artículos del Capítulo IV
MATCH (a:Articulo)-[:PERTENECE_A]->(c:Capitulo)
WHERE c.texto CONTAINS 'IV'
RETURN a.numero, a.titulo ORDER BY a.numero

// Artículos que tratan sobre 'autorización'
MATCH (a:Articulo)-[:TRATA_SOBRE]->(c:Concepto {nombre: 'autorización'})
RETURN a.numero, a.titulo

// Mapa completo del grafo (muestra estructura visual)
MATCH (a:Articulo)-[r]->(b)
RETURN a, r, b LIMIT 50
```
