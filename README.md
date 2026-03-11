# Graph RAG — Ley 22/2014 (BOE) con Neo4j + LangChain

RAG con grafo de conocimiento para documentos legales españoles.  
El grafo modela la jerarquía de la ley y las referencias cruzadas entre artículos.

---

## Arquitectura

```
PDF
 │
 ▼
01_extraer_pdf.py   →  data/articulos.json
                            │
                            ▼
                    02_construir_grafo.py  →  Neo4j
                                                │
                                                ▼
                                        03_consultar.py  →  Respuesta
```

**El grafo contiene:**

| Nodo | Qué representa |
|---|---|
| `Articulo` | Cada artículo con su texto completo |
| `Titulo` / `Capitulo` / `Seccion` | Estructura jerárquica |
| `Entidad` | SGEIC, CNMV, ECR… |
| `Concepto` | autorización, capital mínimo… |
| `Disposicion` | Disposiciones adicionales/finales |

| Relación | Significado |
|---|---|
| `PERTENECE_A` | Artículo → Capítulo → Título |
| `REFERENCIA` | "conforme al artículo X" |
| `MENCIONA` | Artículo → Entidad |
| `TRATA_SOBRE` | Artículo → Concepto |

---

## Instalación

### 1. Neo4j (con Docker)

```powershell
docker run `
  --name neo4j-ley `
  -p 7474:7474 -p 7687:7687 `
  -e NEO4J_AUTH=neo4j/password `
  -d neo4j:5
```

Accede al navegador en http://localhost:7474  
Usuario: `neo4j` / Contraseña: `password`

> Sin Docker: descarga Neo4j Desktop desde https://neo4j.com/download/

### 2. Paquetes Python

```powershell
cd rag_grafos
pip install -r requirements.txt
```

### 3. Configurar `config.py`

Abre [config.py](config.py) y revisa:
- `NEO4J_PASSWORD` — la que pusiste al arrancar Neo4j
- `PDF_PATH` — ruta al PDF del BOE
- `OPENAI_API_KEY` — tu clave de OpenAI
- `USE_VECTOR_INDEX` — si tu Neo4j soporta índices vectoriales (Enterprise/Aura), ponlo en `True`. Si usas Neo4j Community, déjalo en `False` (el sistema usará un fallback local en Python).

---

## Uso paso a paso

### Paso 1 — Extraer y parsear el PDF

```powershell
python 01_extraer_pdf.py
```

Genera `data/articulos.json` con todos los artículos, su jerarquía y referencias cruzadas.

### Paso 2 — Construir el grafo en Neo4j

```powershell
python 02_construir_grafo.py
```

Si ya tienes un grafo anterior y quieres rehacerlo desde cero:

```powershell
python 02_construir_grafo.py --limpiar
```

### Paso 3 — Consultar

**Modo interactivo:**
```powershell
python 03_consultar.py
```

**Pregunta directa:**
```powershell
python 03_consultar.py --query "¿Qué requisitos necesita una SGEIC para obtener autorización?"
```

**Modo grafo** (más robusto para preguntas abiertas):
```powershell
python 03_consultar.py --query "¿Cuáles son las obligaciones de los depositarios?" --modo grafo
```

**Ver el Cypher / nodos recuperados:**
```powershell
python 03_consultar.py --query "..." --verbose
```

---

## Dos modos de recuperación

### `--modo cypher` (defecto)
El LLM genera una consulta Cypher a partir de la pregunta, la ejecuta en Neo4j y usa el resultado para responder.

```
Pregunta  →  LLM genera Cypher  →  Neo4j ejecuta  →  resultados  →  LLM responde
```

Mejor para: preguntas estructuradas, búsqueda por artículo concreto, relaciones explícitas.

### `--modo grafo`
Recupera artículos por múltiples estrategias y expande el grafo.

```
Pregunta
  ├─ números de artículo detectados  → búsqueda directa en grafo
  ├─ entidades detectadas (SGEIC…)   → artículos que las mencionan
  ├─ palabras clave                  → FULLTEXT search
  └─ expansión                       → artículos referenciados por los encontrados
           │
           ▼
      contexto enriquecido  →  LLM responde
```

Mejor para: preguntas abiertas, cuando no sabes el número de artículo.

---

## Ejemplo de queries Cypher en Neo4j Browser

```cypher
// Ver artículos que mencionan SGEIC
MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad {nombre: 'SGEIC'})
RETURN a.numero, a.titulo LIMIT 20

// Ver referencias cruzadas
MATCH (a:Articulo)-[:REFERENCIA]->(b:Articulo)
RETURN a.numero, b.numero LIMIT 30

// Ver estructura del Capítulo IV
MATCH (a:Articulo)-[:PERTENECE_A]->(c:Capitulo)
WHERE c.texto CONTAINS 'IV'
RETURN a.numero, a.titulo ORDER BY a.numero

// Artículos que tratan sobre autorización
MATCH (a:Articulo)-[:TRATA_SOBRE]->(c:Concepto {nombre: 'autorización'})
RETURN a.numero, a.titulo
```

---

## Mejoras opcionales

**Extracción de entidades con LLM:**  
En `config.py`, pon `USAR_LLM_TRANSFORMER = True`.  
Añade al grafo relaciones semánticas como `REQUIERE`, `IMPLICA`, `DEFINE` extraídas por el LLM.  
Controla el coste con `MAX_ARTICULOS_LLM = 30`.

**Modelo más potente:**  
Cambia `LLM_MODEL = "gpt-4o"` en `config.py` para mejor calidad en generación de Cypher.
