"""
Script 3: Consulta el grafo con lenguaje natural.

Tiene dos modos de recuperación:

  --modo cypher  (defecto)
      LangChain genera Cypher a partir de la pregunta, lo ejecuta en Neo4j
      y usa los resultados como contexto para responder.
      Bueno para preguntas estructuradas ("¿qué artículos regulan X?").

  --modo grafo
      Busca artículos por coincidencia de texto, luego expande el grafo
      (artículos referenciados, misma sección) para obtener más contexto.
      Más robusto para preguntas abiertas.

  --modo vector
      Búsqueda semántica por embeddings (vector search en Neo4j).

Uso:
    python 03_consultar.py
    python 03_consultar.py --query "¿Qué requisitos necesita una SGEIC?"
    python 03_consultar.py --query "..." --modo grafo
    python 03_consultar.py --query "..." --verbose   (muestra Cypher / nodos)
"""

import os
import re
import sys
import math
import argparse
import unicodedata

from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OPENAI_API_KEY, LLM_MODEL, EMBED_MODEL, USE_VECTOR_INDEX,
)

# En Windows el terminal por defecto usa cp1252 y puede fallar al imprimir emojis.
# Reconfiguramos stdout/stderr a UTF-8 para evitar UnicodeEncodeError.
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
MAX_ARTICULOS_CONTEXTO = 12
MAX_CHARS_CONTEXTO = 15000
MAX_CHUNKS_RELEVANTES = 10
MAX_CHUNKS_POR_ARTICULO = 2
FULLTEXT_CHUNK_CANDIDATES = 30
FULLTEXT_ARTICULO_CANDIDATES = 12
RERANK_TOP_ARTICULOS = 6
NO_CONTEXT_RESPONSE = "NO ENCONTRADO EN EL CONTEXTO"

_STOPWORDS_RERANK = {
    "a", "al", "algo", "ante", "art", "articulo", "artículo", "bajo", "como", "con",
    "contra", "cual", "cuál", "cuando", "de", "del", "desde", "donde", "dónde", "el",
    "en", "entre", "es", "esa", "ese", "esta", "este", "esto", "hay", "la", "las", "le",
    "ley", "lo", "los", "mas", "más", "me", "mi", "mis", "necesita", "necesitan", "o",
    "para", "pero", "por", "porque", "puede", "pueden", "que", "qué", "quien", "quién",
    "regula", "regulan", "se", "segun", "según", "ser", "si", "sí", "sin", "sobre", "su",
    "sus", "tiene", "tienen", "un", "una", "uno", "unas", "unos", "y", "ya",
}

_SIGLAS_ARTICULOS = {
    "SCR": ["art_26"],
    "FCR": ["art_30"],
    "SICC": ["art_38"],
    "FICC": ["art_38"],
    "SGEIC": ["art_41"],
    "ECR": ["art_3"],
    "EICC": ["art_4"],
    "EICCP": ["art_4_bis"],
    "ECR-PYME": ["art_20", "art_21", "art_22", "art_23"],
}

_TERMINOS_ARTICULOS = {
    "sociedad de capital-riesgo": ["art_26"],
    "sociedad de capital riesgo": ["art_26"],
    "fondo de capital-riesgo": ["art_30"],
    "fondo de capital riesgo": ["art_30"],
    "sociedad gestora de entidades de inversion colectiva de tipo cerrado": ["art_41"],
    "sociedad gestora de entidades de inversión colectiva de tipo cerrado": ["art_41"],
    "entidad de capital-riesgo": ["art_3"],
    "entidad de capital riesgo": ["art_3"],
    "entidades de capital-riesgo-pyme": ["art_20", "art_21", "art_22", "art_23"],
    "entidades de capital riesgo pyme": ["art_20", "art_21", "art_22", "art_23"],
    "ecr-pyme": ["art_20", "art_21", "art_22", "art_23"],
    "ecr pyme": ["art_20", "art_21", "art_22", "art_23"],
    "entidad de inversion colectiva de tipo cerrado": ["art_4"],
    "entidad de inversión colectiva de tipo cerrado": ["art_4"],
    "entidad de inversion colectiva de tipo cerrado de prestamos": ["art_4_bis"],
    "entidad de inversión colectiva de tipo cerrado de préstamos": ["art_4_bis"],
}

LEGAL_SYNONYMS = {
    "sgeic": ["sociedad gestora", "sociedades gestoras"],
    "ecr": ["entidades de capital-riesgo", "capital riesgo"],
    "ecr pyme": ["ecr-pyme", "capital riesgo pyme"],
    "ecr-pyme": ["ecr pyme", "capital riesgo pyme"],
    "eicc": ["entidades de inversión colectiva de tipo cerrado"],
    "excluye": ["entidades excluidas"],
    "excluidas": ["entidades excluidas"],
    "coeficiente": ["porcentaje de inversión", "ratio de inversión"],
    "comercialización": ["venta", "distribución"],
    "comercializacion": ["venta", "distribución"],
}

ARTICLE_TITLE_HINTS = {
    "excluye": "entidades excluidas",
    "excluidas": "entidades excluidas",
    "actividad": "actividad principal",
    "funciones": "funciones",
    "definicion": "definición",
    "definición": "definición",
    "define": "definición",
    "significa": "concepto",
    "que es": "concepto",
    "qué es": "concepto",
}

_PATRONES_DEFINICION = (
    "que es", "que son", "qué es", "qué son", "definicion", "definición",
    "concepto", "significa", "objeto", "naturaleza", "regimen juridico",
    "régimen jurídico", "formas juridicas", "formas jurídicas",
)
_PATRONES_COMPARATIVA = (
    "diferencia", "diferencias", "compar", "vs", "versus", "frente a",
    "relacion entre", "relación entre", "distincion", "distinción",
)
_PATRONES_CONDICIONES = (
    "requisito", "requisitos", "condicion", "condiciones", "autorizacion",
    "autorización", "acceso", "puede", "pueden", "debe", "deben", "limite",
    "límite", "prohib", "obligacion", "obligación", "obligaciones",
)

_RERANKER = None
_RERANKER_FAILED = False

# ──────────────────────────────────────────────────────────────────────────────
# Conexiones
# ──────────────────────────────────────────────────────────────────────────────

class _LLMPayloadLogger(BaseCallbackHandler):
    """Imprime por terminal los mensajes que se envían al modelo de chat."""

    def __init__(self, max_chars: int = 20000):
        self.max_chars = max_chars

    def _content_to_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            bloques = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    bloques.append(str(item["text"]))
                else:
                    bloques.append(str(item))
            return "\n".join(bloques)
        return str(content)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        recorte = len(text) - self.max_chars
        return text[:self.max_chars] + f"\n...[truncado {recorte} caracteres]"

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        print("\n" + "=" * 80)
        print("LOG LLM - Mensajes enviados al modelo")
        print("=" * 80)
        for i, prompt in enumerate(messages, 1):
            if len(messages) > 1:
                print(f"\n--- Prompt {i} ---")
            for msg in prompt:
                role = getattr(msg, "type", msg.__class__.__name__)
                content = self._truncate(self._content_to_text(getattr(msg, "content", "")))
                print(f"\n[{role}]")
                print(content)
        print("=" * 80 + "\n")


_LLM_PAYLOAD_LOGGER = _LLMPayloadLogger()


def get_llm(log_llm: bool = False):
    callbacks = [_LLM_PAYLOAD_LOGGER] if log_llm else None
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
        callbacks=callbacks,
    )


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# Schema manual del grafo (evita la llamada a apoc.meta.data() que requiere el plugin APOC)
_MANUAL_SCHEMA = """
Node properties:
- Articulo: id (STRING), numero (STRING), titulo (STRING), texto (STRING), titulo_padre (STRING), capitulo_padre (STRING), seccion_padre (STRING)
- Chunk: id (STRING), texto (STRING), orden (INTEGER), token_count (INTEGER), articulo_id (STRING), articulo_numero (STRING), articulo_titulo (STRING), titulo_padre (STRING), capitulo_padre (STRING), seccion_padre (STRING), disposicion_id (STRING), disposicion_titulo (STRING)
- Titulo: id (STRING), texto (STRING)
- Capitulo: id (STRING), texto (STRING)
- Seccion: id (STRING), texto (STRING)
- Entidad: nombre (STRING)
- Concepto: nombre (STRING)
- Disposicion: id (STRING), titulo (STRING), texto (STRING)

Relationships:
- (:Articulo)-[:PERTENECE_A]->(:Titulo)
- (:Articulo)-[:PERTENECE_A]->(:Capitulo)
- (:Articulo)-[:PERTENECE_A]->(:Seccion)
- (:Articulo)-[:TIENE_CHUNK]->(:Chunk)
- (:Disposicion)-[:TIENE_CHUNK]->(:Chunk)
- (:Articulo)-[:REFERENCIA]->(:Articulo)
- (:Articulo)-[:MENCIONA]->(:Entidad)
- (:Articulo)-[:TRATA_SOBRE]->(:Concepto)
"""


def get_lang_graph():
    g = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        refresh_schema=False,
    )
    g.schema = _MANUAL_SCHEMA
    return g


# ──────────────────────────────────────────────────────────────────────────────
# MODO A: GraphCypherQAChain
# ──────────────────────────────────────────────────────────────────────────────

# Ejemplos few-shot para guiar la generación de Cypher
FEW_SHOT_CYPHER = """
Ejemplos de preguntas y consultas Cypher para este grafo:

Pregunta: ¿Qué regula la Ley 22/2014 según su artículo 1?
Cypher: MATCH (a:Articulo)
        WHERE a.id = 'art_1'
        RETURN a.numero, a.titulo, a.texto

Pregunta: ¿Qué artículos regulan la autorización de SGEIC?
Cypher: MATCH (a:Articulo)-[:TIENE_CHUNK]->(c:Chunk)
        WHERE (c.texto CONTAINS 'SGEIC' OR c.texto CONTAINS 'sociedad gestora')
          AND (c.texto CONTAINS 'autorización' OR c.texto CONTAINS 'requisito' OR c.texto CONTAINS 'acceso a la actividad')
        RETURN a.numero, a.titulo, c.texto AS fragmento LIMIT 5

Pregunta: ¿Qué artículos están en el Capítulo IV?
Cypher: MATCH (a:Articulo)-[:PERTENECE_A]->(c:Capitulo)
        WHERE c.texto CONTAINS 'IV'
        RETURN a.numero, a.titulo ORDER BY a.numero LIMIT 20

Pregunta: ¿A qué artículos hace referencia el artículo 42?
Cypher: MATCH (a:Articulo)-[:REFERENCIA]->(b:Articulo)
        WHERE a.id = 'art_42'
        RETURN b.numero, b.titulo

Pregunta: ¿Qué artículos tratan sobre capital mínimo?
Cypher: MATCH (a:Articulo)-[:TIENE_CHUNK]->(c:Chunk)
        WHERE c.texto CONTAINS 'capital mínimo' OR c.texto CONTAINS 'capital minimo'
        RETURN a.numero, a.titulo, c.texto AS fragmento LIMIT 3

Pregunta: ¿Cuál es el patrimonio mínimo de un FCR?
Cypher: MATCH (a:Articulo)-[:TIENE_CHUNK]->(c:Chunk)
        WHERE c.texto CONTAINS 'patrimonio' AND (c.texto CONTAINS 'FCR' OR c.texto CONTAINS 'fondo de capital')
        RETURN a.numero, a.titulo, c.texto AS fragmento LIMIT 3

Pregunta: ¿Qué entidades menciona el artículo 48?
Cypher: MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad)
        WHERE a.id = 'art_48'
        RETURN e.nombre

Pregunta: ¿Qué dice el artículo 26?
Cypher: MATCH (a:Articulo)
        WHERE a.id = 'art_26'
        RETURN a.numero, a.titulo, a.texto

Pregunta: ¿Qué significan las siglas SCR, FCR, SICC y FICC y en qué artículo se definen?
Cypher: MATCH (a:Articulo)-[:TIENE_CHUNK]->(c:Chunk)
        WHERE a.titulo CONTAINS 'Definición'
           OR a.titulo CONTAINS 'Régimen jurídico'
           OR a.titulo CONTAINS 'SICC'
           OR a.titulo CONTAINS 'FICC'
           OR c.texto CONTAINS 'SCR'
           OR c.texto CONTAINS 'FCR'
           OR c.texto CONTAINS 'SICC'
           OR c.texto CONTAINS 'FICC'
        RETURN a.numero, a.titulo, c.texto AS fragmento
        ORDER BY a.numero ASC
        LIMIT 5

Pregunta: ¿Qué es una SGEIC y en qué artículo se define?
Cypher: MATCH (a:Articulo)-[:TIENE_CHUNK]->(c:Chunk)
        WHERE a.titulo CONTAINS 'sociedad gestora'
           OR a.titulo CONTAINS 'SGEIC'
           OR a.titulo CONTAINS 'Definición'
           OR (c.texto CONTAINS 'SGEIC' AND a.titulo CONTAINS 'Requisitos')
        RETURN a.numero, a.titulo, c.texto AS fragmento
        ORDER BY a.numero ASC
        LIMIT 5

Pregunta: ¿Cuáles son las condiciones de acceso a la actividad de una SGEIC?
Cypher: MATCH (a:Articulo)
        WHERE a.titulo CONTAINS 'acceso'
           OR a.titulo CONTAINS 'Requisitos'
           OR a.titulo CONTAINS 'Condiciones de acceso'
        RETURN a.numero, a.titulo, a.texto
        ORDER BY a.numero ASC
        LIMIT 5

Pregunta: Diferencia entre ECR y EICC
Cypher: MATCH (a:Articulo)-[:TIENE_CHUNK]->(c:Chunk)
        WHERE a.titulo CONTAINS 'Entidades de capital-riesgo'
           OR a.titulo CONTAINS 'Entidades de inversión colectiva'
           OR a.titulo CONTAINS 'Definición y régimen jurídico'
           OR a.titulo CONTAINS 'Régimen jurídico'
           OR c.texto CONTAINS 'ECR'
           OR c.texto CONTAINS 'EICC'
        RETURN a.numero, a.titulo, c.texto AS fragmento
        ORDER BY a.numero ASC
        LIMIT 6

Pregunta: ¿Qué son las ECR? ¿Cuál es su objeto?
Cypher: MATCH (a:Articulo)-[:TIENE_CHUNK]->(c:Chunk)
        WHERE a.titulo CONTAINS 'Objeto'
           OR a.titulo CONTAINS 'Definición'
           OR (c.texto CONTAINS 'ECR' AND (a.titulo CONTAINS 'capital-riesgo' OR a.titulo CONTAINS 'Concepto'))
        RETURN a.numero, a.titulo, c.texto AS fragmento
        ORDER BY a.numero ASC
        LIMIT 5
"""

SYSTEM_PROMPT_CYPHER = f"""Eres un experto en la Ley 22/2014 y en Neo4j.
Tu tarea es generar consultas Cypher PRECISAS para recuperar los artículos más relevantes del grafo.

Esquema del grafo:
- Nodos:
  - Articulo(id, numero, titulo, texto)
  - Chunk(id, texto, orden, articulo_id, articulo_numero, articulo_titulo)
  - Titulo
  - Capitulo
  - Seccion
  - Entidad(nombre)
  - Concepto(nombre)
  - Disposicion
- Relaciones:
  - PERTENECE_A
  - TIENE_CHUNK
  - REFERENCIA
  - MENCIONA
  - TRATA_SOBRE

OBJETIVO:
- Recuperar POCOS artículos, pero muy relevantes.
- Priorizar artículos definicionales, estructurales y directamente aplicables.
- Evitar ruido.
- No recuperar artículos solo porque mencionan una sigla si no regulan la cuestión preguntada.

REGLAS OBLIGATORIAS:

1. Si la pregunta menciona un artículo concreto (por ejemplo: "artículo 26", "artículo 4 bis"):
   - usa búsqueda exacta por id.
   - patrón de ids:
     - art_1
     - art_26
     - art_4_bis
     - art_15_ter
   - ejemplo:
     MATCH (a:Articulo)
     WHERE a.id = 'art_26'
     RETURN a.numero, a.titulo, a.texto

2. Si la pregunta es definicional o conceptual ("qué es X", "qué significa X", "qué regula", "cuál es el objeto", "define", "formas jurídicas", "naturaleza jurídica", "diferencia entre X e Y"):
   - prioriza artículos con títulos estructurales:
     - 'Objeto'
     - 'Definición'
     - 'Definición y régimen jurídico'
     - 'Actividad principal'
     - 'Entidades'
     - 'Régimen jurídico'
     - 'Concepto'
   - combina búsqueda por titulo y texto.
   - usa pocas coincidencias muy relevantes.
   - usa ORDER BY a.numero ASC.
   - usa LIMIT 5 por defecto.
   - NO uses búsquedas excesivamente amplias por OR si eso trae artículos operativos o de comercialización.

3. Si la pregunta es comparativa entre dos o más conceptos:
   - busca artículos definicionales de cada concepto.
   - busca también artículos sobre sus formas jurídicas o régimen jurídico si la pregunta lo pide.
   - usa ORDER BY a.numero ASC.
   - usa LIMIT 6 como máximo.

4. La búsqueda por contenido en `c.texto CONTAINS ...` es útil, pero NO debe usarse como única estrategia si la pregunta es definicional.
   - En preguntas definicionales, combina:
     - `a.titulo CONTAINS ...`
     - `c.texto CONTAINS ...`
   - Prioriza artículos cuyo título sugiere definición o régimen básico.

5. Para siglas o acrónimos (SCR, FCR, SICC, FICC, SGEIC, ECR, EICC, EICCP):
   - busca primero por `a.titulo CONTAINS 'sigla'`
   - y en segundo lugar por `c.texto CONTAINS 'sigla'`
   - si la pregunta pide significado o artículo definitorio, prioriza artículos de definición/régimen jurídico, no artículos donde solo se mencionan esas siglas.

6. Para SGEIC, además de 'SGEIC', puedes usar 'sociedad gestora', pero evita recuperar artículos meramente operativos si la pregunta pide definición o requisitos de acceso.

7. Para preguntas sobre requisitos, obligaciones, límites o condiciones:
   - prioriza títulos como:
     - 'Requisitos'
     - 'Requisitos de acceso'
     - 'Condiciones'
     - 'Obligaciones'
     - 'Limitaciones'
     - 'Coeficiente'
     - 'Actividad principal'
   - combina titulo y texto.
   - LIMIT 5.

8. Evita artículos de comercialización, supervisión o sanción cuando la pregunta sea definicional o estructural, salvo que la pregunta trate específicamente de comercialización, CNMV, notificación, registro o sanciones.

9. Si recuperas por `Chunk`, incluye `c.texto` en el RETURN como `fragmento`. Si recuperas un artículo concreto por id, incluye `a.texto`.

10. Usa siempre consultas limpias, sin comentarios ni markdown. Devuelve solo Cypher.

PATRONES DE PRIORIZACIÓN:

- Pregunta sobre "qué regula el artículo X" → artículo exacto.
- Pregunta sobre "qué es X" → definición/régimen jurídico.
- Pregunta sobre "diferencia entre X e Y" → definición/régimen de X + definición/régimen de Y + formas jurídicas si aplica.
- Pregunta sobre "qué artículo regula..." → prioriza búsqueda por titulo + texto con el término regulatorio exacto.

11. Si la pregunta requiere combinar varias definiciones o identificar todas las categorías de una materia (por ejemplo: tipos de entidades, clases de vehículos, sujetos regulados o excluidos):
   - recupera los artículos definicionales relevantes para cada categoría.
   - prioriza artículos estructurales del inicio de la ley.
   - evita recuperar artículos operativos si no son necesarios para responder.
   - normalmente estos artículos se encuentran en los primeros títulos de la ley.
   
{FEW_SHOT_CYPHER}
"""

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente jurídico experto en la Ley 22/2014 de entidades de capital-riesgo. "
     "Responde en español basándote SOLO en la información proporcionada. "

     "REGLAS:\n"
     "- Usa exclusivamente el contexto. No completes lagunas con conocimiento externo, intuiciones ni interpretaciones.\n"
     "- Si el contexto no contiene la respuesta suficiente, responde exactamente: "
     f"'{NO_CONTEXT_RESPONSE}'.\n"
     "- Identifica primero qué artículos del contexto soportan la respuesta.\n"
     "- Si varios artículos regulan distintos aspectos, combínalos solo si todos están en el contexto.\n"
     "- Si la pregunta menciona un artículo concreto, cita ese número exacto.\n"
     "- Si el contexto contiene listas legales, reproduce todos sus elementos relevantes y no añadas ninguno.\n"
     "- NUNCA inventes números de artículo ni categorías jurídicas.\n"
     "- La respuesta debe ser breve: artículos relevantes y explicación corta."
    ),
    ("human", "Información recuperada:\n{context}\n\nPregunta: {question}")
])

# ──────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────────────────────────────────────

def _extraer_numeros_articulo(texto: str) -> list[str]:
    """Extrae números de artículo mencionados en la query."""
    patron = re.compile(
        r'(?:art[íi]culo|art\.)\s*(\d+(?:\s*(?:bis|ter|quater))?)',
        re.IGNORECASE
    )
    return [re.sub(r"\s+", "_", m.strip().lower()) for m in patron.findall(texto)]


def _extraer_entidades_query(query: str) -> list[str]:
    """Detecta entidades jurídicas en la query."""
    conocidas = ["SGEIC", "ECR", "SCR", "FCR", "SICC", "FICC", "EICC", "EICCP", "CNMV", "GFIA", "AIFMD"]
    return [e for e in conocidas if e.lower() in query.lower()]


def _articulo_id(numero_normalizado: str) -> str:
    return f"art_{numero_normalizado}"


def _strip_accents(texto: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", texto or "")
        if not unicodedata.combining(ch)
    )


def _normalizar_texto(texto: str) -> str:
    normalizado = _strip_accents(texto).lower()
    return re.sub(r"\s+", " ", normalizado).strip()


def _tokenizar(texto: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", _normalizar_texto(texto))
    return [tok for tok in tokens if len(tok) > 1 and tok not in _STOPWORDS_RERANK]


def _dedupe_preservando_orden(items: list[str]) -> list[str]:
    vistos = set()
    salida = []
    for item in items:
        if not item or item in vistos:
            continue
        vistos.add(item)
        salida.append(item)
    return salida


def expand_query_with_synonyms(query: str) -> str:
    query_lower = (query or "").lower()
    extras = []
    for key, synonyms in LEGAL_SYNONYMS.items():
        if key in query_lower:
            for synonym in synonyms:
                synonym_lower = synonym.lower()
                if synonym_lower not in query_lower and synonym_lower not in extras:
                    extras.append(synonym_lower)
    if not extras:
        return query
    return f"{query} {' '.join(extras)}".strip()


def expand_query_with_article_titles(query: str) -> str:
    query_lower = (query or "").lower()
    extras = []
    for key, article_title in ARTICLE_TITLE_HINTS.items():
        title_lower = article_title.lower()
        if key in query_lower and title_lower not in query_lower and title_lower not in extras:
            extras.append(title_lower)
    if not extras:
        return query
    return f"{query} {' '.join(extras)}".strip()


def expand_query(query: str) -> str:
    expanded_query = expand_query_with_synonyms(query)
    return expand_query_with_article_titles(expanded_query)


def _extraer_siglas_query(query: str) -> list[str]:
    siglas = re.findall(r"\b[A-Z]{2,}(?:-[A-Z]+)?\b", query or "")
    return _dedupe_preservando_orden(siglas)


def _clasificar_pregunta(query: str) -> str:
    query_norm = _normalizar_texto(query)
    if _extraer_numeros_articulo(query):
        return "articulo_directo"
    if any(patron in query_norm for patron in _PATRONES_COMPARATIVA):
        return "comparativa"
    if any(patron in query_norm for patron in _PATRONES_CONDICIONES):
        return "condiciones"
    if any(patron in query_norm for patron in _PATRONES_DEFINICION) or _extraer_siglas_query(query):
        return "definicion"
    return "general"


def _semillas_articulos_query(query: str) -> list[str]:
    query_norm = _normalizar_texto(query)
    semillas = []

    for sigla in _extraer_siglas_query(query):
        if sigla in _SIGLAS_ARTICULOS:
            semillas.extend(_SIGLAS_ARTICULOS[sigla])

    for termino, ids in sorted(_TERMINOS_ARTICULOS.items(), key=lambda item: len(item[0]), reverse=True):
        if _normalizar_texto(termino) in query_norm:
            semillas.extend(ids)

    return _dedupe_preservando_orden(semillas)


def _terminos_busqueda(query: str) -> list[str]:
    tokens = _tokenizar(query)
    siglas = _extraer_siglas_query(query)
    terminos = list(siglas)
    terminos.extend(tokens)
    return _dedupe_preservando_orden(terminos)


def _numero_base_articulo(articulo: dict) -> int:
    numero = str(articulo.get("numero", ""))
    match = re.match(r"(\d+)", numero)
    return int(match.group(1)) if match else 10**9


def _texto_rerank_articulo(articulo: dict) -> str:
    partes = [f"Artículo {articulo.get('numero', '?')}. {articulo.get('titulo', '')}"]
    for chunk in _merge_chunks(articulo.get("chunks_relevantes", []))[:4]:
        texto = chunk.get("texto", "")
        if texto:
            partes.append(texto)
    if len(partes) == 1 and articulo.get("texto"):
        partes.append(str(articulo.get("texto", ""))[:3000])
    return "\n".join(partes)


def _score_texto_para_query(query: str, texto: str) -> float:
    q_tokens = _tokenizar(query)
    if not q_tokens:
        return 0.0

    texto_norm = _normalizar_texto(texto)
    tokens_texto = set(_tokenizar(texto))

    score = 0.0
    for token in q_tokens:
        if token in tokens_texto:
            score += 1.5
        if len(token) >= 4 and token in texto_norm:
            score += 0.6

    for sigla in _extraer_siglas_query(query):
        if sigla.lower() in texto_norm:
            score += 4.0

    frases = [
        frase.strip()
        for frase in re.split(r"[?!.:,;]+", query)
        if len(frase.strip().split()) >= 2
    ]
    for frase in frases:
        frase_norm = _normalizar_texto(frase)
        if len(frase_norm) >= 10 and frase_norm in texto_norm:
            score += 3.0

    return score


def _score_estructura_articulo(articulo: dict, tipo_pregunta: str) -> float:
    titulo_norm = _normalizar_texto(articulo.get("titulo", ""))
    base_num = _numero_base_articulo(articulo)
    score = 0.0

    if tipo_pregunta in {"definicion", "comparativa"}:
        if any(
            patron in titulo_norm
            for patron in ("definicion", "regimen juridico", "concepto", "objeto")
        ):
            score += 3.0
        if base_num <= 45:
            score += 1.0

    if tipo_pregunta == "condiciones":
        if any(
            patron in titulo_norm
            for patron in ("requisito", "condicion", "autorizacion", "acceso", "obligacion")
        ):
            score += 3.0

    return score


def _get_reranker(verbose: bool = False):
    global _RERANKER, _RERANKER_FAILED
    if _RERANKER is not None or _RERANKER_FAILED:
        return _RERANKER
    try:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder("BAAI/bge-reranker-v2-m3")
    except Exception as exc:
        _RERANKER_FAILED = True
        if verbose:
            print(f"[rerank] Reranker no disponible; uso heurístico. Detalle: {exc}")
    return _RERANKER


def _rerank_articulos(
    query: str,
    articulos: list[dict],
    tipo_pregunta: str,
    ids_prioritarios: list[str] | None = None,
    ids_semilla: list[str] | None = None,
    verbose: bool = False,
) -> list[dict]:
    ids_prioritarios = ids_prioritarios or []
    ids_semilla = ids_semilla or []
    prioridad = {art_id: i for i, art_id in enumerate(ids_prioritarios)}
    semillas = {art_id: i for i, art_id in enumerate(ids_semilla)}

    if not articulos:
        return []

    enriquecidos = []
    for articulo in articulos:
        art = dict(articulo)
        texto_rerank = _texto_rerank_articulo(art)
        base_score = float(art.get("score") or 0.0)
        heuristico = _score_texto_para_query(query, texto_rerank) + _score_estructura_articulo(art, tipo_pregunta)
        art["rerank_score"] = heuristico + (base_score * 4.0)
        art["_texto_rerank"] = texto_rerank
        enriquecidos.append(art)

    reranker = _get_reranker(verbose=verbose)
    if reranker is not None:
        pares = [(query, art["_texto_rerank"][:3500]) for art in enriquecidos]
        try:
            scores = reranker.predict(pares)
            for art, score in zip(enriquecidos, scores):
                art["rerank_score"] += float(score) * 5.0
        except Exception as exc:
            if verbose:
                print(f"[rerank] Falló el reranker; sigo con heurístico. Detalle: {exc}")

    def _key(art: dict):
        art_id = art.get("id", "")
        if art_id in prioridad:
            return (0, prioridad[art_id], 0.0, _numero_base_articulo(art), art_id)
        if art_id in semillas:
            return (1, semillas[art_id], -float(art.get("rerank_score") or 0.0), _numero_base_articulo(art), art_id)
        return (2, -float(art.get("rerank_score") or 0.0), _numero_base_articulo(art), art_id)

    ordenados = sorted(enriquecidos, key=_key)
    for art in ordenados:
        art.pop("_texto_rerank", None)
    return ordenados


def _seleccionar_chunks_relevantes(query: str, chunks: list[dict], limit: int = MAX_CHUNKS_POR_ARTICULO) -> list[dict]:
    if not chunks:
        return []
    if not query:
        return _merge_chunks(chunks)[:limit]

    candidatos = []
    for chunk in _merge_chunks(chunks):
        chunk_copy = dict(chunk)
        score = _score_texto_para_query(query, chunk_copy.get("texto", "")) + _score_chunk(chunk_copy)
        chunk_copy["query_score"] = score
        candidatos.append(chunk_copy)

    candidatos.sort(key=lambda c: (-float(c.get("query_score") or 0.0), c.get("orden", 10**9), c.get("id", "")))
    for chunk in candidatos:
        chunk.pop("query_score", None)
    return candidatos[:limit]


def _es_respuesta_no_contexto(respuesta: str) -> bool:
    texto = _normalizar_texto(respuesta)
    if not texto:
        return True
    marcadores = (
        _normalizar_texto(NO_CONTEXT_RESPONSE),
        "no regula expresamente",
        "no tengo informacion",
        "no encontr",
        "no hay informacion",
    )
    return any(marcador in texto for marcador in marcadores)


def _verificar_groundedness(llm, query: str, contexto: str, respuesta: str) -> bool:
    if not contexto or _es_respuesta_no_contexto(respuesta):
        return False

    from langchain_core.messages import HumanMessage, SystemMessage

    verdict = llm.invoke([
        SystemMessage(content=(
            "Verifica si la respuesta propuesta está completamente soportada por el contexto. "
            "Responde solo YES o NO. "
            "Responde NO si hay inferencias no explícitas, generalizaciones, ejemplos inventados o "
            "cualquier dato que no esté claramente contenido en el contexto."
        )),
        HumanMessage(content=(
            f"Pregunta:\n{query}\n\n"
            f"Contexto:\n{contexto}\n\n"
            f"Respuesta propuesta:\n{respuesta}"
        )),
    ]).content.strip().upper()
    return verdict.startswith("YES")


def _articulos_completos_para_query(
    query: str,
    articulos: list[dict],
    ids_objetivo: list[str] | None = None,
    numeros_objetivo: list[str] | None = None,
    ids_semilla: list[str] | None = None,
) -> set[str]:
    ids_objetivo = ids_objetivo or []
    numeros_objetivo = numeros_objetivo or []
    ids_semilla = ids_semilla or []
    tipo_pregunta = _clasificar_pregunta(query)

    completos = set(ids_objetivo) | set(numeros_objetivo)
    if tipo_pregunta == "articulo_directo":
        completos.update(ids_objetivo)
        return completos

    top_n = 2
    if tipo_pregunta in {"definicion", "comparativa"}:
        top_n = 3
    elif tipo_pregunta == "condiciones":
        top_n = 2

    completos.update(ids_semilla[:top_n])
    completos.update(a.get("id") for a in articulos[:top_n] if a.get("id"))
    return completos

def _score_chunk(chunk: dict) -> float:
    try:
        return float(chunk.get("score") or 0.0)
    except Exception:
        return 0.0


def _ordenar_chunks(chunks: list[dict]) -> list[dict]:
    return sorted(
        chunks,
        key=lambda c: (-_score_chunk(c), c.get("orden", 10**9), c.get("id", "")),
    )


def _merge_chunks(*listas_chunks: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for lista in listas_chunks:
        for chunk in lista or []:
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue
            previo = merged.get(chunk_id)
            if previo is None or _score_chunk(chunk) > _score_chunk(previo):
                merged[chunk_id] = dict(chunk)
    return _ordenar_chunks(list(merged.values()))


def _adjuntar_chunk_a_articulo(articulo: dict, chunk: dict | None) -> dict:
    art = dict(articulo)
    if chunk:
        art["chunks_relevantes"] = _merge_chunks(art.get("chunks_relevantes", []), [chunk])
        art["score"] = max(float(art.get("score") or 0.0), _score_chunk(chunk))
    return art


def _merge_articulos(*listas_articulos: list[dict]) -> list[dict]:
    """Une listas de artículos sin duplicados por id, priorizando el texto más largo."""
    merged: dict[str, dict] = {}
    for lista in listas_articulos:
        for art in lista or []:
            art_id = art.get("id")
            if not art_id:
                continue
            actual = dict(art)
            actual["chunks_relevantes"] = _merge_chunks(actual.get("chunks_relevantes", []))

            previo = merged.get(art_id)
            if previo is None:
                merged[art_id] = actual
                continue

            base = dict(previo)
            if len(str(actual.get("texto", ""))) > len(str(base.get("texto", ""))):
                for key, value in base.items():
                    actual.setdefault(key, value)
                base = actual

            base["chunks_relevantes"] = _merge_chunks(
                previo.get("chunks_relevantes", []),
                actual.get("chunks_relevantes", []),
            )
            base["score"] = max(float(previo.get("score") or 0.0), float(actual.get("score") or 0.0))
            merged[art_id] = base
    return list(merged.values())


def _ordenar_articulos(articulos: list[dict], ids_prioritarios: list[str] | None = None) -> list[dict]:
    ids_prioritarios = ids_prioritarios or []
    prioridad = {art_id: i for i, art_id in enumerate(ids_prioritarios)}

    def _key(art: dict):
        art_id = art.get("id", "")
        if art_id in prioridad:
            return (0, prioridad[art_id], 0, art_id)
        score = float(art.get("score") or 0.0)
        numero = str(art.get("numero", ""))
        m = re.match(r"(\d+)", numero)
        base = int(m.group(1)) if m else 10**9
        return (1, -score, base, numero, art_id)

    return sorted(articulos, key=_key)


def _fetch_articulos_by_ids(driver, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    with driver.session() as s:
        rows = s.run(
            "MATCH (a:Articulo) WHERE a.id IN $ids RETURN a",
            ids=ids,
        )
        articulos = [dict(record["a"]) for record in rows]
    return _ordenar_articulos(articulos, ids_prioritarios=ids)


def _fetch_chunks_by_article_ids(driver, ids: list[str], limit_per_article: int | None = None) -> dict[str, list[dict]]:
    if not ids:
        return {}

    with driver.session() as s:
        rows = s.run(
            """
            MATCH (a:Articulo)-[:TIENE_CHUNK]->(c:Chunk)
            WHERE a.id IN $ids
            RETURN a.id AS articulo_id, c
            ORDER BY a.id, c.orden ASC
            """,
            ids=ids,
        ).data()

    chunks_por_articulo: dict[str, list[dict]] = {}
    for row in rows:
        articulo_id = row.get("articulo_id")
        chunk = dict(row.get("c"))
        chunks_por_articulo.setdefault(articulo_id, []).append(chunk)

    if limit_per_article is not None:
        chunks_por_articulo = {
            articulo_id: chunk_list[:limit_per_article]
            for articulo_id, chunk_list in chunks_por_articulo.items()
        }
    return chunks_por_articulo


def _adjuntar_chunks_articulos(driver, articulos: list[dict], limit_per_article: int | None = None) -> list[dict]:
    ids = [art.get("id") for art in articulos if art.get("id")]
    chunks_por_articulo = _fetch_chunks_by_article_ids(driver, ids, limit_per_article=limit_per_article)
    enriched = []
    for art in articulos:
        art_copy = dict(art)
        art_copy["chunks_relevantes"] = _merge_chunks(
            art_copy.get("chunks_relevantes", []),
            chunks_por_articulo.get(art_copy.get("id"), []),
        )
        enriched.append(art_copy)
    return enriched


def _row_to_articulo_con_chunk(row: dict) -> dict | None:
    articulo = row.get("a") or row.get("articulo") or row.get("node")
    if not articulo:
        return None
    art = dict(articulo)
    chunk = row.get("c") or row.get("chunk")
    if chunk:
        chunk_dict = dict(chunk)
        if row.get("score") is not None:
            chunk_dict["score"] = row.get("score")
        return _adjuntar_chunk_a_articulo(art, chunk_dict)
    if row.get("score") is not None:
        art["score"] = row.get("score")
    return art


def _expandir_subgrafo(driver, ids_base: list[str], verbose: bool = False) -> list[dict]:
    """Expande por referencias + misma sección + mismo capítulo."""
    if not ids_base:
        return []

    ids_base_set = set(ids_base)
    with driver.session() as s:
        refs = s.run(
            """
            MATCH (a:Articulo)-[:REFERENCIA]-(b:Articulo)
            WHERE a.id IN $ids
            RETURN DISTINCT b AS a LIMIT 30
            """,
            ids=ids_base,
        ).data()
        secciones = s.run(
            """
            MATCH (a:Articulo)-[:PERTENECE_A]->(sec:Seccion)<-[:PERTENECE_A]-(b:Articulo)
            WHERE a.id IN $ids
            RETURN DISTINCT b AS a LIMIT 20
            """,
            ids=ids_base,
        ).data()
        capitulos = s.run(
            """
            MATCH (a:Articulo)-[:PERTENECE_A]->(cap:Capitulo)<-[:PERTENECE_A]-(b:Articulo)
            WHERE a.id IN $ids
            RETURN DISTINCT b AS a LIMIT 20
            """,
            ids=ids_base,
        ).data()

    articulos_refs = [dict(r["a"]) for r in refs if r.get("a")]
    articulos_sec = [dict(r["a"]) for r in secciones if r.get("a")]
    articulos_cap = [dict(r["a"]) for r in capitulos if r.get("a")]

    if verbose:
        print(
            f"  [expansion] referencias={len(articulos_refs)}, "
            f"misma seccion={len(articulos_sec)}, mismo capitulo={len(articulos_cap)}"
        )

    extra = _merge_articulos(articulos_refs, articulos_sec, articulos_cap)
    extra = [a for a in extra if a.get("id") not in ids_base_set]
    return _ordenar_articulos(extra)


def _recuperar_articulos_objetivo(query: str, expandir: bool = True, verbose: bool = False) -> list[dict]:
    """Recupera artículo(s) pedidos explícitamente en la query."""
    nums = _extraer_numeros_articulo(query)
    if not nums:
        return []

    ids = [_articulo_id(n) for n in nums]
    driver = get_driver()
    try:
        directos = _adjuntar_chunks_articulos(driver, _fetch_articulos_by_ids(driver, ids))
        if verbose:
            print(f"  [articulo directo] solicitados={ids}, encontrados={len(directos)}")
        if not expandir:
            return directos
        vecinos = _adjuntar_chunks_articulos(
            driver,
            _expandir_subgrafo(driver, [a["id"] for a in directos] or ids, verbose=verbose),
            limit_per_article=1,
        )
        return _ordenar_articulos(_merge_articulos(directos, vecinos), ids_prioritarios=ids)
    finally:
        driver.close()


def _contiene_cita_articulo(respuesta: str, numero_normalizado: str) -> bool:
    texto = (respuesta or "").lower()
    numero = numero_normalizado.replace("_", " ").lower()
    variantes = [
        f"artículo {numero}",
        f"articulo {numero}",
        f"art. {numero}",
        f"art {numero}",
    ]
    return any(v in texto for v in variantes)


def _articulos_sin_cita(respuesta: str, articulos_objetivo: list[str]) -> list[str]:
    return [n for n in articulos_objetivo if not _contiene_cita_articulo(respuesta, n)]


# ──────────────────────────────────────────────────────────────────────────────
# Búsqueda y contexto
# ──────────────────────────────────────────────────────────────────────────────

def buscar_articulos(driver, query: str, verbose: bool = False) -> list[dict]:
    """
    Recupera artículos relevantes desde Neo4j usando chunks como unidad primaria:
      1. Números de artículo directos en la query
      2. Semillas jurídicas por siglas y definiciones legales
      3. Entidades mencionadas en la query
      4. Búsqueda FULLTEXT por palabras clave sobre Chunk y Articulo
      5. Expansión: referencias + misma sección + mismo capítulo
      6. Reranking final a nivel artículo
    """
    resultados: dict[str, dict] = {}
    retrieval_query = expand_query(query)
    tipo_pregunta = _clasificar_pregunta(query)

    def _guardar_articulo(art: dict | None):
        if not art or not art.get("id"):
            return
        previo = resultados.get(art["id"])
        if previo is None:
            resultados[art["id"]] = art
        else:
            resultados[art["id"]] = _merge_articulos([previo], [art])[0]

    numeros_objetivo = _extraer_numeros_articulo(query)
    ids_objetivo = [_articulo_id(n) for n in numeros_objetivo]
    ids_semilla = _semillas_articulos_query(retrieval_query)
    entity_ids: set[str] = set()
    terminos_busqueda = _terminos_busqueda(retrieval_query)

    with driver.session() as s:
        if verbose and retrieval_query != query:
            print(f"  [expansion] Query expandida: {retrieval_query}")
        if numeros_objetivo and verbose:
            print(f"  [1] Artículos directos solicitados: {ids_objetivo}")
        direct_ids = _dedupe_preservando_orden(ids_objetivo + ids_semilla)
        if ids_semilla and verbose:
            print(f"  [2] Semillas jurídicas: {ids_semilla}")
        directos = _adjuntar_chunks_articulos(driver, _fetch_articulos_by_ids(driver, direct_ids))
        for art in directos:
            _guardar_articulo(art)

        entidades = _extraer_entidades_query(retrieval_query)
        if entidades and verbose:
            print(f"  [3] Entidades: {entidades}")
        for ent in entidades:
            res = s.run("""
                MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad {nombre: $nombre})
                RETURN a.id AS articulo_id LIMIT 15
            """, nombre=ent)
            for record in res:
                if record.get("articulo_id"):
                    entity_ids.add(record["articulo_id"])

        if terminos_busqueda:
            lucene_query = " OR ".join(terminos_busqueda[:8])
            if verbose:
                print(f"  [4] Fulltext sobre chunks: {lucene_query}")
            try:
                res = s.run("""
                    CALL db.index.fulltext.queryNodes('chunk_texto', $q)
                    YIELD node, score
                    MATCH (a:Articulo)-[:TIENE_CHUNK]->(node)
                    WHERE score > 0.2
                    RETURN a, node AS c, score
                    ORDER BY score DESC, c.orden ASC
                    LIMIT $limit
                """, q=lucene_query, limit=FULLTEXT_CHUNK_CANDIDATES)
                for record in res:
                    _guardar_articulo(_row_to_articulo_con_chunk(record))
            except Exception:
                for p in terminos_busqueda[:6]:
                    res = s.run("""
                        MATCH (a:Articulo)-[:TIENE_CHUNK]->(c:Chunk)
                        WHERE toLower(c.texto) CONTAINS toLower($p)
                        RETURN a, c, 0.0 AS score
                        ORDER BY c.orden ASC
                        LIMIT 8
                    """, p=p)
                    for record in res:
                        _guardar_articulo(_row_to_articulo_con_chunk(record))

            try:
                res = s.run("""
                    CALL db.index.fulltext.queryNodes('articulo_texto', $q)
                    YIELD node, score
                    RETURN node AS a, score
                    ORDER BY score DESC
                    LIMIT $limit
                """, q=lucene_query, limit=FULLTEXT_ARTICULO_CANDIDATES)
                for record in res:
                    _guardar_articulo(_row_to_articulo_con_chunk(record))
            except Exception:
                pass

        if tipo_pregunta in {"definicion", "comparativa"}:
            for kw in ["Definición", "Régimen jurídico", "Concepto", "Objeto"]:
                res = s.run(
                    """
                    MATCH (a:Articulo)
                    WHERE a.titulo CONTAINS $kw
                    RETURN a LIMIT 8
                    """,
                    kw=kw,
                )
                for record in res:
                    _guardar_articulo(_row_to_articulo_con_chunk(record))

        if entity_ids:
            entity_articles = _adjuntar_chunks_articulos(
                driver,
                _fetch_articulos_by_ids(driver, list(entity_ids)),
                limit_per_article=2,
            )
            for art in entity_articles:
                _guardar_articulo(art)

        if verbose:
            print(f"  Recuperados inicialmente: {len(resultados)} artículos")

    ids_base = list(resultados.keys()) or ids_objetivo
    if ids_base:
        expandidos = _adjuntar_chunks_articulos(
            driver,
            _expandir_subgrafo(driver, ids_base, verbose=verbose),
            limit_per_article=1,
        )
        nuevos = 0
        for art in expandidos:
            art_id = art.get("id")
            if art_id and art_id not in resultados:
                resultados[art_id] = art
                nuevos += 1
        if verbose and nuevos > 0:
            print(f"  [5] Expandidos (referencias/seccion/capitulo): +{nuevos} artículos")

    reordenados = _rerank_articulos(
        query,
        list(resultados.values()),
        tipo_pregunta=tipo_pregunta,
        ids_prioritarios=ids_objetivo,
        ids_semilla=ids_semilla,
        verbose=verbose,
    )
    return reordenados[:MAX_ARTICULOS_CONTEXTO]


def construir_contexto(
    articulos: list[dict],
    query: str | None = None,
    max_chars: int = MAX_CHARS_CONTEXTO,
    articulos_completos: set[str] | None = None,
    ids_prioritarios: list[str] | None = None,
) -> str:
    """Construye el bloque de contexto a pasar al LLM."""
    articulos_completos = articulos_completos or set()
    articulos_ordenados = _ordenar_articulos(articulos, ids_prioritarios=ids_prioritarios)
    partes = []
    total = 0
    for art in articulos_ordenados:
        header = f"\n=== Artículo {art.get('numero', '?')}. {art.get('titulo', '')} ==="
        art_id = str(art.get("id", ""))
        numero_norm = str(art.get("numero", "")).lower().replace(" ", "_")
        incluir_completo = art_id in articulos_completos or numero_norm in articulos_completos
        if incluir_completo:
            cuerpo = art.get("texto", "")
        else:
            fragmentos = _seleccionar_chunks_relevantes(query or "", art.get("chunks_relevantes", []))
            if fragmentos:
                lineas = []
                for idx, chunk in enumerate(fragmentos, 1):
                    score = chunk.get("score")
                    tag_score = f" | score={score:.3f}" if isinstance(score, (int, float)) else ""
                    lineas.append(f"[fragmento {idx}{tag_score}]")
                    lineas.append(chunk.get("texto", ""))
                cuerpo = "\n\n".join(lineas)
            else:
                cuerpo = art.get("texto", "")[:4000]

        bloque = header + "\n" + cuerpo
        if total + len(bloque) > max_chars and not incluir_completo:
            break
        partes.append(bloque)
        total += len(bloque)
    return "\n".join(partes)


# ──────────────────────────────────────────────────────────────────────────────
# Síntesis y validación de respuestas
# ──────────────────────────────────────────────────────────────────────────────

def _sintetizar_respuesta(
    llm,
    query: str,
    contexto: str,
    articulos_objetivo: list[str] | None = None,
) -> str:
    """Usa el LLM para sintetizar una respuesta a partir del contexto recuperado."""
    from langchain_core.messages import HumanMessage, SystemMessage
    articulos_objetivo = articulos_objetivo or []
    instruccion_objetivo = ""
    if articulos_objetivo:
        lista = ", ".join(f"artículo {n.replace('_', ' ')}" for n in articulos_objetivo)
        instruccion_objetivo = (
            f" El usuario pide explícitamente {lista}. "
            "Debes mencionar esos números exactos y responder solo con lo que esté soportado por esos artículos."
        )

    messages = [
        SystemMessage(content=(
            "Eres un asistente jurídico experto en la Ley 22/2014 de entidades de capital-riesgo. "
            "Responde en español basándote SOLO en el contexto proporcionado. "
            "No uses conocimiento externo, no extrapoles y no completes huecos. "
            "Si el contexto no contiene la respuesta suficiente, responde exactamente: "
            f"'{NO_CONTEXT_RESPONSE}'. "
            "Cita solo artículos que aparezcan en el contexto. "
            "Si el contexto incluye listas legales, reproduce todos los elementos relevantes sin añadir otros. "
            "Responde de forma breve: primero los artículos relevantes y después una explicación corta, sin doctrinas ni ejemplos inventados."
            f"{instruccion_objetivo}"
        )),
        HumanMessage(content=(
            f"Contexto de la ley:\n{contexto}\n\n"
            f"Pregunta del usuario: {query}\n"
            "Formato esperado:\n"
            "Artículos relevantes: ...\n"
            "Respuesta: ...\n"
            f"Si no está soportado, responde solo: {NO_CONTEXT_RESPONSE}"
        )),
    ]
    return llm.invoke(messages).content


def _validar_y_reforzar_citas(
    respuesta: str,
    query: str,
    contexto_base: str,
    log_llm: bool = False,
    verbose: bool = False,
    trace: bool = False,
    articulos_contexto: list[dict] | None = None,
) -> str:
    """Verifica citas de artículos pedidos; si faltan, recupera directo y re-sintetiza."""
    articulos_objetivo = _extraer_numeros_articulo(query)
    if not articulos_objetivo:
        return respuesta

    faltan = _articulos_sin_cita(respuesta, articulos_objetivo)
    if not faltan:
        return respuesta

    if verbose or trace:
        print(f"[validacion citas] faltan artículos en la respuesta: {faltan}. Reforzando retrieval directo...")

    ids_objetivo = [_articulo_id(n) for n in articulos_objetivo]
    ids_faltan = {_articulo_id(n) for n in faltan}

    base_contexto = articulos_contexto or []
    directos: list[dict] = [a for a in base_contexto if a.get("id") in ids_faltan]

    driver = get_driver()
    try:
        ids_directos = [a.get("id") for a in directos if a.get("id")]
        missing_ids = [art_id for art_id in ids_faltan if art_id not in ids_directos]
        fetched = _adjuntar_chunks_articulos(driver, _fetch_articulos_by_ids(driver, missing_ids)) if missing_ids else []
        semilla_expansion = list(ids_faltan) if missing_ids else ids_directos
        expandidos = _adjuntar_chunks_articulos(
            driver,
            _expandir_subgrafo(driver, semilla_expansion, verbose=verbose or trace),
            limit_per_article=1,
        ) if semilla_expansion else []
    finally:
        driver.close()

    reforzados = _merge_articulos(base_contexto, directos, fetched, expandidos)
    if not reforzados:
        return NO_CONTEXT_RESPONSE

    articulos_completos = set(faltan) | ids_faltan
    contexto_reforzado = construir_contexto(
        reforzados,
        query=query,
        articulos_completos=articulos_completos,
        ids_prioritarios=ids_objetivo,
    )
    contexto_final = contexto_reforzado if not contexto_base else f"{contexto_base}\n\n---\n\n{contexto_reforzado}"
    respuesta_reforzada = _sintetizar_respuesta(
        get_llm(log_llm=log_llm),
        query,
        contexto_final,
        articulos_objetivo=articulos_objetivo,
    )

    faltan_despues = _articulos_sin_cita(respuesta_reforzada, articulos_objetivo)
    if faltan_despues:
        if verbose or trace:
            print(f"[validacion citas] sigue faltando citar artículos: {faltan_despues}.")
        return NO_CONTEXT_RESPONSE

    return respuesta_reforzada


def _filtrar_respuesta_por_groundedness(
    query: str,
    respuesta: str,
    contexto: str,
    log_llm: bool = False,
    verbose: bool = False,
    trace: bool = False,
) -> str:
    if _es_respuesta_no_contexto(respuesta):
        return NO_CONTEXT_RESPONSE
    if not contexto.strip():
        return NO_CONTEXT_RESPONSE

    try:
        soportada = _verificar_groundedness(
            get_llm(log_llm=log_llm),
            query=query,
            contexto=contexto,
            respuesta=respuesta,
        )
    except Exception as exc:
        if verbose or trace:
            print(f"[groundedness] No se pudo verificar; mantengo respuesta. Detalle: {exc}")
        return respuesta

    if not soportada:
        if verbose or trace:
            print("[groundedness] Respuesta no soportada por el contexto. Se devuelve fallback estricto.")
        return NO_CONTEXT_RESPONSE
    return respuesta


# ──────────────────────────────────────────────────────────────────────────────
# Fallback por texto
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_texto(query: str) -> str:
    """Busca contexto con el mismo retrieval jurídico del modo grafo."""
    retrieval_query = expand_query(query)
    driver = get_driver()
    try:
        articulos = buscar_articulos(driver, query, verbose=False)
    finally:
        driver.close()

    if not articulos:
        return ""

    numeros_objetivo = _extraer_numeros_articulo(query)
    ids_objetivo = [_articulo_id(n) for n in numeros_objetivo]
    ids_semilla = _semillas_articulos_query(retrieval_query)
    articulos_completos = _articulos_completos_para_query(
        query,
        articulos,
        ids_objetivo=ids_objetivo,
        numeros_objetivo=numeros_objetivo,
        ids_semilla=ids_semilla,
    )
    return construir_contexto(
        articulos,
        query=query,
        articulos_completos=articulos_completos,
        ids_prioritarios=ids_objetivo,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MODO A: consulta_cypher
# ──────────────────────────────────────────────────────────────────────────────

def consulta_cypher(query: str, verbose: bool = False, trace: bool = False, log_llm: bool = False) -> str:
    """Usa GraphCypherQAChain: pregunta -> Cypher -> Neo4j -> respuesta."""
    retrieval_query = expand_query(query)
    articulos_objetivo = _extraer_numeros_articulo(query)
    ids_objetivo = [_articulo_id(n) for n in articulos_objetivo]
    articulos_directos = _recuperar_articulos_objetivo(query, expandir=True, verbose=verbose or trace)
    ids_semilla = _semillas_articulos_query(query)

    llm = get_llm(log_llm=log_llm)
    graph = get_lang_graph()

    cypher_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_CYPHER.replace("{", "{{").replace("}", "}}")),
        ("human", "Schema del grafo:\n{schema}\n\nPregunta: {question}\n\nDevuelve solo la consulta Cypher sin explicacion."),
    ])

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=verbose,
        allow_dangerous_requests=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=_QA_PROMPT,
        return_intermediate_steps=True,
    )

    # Si la query contiene siglas conocidas, inyectar expansiones exactas para evitar alucinaciones
    _SIGLAS_LEY = {
        "SCR": "Sociedad de Capital-Riesgo (art. 26)",
        "FCR": "Fondo de Capital-Riesgo (art. 30)",
        "SICC": "Sociedad de Inversión Colectiva de Tipo Cerrado (art. 38)",
        "FICC": "Fondo de Inversión Colectiva de Tipo Cerrado (art. 38)",
        "SGEIC": "Sociedad Gestora de Entidades de Inversión Colectiva de Tipo Cerrado (art. 41)",
        "ECR": "Entidad de Capital-Riesgo (art. 3)",
        "EICC": "Entidad de Inversión Colectiva de Tipo Cerrado (art. 4)",
        "EICCP": "Entidad de Inversión Colectiva de Tipo Cerrado de Préstamos (art. 4 bis)",
        "CNMV": "Comisión Nacional del Mercado de Valores",
    }
    siglas_en_query = [s for s in _SIGLAS_LEY if s in query.upper()]
    nota_siglas = ""
    if siglas_en_query:
        expansiones = "; ".join(f"{s} = {_SIGLAS_LEY[s]}" for s in siglas_en_query)
        nota_siglas = f"\n[Referencia obligatoria — expansiones exactas de la Ley 22/2014: {expansiones}]"

    if (verbose or trace) and retrieval_query != query:
        print(f"[expansion] Query expandida para generación de Cypher: {retrieval_query}")

    result = chain.invoke({
        "query": f"Responde en español. Ley 22/2014 (BOE).{nota_siglas}\nPregunta: {retrieval_query}"
    })

    # Extraer steps siempre (se usan tanto para trace como para resintetizar)
    steps = result.get("intermediate_steps", [])
    cypher_rows = steps[1].get("context", []) if len(steps) > 1 else []

    # -- TRACE: mostrar nodos accedidos en Neo4j
    if trace:
        cypher_generado = steps[0].get("query", "(no disponible)") if steps else "(no disponible)"
        contexto_neo4j = cypher_rows
        print("\n" + "=" * 60)
        print("TRAZA DE ACCESO AL GRAFO")
        print("=" * 60)
        print(f"\nCypher generado por el LLM:")
        print(f"   {cypher_generado.strip()}")
        print(f"\nNodos/filas devueltos por Neo4j ({len(contexto_neo4j)}):")
        if contexto_neo4j:
            for i, row in enumerate(contexto_neo4j, 1):
                print(f"\n  [{i}]")
                for k, v in row.items():
                    val = str(v)
                    if len(val) > 120:
                        val = val[:120] + "..."
                    print(f"      {k}: {val}")
        else:
            print("   (ninguno — la consulta no devolvio resultados)")
        print("=" * 60 + "\n")

    contexto_directo = ""
    if articulos_directos:
        articulos_completos = _articulos_completos_para_query(
            query,
            articulos_directos,
            ids_objetivo=ids_objetivo,
            numeros_objetivo=articulos_objetivo,
            ids_semilla=ids_semilla,
        )
        contexto_directo = construir_contexto(
            articulos_directos,
            query=query,
            articulos_completos=articulos_completos,
            ids_prioritarios=ids_objetivo,
        )
        if verbose or trace:
            print(f"[articulo directo] Contexto anadido: {len(articulos_directos)} articulos")

    contexto_cypher = "\n\n---\n\n".join(
        "\n".join(f"{k}: {v}" for k, v in row.items())
        for row in cypher_rows
    ) if cypher_rows else ""

    contexto_principal = "\n\n---\n\n".join(c for c in [contexto_cypher, contexto_directo] if c)
    if not contexto_principal:
        contexto_principal = _fallback_texto(query)
        if contexto_principal and (verbose or trace):
            print(f"[fallback] Contexto recuperado por texto ({len(contexto_principal)} chars)")

    if not contexto_principal:
        return NO_CONTEXT_RESPONSE

    if verbose or trace:
        print(f"[sintesis] Contexto final para respuesta: {len(contexto_principal)} chars")

    respuesta = _sintetizar_respuesta(
        llm,
        query,
        contexto_principal,
        articulos_objetivo=articulos_objetivo,
    )
    respuesta = _validar_y_reforzar_citas(
        respuesta,
        query,
        contexto_principal,
        log_llm=log_llm,
        verbose=verbose,
        trace=trace,
        articulos_contexto=articulos_directos,
    )
    return _filtrar_respuesta_por_groundedness(
        query=query,
        respuesta=respuesta,
        contexto=contexto_principal,
        log_llm=log_llm,
        verbose=verbose,
        trace=trace,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MODO B: Recuperación manual por grafo + expansión
# ──────────────────────────────────────────────────────────────────────────────

def generar_respuesta_grafo(query: str, articulos: list[dict], log_llm: bool = False) -> str:
    """Genera respuesta con el LLM dado el contexto del grafo."""
    articulos_objetivo = _extraer_numeros_articulo(query)
    ids_objetivo = [_articulo_id(n) for n in articulos_objetivo]
    ids_semilla = _semillas_articulos_query(query)
    tipo_pregunta = _clasificar_pregunta(query)
    articulos_directos = _recuperar_articulos_objetivo(query, expandir=True, verbose=False)
    articulos = _merge_articulos(articulos, articulos_directos)
    articulos = _rerank_articulos(
        query,
        articulos,
        tipo_pregunta=tipo_pregunta,
        ids_prioritarios=ids_objetivo,
        ids_semilla=ids_semilla,
        verbose=False,
    )[:RERANK_TOP_ARTICULOS]

    if not articulos:
        return NO_CONTEXT_RESPONSE

    articulos_completos = _articulos_completos_para_query(
        query,
        articulos,
        ids_objetivo=ids_objetivo,
        numeros_objetivo=articulos_objetivo,
        ids_semilla=ids_semilla,
    )
    contexto = construir_contexto(
        articulos,
        query=query,
        articulos_completos=articulos_completos,
        ids_prioritarios=ids_objetivo,
    )
    llm = get_llm(log_llm=log_llm)

    respuesta = _sintetizar_respuesta(
        llm,
        query,
        contexto,
        articulos_objetivo=articulos_objetivo,
    )
    respuesta = _validar_y_reforzar_citas(
        respuesta,
        query,
        contexto,
        log_llm=log_llm,
        articulos_contexto=articulos,
    )
    return _filtrar_respuesta_por_groundedness(
        query=query,
        respuesta=respuesta,
        contexto=contexto,
        log_llm=log_llm,
    )


def consulta_grafo(query: str, verbose: bool = False, trace: bool = False, log_llm: bool = False) -> str:
    """Recuperación manual por grafo + expansión de subgrafo."""
    if _clasificar_pregunta(query) == "articulo_directo":
        articulos = _recuperar_articulos_objetivo(query, expandir=False, verbose=verbose or trace)
        return generar_respuesta_grafo(query, articulos, log_llm=log_llm)

    driver = get_driver()
    try:
        articulos = buscar_articulos(driver, query, verbose=verbose or trace)
    finally:
        driver.close()

    if verbose or trace:
        print(f"\n  Total artículos en contexto (grafo): {len(articulos)}")
        for a in articulos:
            print(f"    - Art. {a.get('numero', '?')}: {a.get('titulo', '')[:80]}")

    if trace:
        print("\n" + "=" * 60)
        print("TRAZA DE ACCESO AL GRAFO (modo: GRAFO)")
        print("=" * 60)
        for i, a in enumerate(articulos, 1):
            texto_preview = (a.get("texto", "") or "")[:150].replace("\n", " ")
            print(f"\n  [{i}] Art. {a.get('numero', '?')} — {a.get('titulo', '')[:70]}")
            print(f"       id: {a.get('id')}")
            print(f"       texto: {texto_preview}...")
        print("=" * 60 + "\n")

    return generar_respuesta_grafo(query, articulos, log_llm=log_llm)


# ──────────────────────────────────────────────────────────────────────────────
# MODO C: Recuperación semántica por vector
# ──────────────────────────────────────────────────────────────────────────────

_VECTOR_INDEX_AVAILABLE: bool | None = None


def _has_vector_index(driver) -> bool:
    """Comprueba si el índice vectorial `chunk_vector` existe en Neo4j."""
    global _VECTOR_INDEX_AVAILABLE
    if _VECTOR_INDEX_AVAILABLE is not None:
        return _VECTOR_INDEX_AVAILABLE

    try:
        with driver.session() as s:
            res = s.run("SHOW INDEXES")
            for r in res:
                info = r.data()
                if info.get("name") == "chunk_vector":
                    _VECTOR_INDEX_AVAILABLE = True
                    return True
    except Exception:
        pass

    _VECTOR_INDEX_AVAILABLE = False
    return False


def consulta_vector(query: str, verbose: bool = False, trace: bool = False, log_llm: bool = False) -> str:
    """Recuperación semántica con embeddings a nivel Chunk."""
    if _clasificar_pregunta(query) == "articulo_directo":
        articulos = _recuperar_articulos_objetivo(query, expandir=False, verbose=verbose or trace)
        return generar_respuesta_grafo(query, articulos, log_llm=log_llm)

    retrieval_query = expand_query(query)
    articulos_objetivo = _extraer_numeros_articulo(query)
    ids_objetivo = [_articulo_id(n) for n in articulos_objetivo]
    ids_semilla = _semillas_articulos_query(query)
    tipo_pregunta = _clasificar_pregunta(query)

    def _get_emb(model_name: str):
        try:
            return OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
        except Exception as e:
            if verbose or trace:
                print(f"[embed] No se pudo inicializar modelo '{model_name}': {e}")
            return None

    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    def _vector_search_fallback(driver, query_vector, top_k=MAX_CHUNKS_RELEVANTES):
        """Busca en Neo4j cargando embeddings de chunks y ordenando por similitud en Python."""
        rows = driver.session().run(
            """
            MATCH (a:Articulo)-[:TIENE_CHUNK]->(c:Chunk)
            WHERE c.embedding IS NOT NULL AND size(c.embedding) > 0
            RETURN a, c
            """
        ).data()
        candidates = []
        for r in rows:
            chunk = r["c"]
            emb = chunk.get("embedding") or []
            if not emb:
                continue
            score = _cosine_similarity(query_vector, emb)
            candidates.append((score, r["a"], chunk))
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, articulo, chunk in candidates[:top_k]:
            results.append(_row_to_articulo_con_chunk({"a": articulo, "c": chunk, "score": score}))
        return results

    emb = _get_emb(EMBED_MODEL)
    if emb is None and EMBED_MODEL != "text-embedding-3-small":
        if verbose or trace:
            print("[embed] Reintentando con text-embedding-3-small...")
        emb = _get_emb("text-embedding-3-small")
    if emb is None:
        raise RuntimeError("No se pudo inicializar ningún modelo de embeddings para vector search.")

    if (verbose or trace) and retrieval_query != query:
        print(f"[expansion] Query expandida para vector search: {retrieval_query}")
    vector = emb.embed_query(retrieval_query)

    driver = get_driver()
    resultados: dict[str, dict] = {}

    def _guardar_articulo(art: dict | None):
        if not art or not art.get("id"):
            return
        previo = resultados.get(art["id"])
        if previo is None:
            resultados[art["id"]] = art
        else:
            resultados[art["id"]] = _merge_articulos([previo], [art])[0]

    with driver.session() as s:
        if USE_VECTOR_INDEX and _has_vector_index(driver):
            try:
                rows = s.run(
                    "CALL db.index.vector.queryNodes('chunk_vector', $k, $vector) "
                    "YIELD node, score "
                    "MATCH (a:Articulo)-[:TIENE_CHUNK]->(node) "
                    "RETURN a, node AS c, score ORDER BY score DESC, c.orden ASC LIMIT $k",
                    vector=vector,
                    k=MAX_CHUNKS_RELEVANTES,
                ).data()
                for r in rows:
                    _guardar_articulo(_row_to_articulo_con_chunk(r))

                articulos_semilla = list(resultados.values())
                if len(articulos_semilla) < 4 or (
                    articulos_semilla
                    and max(float(a.get("score", 0) or 0) for a in articulos_semilla) < 0.35
                ):
                    if verbose:
                        print("[vector] Resultados limitados; complemento con busqueda por texto...")
                    pivot_ids = {a["id"] for a in articulos_semilla}
                    texto_fallback = buscar_articulos(driver, query, verbose=verbose)
                    for a in texto_fallback:
                        if a["id"] not in pivot_ids:
                            _guardar_articulo(a)
            except Exception as e:
                if verbose or trace:
                    print(f"[vector] Error ejecutando vector search en Neo4j: {e}")
                for art in _vector_search_fallback(driver, vector):
                    _guardar_articulo(art)
        else:
            if verbose or trace:
                if not USE_VECTOR_INDEX:
                    print("[vector] Vector index deshabilitado via config (USE_VECTOR_INDEX=False). Usando fallback local (cosine similarity).")
                else:
                    print("[vector] No hay indice vectorial disponible; usando fallback local (cosine similarity).")
            for art in _vector_search_fallback(driver, vector):
                _guardar_articulo(art)

        direct_ids = _dedupe_preservando_orden(ids_objetivo + ids_semilla)
        directos = _adjuntar_chunks_articulos(driver, _fetch_articulos_by_ids(driver, direct_ids)) if direct_ids else []
        ids_expansion = list(dict.fromkeys(direct_ids + [a.get("id") for a in resultados.values() if a.get("id")]))
        expandidos = _adjuntar_chunks_articulos(
            driver,
            _expandir_subgrafo(driver, ids_expansion, verbose=verbose or trace),
            limit_per_article=1,
        ) if ids_expansion else []
        articulos = _rerank_articulos(
            query,
            _merge_articulos(directos, list(resultados.values()), expandidos),
            tipo_pregunta=tipo_pregunta,
            ids_prioritarios=ids_objetivo,
            ids_semilla=ids_semilla,
            verbose=verbose or trace,
        )[:MAX_ARTICULOS_CONTEXTO]

    driver.close()

    if verbose or trace:
        print(f"\n  Total artículos en contexto (vector): {len(articulos)}")
        for a in articulos:
            print(f"    - Art. {a.get('numero', '?')}: {a.get('titulo', '')[:60]} (score={a.get('score')})")

    if trace:
        print("\n" + "=" * 60)
        print("TRAZA DE ACCESO AL GRAFO (modo: VECTOR)")
        print("=" * 60)
        for i, a in enumerate(articulos, 1):
            print(f"\n  [{i}] Art. {a.get('numero','?')} — {a.get('titulo','')[:70]} (score={a.get('score')})")
            print(f"       id: {a.get('id')}")
            texto_preview = a.get('texto', '')[:150].replace('\n', ' ')
            print(f"       texto: {texto_preview}...")
        print("=" * 60 + "\n")

    return generar_respuesta_grafo(query, articulos, log_llm=log_llm)


# ──────────────────────────────────────────────────────────────────────────────
# Main interactivo
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph RAG sobre la Ley 22/2014 (BOE)")
    parser.add_argument("--query", "-q", type=str, help="Pregunta directa")
    parser.add_argument("--modo", choices=["cypher", "grafo", "vector"], default="cypher",
                        help="Modo de recuperación (defecto: cypher). vector = búsqueda semántica por embeddings")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Mostrar Cypher generado / nodos recuperados")
    parser.add_argument("--trace", "-t", action="store_true",
                        help="Mostrar trazabilidad completa de nodos accedidos en Neo4j")
    parser.add_argument("--log-llm", action="store_true",
                        help="Imprimir en terminal los mensajes enviados al LLM")
    args = parser.parse_args()

    print(f"Modo: {args.modo.upper()}")

    def responder(query: str):
        print(f"\n{'='*60}")
        print(f"PREGUNTA: {query}")
        print(f"{'='*60}")
        if args.modo == "cypher":
            respuesta = consulta_cypher(query, verbose=args.verbose, trace=args.trace, log_llm=args.log_llm)
        elif args.modo == "vector":
            respuesta = consulta_vector(query, verbose=args.verbose, trace=args.trace, log_llm=args.log_llm)
        else:
            respuesta = consulta_grafo(query, verbose=args.verbose, trace=args.trace, log_llm=args.log_llm)
        print(f"\nRESPUESTA:\n{respuesta}\n")

    if args.query:
        responder(args.query)
    else:
        print("Modo interactivo (escribe 'salir' para terminar)\n")
        while True:
            query = input("Tu pregunta: ").strip()
            if query.lower() in ("salir", "exit", "quit", "q"):
                break
            if not query:
                continue
            responder(query)


if __name__ == "__main__":
    main()
