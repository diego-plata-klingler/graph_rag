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
Cypher: MATCH (a:Articulo)
        WHERE (a.texto CONTAINS 'SGEIC' OR a.texto CONTAINS 'sociedad gestora')
          AND (a.texto CONTAINS 'autorización' OR a.texto CONTAINS 'requisito' OR a.texto CONTAINS 'acceso a la actividad')
        RETURN a.numero, a.titulo, a.texto LIMIT 5

Pregunta: ¿Qué artículos están en el Capítulo IV?
Cypher: MATCH (a:Articulo)-[:PERTENECE_A]->(c:Capitulo)
        WHERE c.texto CONTAINS 'IV'
        RETURN a.numero, a.titulo ORDER BY a.numero LIMIT 20

Pregunta: ¿A qué artículos hace referencia el artículo 42?
Cypher: MATCH (a:Articulo)-[:REFERENCIA]->(b:Articulo)
        WHERE a.id = 'art_42'
        RETURN b.numero, b.titulo

Pregunta: ¿Qué artículos tratan sobre capital mínimo?
Cypher: MATCH (a:Articulo)
        WHERE a.texto CONTAINS 'capital mínimo' OR a.texto CONTAINS 'capital minimo'
        RETURN a.numero, a.titulo, a.texto LIMIT 3

Pregunta: ¿Cuál es el patrimonio mínimo de un FCR?
Cypher: MATCH (a:Articulo)
        WHERE a.texto CONTAINS 'patrimonio' AND (a.texto CONTAINS 'FCR' OR a.texto CONTAINS 'fondo de capital')
        RETURN a.numero, a.titulo, a.texto LIMIT 3

Pregunta: ¿Qué entidades menciona el artículo 48?
Cypher: MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad)
        WHERE a.id = 'art_48'
        RETURN e.nombre

Pregunta: ¿Qué dice el artículo 26?
Cypher: MATCH (a:Articulo)
        WHERE a.id = 'art_26'
        RETURN a.numero, a.titulo, a.texto

Pregunta: ¿Qué significan las siglas SCR, FCR, SICC y FICC y en qué artículo se definen?
Cypher: MATCH (a:Articulo)
        WHERE a.titulo CONTAINS 'Definición'
           OR a.titulo CONTAINS 'Régimen jurídico'
           OR a.titulo CONTAINS 'SICC'
           OR a.titulo CONTAINS 'FICC'
           OR a.texto CONTAINS 'SCR'
           OR a.texto CONTAINS 'FCR'
           OR a.texto CONTAINS 'SICC'
           OR a.texto CONTAINS 'FICC'
        RETURN a.numero, a.titulo, a.texto
        ORDER BY a.numero ASC
        LIMIT 5

Pregunta: ¿Qué es una SGEIC y en qué artículo se define?
Cypher: MATCH (a:Articulo)
        WHERE a.titulo CONTAINS 'sociedad gestora'
           OR a.titulo CONTAINS 'SGEIC'
           OR a.titulo CONTAINS 'Definición'
           OR (a.texto CONTAINS 'SGEIC' AND a.titulo CONTAINS 'Requisitos')
        RETURN a.numero, a.titulo, a.texto
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
Cypher: MATCH (a:Articulo)
        WHERE a.titulo CONTAINS 'Entidades de capital-riesgo'
           OR a.titulo CONTAINS 'Entidades de inversión colectiva'
           OR a.titulo CONTAINS 'Definición y régimen jurídico'
           OR a.titulo CONTAINS 'Régimen jurídico'
           OR a.texto CONTAINS 'ECR'
           OR a.texto CONTAINS 'EICC'
        RETURN a.numero, a.titulo, a.texto
        ORDER BY a.numero ASC
        LIMIT 6

Pregunta: ¿Qué son las ECR? ¿Cuál es su objeto?
Cypher: MATCH (a:Articulo)
        WHERE a.titulo CONTAINS 'Objeto'
           OR a.titulo CONTAINS 'Definición'
           OR (a.texto CONTAINS 'ECR' AND (a.titulo CONTAINS 'capital-riesgo' OR a.titulo CONTAINS 'Concepto'))
        RETURN a.numero, a.titulo, a.texto
        ORDER BY a.numero ASC
        LIMIT 5
"""

SYSTEM_PROMPT_CYPHER = f"""Eres un experto en la Ley 22/2014 y en Neo4j.
Tu tarea es generar consultas Cypher PRECISAS para recuperar los artículos más relevantes del grafo.

Esquema del grafo:
- Nodos:
  - Articulo(id, numero, titulo, texto)
  - Titulo
  - Capitulo
  - Seccion
  - Entidad(nombre)
  - Concepto(nombre)
  - Disposicion
- Relaciones:
  - PERTENECE_A
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

4. La búsqueda por contenido en `a.texto CONTAINS ...` es útil, pero NO debe usarse como única estrategia si la pregunta es definicional.
   - En preguntas definicionales, combina:
     - `a.titulo CONTAINS ...`
     - `a.texto CONTAINS ...`
   - Prioriza artículos cuyo título sugiere definición o régimen básico.

5. Para siglas o acrónimos (SCR, FCR, SICC, FICC, SGEIC, ECR, EICC, EICCP):
   - busca primero por `a.titulo CONTAINS 'sigla'`
   - y en segundo lugar por `a.texto CONTAINS 'sigla'`
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

9. Incluye SIEMPRE `a.texto` en el RETURN.

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

     # NUEVA REGLA 1
     "- Antes de responder, identifica qué artículos del contexto contienen la información relevante para la pregunta.\n"
     "- Si varios artículos regulan distintos aspectos de la respuesta, debes combinarlos en la respuesta final.\n"

     # NUEVA REGLA 2
     "- Si la pregunta se refiere a tipos de entidades, categorías o clases reguladas por la ley, debes identificar TODAS las categorías definidas en los artículos relevantes.\n"

     "- Si la información proporcionada contiene la respuesta, respóndela de forma clara y cita los artículos.\n"
     "- Si la pregunta menciona un artículo concreto (por ejemplo, artículo 26 o 4 bis), cita ESE número exacto y, cuando esté en el contexto, incluye su texto completo antes de interpretar.\n"

     "- Cuando el artículo tenga listas con letras (a), b), c)... o listas numeradas, EXTRAE Y CITA TODOS los puntos.\n"
     "- Debes reproducir TODOS los elementos de la lista exactamente como aparecen en el texto.\n"
     "- No omitas ninguno.\n"
     "- No añadas elementos nuevos.\n"

     "- El usuario puede usar términos genéricos; búscalos en el texto aunque no coincidan exactamente. "
     "'procedimientos' puede referirse a 'procedimiento de gestión del riesgo de crédito', etc.\n"

     "- CRÍTICO — NUNCA inventes números de artículo: solo cita los que aparezcan en la información.\n"
     "- Antes de responder, valida que cada artículo citado exista en la información recuperada.\n"

     "- Si la información no es suficiente, di exactamente: "
     "'La Ley 22/2014 no regula expresamente este aspecto en los artículos consultados.'\n"

     "- NUNCA digas 'no encontrado en el documento', 'no tengo información' ni frases que suenen a error técnico."
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
    conocidas = ["SGEIC", "ECR", "SICC", "FICC", "CNMV", "GFIA", "AIFMD"]
    return [e for e in conocidas if e.lower() in query.lower()]


def _articulo_id(numero_normalizado: str) -> str:
    return f"art_{numero_normalizado}"


def _merge_articulos(*listas_articulos: list[dict]) -> list[dict]:
    """Une listas de artículos sin duplicados por id, priorizando el texto más largo."""
    merged: dict[str, dict] = {}
    for lista in listas_articulos:
        for art in lista or []:
            art_id = art.get("id")
            if not art_id:
                continue
            previo = merged.get(art_id)
            if previo is None or len(str(art.get("texto", ""))) > len(str(previo.get("texto", ""))):
                merged[art_id] = art
    return list(merged.values())


def _ordenar_articulos(articulos: list[dict], ids_prioritarios: list[str] | None = None) -> list[dict]:
    ids_prioritarios = ids_prioritarios or []
    prioridad = {art_id: i for i, art_id in enumerate(ids_prioritarios)}

    def _key(art: dict):
        art_id = art.get("id", "")
        if art_id in prioridad:
            return (0, prioridad[art_id], 0, art_id)
        numero = str(art.get("numero", ""))
        m = re.match(r"(\d+)", numero)
        base = int(m.group(1)) if m else 10**9
        return (1, base, numero, art_id)

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
        directos = _fetch_articulos_by_ids(driver, ids)
        if verbose:
            print(f"  [articulo directo] solicitados={ids}, encontrados={len(directos)}")
        if not expandir:
            return directos
        vecinos = _expandir_subgrafo(driver, [a["id"] for a in directos] or ids, verbose=verbose)
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
    Recupera artículos relevantes desde Neo4j usando múltiples estrategias:
      1. Números de artículo directos en la query
      2. Entidades mencionadas en la query
      3. Búsqueda FULLTEXT por palabras clave de la query
      4. Expansión: referencias + misma sección + mismo capítulo
    """
    resultados = {}  # id -> dict

    numeros_objetivo = _extraer_numeros_articulo(query)
    ids_objetivo = [_articulo_id(n) for n in numeros_objetivo]

    with driver.session() as s:

        # -- Estrategia 1: artículo por número directo
        if numeros_objetivo and verbose:
            print(f"  [1] Artículos directos solicitados: {ids_objetivo}")
        for art_id in ids_objetivo:
            res = s.run("MATCH (a:Articulo {id: $id}) RETURN a", id=art_id)
            for record in res:
                a = dict(record["a"])
                resultados[a["id"]] = a

        # -- Estrategia 2: por entidad mencionada en la query
        entidades = _extraer_entidades_query(query)
        if entidades and verbose:
            print(f"  [2] Entidades: {entidades}")
        for ent in entidades:
            res = s.run("""
                MATCH (a:Articulo)-[:MENCIONA]->(e:Entidad {nombre: $nombre})
                RETURN a LIMIT 15
            """, nombre=ent)
            for record in res:
                a = dict(record["a"])
                resultados[a["id"]] = a

        # -- Estrategia 3: FULLTEXT sobre titulo + texto
        palabras = [p for p in re.split(r'\W+', query) if len(p) > 4]
        if palabras:
            lucene_query = " OR ".join(palabras[:6])
            if verbose:
                print(f"  [3] Fulltext: {lucene_query}")
            try:
                res = s.run("""
                    CALL db.index.fulltext.queryNodes('articulo_texto', $q)
                    YIELD node, score
                    WHERE score > 0.2
                    RETURN node AS a ORDER BY score DESC LIMIT 15
                """, q=lucene_query)
                for record in res:
                    a = dict(record["a"])
                    resultados[a["id"]] = a
            except Exception:
                # Fallback si fulltext no está disponible: CONTAINS
                for p in palabras[:5]:
                    res = s.run("""
                        MATCH (a:Articulo)
                        WHERE toLower(a.titulo) CONTAINS toLower($p)
                           OR toLower(a.texto)  CONTAINS toLower($p)
                        RETURN a LIMIT 8
                    """, p=p)
                    for record in res:
                        a = dict(record["a"])
                        resultados[a["id"]] = a

        if verbose:
            print(f"  Recuperados inicialmente: {len(resultados)} artículos")

    # -- Estrategia 4: expansión de subgrafo (referencias + sección + capítulo)
    ids_base = list(resultados.keys()) or ids_objetivo
    if ids_base:
        expandidos = _expandir_subgrafo(driver, ids_base, verbose=verbose)
        nuevos = 0
        for art in expandidos:
            art_id = art.get("id")
            if art_id and art_id not in resultados:
                resultados[art_id] = art
                nuevos += 1
        if verbose and nuevos > 0:
            print(f"  [4] Expandidos (referencias/seccion/capitulo): +{nuevos} artículos")

    ordenados = _ordenar_articulos(list(resultados.values()), ids_prioritarios=ids_objetivo)
    return ordenados[:MAX_ARTICULOS_CONTEXTO]


def construir_contexto(
    articulos: list[dict],
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
        texto = art.get("texto", "") if incluir_completo else art.get("texto", "")[:8000]
        bloque = header + "\n" + texto
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
            "Debes mencionar esos números exactos y, cuando el contexto incluya su contenido íntegro, "
            "citar el texto completo antes de interpretar."
        )

    messages = [
        SystemMessage(content=(
            "Eres un asistente jurídico experto en la Ley 22/2014 de entidades de capital-riesgo. "
            "Responde en español basándote SOLO en el contexto proporcionado. "
            "El usuario puede usar términos coloquiales o genéricos; interpreta la intención y usa la información equivalente del contexto. "
            "Cuando el contexto tenga listas con (a), b), c)...) o números, EXTRAE Y CITA TODOS los puntos: son obligaciones o requisitos legales que debes incluir en la respuesta. "
            "CRÍTICO: solo cita artículos que aparezcan en el texto. "
            "Valida internamente que cada artículo mencionado en tu salida exista en el contexto. "
            "Si realmente no hay información relacionada, responde: "
            "'La Ley 22/2014 no regula expresamente este aspecto en los artículos consultados.' "
            "No digas 'no encontrado en el documento' ni frases que suenen a error técnico."
            f"{instruccion_objetivo}"
        )),
        HumanMessage(content=(
            f"Contexto de la ley:\n{contexto}\n\n"
            f"Pregunta del usuario: {query}\n"
            "Menciona siempre el número de artículo en la respuesta."
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
        fetched = _fetch_articulos_by_ids(driver, missing_ids) if missing_ids else []
        semilla_expansion = list(ids_faltan) if missing_ids else ids_directos
        expandidos = _expandir_subgrafo(driver, semilla_expansion, verbose=verbose or trace) if semilla_expansion else []
    finally:
        driver.close()

    reforzados = _merge_articulos(base_contexto, directos, fetched, expandidos)
    if not reforzados:
        return "La Ley 22/2014 no regula expresamente este aspecto en los artículos consultados."

    articulos_completos = set(faltan) | ids_faltan
    contexto_reforzado = construir_contexto(
        reforzados,
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
        return "La Ley 22/2014 no regula expresamente este aspecto en los artículos consultados."

    return respuesta_reforzada


# ──────────────────────────────────────────────────────────────────────────────
# Fallback por texto
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_texto(query: str) -> str:
    """Busca artículos por coincidencia de palabras clave en su texto."""
    SINONIMOS = {
        "presupuesto": "patrimonio",
        "presupuesto minimo": "patrimonio comprometido",
        "capital minimo": "patrimonio comprometido",
        "capital inicial": "patrimonio comprometido",
        "inversion minima": "patrimonio comprometido",
        "dinero minimo": "patrimonio comprometido",
        "importe minimo": "patrimonio comprometido",
        "cuota minima": "patrimonio comprometido",
        "aportacion minima": "patrimonio comprometido",
        "administrador": "gestor",
        "director": "gestor",
        "multa": "sancion",
        "penalizacion": "sancion",
        "socio": "participe",
        "accionista": "participe",
        "inversor": "participe",
    }

    PALABRAS_DEFINICION = {"qué es", "que es", "qué son", "que son", "definicion",
                           "definición", "naturaleza", "objeto", "formas jurídicas",
                           "formas juridicas", "diferencia entre", "diferencia",
                           "concepto", "caracterizan", "constituyen", "cuál es"}
    es_pregunta_definitoria = any(expr in query.lower() for expr in PALABRAS_DEFINICION)

    stopwords = {"cual", "cuál", "como", "cómo", "qué", "que", "para", "sobre",
                 "tiene", "debe", "deben", "puede", "pueden", "artículo",
                 "articulo", "ley", "regulan", "regula", "dice", "indica", "fondo"}

    query_norm = query.lower().strip("?¿.,:")
    for sinonimo, legal in SINONIMOS.items():
        if sinonimo in query_norm:
            query_norm = query_norm.replace(sinonimo, legal)

    palabras = [w.strip("?¿.,:") for w in query_norm.split()
                if (len(w) > 4 or w.strip("?¿.,:").isupper()) and w.strip("?¿.,:") not in stopwords]
    acronimos = [w.strip("?¿.,:;") for w in query.split()
                 if w.strip("?¿.,:;").isupper() and len(w.strip("?¿.,:;")) >= 2]
    if not palabras and not acronimos:
        return ""

    ACRONIMOS_LEY = {
        "SCR":   "Sociedad de Capital-Riesgo (artículo 26)",
        "FCR":   "Fondo de Capital-Riesgo (artículo 30)",
        "SICC":  "Sociedad de Inversión Colectiva de Tipo Cerrado (artículo 38)",
        "FICC":  "Fondo de Inversión Colectiva de Tipo Cerrado (artículo 38)",
        "SGEIC": "Sociedad Gestora de Entidades de Inversión Colectiva de Tipo Cerrado (artículo 41)",
        "ECR":   "Entidad de Capital-Riesgo (artículo 3)",
        "EICC":  "Entidad de Inversión Colectiva de Tipo Cerrado (artículo 4)",
        "EICCP": "Entidad de Inversión Colectiva de Tipo Cerrado de Préstamos (artículo 4 bis)",
        "CNMV":  "Comisión Nacional del Mercado de Valores",
    }
    acr_encontrados = [acr for acr in acronimos if acr in ACRONIMOS_LEY]
    contexto_acronimos = ""
    if acr_encontrados:
        lineas = [f"- {acr}: {ACRONIMOS_LEY[acr]}" for acr in acr_encontrados]
        contexto_acronimos = "Expansión de siglas según la Ley 22/2014:\n" + "\n".join(lineas) + "\n\n"

    numeros_objetivo = _extraer_numeros_articulo(query)
    ids_objetivo = [_articulo_id(n) for n in numeros_objetivo]

    driver = get_driver()
    resultados: dict[str, dict] = {}
    try:
        with driver.session() as s:
            # Artículo directo cuando viene explícito en la pregunta.
            if ids_objetivo:
                directos = _fetch_articulos_by_ids(driver, ids_objetivo)
                for art in directos:
                    resultados[art["id"]] = art

            # Búsqueda prioritaria para preguntas definitorias: por título del artículo.
            if es_pregunta_definitoria:
                titulos_def = [
                    "Objeto", "Naturaleza", "Formas", "Definición", "Concepto",
                    "Régimen jurídico", "Inversión colectiva", "Entidades de inversión",
                    "Ámbito de aplicación",
                ]
                for titulo_kw in titulos_def:
                    rows = s.run(
                        "MATCH (a:Articulo) WHERE a.titulo CONTAINS $kw "
                        "RETURN a ORDER BY a.numero LIMIT 6",
                        kw=titulo_kw,
                    )
                    for r in rows:
                        art = dict(r["a"])
                        resultados[art["id"]] = art

                for palabra in palabras[:4]:
                    rows = s.run(
                        "MATCH (a:Articulo) WHERE toLower(a.texto) CONTAINS toLower($kw) "
                        "AND toInteger(split(a.numero, ' ')[0]) <= 15 "
                        "RETURN a ORDER BY a.numero LIMIT 4",
                        kw=palabra,
                    )
                    for r in rows:
                        art = dict(r["a"])
                        resultados[art["id"]] = art

            # Búsqueda específica por acrónimos.
            for acr in acronimos[:6]:
                rows = s.run(
                    "MATCH (a:Articulo) WHERE a.titulo CONTAINS $kw OR a.texto CONTAINS $kw "
                    "RETURN a ORDER BY a.numero LIMIT 4",
                    kw=acr,
                )
                for r in rows:
                    art = dict(r["a"])
                    resultados[art["id"]] = art

            # Búsqueda por palabras clave.
            for palabra in palabras[:5]:
                rows = s.run(
                    "MATCH (a:Articulo) WHERE toLower(a.texto) CONTAINS toLower($kw) "
                    "RETURN a LIMIT 4",
                    kw=palabra,
                )
                for r in rows:
                    art = dict(r["a"])
                    resultados[art["id"]] = art

            # Si pide cantidades, priorizar artículos con importes.
            pide_cantidad = any(
                w in query.lower()
                for w in ["minimo", "mínimo", "cuanto", "cuánto", "importe", "cifra", "cantidad", "euros", "presupuesto"]
            )
            if pide_cantidad:
                for palabra in palabras[:4]:
                    rows = s.run(
                        "MATCH (a:Articulo) WHERE toLower(a.texto) CONTAINS toLower($kw) "
                        "AND a.texto CONTAINS 'euros' "
                        "RETURN a LIMIT 4",
                        kw=palabra,
                    )
                    for r in rows:
                        art = dict(r["a"])
                        resultados[art["id"]] = art

        ids_semilla = list(resultados.keys()) or ids_objetivo
        if ids_semilla:
            for art in _expandir_subgrafo(driver, ids_semilla, verbose=False):
                resultados[art["id"]] = art
    finally:
        driver.close()

    articulos = _ordenar_articulos(list(resultados.values()), ids_prioritarios=ids_objetivo)[:MAX_ARTICULOS_CONTEXTO]
    if not articulos and not contexto_acronimos:
        return ""

    contexto_articulos = construir_contexto(
        articulos,
        articulos_completos=set(ids_objetivo) | set(numeros_objetivo),
        ids_prioritarios=ids_objetivo,
    )
    if contexto_acronimos and contexto_articulos:
        return f"{contexto_acronimos}{contexto_articulos}"
    return contexto_acronimos or contexto_articulos


# ──────────────────────────────────────────────────────────────────────────────
# MODO A: consulta_cypher
# ──────────────────────────────────────────────────────────────────────────────

def consulta_cypher(query: str, verbose: bool = False, trace: bool = False, log_llm: bool = False) -> str:
    """Usa GraphCypherQAChain: pregunta -> Cypher -> Neo4j -> respuesta."""
    articulos_objetivo = _extraer_numeros_articulo(query)
    ids_objetivo = [_articulo_id(n) for n in articulos_objetivo]
    articulos_directos = _recuperar_articulos_objetivo(query, expandir=True, verbose=verbose or trace)

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

    result = chain.invoke({
        "query": f"Responde en español. Ley 22/2014 (BOE).{nota_siglas}\nPregunta: {query}"
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

    respuesta = result.get("result", str(result))
    contexto_directo = ""
    if articulos_directos:
        contexto_directo = construir_contexto(
            articulos_directos,
            articulos_completos=set(articulos_objetivo) | set(ids_objetivo),
            ids_prioritarios=ids_objetivo,
        )
        if verbose or trace:
            print(f"[articulo directo] Contexto anadido: {len(articulos_directos)} articulos")

    # -- Si el Cypher devolvió datos pero la chain respondió "no regula",
    #    sintetizar directamente desde el contexto de Neo4j (más fiable que la QA interna)
    respuesta_lower = respuesta.lower()
    no_encontro = (
        not respuesta
        or "no s" in respuesta_lower
        or "no tengo" in respuesta_lower
        or "no encontr" in respuesta_lower
        or "no hay inform" in respuesta_lower
        or "no regula expresamente" in respuesta_lower
    )

    if no_encontro and cypher_rows:
        contexto_cypher = "\n\n---\n\n".join(
            "\n".join(f"{k}: {v}" for k, v in row.items())
            for row in cypher_rows
        )
        if contexto_directo:
            contexto_cypher = f"{contexto_cypher}\n\n---\n\n{contexto_directo}"
        if verbose or trace:
            print(f"[resintetizar desde Cypher] {len(cypher_rows)} filas, {len(contexto_cypher)} chars")
        respuesta = _sintetizar_respuesta(
            get_llm(log_llm=log_llm),
            query,
            contexto_cypher,
            articulos_objetivo=articulos_objetivo,
        )
        respuesta_lower = respuesta.lower()
        no_encontro = "no regula expresamente" in respuesta_lower

    if no_encontro and contexto_directo:
        if verbose or trace:
            print("[resintetizar directo] Usando texto integro del/los articulo(s) solicitado(s).")
        respuesta = _sintetizar_respuesta(
            get_llm(log_llm=log_llm),
            query,
            contexto_directo,
            articulos_objetivo=articulos_objetivo,
        )
        respuesta_lower = respuesta.lower()
        no_encontro = "no regula expresamente" in respuesta_lower

    if no_encontro:
        contexto_fallback = _fallback_texto(query)
        if contexto_fallback:
            if verbose or trace:
                print(f"[fallback] Buscando por palabras clave... ({len(contexto_fallback)} chars de contexto)")
            respuesta = _sintetizar_respuesta(
                get_llm(log_llm=log_llm),
                query,
                contexto_fallback,
                articulos_objetivo=articulos_objetivo,
            )
        else:
            respuesta = "La Ley 22/2014 no regula expresamente este aspecto en los artículos consultados."

    contexto_cypher = "\n\n---\n\n".join(
        "\n".join(f"{k}: {v}" for k, v in row.items())
        for row in cypher_rows
    ) if cypher_rows else ""
    contexto_validacion = "\n\n---\n\n".join(c for c in [contexto_cypher, contexto_directo] if c)
    return _validar_y_reforzar_citas(
        respuesta,
        query,
        contexto_validacion,
        log_llm=log_llm,
        verbose=verbose,
        trace=trace,
        articulos_contexto=articulos_directos,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MODO B: Recuperación manual por grafo + expansión
# ──────────────────────────────────────────────────────────────────────────────

def generar_respuesta_grafo(query: str, articulos: list[dict], log_llm: bool = False) -> str:
    """Genera respuesta con el LLM dado el contexto del grafo."""
    articulos_objetivo = _extraer_numeros_articulo(query)
    ids_objetivo = [_articulo_id(n) for n in articulos_objetivo]
    articulos_directos = _recuperar_articulos_objetivo(query, expandir=True, verbose=False)
    articulos = _merge_articulos(articulos, articulos_directos)

    if not articulos:
        return "No se encontraron artículos relevantes en el grafo para responder esta pregunta."

    contexto = construir_contexto(
        articulos,
        articulos_completos=set(ids_objetivo) | set(articulos_objetivo),
        ids_prioritarios=ids_objetivo,
    )
    llm = get_llm(log_llm=log_llm)

    from langchain_core.messages import HumanMessage, SystemMessage

    system = """Eres un asistente jurídico experto en la Ley 22/2014 de entidades de capital-riesgo \
y entidades de inversión colectiva de tipo cerrado (España).

Responde la pregunta basándote EXCLUSIVAMENTE en los artículos proporcionados.
- Cita siempre el número de artículo.
- Si la pregunta menciona un artículo concreto, debes citar ese número exacto.
- Si el contexto trae el texto completo del artículo solicitado, inclúyelo de forma íntegra.
-Después de citar el artículo, limita la explicación a reformular o resumir su contenido.
-No añadas información externa ni explicaciones doctrinales.
- Si la información es insuficiente, indícalo claramente.
- Responde en español, de forma estructurada."""

    human = f"""Pregunta: {query}

Artículos del grafo:
{contexto}"""

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return _validar_y_reforzar_citas(
        response.content,
        query,
        contexto,
        log_llm=log_llm,
        articulos_contexto=articulos,
    )


def consulta_grafo(query: str, verbose: bool = False, trace: bool = False, log_llm: bool = False) -> str:
    """Recuperación manual por grafo + expansión de subgrafo."""
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
    """Comprueba si el índice vectorial `articulo_vector` existe en Neo4j."""
    global _VECTOR_INDEX_AVAILABLE
    if _VECTOR_INDEX_AVAILABLE is not None:
        return _VECTOR_INDEX_AVAILABLE

    try:
        with driver.session() as s:
            res = s.run("SHOW INDEXES")
            for r in res:
                info = r.data()
                if info.get("name") == "articulo_vector":
                    _VECTOR_INDEX_AVAILABLE = True
                    return True
    except Exception:
        pass

    _VECTOR_INDEX_AVAILABLE = False
    return False


def consulta_vector(query: str, verbose: bool = False, trace: bool = False, log_llm: bool = False) -> str:
    """Recuperación semántica con embeddings (vector search en Neo4j)."""
    articulos_objetivo = _extraer_numeros_articulo(query)
    ids_objetivo = [_articulo_id(n) for n in articulos_objetivo]

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

    def _vector_search_fallback(driver, query_vector, top_k=MAX_ARTICULOS_CONTEXTO):
        """Busca en Neo4j cargando embeddings y ordenando por similitud en Python."""
        rows = driver.session().run(
            "MATCH (a:Articulo) WHERE a.embedding IS NOT NULL AND size(a.embedding) > 0 RETURN a",
        ).data()
        candidates = []
        for r in rows:
            node = r["a"]
            emb = node.get("embedding") or []
            if not emb:
                continue
            score = _cosine_similarity(query_vector, emb)
            candidates.append((score, node))
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, node in candidates[:top_k]:
            results.append({
                "id": node.get("id"),
                "numero": node.get("numero"),
                "titulo": node.get("titulo"),
                "texto": node.get("texto"),
                "score": score,
            })
        return results

    emb = _get_emb(EMBED_MODEL)
    if emb is None and EMBED_MODEL != "text-embedding-3-small":
        if verbose or trace:
            print("[embed] Reintentando con text-embedding-3-small...")
        emb = _get_emb("text-embedding-3-small")
    if emb is None:
        raise RuntimeError("No se pudo inicializar ningún modelo de embeddings para vector search.")

    vector = emb.embed_query(query)

    driver = get_driver()
    articulos = []
    with driver.session() as s:
        if USE_VECTOR_INDEX and _has_vector_index(driver):
            try:
                rows = s.run(
                    "CALL db.index.vector.queryNodes('articulo_vector', $k, $vector) "
                    "YIELD node, score "
                    "RETURN node, score ORDER BY score DESC LIMIT $k",
                    vector=vector,
                    k=MAX_ARTICULOS_CONTEXTO,
                ).data()
                for r in rows:
                    node = r["node"]
                    articulos.append({
                        "id": node.get("id"),
                        "numero": node.get("numero"),
                        "titulo": node.get("titulo"),
                        "texto": node.get("texto"),
                        "score": r.get("score"),
                    })

                # Si la búsqueda vectorial no produce suficientes resultados, hacer una
                # búsqueda de respaldo por texto para aumentar la cobertura.
                if len(articulos) < 4 or (articulos and max(a.get("score", 0) or 0 for a in articulos) < 0.35):
                    if verbose:
                        print("[vector] Resultados limitados; complemento con busqueda por texto...")
                    pivot_ids = {a["id"] for a in articulos}
                    texto_fallback = buscar_articulos(driver, query, verbose=verbose)
                    for a in texto_fallback:
                        if a["id"] not in pivot_ids:
                            articulos.append(a)
            except Exception as e:
                if verbose or trace:
                    print(f"[vector] Error ejecutando vector search en Neo4j: {e}")
                articulos = _vector_search_fallback(driver, vector)
        else:
            if verbose or trace:
                if not USE_VECTOR_INDEX:
                    print("[vector] Vector index deshabilitado via config (USE_VECTOR_INDEX=False). Usando fallback local (cosine similarity).")
                else:
                    print("[vector] No hay indice vectorial disponible; usando fallback local (cosine similarity).")
            articulos = _vector_search_fallback(driver, vector)

        directos = _fetch_articulos_by_ids(driver, ids_objetivo) if ids_objetivo else []
        ids_semilla = list(dict.fromkeys(ids_objetivo + [a.get("id") for a in articulos if a.get("id")]))
        expandidos = _expandir_subgrafo(driver, ids_semilla, verbose=verbose or trace) if ids_semilla else []
        articulos = _ordenar_articulos(
            _merge_articulos(directos, articulos, expandidos),
            ids_prioritarios=ids_objetivo,
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
