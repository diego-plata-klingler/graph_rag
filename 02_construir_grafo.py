"""
Script 2: Construye el grafo en Neo4j.

Lee data/articulos.json y crea el siguiente esquema:

  NODOS
  ─────
  (:Titulo   {id, texto})
  (:Capitulo {id, texto, titulo_padre})
  (:Seccion  {id, texto})
  (:Articulo {id, numero, titulo, texto, titulo_padre, capitulo_padre, seccion_padre})
  (:Disposicion {id, titulo, texto})
  (:Entidad  {nombre})          ← SGEIC, CNMV, ECR…
  (:Concepto {nombre})          ← autorización, capital mínimo…

  RELACIONES
  ──────────
  (Articulo)-[:PERTENECE_A]->(Titulo)
  (Articulo)-[:PERTENECE_A]->(Capitulo)
  (Articulo)-[:REFERENCIA]->(Articulo)   ← "conforme al artículo X"
  (Articulo)-[:MENCIONA]->(Entidad)
  (Articulo)-[:TRATA_SOBRE]->(Concepto)
  (Capitulo)-[:PERTENECE_A]->(Titulo)

Opciones en config.py:
  USAR_LLM_TRANSFORMER = True  → añade entidades/relaciones extraídas por LLM

Uso:
    python 02_construir_grafo.py
    python 02_construir_grafo.py --limpiar   (borra el grafo anterior)
"""

import json
import os
import sys
import argparse
from tqdm import tqdm

from neo4j import GraphDatabase

from langchain_openai import OpenAIEmbeddings

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OPENAI_API_KEY, LLM_MODEL, EMBED_MODEL, USE_VECTOR_INDEX,
    ARTICULOS_JSON, USAR_LLM_TRANSFORMER, MAX_ARTICULOS_LLM,
)


# ──────────────────────────────────────────────────────────────────────────────
# Conexión
# ──────────────────────────────────────────────────────────────────────────────

def get_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


# ──────────────────────────────────────────────────────────────────────────────
# Schema: constraints e índices
# ──────────────────────────────────────────────────────────────────────────────

CONSTRAINTS = [
    "CREATE CONSTRAINT articulo_id IF NOT EXISTS FOR (a:Articulo) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT titulo_id   IF NOT EXISTS FOR (t:Titulo)   REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT capitulo_id IF NOT EXISTS FOR (c:Capitulo) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT seccion_id  IF NOT EXISTS FOR (s:Seccion)  REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT entidad_nom IF NOT EXISTS FOR (e:Entidad)  REQUIRE e.nombre IS UNIQUE",
    "CREATE CONSTRAINT concepto_nom IF NOT EXISTS FOR (c:Concepto) REQUIRE c.nombre IS UNIQUE",
    "CREATE CONSTRAINT disposicion_id IF NOT EXISTS FOR (d:Disposicion) REQUIRE d.id IS UNIQUE",
]


def crear_schema(driver, embed_dim=None, use_vector_index: bool = True):
    """Crea constraints e índices en Neo4j.

    Si `embed_dim` se proporciona y `use_vector_index` es True, intenta crear un índice vectorial.
    """
    with driver.session() as s:
        for q in CONSTRAINTS:
            s.run(q)

        indices = [
            "CREATE FULLTEXT INDEX articulo_texto IF NOT EXISTS FOR (a:Articulo) ON EACH [a.titulo, a.texto]",
            "CREATE FULLTEXT INDEX entidad_nombre IF NOT EXISTS FOR (e:Entidad)  ON EACH [e.nombre]",
        ]

        if embed_dim is not None and use_vector_index:
            indices.append(
                f"CREATE INDEX articulo_vector IF NOT EXISTS FOR (a:Articulo) "
                f"ON (a.embedding) OPTIONS {{indexProvider: 'vector-1', dimensions: {embed_dim}}}"
            )
        elif use_vector_index:
            # Si se solicitó vector index pero no podemos determinar dimensiones,
            # intentamos crear el índice con la dimensión por defecto.
            indices.append(
                f"CREATE INDEX articulo_vector IF NOT EXISTS FOR (a:Articulo) "
                f"ON (a.embedding) OPTIONS {{indexProvider: 'vector-1', dimensions: {embed_dim or 1536}}}"
            )

        for q in indices:
            try:
                s.run(q)
            except Exception as e:
                # Algunos despliegues de Neo4j (Community / sin plugins) no soportan
                # índices fulltext o vectoriales; no queremos abortar el proceso.
                if "vector" in q.lower():
                    print("[warn] No se pudo crear el índice vectorial (probablemente no soportado en esta versión de Neo4j). Seguiremos sin vector search en Neo4j.")
                elif "fulltext" in q.lower():
                    print("[warn] No se pudo crear el índice fulltext (probablemente no soportado en esta versión de Neo4j).")
                else:
                    print(f"[warn] No se pudo crear un índice: {e}")

        # Comprobar si el índice vectorial realmente se creó (para dar feedback útil)
        if use_vector_index and embed_dim is not None:
            try:
                result = s.run("SHOW INDEXES")
                found = any(r.data().get("name") == "articulo_vector" for r in result)
                if not found:
                    print("[info] El índice vectorial 'articulo_vector' no está disponible en este Neo4j; la búsqueda vectorial nativa no funcionará.")
            except Exception:
                # En algunas versiones/ediciones SHOW INDEXES puede no existir, ignoramos.
                pass


def infer_embedding_dim(emb_model, sample_text: str = "hola"):
    """Intenta inferir la dimensión de los embeddings consultando al modelo."""
    try:
        emb = emb_model.embed_query(sample_text)
        return len(emb)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Nodos y relaciones desde el JSON parseado
# ──────────────────────────────────────────────────────────────────────────────

def _id_seguro(texto: str) -> str:
    """Genera un id slug desde un texto largo."""
    import re
    return re.sub(r'[^a-z0-9_]', '_', texto.lower().strip())[:60]


def crear_titulos_capitulos(driver, estructura: dict):
    """Crea nodos Titulo y Capitulo con relación CONTIENE."""
    with driver.session() as s:
        for titulo in estructura["titulos"]:
            tid = _id_seguro(titulo)
            s.run("MERGE (n:Titulo {id: $id}) SET n.texto = $texto", id=tid, texto=titulo)

        for cap in estructura["capitulos"]:
            cid = _id_seguro(cap)
            s.run("MERGE (n:Capitulo {id: $id}) SET n.texto = $texto", id=cid, texto=cap)

        seen_sec = set()
        for sec in estructura["secciones"]:
            sid = _id_seguro(sec)
            if sid in seen_sec:
                continue
            seen_sec.add(sid)
            s.run("MERGE (n:Seccion {id: $id}) SET n.texto = $texto", id=sid, texto=sec)


def crear_articulos(driver, articulos: list[dict], emb_model):
    """Crea nodos Articulo y sus relaciones jerárquicas.

    Para mejorar la calidad de la búsqueda semántica, los embeddings se calculan
en batches y se basan en el título + el texto del artículo.
    """

    batch_size = 50

    with driver.session() as s:
        for inicio in range(0, len(articulos), batch_size):
            batch = articulos[inicio:inicio + batch_size]

            textos_para_embed = []
            for art in batch:
                texto = art.get("texto", "")[:8000]  # límite para Neo4j
                titulo = art.get("titulo", "")
                textos_para_embed.append(f"{titulo}\n\n{texto}" if titulo else texto)

            try:
                embeddings = emb_model.embed_documents(textos_para_embed)
            except Exception as e:
                msg = str(e)
                print(f"[embed] Error al crear embeddings para artículos {inicio}-{inicio+len(batch)-1}: {e}")

                # Si el modelo no está disponible para el proyecto, intentar con el modelo pequeño.
                if "model_not_found" in msg or "does not have access" in msg:
                    print("[embed] Modelo no accesible; reintentando con text-embedding-3-small...")
                    emb_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
                    try:
                        embeddings = emb_model.embed_documents(textos_para_embed)
                    except Exception as e2:
                        print(f"[embed] Error al reintentar con text-embedding-3-small: {e2}")
                        embeddings = [None] * len(batch)
                else:
                    embeddings = [None] * len(batch)

            for art, embedding in zip(batch, embeddings):
                texto = art.get("texto", "")[:8000]
                if embedding is None:
                    embedding = []

                # Nodo artículo
                s.run("""
                    MERGE (a:Articulo {id: $id})
                    SET a.numero       = $numero,
                        a.titulo       = $titulo,
                        a.texto        = $texto,
                        a.titulo_padre   = $titulo_padre,
                        a.capitulo_padre = $capitulo_padre,
                        a.seccion_padre  = $seccion_padre,
                        a.embedding     = $embedding
                """,
                    id=art["id"],
                    numero=art["numero"],
                    titulo=art["titulo"],
                    texto=texto,
                    titulo_padre=art.get("titulo_padre") or "",
                    capitulo_padre=art.get("capitulo_padre") or "",
                    seccion_padre=art.get("seccion_padre") or "",
                    embedding=embedding,
                )

                # Relación con Título padre
                if art.get("titulo_padre"):
                    tid = _id_seguro(art["titulo_padre"])
                    s.run("""
                        MATCH (a:Articulo {id: $aid}), (t:Titulo {id: $tid})
                        MERGE (a)-[:PERTENECE_A]->(t)
                    """, aid=art["id"], tid=tid)

                # Relación con Capítulo padre
                if art.get("capitulo_padre"):
                    cid = _id_seguro(art["capitulo_padre"])
                    s.run("""
                        MATCH (a:Articulo {id: $aid}), (c:Capitulo {id: $cid})
                        MERGE (a)-[:PERTENECE_A]->(c)
                    """, aid=art["id"], cid=cid)


def crear_referencias_cruzadas(driver, articulos: list[dict]):
    """Crea relaciones REFERENCIA entre artículos."""
    # Índice rápido de id existentes
    ids_existentes = {a["id"] for a in articulos}

    with driver.session() as s:
        count = 0
        for art in tqdm(articulos, desc="  Referencias cruzadas", unit="art"):
            for ref_num in art.get("referencias", []):
                ref_id = f"art_{ref_num}"
                if ref_id in ids_existentes and ref_id != art["id"]:
                    s.run("""
                        MATCH (a:Articulo {id: $from_id}), (b:Articulo {id: $to_id})
                        MERGE (a)-[:REFERENCIA]->(b)
                    """, from_id=art["id"], to_id=ref_id)
                    count += 1
        print(f"    {count} referencias cruzadas creadas")


def crear_entidades_y_conceptos(driver, articulos: list[dict]):
    """Crea nodos Entidad/Concepto y relaciones MENCIONA / TRATA_SOBRE."""
    with driver.session() as s:
        for art in tqdm(articulos, desc="  Entidades y conceptos", unit="art"):
            for ent in art.get("entidades_mencionadas", []):
                s.run("MERGE (:Entidad {nombre: $nombre})", nombre=ent)
                s.run("""
                    MATCH (a:Articulo {id: $aid}), (e:Entidad {nombre: $nombre})
                    MERGE (a)-[:MENCIONA]->(e)
                """, aid=art["id"], nombre=ent)

            for con in art.get("conceptos_mencionados", []):
                s.run("MERGE (:Concepto {nombre: $nombre})", nombre=con)
                s.run("""
                    MATCH (a:Articulo {id: $aid}), (c:Concepto {nombre: $nombre})
                    MERGE (a)-[:TRATA_SOBRE]->(c)
                """, aid=art["id"], nombre=con)


def crear_disposiciones(driver, disposiciones: list[dict]):
    """Crea nodos Disposicion."""
    with driver.session() as s:
        for disp in disposiciones:
            s.run("""
                MERGE (d:Disposicion {id: $id})
                SET d.titulo = $titulo, d.texto = $texto
            """, id=disp["id"], titulo=disp["titulo"], texto=disp.get("texto", "")[:4000])


# ──────────────────────────────────────────────────────────────────────────────
# Opcional: LLMGraphTransformer
# ──────────────────────────────────────────────────────────────────────────────

def enriquecer_con_llm(driver, articulos: list[dict]):
    """Usa LLMGraphTransformer (LangChain) para extraer entidades y relaciones.

    Cuando Neo4j no tiene APOC instalado (como en este setup), LangChain intenta
    ejecutar `apoc.meta.data()` al inicializar el grafo, lo que falla.

    Por eso se fuerza `refresh_schema=False` y se establece un schema manual simple.
    """
    from langchain_openai import ChatOpenAI
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_core.documents import Document
    from langchain_neo4j import Neo4jGraph

    print(f"\n🤖  LLMGraphTransformer (máx. {MAX_ARTICULOS_LLM} artículos)...")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    transformer = LLMGraphTransformer(
        llm=llm,
        # Tipos de nodo y relación que queremos extraer
        allowed_nodes=["Entidad", "Concepto", "Requisito", "Obligacion", "Plazo"],
        allowed_relationships=["REQUIERE", "IMPLICA", "DEFINE", "COMPLEMENTA", "EXCLUYE"],
        node_properties=["descripcion"],
    )

    # LangChain Neo4jGraph para add_graph_documents
    # Evitar que intente usar APOC (apoc.meta.data) si no está instalado
    schema_manual = """
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

    # Verificar si APOC está disponible (Neo4j Desktop / Docker suelen no incluirlo por defecto)
    try:
        with driver.session() as s:
            s.run("CALL apoc.help('meta.data') YIELD name RETURN name LIMIT 1").consume()
    except Exception:
        print("[LLMGraphTransformer] APOC no está disponible en Neo4j. Saltando extracción de entidades/relaciones con LLM.")
        print("  - Para habilitarla, instala el plugin APOC en Neo4j o pon USAR_LLM_TRANSFORMER=False en config.py")
        return

    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            refresh_schema=False,
        )
        graph.schema = schema_manual
    except Exception as e:
        print("[LLMGraphTransformer] No se pudo inicializar Neo4jGraph (posible falta de APOC).")
        print("  - Si quieres usar LLMGraphTransformer, instala APOC en Neo4j o pon USAR_LLM_TRANSFORMER=False en config.py")
        print(f"  - Error: {e}")
        return

    lote = articulos if MAX_ARTICULOS_LLM == -1 else articulos[:MAX_ARTICULOS_LLM]

    for art in tqdm(lote, desc="  LLMTransformer", unit="art"):
        doc = Document(
            page_content=art.get("texto", ""),
            metadata={"source": art["id"], "articulo": art["numero"]},
        )
        try:
            graph_docs = transformer.convert_to_graph_documents([doc])
            graph.add_graph_documents(
                graph_docs,
                baseEntityLabel=True,
                include_source=True,
            )
        except Exception as e:
            print(f"    ⚠️  Error en {art['id']}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limpiar", action="store_true",
                        help="Borra todo el grafo antes de construirlo")
    args = parser.parse_args()

    # Cargar datos
    print(f"📂  Cargando {ARTICULOS_JSON}...")
    with open(ARTICULOS_JSON, "r", encoding="utf-8") as f:
        estructura = json.load(f)

    articulos    = estructura["articulos"]
    disposiciones = estructura["disposiciones"]
    print(f"   {len(articulos)} artículos, {len(disposiciones)} disposiciones")

    # Conectar a Neo4j
    print(f"\n🔌  Conectando a Neo4j ({NEO4J_URI})...")
    driver = get_driver()
    print("   Conexión OK")

    # Limpiar si se pide
    if args.limpiar:
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        print("   Grafo vaciado")

    # Crear modelo de embeddings y determinar la dimensión para el índice vectorial
    def _get_emb_model(model_name: str):
        try:
            return OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
        except Exception as e:
            print(f"[embed] No se pudo inicializar el modelo '{model_name}': {e}")
            return None

    used_model = EMBED_MODEL
    emb_model = _get_emb_model(used_model)
    if emb_model is None and used_model != "text-embedding-3-small":
        used_model = "text-embedding-3-small"
        print(f"[embed] Reintentando con '{used_model}'...")
        emb_model = _get_emb_model(used_model)

    if emb_model is None:
        raise RuntimeError("No se pudo inicializar ningún modelo de embeddings. Revisa tu clave/API y acceso a modelos.")

    embed_dim = infer_embedding_dim(emb_model) or 1536
    print(f"   Usando embeddings '{used_model}' (dim={embed_dim})")

    # Schema
    print("\n⚙️  Creando constraints e índices...")
    crear_schema(driver, embed_dim=embed_dim, use_vector_index=USE_VECTOR_INDEX)

    # Nodos y relaciones
    print("\n🏗️  Construyendo grafo:")
    print("  [1/5] Títulos, capítulos y secciones...")
    crear_titulos_capitulos(driver, estructura)

    print("  [2/5] Artículos...")
    crear_articulos(driver, articulos, emb_model)

    print("  [3/5] Referencias cruzadas...")
    crear_referencias_cruzadas(driver, articulos)

    print("  [4/5] Entidades y conceptos...")
    crear_entidades_y_conceptos(driver, articulos)

    print("  [5/5] Disposiciones...")
    crear_disposiciones(driver, disposiciones)

    # Opcional: LLMGraphTransformer
    if USAR_LLM_TRANSFORMER:
        enriquecer_con_llm(driver, articulos)

    driver.close()

    # Estadísticas finales
    print("\n📊  Grafo construido. Puedes explorarlo en:")
    print(f"    http://localhost:7474  (Neo4j Browser)")
    print(f"    Query de prueba:")
    print(f"    MATCH (a:Articulo)-[:REFERENCIA]->(b:Articulo) RETURN a.numero, b.numero LIMIT 20")
    print(f"\n✅  Siguiente paso: python 03_consultar.py")


if __name__ == "__main__":
    main()
