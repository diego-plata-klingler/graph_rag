"""Motor RAG simplificado - Ingesta, Chunking, Embeddings, Búsqueda y Generación."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP,
    DATA_DIR,
    DOCS_DIR,
    EMBED_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MILVUS_COLLECTION,
    MILVUS_URI,
    OPENAI_API_KEY,
    get_device,
)

logger = logging.getLogger(__name__)

# ==================== MODELOS DE DATOS ====================


@dataclass
class Chunk:
    """Fragmento de documento con su embedding."""
    id: str
    doc_slug: str
    filename: str  # Nombre original del fichero
    text: str
    page: int | None = None
    score: float | None = None  # Score de similitud (solo en búsquedas)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """Resultado de una consulta RAG."""
    answer: str | None
    chunks: list[Chunk]
    sources: list[str]  # Nombres de ficheros fuente


# ==================== CONVERSIÓN PDF -> MARKDOWN ====================


class DocumentConverter:
    """Convierte documentos (PDF, DOCX, etc.) a Markdown usando Docling."""

    def __init__(self):
        self._converter = None
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Crea directorios necesarios."""
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)

    def _get_converter(self):
        """Inicializa el converter de Docling de forma lazy."""
        if self._converter is not None:
            return self._converter

        try:
            self._configure_rapidocr()
            from docling.document_converter import DocumentConverter as DoclingConverter
            self._converter = DoclingConverter()
            return self._converter
        except ImportError:
            raise ImportError("Docling no está instalado. Instala con: pip install docling")

    def _configure_rapidocr(self) -> None:
        """Apunta RapidOCR a un directorio escribible para modelos."""
        model_dir = Path(os.getenv("RAPIDOCR_MODEL_DIR", f"{DATA_DIR}/rapidocr_models"))
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            from rapidocr.inference_engine.base import InferSession
            InferSession.DEFAULT_MODEL_PATH = model_dir
        except Exception:
            # Si rapidocr no está disponible, continuar sin fallar
            pass

    def convert_to_markdown(
        self,
        file_path: Path | str,
        filename: str,
        output_dir: Path | str | None = None,
    ) -> tuple[str, str]:
        """
        Convierte un documento a Markdown.
        
        Returns:
            Tuple[markdown_content, doc_slug]
        """
        file_path = Path(file_path)
        doc_slug = self._create_slug(filename)
        
        # Si ya es Markdown, simplemente leer
        if file_path.suffix.lower() in (".md", ".markdown"):
            markdown = file_path.read_text(encoding="utf-8")
        else:
            # Convertir con Docling
            converter = self._get_converter()
            result = converter.convert(str(file_path))
            markdown = result.document.export_to_markdown()

        # Guardar el markdown procesado
        md_dir = Path(output_dir) if output_dir else Path(DOCS_DIR)
        md_dir.mkdir(parents=True, exist_ok=True)
        md_path = md_dir / f"{doc_slug}.md"
        md_path.write_text(markdown, encoding="utf-8")

        return markdown, doc_slug

    def convert_bytes_to_markdown(self, content: bytes, filename: str) -> tuple[str, str]:
        """Convierte bytes a Markdown (para uploads via API)."""
        # Guardar temporalmente
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            return self.convert_to_markdown(tmp_path, filename)
        finally:
            os.unlink(tmp_path)

    def _create_slug(self, filename: str) -> str:
        """Crea un slug único basado en el nombre del fichero."""
        base = Path(filename).stem
        # Limpiar caracteres especiales
        slug = re.sub(r"[^\w\s-]", "", base.lower())
        slug = re.sub(r"[\s_]+", "_", slug).strip("_")
        # Añadir hash corto para unicidad
        hash_suffix = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"{slug}_{hash_suffix}"


# ==================== CHUNKING ====================


class MarkdownChunker:
    """Divide Markdown en chunks respetando la estructura."""

    def __init__(self, max_tokens: int = CHUNK_MAX_TOKENS, overlap: int = CHUNK_OVERLAP):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self._tokenizer = None

    def _get_tokenizer(self):
        """Obtiene el tokenizer del modelo de embeddings."""
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
            return self._tokenizer
        except Exception:
            return None

    def _count_tokens(self, text: str) -> int:
        """Cuenta tokens en el texto."""
        tokenizer = self._get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text, add_special_tokens=False))
        # Fallback: aproximación
        return int(len(text.split()) * 1.3)

    def chunk(self, markdown: str, doc_slug: str, filename: str) -> list[Chunk]:
        """Divide el markdown en chunks."""
        if not markdown.strip():
            return []

        # Limpiar artefactos de conversión PDF
        markdown = re.sub(r"<!--\s*pagebreak\s*-->", "", markdown)

        chunks = []
        # Dividir por secciones (headers)
        sections = self._split_by_headers(markdown)

        for section_text in sections:
            if not section_text.strip():
                continue

            # Si la sección es muy larga, dividir por párrafos
            if self._count_tokens(section_text) > self.max_tokens:
                # Extraer header para propagarlo a cada sub-chunk
                header, body = self._extract_header(section_text)
                sub_chunks = self._split_long_section(body, header)
            else:
                sub_chunks = [section_text]

            for text in sub_chunks:
                if text.strip():
                    chunk = Chunk(
                        id=uuid4().hex,
                        doc_slug=doc_slug,
                        filename=filename,
                        text=text.strip(),
                        metadata={"token_count": self._count_tokens(text)},
                    )
                    chunks.append(chunk)

        return chunks

    def _split_by_headers(self, text: str) -> list[str]:
        """Divide el texto por headers de Markdown."""
        # Pattern para encontrar headers
        pattern = r"(^#{1,6}\s+.+$)"
        parts = re.split(pattern, text, flags=re.MULTILINE)
        
        sections = []
        current = ""
        
        for part in parts:
            if re.match(r"^#{1,6}\s+", part):
                if current.strip():
                    sections.append(current.strip())
                current = part + "\n"
            else:
                current += part
        
        if current.strip():
            sections.append(current.strip())
        
        return sections if sections else [text]

    def _extract_header(self, section_text: str) -> tuple[str, str]:
        """Extrae el header de una sección.

        Returns:
            (header, body) — header es la línea con ``#`` y body el resto.
            Si no hay header, header es cadena vacía y body es el texto completo.
        """
        first_nl = section_text.find("\n")
        first_line = section_text[:first_nl] if first_nl != -1 else section_text
        if re.match(r"^#{1,6}\s+", first_line):
            body = section_text[first_nl + 1:].strip() if first_nl != -1 else ""
            return first_line.strip(), body
        return "", section_text

    def _split_long_section(self, text: str, header: str = "") -> list[str]:
        """Divide una sección larga en chunks más pequeños.

        Si se proporciona *header*, se antepone a cada sub-chunk resultante
        para que todos conserven el contexto semántico de la sección.
        Los tokens del header se descuentan del presupuesto de cada chunk.

        Estrategia de splitting en dos niveles:
        1. Primero intenta agrupar párrafos (separados por ``\\n\\n``).
        2. Si un párrafo individual excede el presupuesto, lo subdivide por
           líneas (``\\n``) para que los ítems de listas numeradas, etc.,
           se distribuyan correctamente entre chunks.
        """
        # Reservar tokens para el header que se prefijará
        header_tokens = self._count_tokens(header + "\n\n") if header else 0
        budget = max(self.max_tokens - header_tokens, 50)  # mínimo razonable

        # ── Paso 1: dividir por párrafos (\n\n) ──
        paragraphs = text.split("\n\n")

        # ── Paso 2: expandir párrafos que solos exceden el presupuesto ──
        # Si un párrafo es demasiado grande, lo subdividimos por líneas (\n)
        # para que ítems de listas (separados por \n) puedan repartirse.
        expanded: list[str] = []
        for para in paragraphs:
            if self._count_tokens(para) > budget:
                lines = para.split("\n")
                expanded.extend(ln for ln in lines if ln.strip())
            else:
                expanded.append(para)

        # ── Paso 3: agrupar fragmentos hasta llenar el presupuesto ──
        chunks: list[str] = []
        current = ""

        for piece in expanded:
            if not piece.strip():
                continue
            test_text = current + "\n" + piece if current else piece
            if self._count_tokens(test_text) <= budget:
                current = test_text
            else:
                if current:
                    chunks.append(current)
                current = piece

        if current:
            chunks.append(current)

        # ── Paso 4: prefijar el header a CADA sub-chunk ──
        if header:
            chunks = [f"{header}\n\n{c}" for c in chunks]

        return chunks


# ==================== EMBEDDINGS ====================


class EmbeddingService:
    """Genera embeddings usando sentence-transformers."""

    def __init__(self):
        self._embedder = None

    def _get_embedder(self):
        """Inicializa el embedder de forma lazy."""
        if self._embedder is not None:
            return self._embedder

        try:
            from sentence_transformers import SentenceTransformer
            device = get_device()
            self._embedder = SentenceTransformer(EMBED_MODEL, device=device)
            return self._embedder
        except ImportError:
            raise ImportError("sentence-transformers no instalado")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings para una lista de textos."""
        if not texts:
            return []
        
        embedder = self._get_embedder()
        # Añadir prefijo para modelos E5
        if "e5" in EMBED_MODEL.lower():
            texts = [f"passage: {t}" for t in texts]
        
        embeddings = embedder.encode(texts, show_progress_bar=len(texts) > 10)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Genera embedding para una query."""
        embedder = self._get_embedder()
        # Añadir prefijo para modelos E5
        if "e5" in EMBED_MODEL.lower():
            query = f"query: {query}"
        
        embedding = embedder.encode(query)
        return embedding.tolist()


# ==================== VECTOR STORE (MILVUS) ====================


class VectorStore:
    """Almacén vectorial usando Milvus."""

    def __init__(self):
        self._collection = None
        self._embedding_dim = 768  # Dimensión por defecto para E5-base

    def _get_collection(self):
        """Obtiene o crea la colección en Milvus."""
        if self._collection is not None:
            return self._collection

        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            utility,
        )

        # Conectar a Milvus
        connections.connect(uri=MILVUS_URI)

        # Definir esquema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="doc_slug", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._embedding_dim),
        ]
        schema = CollectionSchema(fields, description="RAG Document Store")

        # Crear o cargar colección
        if utility.has_collection(MILVUS_COLLECTION):
            self._collection = Collection(MILVUS_COLLECTION)
        else:
            self._collection = Collection(MILVUS_COLLECTION, schema)
            # Crear índice
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            self._collection.create_index("embedding", index_params)

        self._collection.load()
        return self._collection

    def insert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Inserta chunks con sus embeddings en Milvus."""
        if not chunks:
            return 0

        collection = self._get_collection()
        
        data = [
            [c.id for c in chunks],                    # id
            [c.doc_slug for c in chunks],              # doc_slug
            [c.filename for c in chunks],              # filename
            [c.text[:65000] for c in chunks],          # text (truncado)
            embeddings,                                 # embedding
        ]

        result = collection.insert(data)
        collection.flush()
        return len(result.primary_keys)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_slugs: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Busca chunks similares a la query."""
        collection = self._get_collection()

        # Construir filtro
        expr = None
        if doc_slugs:
            slugs_str = ", ".join(f'"{s}"' for s in doc_slugs)
            expr = f"doc_slug in [{slugs_str}]"

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=expr,
            output_fields=["id", "doc_slug", "filename", "text"],
        )

        chunks_with_scores = []
        for hits in results:
            for hit in hits:
                chunk = Chunk(
                    id=hit.entity.get("id"),
                    doc_slug=hit.entity.get("doc_slug"),
                    filename=hit.entity.get("filename"),
                    text=hit.entity.get("text"),
                )
                chunks_with_scores.append((chunk, hit.score))

        return chunks_with_scores

    def delete_by_doc_slug(self, doc_slug: str) -> int:
        """Elimina todos los chunks de un documento."""
        collection = self._get_collection()
        expr = f'doc_slug == "{doc_slug}"'
        result = collection.delete(expr)
        collection.flush()
        return result.delete_count if hasattr(result, 'delete_count') else 0

    def count(self) -> int:
        """Cuenta el total de documentos."""
        collection = self._get_collection()
        return collection.num_entities


# ==================== GENERADOR LLM ====================


class LLMGenerator:
    """Genera respuestas usando OpenAI/LLM."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Obtiene el cliente de OpenAI."""
        if self._client is not None:
            return self._client

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY no configurada")

        from openai import OpenAI
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        return self._client

    def generate(
        self,
        question: str,
        chunks: list[Chunk],
        chat_history: list[dict] | None = None,
    ) -> str:
        """Genera una respuesta basada en los chunks recuperados."""
        if not chunks:
            return "No se encontraron documentos relevantes para responder tu pregunta."

        client = self._get_client()

        # Construir contexto con fuentes (numeradas para citación)
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            score_str = f" (relevancia: {chunk.score:.1%})" if chunk.score is not None else ""
            context_parts.append(
                f"<fragmento id=\"{i}\" fuente=\"{chunk.filename}\"{score_str}>\n"
                f"{chunk.text}\n"
                f"</fragmento>"
            )

        context = "\n\n".join(context_parts)

        # ── System prompt ──
        # Instrucciones fijas del sistema (sin contexto documental).
        system_prompt = (
            "Eres un asistente experto en análisis documental y normativa CNMV "
            "(Comisión Nacional del Mercado de Valores) de España.\n\n"
            "REGLAS OBLIGATORIAS — léelas ANTES de generar cualquier respuesta:\n\n"
            "1.  ÚNICA DE VERDAD: responde EXCLUSIVAMENTE con la información "
            "contenida en los fragmentos documentales que se proporcionan dentro de "
            "las etiquetas <contexto>…</contexto>. Está PROHIBIDO usar tu "
            "conocimiento paramétrico, inferencias externas o completar información "
            "que no aparezca literalmente en los fragmentos.\n\n"
            "2. VERIFICACIÓN: antes de incluir cualquier dato en tu respuesta, "
            "confirma que aparece de forma explícita en al menos un fragmento. Si no "
            "puedes verificarlo, NO lo incluyas.\n\n"
            "3. ARTÍCULOS Y TEXTOS LEGALES: reproduce fielmente el texto tal como "
            "aparece en los fragmentos. Nunca completes, resumas ni parafrasees "
            "artículos legales. Si un artículo aparece cortado o incompleto en los "
            "fragmentos, indica que el texto disponible está incompleto.\n\n"
            "4. CITACIÓN: cita SIEMPRE la fuente usando el nombre exacto del fichero "
            "y el número de fragmento. Formato: «Según [nombre_fichero] (fragmento N), …».\n\n"
            "5. INFORMACIÓN INSUFICIENTE: si la respuesta NO aparece de forma "
            "explícita en los fragmentos, responde EXACTAMENTE:\n"
            "   «NO ENCONTRADO EN EL DOCUMENTO»\n\n"
            "6. FORMATO: estructura la respuesta en texto plano o Markdown ligero "
            "(listas, negritas). No inventes encabezados de secciones que no existan "
            "en el documento original.\n\n"
            "7. TRANSPARENCIA: si varios fragmentos se contradicen, señálalo "
            "explícitamente indicando qué dice cada fuente.\n\n"
            "8. IDIOMA: responde siempre en español."
        )

        # ── User message con contexto + pregunta ──
        # Inyectar el contexto documental junto con la pregunta del usuario
        # para que el LLM lo vea de forma explícita.
        user_content = (
            "<contexto>\n"
            f"{context}\n"
            "</contexto>\n\n"
            f"Pregunta: {question}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Añadir historial si existe
        if chat_history:
            messages.extend(chat_history)

        messages.append({"role": "user", "content": user_content})

        # ── DEBUG: imprimir mensajes completos que se envían al LLM ──
        logger.info("=" * 80)
        logger.info(" MENSAJES ENVIADOS AL LLM (model=%s, temperature=%s)", LLM_MODEL, LLM_TEMPERATURE)
        logger.info("=" * 80)
        for idx, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            logger.info("── [%d] role=%s  (chars=%d) ──", idx, role, len(content))
            logger.info(content)
            logger.info("── fin [%d] ──", idx)
        logger.info("=" * 80)

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
        )

        return response.choices[0].message.content


# ==================== RAG ENGINE PRINCIPAL ====================


class RAGEngine:
    """Motor RAG completo: ingesta, búsqueda y generación."""

    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = MarkdownChunker()
        self.embedder = EmbeddingService()
        self.vector_store = VectorStore()
        self.generator = LLMGenerator()

    def ingest_file(self, file_path: str | Path, filename: str | None = None) -> dict:
        """
        Ingesta un fichero: convierte a MD, chunquea, genera embeddings e indexa.
        
        Returns:
            Dict con estadísticas de la ingesta
        """
        file_path = Path(file_path)
        filename = filename or file_path.name

        # 1. Convertir a Markdown
        markdown, doc_slug = self.converter.convert_to_markdown(file_path, filename)
        return self._ingest_markdown(markdown, doc_slug, filename)

    def ingest_bytes(self, content: bytes, filename: str) -> dict:
        """Ingesta contenido en bytes (para API)."""
        markdown, doc_slug = self.converter.convert_bytes_to_markdown(content, filename)
        return self._ingest_markdown(markdown, doc_slug, filename)

    def ingest_markdown_file(
        self,
        md_path: str | Path,
        filename: str,
        doc_slug: str | None = None,
    ) -> dict:
        """Ingesta un Markdown existente (sin reconvertir)."""
        md_path = Path(md_path)
        markdown = md_path.read_text(encoding="utf-8")
        slug = doc_slug or self.converter._create_slug(filename)
        return self._ingest_markdown(markdown, slug, filename)

    def _ingest_markdown(self, markdown: str, doc_slug: str, filename: str) -> dict:
        """Ingesta un Markdown ya disponible."""
        chunks = self.chunker.chunk(markdown, doc_slug, filename)

        if not chunks:
            return {"doc_slug": doc_slug, "chunks": 0, "status": "empty"}

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        inserted = self.vector_store.insert(chunks, embeddings)

        return {
            "doc_slug": doc_slug,
            "filename": filename,
            "chunks": len(chunks),
            "indexed": inserted,
            "status": "success",
        }

    def query(
        self,
        question: str,
        top_k: int = 5,
        doc_slugs: list[str] | None = None,
        chat_history: list[dict] | None = None,
    ) -> RAGResult:
        """
        Ejecuta una consulta RAG completa.
        
        Returns:
            RAGResult con respuesta, chunks usados y fuentes
        """
        # 1. Generar embedding de la query
        query_embedding = self.embedder.embed_query(question)

        # 2. Buscar chunks relevantes
        results = self.vector_store.search(query_embedding, top_k=top_k, doc_slugs=doc_slugs)
        # Añadir score a cada chunk
        chunks = []
        for chunk, score in results:
            chunk.score = score
            chunks.append(chunk)

        # 3. Generar respuesta
        answer = self.generator.generate(question, chunks, chat_history)

        # 4. Extraer fuentes únicas
        sources = list(set(c.filename for c in chunks))

        return RAGResult(answer=answer, chunks=chunks, sources=sources)

    def delete_document(self, doc_slug: str) -> int:
        """Elimina un documento del índice."""
        return self.vector_store.delete_by_doc_slug(doc_slug)

    def get_stats(self) -> dict:
        """Obtiene estadísticas del sistema."""
        return {
            "total_chunks": self.vector_store.count(),
            "milvus_uri": MILVUS_URI,
            "embed_model": EMBED_MODEL,
            "llm_model": LLM_MODEL,
        }
