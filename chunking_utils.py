"""Chunking estructurado inspirado en rag_engine para textos legales."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    from config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP
except Exception:
    CHUNK_MAX_TOKENS = 350
    CHUNK_OVERLAP = 60


@dataclass
class StructuredChunk:
    """Fragmento estructurado listo para indexar en el grafo."""

    id: str
    text: str
    order: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class MarkdownChunker:
    """Divide texto Markdown respetando encabezados y estructura interna."""

    def __init__(self, max_tokens: int = CHUNK_MAX_TOKENS, overlap: int = CHUNK_OVERLAP):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
            return self._tokenizer
        except Exception:
            return None

    def count_tokens(self, text: str) -> int:
        tokenizer = self._get_tokenizer()
        if tokenizer:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return max(1, int(len(text.split()) * 1.3))

    def chunk(self, markdown: str, metadata: dict[str, Any] | None = None) -> list[StructuredChunk]:
        """Divide un markdown en fragmentos estructurados."""
        if not markdown.strip():
            return []

        markdown = re.sub(r"<!--\s*pagebreak\s*-->", "", markdown)
        metadata = metadata or {}

        chunks: list[StructuredChunk] = []
        order = 0
        for section_text in self._split_by_headers(markdown):
            if not section_text.strip():
                continue

            prefix, body = self._extract_prefix(section_text)
            complete_text = prefix if not body else f"{prefix}\n\n{body}".strip()
            if self.count_tokens(complete_text) <= self.max_tokens:
                pieces = [complete_text]
            else:
                pieces = self._split_long_section(body, prefix=prefix)

            for piece in pieces:
                cleaned = piece.strip()
                if not cleaned:
                    continue
                chunks.append(
                    StructuredChunk(
                        id="",
                        text=cleaned,
                        order=order,
                        token_count=self.count_tokens(cleaned),
                        metadata=dict(metadata),
                    )
                )
                order += 1

        return chunks

    def _split_by_headers(self, text: str) -> list[str]:
        pattern = r"(^#{1,6}\s+.+$)"
        parts = re.split(pattern, text, flags=re.MULTILINE)

        sections: list[str] = []
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

    def _extract_prefix(self, section_text: str) -> tuple[str, str]:
        """Extrae el header y su bloque contextual inicial."""
        lines = section_text.splitlines()
        if not lines:
            return "", ""

        first_line = lines[0].strip()
        if not re.match(r"^#{1,6}\s+", first_line):
            return "", section_text.strip()

        prefix_lines = [first_line]
        idx = 1
        while idx < len(lines):
            line = lines[idx].rstrip()
            if not line.strip():
                idx += 1
                break
            prefix_lines.append(line)
            idx += 1

        body = "\n".join(lines[idx:]).strip()
        return "\n".join(prefix_lines).strip(), body

    def _split_long_section(self, text: str, prefix: str = "") -> list[str]:
        prefix_tokens = self.count_tokens(prefix + "\n\n") if prefix else 0
        budget = max(self.max_tokens - prefix_tokens, 50)

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        expanded: list[str] = []
        for para in paragraphs:
            if self.count_tokens(para) > budget:
                lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
                expanded.extend(self._split_oversized_lines(lines, budget))
            else:
                expanded.append(para)

        chunks: list[str] = []
        current_parts: list[str] = []
        current_text = ""

        for piece in expanded:
            candidate = f"{current_text}\n\n{piece}".strip() if current_text else piece
            if self.count_tokens(candidate) <= budget:
                current_parts.append(piece)
                current_text = candidate
                continue

            if current_text:
                chunks.append(current_text)

            overlap_parts = self._build_overlap(current_parts)
            current_parts = overlap_parts + [piece]
            current_text = "\n\n".join(current_parts).strip()

            if self.count_tokens(current_text) > budget:
                if overlap_parts:
                    current_parts = [piece]
                    current_text = piece
                if self.count_tokens(current_text) > budget:
                    chunks.extend(self._split_oversized_piece(piece, budget))
                    current_parts = []
                    current_text = ""

        if current_text:
            chunks.append(current_text)

        if prefix:
            return [f"{prefix}\n\n{chunk}".strip() for chunk in chunks]
        return chunks

    def _build_overlap(self, parts: list[str]) -> list[str]:
        if self.overlap <= 0 or not parts:
            return []

        selected: list[str] = []
        total = 0
        for part in reversed(parts):
            tokens = self.count_tokens(part)
            if selected and total + tokens > self.overlap:
                break
            selected.insert(0, part)
            total += tokens
            if total >= self.overlap:
                break
        return selected

    def _split_oversized_lines(self, lines: list[str], budget: int) -> list[str]:
        pieces: list[str] = []
        for line in lines:
            if self.count_tokens(line) <= budget:
                pieces.append(line)
            else:
                pieces.extend(self._split_oversized_piece(line, budget))
        return pieces

    def _split_oversized_piece(self, text: str, budget: int) -> list[str]:
        sentences = re.split(r"(?<=[.!?;:])\s+", text)
        pieces: list[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if self.count_tokens(candidate) <= budget:
                current = candidate
                continue

            if current:
                pieces.append(current)
            current = sentence

            if self.count_tokens(current) > budget:
                words = current.split()
                current = ""
                for word in words:
                    candidate = f"{current} {word}".strip() if current else word
                    if self.count_tokens(candidate) <= budget:
                        current = candidate
                    else:
                        if current:
                            pieces.append(current)
                        current = word

        if current:
            pieces.append(current)
        return pieces


def clean_legal_text(text: str) -> str:
    """Limpia artefactos frecuentes del PDF antes del chunking."""
    text = text or ""
    text = re.sub(r"BOLET[ÍI]N OFICIAL DEL ESTADO LEGISLACI[ÓO]N CONSOLIDADA P[aá]gina \d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[PAGINA_\d+\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prepare_legal_body(text: str) -> str:
    """Introduce saltos útiles para listas y apartados legales."""
    text = clean_legal_text(text)

    patterns = [
        r"(?<=[.;:])\s+(?=(\d+\.\s+[A-ZÁÉÍÓÚÑ]))",
        r"(?<=[.;:])\s+(?=([a-z]\)\s+[A-ZÁÉÍÓÚÑ]))",
        r"(?<=[.;:])\s+(?=(\d+\.\s?[º°ª]\s+[A-ZÁÉÍÓÚÑ]))",
        r"(?<=[.;:])\s+(?=(\d+\.\s?[º°ª]\s*[a-záéíóúñ]))",
        r"(?<=\))\s+(?=([a-z]\)\s+[A-ZÁÉÍÓÚÑ]))",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "\n", text)

    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def build_article_markdown(article: dict) -> str:
    """Construye un markdown autocontenido para chunkear un artículo."""
    header = f"#### Artículo {article.get('numero', '?')}. {article.get('titulo', '').strip()}".strip()
    context = [
        f"Título: {article.get('titulo_padre') or 'Sin título'}",
        f"Capítulo: {article.get('capitulo_padre') or 'Sin capítulo'}",
        f"Sección: {article.get('seccion_padre') or 'Sin sección'}",
    ]

    text = article.get("texto", "")
    body = text.split("\n", 1)[1] if "\n" in text else text
    body = prepare_legal_body(body)

    return f"{header}\n" + "\n".join(context) + f"\n\n{body}"


def build_disposition_markdown(disposition: dict) -> str:
    """Construye un markdown autocontenido para chunkear una disposición."""
    header = f"#### {disposition.get('titulo', 'Disposición').strip()}"
    body = prepare_legal_body(disposition.get("texto", ""))
    return f"{header}\nTipo: disposición\n\n{body}".strip()
