from __future__ import annotations

from pathlib import Path
import json
import logging
from typing import Iterable

from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from core.config import AppConfig
from core.exceptions import KnowledgeBaseNotReadyError

logger = logging.getLogger(__name__)

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".url"}
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv", ".json", ".txt", ".md", ".url"}


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


class KnowledgeBaseManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embeddings = SentenceTransformerEmbeddings(config.embedding_model)

    def _iter_knowledge_files(self) -> Iterable[Path]:
        if not self.config.knowledge_dir.exists():
            raise KnowledgeBaseNotReadyError(
                f"Cartella knowledge non trovata: {self.config.knowledge_dir}"
            )

        for path in sorted(self.config.knowledge_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path

    def _load_json_document(self, path: Path) -> list[Document]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            raw = json.loads(path.read_text(encoding="latin-1"))

        if isinstance(raw, (dict, list)):
            content = json.dumps(raw, ensure_ascii=False, indent=2)
        else:
            content = str(raw)

        return [
            Document(
                page_content=content,
                metadata={"source": str(path), "file_type": path.suffix.lower()},
            )
        ]

    def _load_text_document(self, path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [
            Document(
                page_content=text,
                metadata={"source": str(path), "file_type": path.suffix.lower()},
            )
        ]

    def load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for path in self._iter_knowledge_files():
            logger.info("Caricamento file: %s", path)
            if path.suffix.lower() == ".pdf":
                documents.extend(PyPDFLoader(str(path)).load())
            elif path.suffix.lower() == ".docx":
                documents.extend(Docx2txtLoader(str(path)).load())
            elif path.suffix.lower() == ".csv":
                documents.extend(CSVLoader(file_path=str(path), encoding="utf-8").load())
            elif path.suffix.lower() == ".json":
                documents.extend(self._load_json_document(path))
            else:
                documents.extend(self._load_text_document(path))

        if not documents:
            raise KnowledgeBaseNotReadyError(
                f"Nessun documento supportato trovato in {self.config.knowledge_dir}"
            )

        return documents

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        text = text.strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunks.append(text[start:end].strip())
            if end >= length:
                break
            start = max(0, end - chunk_overlap)
        return [chunk for chunk in chunks if chunk]

    def _split_documents(self, documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            metadata = dict(document.metadata or {})
            source = metadata.get("source", "sorgente sconosciuta")
            for index, chunk in enumerate(self._split_text(document.page_content, chunk_size, chunk_overlap), start=1):
                chunk_metadata = {**metadata, "source": source, "chunk": index}
                chunks.append(Document(page_content=chunk, metadata=chunk_metadata))
        return chunks

    def build_index(self) -> Chroma:
        documents = self.load_documents()
        chunks = self._split_documents(documents)

        self.config.vector_db_dir.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.config.vector_db_dir),
        )
        logger.info("Indice vettoriale creato in %s", self.config.vector_db_dir)
        return vector_store

    def load_index(self) -> Chroma:
        if not self.config.vector_db_dir.exists() or not any(self.config.vector_db_dir.iterdir()):
            raise KnowledgeBaseNotReadyError(
                f"Indice vettoriale non trovato in {self.config.vector_db_dir}. Esegui prima ingest.py."
            )

        return Chroma(
            persist_directory=str(self.config.vector_db_dir),
            embedding_function=self.embeddings,
        )
