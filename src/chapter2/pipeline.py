"""Каждая функция соответствует отдельному этапу из главы 2."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager, index
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Settings:
    base_url: str = "https://api.vsellm.ru/v1"
    api_key: str = ""
    embedding_model: str = ""
    chat_model: str = ""
    chunk_size: int = 500
    chunk_overlap: int = 100

    def __post_init__(self):
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Нужно: CHUNK_SIZE > 0 и 0 <= CHUNK_OVERLAP < CHUNK_SIZE")

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            base_url=os.getenv("OPENAI_BASE_URL", cls.base_url).rstrip("/"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", ""),
            chat_model=os.getenv("CHAT_MODEL", ""),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        )

    def require_key(self):
        if not self.api_key.strip():
            raise ValueError("Заполните OPENAI_API_KEY в .env или запустите с --demo")


class KeywordDemoEmbeddings(Embeddings):
    """Учебный хеш-вектор слов. Это НЕ семантическая модель и НЕ вызов API."""
    dimensions = 256

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.lower()):
            bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
            vector[bucket % self.dimensions] += 1
        length = math.sqrt(sum(x * x for x in vector)) or 1
        return [x / length for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def make_embeddings(settings: Settings, demo: bool = False) -> Embeddings:
    if demo:
        return KeywordDemoEmbeddings()
    settings.require_key()
    if not settings.embedding_model.strip():
        raise ValueError("Укажите EMBEDDING_MODEL: ID модели для /embeddings, а не чата")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        # Провайдер получает обычные строки, а не массивы токенов OpenAI.
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
        chunk_size=32,
        request_timeout=60,
        max_retries=1,
    )


def load_documents(data_dir: Path) -> list[Document]:
    if not data_dir.is_dir():
        raise ValueError(f"Нет каталога с данными: {data_dir}")
    documents = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        loaded = (PyPDFLoader(str(path)).load() if path.suffix.lower() == ".pdf"
                  else TextLoader(str(path), encoding="utf-8").load())
        for doc in loaded:
            # Стабильный source: перемещение всего проекта не меняет идентификатор.
            doc.metadata = {"source": path.relative_to(data_dir).as_posix(),
                            **({"page": doc.metadata["page"]} if "page" in doc.metadata else {})}
            if doc.page_content.strip():
                documents.append(doc)
    return documents


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def split_markdown(text: str) -> list[Document]:
    """Отдельный эксперимент: структура заголовков вместо фиксированных размеров."""
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=False
    ).split_text(text)


def open_store(settings: Settings, embeddings: Embeddings, root: Path, demo: bool):
    directory = root / ("demo" if demo else "live")
    directory.mkdir(parents=True, exist_ok=True)
    signature = {
        "embedding": "keyword-demo-v1-256" if demo else settings.embedding_model,
        "base_url": None if demo else settings.base_url,
        "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
    }
    manifest = directory / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Настройки индекса изменились. Используйте новый --index-dir.")
    manifest.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    store = Chroma(collection_name="chapter2", embedding_function=embeddings,
                   persist_directory=str(directory / "chroma"),
                   collection_metadata={"hnsw:space": "cosine"})
    manager = SQLRecordManager("chapter2", db_url=f"sqlite:///{directory / 'records.sqlite'}")
    manager.create_schema()
    return store, manager


def synchronize(chunks: list[Document], store: Chroma, manager: SQLRecordManager):
    # full: вход — полный снимок каталога. Удаленные файлы исчезают из индекса.
    # incremental из книги удаляет старые версии только переданных source.
    return index(chunks, manager, store, cleanup="full", source_id_key="source", key_encoder="sha256")
