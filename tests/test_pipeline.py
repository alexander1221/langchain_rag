import json

import httpx
import pytest
from openai import OpenAI

from chapter2.advanced import build_multivector
from chapter2.pipeline import (KeywordDemoEmbeddings, Settings, load_documents,
                                make_embeddings, open_store, split_documents, synchronize)


class CountingEmbeddings(KeywordDemoEmbeddings):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return super().embed_documents(texts)


def test_index_lifecycle_and_persistence(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    first = data / "scholarship.txt"
    first.write_text("Стипендия составляет 7000 рублей.", encoding="utf-8")
    second = data / "library.md"
    second.write_text("Библиотека закрыта в воскресенье.", encoding="utf-8")
    settings = Settings()
    embeddings = CountingEmbeddings()
    store, manager = open_store(settings, embeddings, tmp_path / "index", True)

    def sync():
        return synchronize(split_documents(load_documents(data), settings), store, manager)

    assert sync()["num_added"] == 2
    initial_calls = embeddings.calls
    assert sync()["num_skipped"] == 2
    assert embeddings.calls == initial_calls  # повторная индексация не тратит API
    first.write_text("Стипендия составляет 9000 рублей.", encoding="utf-8")
    result = sync()
    assert result["num_added"] == 1 and result["num_deleted"] == 1
    assert embeddings.calls == initial_calls + 1
    second.unlink()
    assert sync()["num_deleted"] == 1
    reopened, _ = open_store(settings, embeddings, tmp_path / "index", True)
    docs = reopened.similarity_search("Стипендия", k=1)
    assert "9000" in docs[0].page_content
    assert reopened._collection.count() == 1
    first.unlink()
    assert sync()["num_deleted"] == 1
    assert store._collection.count() == 0


def test_settings_and_index_signature(tmp_path):
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=100)
    open_store(Settings(), KeywordDemoEmbeddings(), tmp_path, True)
    with pytest.raises(ValueError, match="Настройки индекса"):
        open_store(Settings(chunk_size=600), KeywordDemoEmbeddings(), tmp_path, True)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_embeddings(Settings())


def test_multivector_returns_original(tmp_path):
    (tmp_path / "library.txt").write_text("Библиотека. " * 40, encoding="utf-8")
    docs = load_documents(tmp_path)
    retriever, summaries = build_multivector(docs, KeywordDemoEmbeddings(), Settings(), demo=True)
    result = retriever.invoke("Библиотека")
    assert result[0].page_content == docs[0].page_content
    assert len(result[0].page_content) > len(summaries[0].page_content)


def test_provider_receives_raw_strings():
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        return httpx.Response(200, json={"object": "list", "model": "test-embedding",
            "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.0, 0.0]}
                     for i, _ in enumerate(body["input"])],
            "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    embeddings = make_embeddings(Settings(api_key="test-key", embedding_model="test-embedding"))
    # Подменяется только транспорт: выполняются настоящие методы LangChain и SDK.
    client = OpenAI(api_key="test-key", base_url="https://api.vsellm.ru/v1",
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    embeddings.client = client.embeddings
    assert embeddings.embed_documents(["Стипендия", "Библиотека"]) == [[1., 0., 0.]] * 2
    assert embeddings.embed_query("Запрос") == [1., 0., 0.]
    assert requests[0][0] == "https://api.vsellm.ru/v1/embeddings"
    assert requests[0][1]["input"] == ["Стипендия", "Библиотека"]
    assert requests[0][1]["encoding_format"] == "float"


def test_pdf_loader_keeps_page_and_source():
    from pathlib import Path
    docs = load_documents(Path(__file__).resolve().parents[1] / "data")
    pdf = [doc for doc in docs if doc.metadata["source"] == "cafeteria.pdf"]
    assert len(pdf) == 1
    assert pdf[0].metadata["page"] == 0
    assert "250 RUB" in pdf[0].page_content
