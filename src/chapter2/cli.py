import argparse
import json
import sys
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI
from chapter2.pipeline import (Settings, load_documents, make_embeddings,
                                open_store, split_documents, synchronize)


def main():
    parser = argparse.ArgumentParser(description="Глава 2: лаборатория индексации")
    parser.add_argument("command", choices=["inspect", "index", "search", "models", "multivector"])
    parser.add_argument("query", nargs="?", default="Как получить стипендию?")
    parser.add_argument("--demo", action="store_true", help="Без API: хеш-векторы слов")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--index-dir", type=Path, default=Path(".index"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--allow-empty", action="store_true", help="Разрешить очистку индекса пустым каталогом")
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        if args.k <= 0:
            raise ValueError("--k должен быть положительным")
        if args.command == "models":
            settings.require_key()
            client = OpenAI(base_url=settings.base_url, api_key=settings.api_key,
                            timeout=30, max_retries=1)
            for model in client.models.list():
                print(model.id)
            return
        if args.command == "search":
            mode_dir = args.index_dir / ("demo" if args.demo else "live")
            if not (mode_dir / "config.json").exists():
                raise ValueError("Сначала выполните chapter2 index с тем же режимом и --index-dir")
        else:
            docs = load_documents(args.data_dir)
            chunks = split_documents(docs, settings)
            if args.command == "inspect":
                print(f"Документов/страниц: {len(docs)}; чанков: {len(chunks)}")
                for i, chunk in enumerate(chunks):
                    print(f"\n[{i}] {chunk.metadata} | {len(chunk.page_content)} символов")
                    print(chunk.page_content)
                return
            if not chunks and not args.allow_empty:
                raise ValueError("Нет текстовых чанков. Для намеренной очистки используйте index --allow-empty")
        embeddings = make_embeddings(settings, args.demo)
        if args.demo:
            print("DEMO: хеш-векторы слов; семантическая модель и API не используются.")
        if args.command == "multivector":
            from chapter2.advanced import build_multivector
            # Родители крупнее обычных чанков: ищем по короткому описанию, возвращаем контекст.
            retriever, summaries = build_multivector(docs, embeddings, settings, args.demo)
            print("Индексируемые описания:")
            for summary in summaries:
                print(summary.metadata, summary.page_content)
            print("\nНайденные полные документы:")
            for doc in retriever.invoke(args.query):
                print(doc.metadata, doc.page_content)
            return
        store, manager = open_store(settings, embeddings, args.index_dir, args.demo)
        if args.command == "index":
            print(json.dumps(synchronize(chunks, store, manager), ensure_ascii=False, indent=2))
            print(f"Всего чанков в хранилище: {store._collection.count()}")
        elif args.command == "search":
            # Для cosine расстояние меньше означает более близкий документ.
            for doc, distance in store.similarity_search_with_score(args.query, k=args.k):
                print(f"\ncosine distance={distance:.4f} | {doc.metadata}")
                print(doc.page_content)
    except APIStatusError as exc:
        print(f"Ошибка API HTTP {exc.status_code}. 401/403: ключ или права; "
              "404: URL/ID модели; 429: лимит; 400: совместимость модели/параметров. "
              "Проверьте .env и поддержку /embeddings у провайдера.", file=sys.stderr)
        raise SystemExit(1) from None
    except APIConnectionError:
        print("Нет соединения с провайдером. Проверьте OPENAI_BASE_URL, сеть и TLS.", file=sys.stderr)
        raise SystemExit(1) from None
    except (ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
