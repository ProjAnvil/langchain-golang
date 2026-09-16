# rag-fullchain

The full RAG chain: HTML loader -> recursive splitter -> embeddings -> vector store -> retrieval -> optional Cohere rerank -> tool-calling agent answer.

## Run

```sh
go run ./examples/rag-fullchain
```

Runs offline with fake embeddings, an in-memory store, and a scripted model.

Optional env (each upgrades one stage):

| Env | Effect when set |
| --- | --- |
| `OPENAI_API_KEY` | real OpenAI embeddings (`text-embedding-3-small`) |
| `PGVECTOR_DSN` | pgvector collection instead of the in-memory store |
| `COHERE_API_KEY` | Cohere rerank (`rerank-v3.5`) as a compression retriever |
