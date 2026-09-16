# advanced-retrievers

Retriever composition on an in-memory store: a dense vector retriever and a keyword retriever are fused with `EnsembleRetriever` (reciprocal rank fusion), and `MultiQueryRetriever` paraphrases the question (fake model) to widen recall across the ensemble.

## Run

```sh
go run ./examples/advanced-retrievers
```

No environment variables required (fake embeddings + fake query model).
