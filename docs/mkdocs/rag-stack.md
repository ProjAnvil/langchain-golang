# RAG stack: filters, pgvector, Redis, rerankers, advanced retrievers

**Languages:** English | [简体中文](zh-CN/rag-stack.zh-CN.md)

The production RAG layer on top of `core/vectorstores`: a declarative
metadata-filter DSL shared by every store, two server-backed vector stores
(pgvector, Redis), hosted rerankers (Cohere, Jina), and four advanced
retriever combinators. For the full end-to-end pipeline, see
[`examples/rag-fullchain`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/rag-fullchain)
and [`examples/advanced-retrievers`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/advanced-retrievers).

## Installation

`partners/pgvector` and `partners/redisvector` are part of the root module:

```bash
go get github.com/projanvil/langchain-golang
```

Server prerequisites:

- **pgvector**: a PostgreSQL database with the `vector` extension available
  (`CREATE EXTENSION` is attempted automatically; a user with privileges for
  the target database is enough).
- **Redis**: a Redis server with the RediSearch module (vector similarity),
  e.g. the `redis/redis-stack-server` image.

## The filter DSL (SearchOptions / OptionSearcher)

`SearchOptions` carries declarative search parameters — the Go counterpart of
Python's `SearchArgs` (`k`, `filter`, `score_threshold`, `fetch_k`). Stores
that support it implement the optional `OptionSearcher` capability interface,
so you type-assert it the same way you assert `TextAdder` or the MMR searcher:

```go
import "github.com/projanvil/langchain-golang/core/vectorstores"

if searcher, ok := store.(vectorstores.OptionSearcher); ok {
    docs, err := searcher.SimilaritySearchWithOptions(ctx, "refund policy",
        vectorstores.SearchOptions{
            K:              4,
            ScoreThreshold: 0.5,
            Filter: map[string]any{
                "tenant": "acme",                    // equality shorthand
                "page":   map[string]any{"$gt": 2},   // operator form
            },
        })
}
```

The operator set (validated by `ValidateFilter`, evaluated by `MatchFilter`):
`$eq` `$ne` `$gt` `$gte` `$lt` `$lte` `$in` `$nin` `$between` `$exists`
`$like`. Top-level fields are ANDed; one operator per condition. The in-memory
store, pgvector, and Redis all speak the same DSL.

## pgvector

```go
import "github.com/projanvil/langchain-golang/partners/pgvector"

store, err := pgvector.New(ctx, "docs_collection",
    pgvector.WithDSN(os.Getenv("PGVECTOR_DSN")), // or WithURL / WithPool
    pgvector.WithEmbedder(embedder),             // required
    // pgvector.WithDistanceMetric(pgvector.DistanceL2), // default cosine
    // pgvector.WithIndexKind(pgvector.IndexIVFFlat),    // default hnsw
)
defer store.Close()

store.AddDocuments(ctx, chunks)
store.SimilaritySearch(ctx, "how long does a refund take?", 4)
```

`New` creates the collection table and its ANN index idempotently, and
auto-detects the vector dimension from any embedder exposing `Dimensions()`.
To reuse an existing `pgxpool`, pass `WithPool`.

## Redis vector store

```go
import "github.com/projanvil/langchain-golang/partners/redisvector"

store, err := redisvector.New(ctx, "docs",
    redisvector.WithAddr("localhost:6379"), // default
    redisvector.WithPassword(os.Getenv("REDIS_PASSWORD")),
    redisvector.WithEmbedder(embedder),
    // Declare filterable metadata fields (tag or numeric) up front;
    // they become the FT.SCHEMA attributes:
    redisvector.WithMetadataField("tenant", redisvector.MetadataTag),
    redisvector.WithMetadataField("page", redisvector.MetadataNumeric),
)
defer store.Close()
```

Documents are stored as RedisJSON under the `docs:` prefix; `New` creates the
RediSearch index (`FT.CREATE ... ON JSON`) if it does not exist yet. To bring
your own client, pass `WithClient`.

## Rerankers (Cohere / Jina)

Both rerankers are `retrievers.DocumentCompressor` implementations — plug them
into `ContextualCompressionRetriever` to reorder raw retrieval hits by true
relevance:

```go
import (
    "github.com/projanvil/langchain-golang/core/retrievers"
    "github.com/projanvil/langchain-golang/partners/cohere"
)

reranker := cohere.New(
    cohere.WithAPIKey(os.Getenv("COHERE_API_KEY")), // model rerank-v3.5
    cohere.WithTopN(3),
)
base, _ := retrievers.AsRetriever(store, retrievers.WithSearchKwargs(
    map[string]any{"k": 8})) // recall wide, keep the top 3 after rerank
final, err := retrievers.NewContextualCompressionRetriever(base, reranker)
```

`partners/jina` is the same surface (`jina.New`, model `jina-reranker-v2-base-multilingual`).
`WithBaseURL`, `WithMaxRetries`, and `WithHTTPClient` are available on both.

## Advanced retrievers

`core/retrievers` ports the four production retrievers:

```go
// Ensemble: reciprocal-rank fusion of dense + keyword retrievers.
ens, _ := retrievers.NewEnsembleRetriever(
    []retrievers.Retriever{dense, keyword},
    retrievers.WithWeights([]float64{0.6, 0.4}),
)

// MultiQuery: a chat model paraphrases the question; hits are merged + deduped.
mq, _ := retrievers.NewMultiQueryRetriever(dense,
    retrievers.WithQueryModel(chatModel))

// ParentDocument: embed small chunks, return the whole parent from a docstore.
pd, _ := retrievers.NewParentDocumentRetriever(store, docstore,
    retrievers.WithChildSplitter(splitter))
pd.AddDocuments(ctx, docs) // parents to docstore, chunks to vector store
```

`MultiQueryRetriever` needs either `WithQueryModel` or a deterministic
`WithQueryVariants` function. Any partner chat model works as the query model.

## Switching points

- **Store**: swap `pgvector.New` / `redisvector.New` /
  `vectorstores.NewInMemory` — everything downstream takes the
  `vectorstores.VectorStore` interface.
- **Embeddings**: any `embeddings.Embeddings` (openai, ollama,
  `embeddings.NewFake(dim)` for tests); dimension is inferred automatically.
- **Rerank on/off**: wrap the base retriever in
  `NewContextualCompressionRetriever` only when an API key is present.
