# RAG 栈：过滤器、pgvector、Redis、重排器、高级检索器

**Languages:** [English](../rag-stack.md) | 简体中文

构建在 `core/vectorstores` 之上的生产级 RAG 层：全库共享的声明式元数据
过滤 DSL、两个服务端向量存储（pgvector、Redis）、托管重排器（Cohere、
Jina），以及四个高级检索器组合件。完整端到端流水线见
[`examples/rag-fullchain`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/rag-fullchain)
与
[`examples/advanced-retrievers`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/advanced-retrievers)。

## 安装

`partners/pgvector` 与 `partners/redisvector` 属于根模块：

```bash
go get github.com/projanvil/langchain-golang
```

服务端前置条件：

- **pgvector**：装有 `vector` 扩展的 PostgreSQL 数据库（`CREATE EXTENSION`
  会自动尝试执行；对目标数据库有权限的用户即可）。
- **Redis**：带 RediSearch 模块（向量相似度）的 Redis 服务器，例如
  `redis/redis-stack-server` 镜像。

## 过滤 DSL（SearchOptions / OptionSearcher）

`SearchOptions` 携带声明式搜索参数——对应 Python 的 `SearchArgs`（`k`、
`filter`、`score_threshold`、`fetch_k`）。支持它的存储实现可选能力接口
`OptionSearcher`，调用方按断言 `TextAdder`、MMR 检索器同样的方式断言即可：

```go
import "github.com/projanvil/langchain-golang/core/vectorstores"

if searcher, ok := store.(vectorstores.OptionSearcher); ok {
    docs, err := searcher.SimilaritySearchWithOptions(ctx, "退款政策",
        vectorstores.SearchOptions{
            K:              4,
            ScoreThreshold: 0.5,
            Filter: map[string]any{
                "tenant": "acme",                    // 相等简写
                "page":   map[string]any{"$gt": 2},   // 操作符形式
            },
        })
}
```

操作符集合（`ValidateFilter` 校验、`MatchFilter` 求值）：`$eq` `$ne` `$gt`
`$gte` `$lt` `$lte` `$in` `$nin` `$between` `$exists` `$like`。顶层字段之间
为 AND；每个条件只允许一个操作符。内存存储、pgvector、Redis 讲同一套 DSL。

## pgvector

```go
import "github.com/projanvil/langchain-golang/partners/pgvector"

store, err := pgvector.New(ctx, "docs_collection",
    pgvector.WithDSN(os.Getenv("PGVECTOR_DSN")), // 或 WithURL / WithPool
    pgvector.WithEmbedder(embedder),             // 必填
    // pgvector.WithDistanceMetric(pgvector.DistanceL2), // 默认 cosine
    // pgvector.WithIndexKind(pgvector.IndexIVFFlat),    // 默认 hnsw
)
defer store.Close()

store.AddDocuments(ctx, chunks)
store.SimilaritySearch(ctx, "退款要多久？", 4)
```

`New` 幂等地创建 collection 表与 ANN 索引，并能从暴露 `Dimensions()` 的
embedder 自动推断向量维度。复用已有 `pgxpool` 时传 `WithPool`。

## Redis 向量存储

```go
import "github.com/projanvil/langchain-golang/partners/redisvector"

store, err := redisvector.New(ctx, "docs",
    redisvector.WithAddr("localhost:6379"), // 默认值
    redisvector.WithPassword(os.Getenv("REDIS_PASSWORD")),
    redisvector.WithEmbedder(embedder),
    // 预先声明可过滤的元数据字段（tag 或 numeric），
    // 它们会成为 FT.SCHEMA 属性：
    redisvector.WithMetadataField("tenant", redisvector.MetadataTag),
    redisvector.WithMetadataField("page", redisvector.MetadataNumeric),
)
defer store.Close()
```

文档以 RedisJSON 形式存放在 `docs:` 前缀下；`New` 会在索引不存在时创建
RediSearch 索引（`FT.CREATE ... ON JSON`）。要自带客户端，传 `WithClient`。

## 重排器（Cohere / Jina）

两个重排器都是 `retrievers.DocumentCompressor` 实现——插入
`ContextualCompressionRetriever`，按真实相关度重排原始检索命中：

```go
import (
    "github.com/projanvil/langchain-golang/core/retrievers"
    "github.com/projanvil/langchain-golang/partners/cohere"
)

reranker := cohere.New(
    cohere.WithAPIKey(os.Getenv("COHERE_API_KEY")), // 模型 rerank-v3.5
    cohere.WithTopN(3),
)
base, _ := retrievers.AsRetriever(store, retrievers.WithSearchKwargs(
    map[string]any{"k": 8})) // 大召回，重排后留前 3
final, err := retrievers.NewContextualCompressionRetriever(base, reranker)
```

`partners/jina` 接口相同（`jina.New`，模型
`jina-reranker-v2-base-multilingual`）。两者都提供 `WithBaseURL`、
`WithMaxRetries`、`WithHTTPClient`。

## 高级检索器

`core/retrievers` 移植了四个生产级检索器：

```go
// Ensemble：稠密 + 关键词检索器的倒数排名融合（RRF）。
ens, _ := retrievers.NewEnsembleRetriever(
    []retrievers.Retriever{dense, keyword},
    retrievers.WithWeights([]float64{0.6, 0.4}),
)

// MultiQuery：聊天模型改写问题；命中合并去重。
mq, _ := retrievers.NewMultiQueryRetriever(dense,
    retrievers.WithQueryModel(chatModel))

// ParentDocument：嵌入小块，从 docstore 返回完整父文档。
pd, _ := retrievers.NewParentDocumentRetriever(store, docstore,
    retrievers.WithChildSplitter(splitter))
pd.AddDocuments(ctx, docs) // 父文档进 docstore，块进向量存储
```

`MultiQueryRetriever` 必须提供 `WithQueryModel` 或确定性的
`WithQueryVariants` 函数。任何 partner 聊天模型都能当查询模型。

## 切换点

- **存储**：互换 `pgvector.New` / `redisvector.New` /
  `vectorstores.NewInMemory`——下游全部面向
  `vectorstores.VectorStore` 接口。
- **嵌入**：任意 `embeddings.Embeddings`（openai、ollama、测试用
  `embeddings.NewFake(dim)`）；维度自动推断。
- **重排开关**：仅在配置了 API key 时才用
  `NewContextualCompressionRetriever` 包裹基础检索器。
