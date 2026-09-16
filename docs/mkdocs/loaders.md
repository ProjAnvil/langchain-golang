# Document loaders

**Languages:** English | [简体中文](zh-CN/loaders.zh-CN.md)

`core/documentloaders` provides the three loaders most RAG pipelines start
with: HTML (goquery), web pages over HTTP, and PDF (ledongthuc/pdf). All
implement the lazy `LazyLoader` interface, so they compose with the shared
`Load` / `LoadAndSplit` helpers and the recursive splitters.

## Installation

```bash
go get github.com/projanvil/langchain-golang
```

## HTML loader

Turns an HTML document into one `Document` of readable body text (scripts,
styles, and tags stripped; headings collapsed onto lines):

```go
import "github.com/projanvil/langchain-golang/core/documentloaders"

loader := documentloaders.NewHTMLLoader(strings.NewReader(htmlSource),
    documentloaders.WithMetadata(map[string]any{"source": "in-app page"}))
docs, err := documentloaders.Load(ctx, loader) // docs[0].Metadata["title"] carries <title>
```

## Web loader

Fetches a page over HTTP and runs the shared HTML extraction — the counterpart
of Python's `WebBaseLoader` (same default header template; the `USER_AGENT`
env variable overrides the default User-Agent). Metadata follows upstream
`_build_metadata`: `source`, `title`, `description`, `language`:

```go
web := documentloaders.NewWebLoader(&http.Client{Timeout: 15 * time.Second},
    documentloaders.WithMaxBodyBytes(1 << 20)) // 1 MiB cap, default 10 MiB

// One-shot:
docs, err := web.Load(ctx, "https://example.com/docs")

// Or bind the URL, then use the shared lazy helpers:
bound := web.ForURL("https://example.com/docs")
docs, err = documentloaders.Load(ctx, bound)
```

## PDF loader

Extracts text per page — one document per page, with the 0-based `page` and
`total_pages` in metadata (Python `PyPDFLoader` parity; encrypted or
malformed PDFs fail loudly):

```go
f, _ := os.Open("report.pdf")
defer f.Close()
loader := documentloaders.NewPDFLoader(f) // any io.Reader, streamed
docs, err := documentloaders.Load(ctx, loader)
```

## Load and split in one step

The usual RAG on-ramp — load then split into retrieval-sized chunks:

```go
splitter, _ := textsplitters.NewRecursiveCharacter(nil, false, textsplitters.Config{
    ChunkSize:    1000,
    ChunkOverlap: 200,
})
chunks, err := documentloaders.LoadAndSplit(ctx, loader, splitter)
store.AddDocuments(ctx, chunks)
```

`LoadAndSplit` with a `nil` splitter uses the factory registered via
`RegisterDefaultTextSplitterFactory` (and errors when none is registered).

## Files and blobs

`NewBlobFromPath(path, mimetype, metadata)` / `NewBlobFromData(...)` give
lazy, source-tagged inputs for file-walking indexers; `blob.Reader()` feeds
any of the three loaders.

## Switching points

- **Loader**: all three share the `LazyLoader` shape — swap HTML for web or
  PDF without touching the split/embed/store stages.
- **Splitter**: `LoadAndSplit` accepts any `TextSplitter`; recursive
  character is the default, token-based splitters drop in.
- **Fetching policy**: inject your own `*http.Client` (timeouts, proxies,
  auth) into `NewWebLoader`; cap untrusted bodies with `WithMaxBodyBytes`.
