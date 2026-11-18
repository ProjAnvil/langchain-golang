# rag_langgraph_crag

Go port of `examples/rag/langgraph_crag.ipynb`.

Corrective RAG retrieves, grades each document, and if any hit is irrelevant
rewrites the question and falls back to web search before generating.

What the Go port demonstrates:

- The CRAG `StateGraph`: START to retrieve, a conditional edge after document
  grading that either goes straight to generate or through transform_query and
  web_search_node, and a terminal generate node.
- A deterministic in-memory corpus and token-hashing fake embeddings in place
  of Chroma and OpenAI embeddings.
- Offline keyword-overlap graders, a query rewriter and an emulated Tavily web
  search.

Offline stub: no OpenAI or Tavily key, LangSmith Hub prompt or network call is
used. Those steps are substituted by deterministic Go functions that preserve
the notebook's control flow.

## Run

```sh
cd examples/langgraph/rag_langgraph_crag
go run .
```
