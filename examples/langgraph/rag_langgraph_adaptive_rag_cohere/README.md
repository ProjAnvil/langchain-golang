# rag_langgraph_adaptive_rag_cohere

Go port of `examples/rag/langgraph_adaptive_rag_cohere.ipynb`.

This Adaptive RAG variant uses Cohere embeddings and Command-R for routing,
document grading, generation and answer grading, and adds an `llm_fallback`
node for conversational questions that fit neither the vectorstore nor web
search.

What the Go port demonstrates:

- A three-way conditional START router: web search, vectorstore retrieval or
  llm_fallback, plus conditional edges after grading and generation.
- The `llm_fallback` branch that terminates directly at END, mirroring the
  notebook's "no tool call" route.
- A deterministic in-memory corpus and token-hashing fake embeddings instead
  of Chroma and Cohere `embed-v4.0`.
- Offline keyword-overlap functions for all graders/routers, an emulated web
  search, and a scripted fallback answer.

Offline stub: no Cohere or Tavily key and no network access is used. The
Cohere model calls are replaced by deterministic Go functions that preserve
the notebook's control flow.

## Run

```sh
cd examples/langgraph/rag_langgraph_adaptive_rag_cohere
go run .
```
