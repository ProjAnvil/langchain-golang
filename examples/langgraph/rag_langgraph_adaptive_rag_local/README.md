# rag_langgraph_adaptive_rag_local

Go port of `examples/rag/langgraph_adaptive_rag_local.ipynb`.

The notebook is the local-model version of Adaptive RAG: it keeps the same
START router / retrieval / grading / rewrite / generation graph but replaces
OpenAI embeddings and chat models with Nomic local embeddings and a ChatOllama
`mistral` model.

What the Go port demonstrates:

- The same `StateGraph` topology as `rag_langgraph_adaptive_rag`: conditional
  START routing, document grading, query-rewrite loop and generation grading.
- A deterministic in-memory corpus and token-hashing fake embeddings instead
  of Chroma plus Nomic local embeddings.
- Offline keyword-overlap functions instead of ChatOllama-based router,
  graders, rewriter and answer generator, and an emulated web search.

Offline stub: no Ollama server, Nomic model or API key is needed. Nothing is
downloaded and no network call is made; the local-model role is played by
deterministic Go code with the same graph wiring.

## Run

```sh
cd examples/langgraph/rag_langgraph_adaptive_rag_local
go run .
```
