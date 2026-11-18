# rag_langgraph_crag_local

Go port of `examples/rag/langgraph_crag_local.ipynb`.

The local-model CRAG variant keeps a `steps` trace and conditionally falls back
to web search when retrieved documents are judged irrelevant. It uses Ollama
`llama3` and (optionally) Nomic local embeddings in the notebook.

What the Go port demonstrates:

- The local CRAG `StateGraph`: retrieve to grade_documents, a conditional edge
  that routes either to web_search or generate, then generate to END.
- The `steps` list threaded through each node, printed at the end, mirroring
  the trajectory the notebook evaluates.
- A deterministic in-memory corpus and token-hashing fake embeddings instead
  of SKLearn/Chroma plus hosted or local embeddings.
- Offline keyword-overlap grading and an emulated web search.

Offline stub: no Ollama server, Nomic model, Tavily key or network call is
required. The local model steps are deterministic Go functions with the same
graph wiring and state shape.

## Run

```sh
cd examples/langgraph/rag_langgraph_crag_local
go run .
```
