# rag_langgraph_self_rag_local

Go port of `examples/rag/langgraph_self_rag_local.ipynb`.

The notebook runs the Self-RAG graph with Nomic local embeddings and a
ChatOllama `mistral` model rather than hosted OpenAI APIs. The graph itself is
the same retrieve / grade / generate / rewrite topology.

What the Go port demonstrates:

- The Self-RAG `StateGraph`: START to retrieve, document grading with a
  conditional edge, a rewrite loop back to retrieval, and answer grading with
  a conditional edge to END.
- A deterministic in-memory corpus and token-hashing fake embeddings instead
  of Chroma plus Nomic local embeddings.
- Offline keyword-overlap functions for the retrieval grader, answer grader,
  hallucination grader, rewriter and generator.

Offline stub: no Ollama server, Nomic model or API key is required. Nothing is
downloaded and no network call is made; the local model role is represented by
deterministic Go code with the same wiring.

## Run

```sh
cd examples/langgraph/rag_langgraph_self_rag_local
go run .
```
