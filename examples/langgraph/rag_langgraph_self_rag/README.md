# rag_langgraph_self_rag

Go port of `examples/rag/langgraph_self_rag.ipynb`.

The Self-RAG graph retrieves documents, grades them for relevance, generates
an answer, and grades that answer for grounding and usefulness. Irrelevant
documents or a non-useful answer route through a query rewriter that feeds
back into retrieval.

What the Go port demonstrates:

- A `StateGraph` with two conditional edges: one after grading documents and
  one after generating the answer (with a `not supported` retry back to
  `generate` and a `not useful` branch back to `transform_query`).
- A deterministic in-memory document corpus with token-hashing fake embeddings
  standing in for Chroma plus OpenAI embeddings.
- Offline keyword-overlap functions for every grader, the rewriter and the
  generator, so the graph runs with no external services.

Offline stub: no API key or network access is used. The hosted LLM graders and
LangSmith Hub prompts are replaced by deterministic Go functions that keep the
same control flow.

## Run

```sh
cd examples/langgraph/rag_langgraph_self_rag
go run .
```
