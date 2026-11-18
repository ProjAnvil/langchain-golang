# rag_langgraph_adaptive_rag

Go port of `examples/rag/langgraph_adaptive_rag.ipynb`.

The notebook builds the Adaptive RAG graph: a START router chooses between
web search and a vectorstore; retrieved documents are graded for relevance;
irrelevant results trigger a query rewrite that loops back to retrieval; and a
generated answer is graded for grounding and usefulness before the run ends.

What the Go port demonstrates:

- `StateGraph` with a conditional entry point and conditional edges.
- An in-memory vector store over a deterministic, built-in document corpus
  (no web loading, no Chroma) using token-hashing fake embeddings.
- The router, retrieval grader, hallucination grader, answer grader and query
  rewriter expressed as deterministic keyword-overlap functions instead of
  OpenAI/Cohere/Tavily calls.
- Both graph branches: an in-domain question goes through retrieval; an
  off-topic question goes through the emulated web search.

Offline stub: this example never calls an external API, downloads a model, or
requires an API key. The LLM and web-search steps are offline substitutions
whose only purpose is to exercise the same graph topology deterministically.

## Run

```sh
cd examples/langgraph/rag_langgraph_adaptive_rag
go run .
```
