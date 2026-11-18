# rag_langgraph_self_rag_pinecone_movies

Go port of `examples/rag/langgraph_self_rag_pinecone_movies.ipynb`.

The notebook applies the Self-RAG graph to a hosted Pinecone `sample-movies`
index using OpenAI embeddings and LangSmith Hub grader prompts.

What the Go port demonstrates:

- The unchanged Self-RAG `StateGraph` topology (retrieve, grade documents,
  rewrite loop, generate, answer grading).
- A deterministic in-memory movie-summary corpus with token-hashing fake
  embeddings in place of the Pinecone index and OpenAI embeddings.
- Offline keyword-overlap functions for the retrieval grader, hallucination
  grader, answer grader and rewriter, exercised with movie questions such as
  "Movies that star Daniel Craig".

Offline stub: no Pinecone account, API key, LangSmith Hub access or network
call is used. The hosted vector store and models are represented by local
deterministic code while the graph wiring stays the same.

## Run

```sh
cd examples/langgraph/rag_langgraph_self_rag_pinecone_movies
go run .
```
