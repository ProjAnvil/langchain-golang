# rag_langgraph_agentic_rag

Go port of `examples/rag/langgraph_agentic_rag.ipynb`.

Agentic RAG cycles between an agent and a retriever tool: the agent decides
whether to retrieve, retrieved documents are graded, and an irrelevant result
rewrites the question and returns to the agent.

What the Go port demonstrates:

- Message-based state using `channels.MessagesReducer`, the Go form of Python's
  `Annotated[Sequence[BaseMessage], add_messages]`.
- A `StateGraph` with the `tools_condition`-style conditional edge from the
  agent (retrieve or END) and a grading conditional edge after retrieval
  (generate or rewrite).
- A scripted offline agent that emits a `retrieve_blog_posts` tool call and
  later a final answer, plus a deterministic in-memory corpus with
  token-hashing fake embeddings.
- Offline keyword-overlap relevance grading in place of the structured-output
  grader.

Offline stub: no OpenAI key, ToolNode chat model, Chroma store or LangSmith Hub
prompt is used. Those roles are played by deterministic Go functions that keep
the same graph wiring and message flow.

## Run

```sh
cd examples/langgraph/rag_langgraph_agentic_rag
go run .
```
