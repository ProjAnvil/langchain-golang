# tutorials_tnt-llm_tnt-llm

Go port of `examples/tutorials/tnt-llm/tnt-llm.ipynb`.

## What the Go port demonstrates

The notebook's taxonomy-generation graph is rebuilt with the Go `StateGraph`
API as a plan/search/reflect-style cycle:

```
START -> summarize -> get_minibatches -> generate_taxonomy
      -> update_taxonomy -> (update_taxonomy | review_taxonomy)* -> END
```

- `clusters` uses `channels.AppendSliceReducer`, the Go equivalent of
  Python's `Annotated[List[List[dict]], operator.add]`: every generation,
  update, or review appends one full taxonomy version
- the conditional edge repeats `update_taxonomy` while fewer taxonomy
  versions exist than minibatches, then runs `review_taxonomy`
- summary generation and taxonomy generation both go through an offline
  `language.ChatModel` double; the taxonomy chain emits the same XML shape
  the notebook parses with regexes
- a `checkpoint.NewMemorySaver()` is installed with a `ThreadID`

## Offline safe substitution

The notebook pulls prompts from LangSmith Hub, calls Anthropic/OpenAI models,
loads real traces, computes embeddings, and fits a scikit-learn classifier.
None of that is reproduced:

- the corpus is a fixed slice of six in-memory documents
- summaries are derived by keyword lookup, not by an LLM
- taxonomy clusters are deterministic strings emitted by the offline model
- the final classifier/embedding sections are intentionally omitted; the
  port focuses on the graph's generate -> update -> review loop

No API key, network request, or external service is used.

## Run

```sh
cd examples/langgraph/tutorials_tnt-llm_tnt-llm
go run .
```
