# run-id-langsmith

Go port of `langgraph/examples/run-id-langsmith.ipynb`.

The Python notebook adds `run_name` and `tags` to the `RunnableConfig` so the
LangSmith trace of a ReAct agent is labeled. This Go version replaces the
remote tracing service with an in-memory trace recorder that:

- assigns a deterministic run ID,
- stores the run name and tags,
- records model and tool events during the agent run,
- prints the collected trace at the end.

The weather ReAct agent itself is built with `agents.CreateAgent` and a
scripted `ChatModel`, and its state is observed via `StreamValues`.

## Offline stub

No LangSmith client, API key, model provider, or network call is used. The
trace is a local data structure, not a cloud run.

## Run

```sh
cd examples/langgraph/run-id-langsmith && go run .
```
