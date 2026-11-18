# create-react-agent-memory

Go port of `langgraph/examples/create-react-agent-memory.ipynb`.

The notebook adds a `MemorySaver` checkpointer to `create_react_agent` so a
follow-up question on the same `thread_id` can see earlier turns. The Go
version uses `agents.WithAgentCheckpointer(checkpoint.NewMemorySaver())` and
`Graph.InvokeWithOptions` with a fixed `ThreadID`, printing the retained
message history after each turn to show it grows across calls.

This example is fully offline. The ChatModel is a deterministic in-process
double standing in for the notebook's `ChatOpenAI("gpt-4o")`; no API key,
network call, or external service is involved.

## Run

```sh
cd examples/langgraph/create-react-agent-memory
go run .
```
