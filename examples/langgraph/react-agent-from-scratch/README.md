# react-agent-from-scratch

Go port of `langgraph/examples/react-agent-from-scratch.ipynb`.

The notebook builds a ReAct loop by hand: an `AgentState` with an
`add_messages` reducer, a `call_model` node that prepends a system message, a
`tool_node`, a `should_continue` conditional edge, and the `agent <-> tools`
cycle. The Go version uses `langgraph/graph.StateGraph` with
`channels.MessagesReducer`, `langgraph/prebuilt.ToolNode`, and a conditional
router, driven by an offline scripted model.

This example is fully offline. The ChatModel is a deterministic in-process
double standing in for the notebook's `ChatOpenAI("gpt-4o-mini")`; no API key,
network call, or external service is involved.

## Run

```sh
cd examples/langgraph/react-agent-from-scratch
go run .
```
