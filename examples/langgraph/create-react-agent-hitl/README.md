# create-react-agent-hitl

Go port of `langgraph/examples/create-react-agent-hitl.ipynb`.

The notebook pauses a ReAct agent before its tools node using
`interrupt_before=["tools"]` and a `MemorySaver`, inspects the pending tool
call, edits its arguments, and resumes. The Go version uses
`agents.WithAgentInterruptBefore(agents.ToolsNodeName)` plus a checkpointer,
`Graph.GetState` to inspect the next step, and `Graph.UpdateState` to edit the
pending tool call before resuming with nil input.

Deviation from the notebook: in Go, a manual `UpdateState` re-arms the
boundary interrupt, so a first resume pauses at the tools node again and a
second resume actually executes the edited tool call. The Python notebook
completes after a single resume because its update does not re-arm the
interrupt. The observable outcome (the tool runs with the edited arguments)
is the same.

This example is fully offline. The ChatModel is a deterministic in-process
double standing in for the notebook's `ChatOpenAI("gpt-4o")`; no API key,
network call, or external service is involved.

## Run

```sh
cd examples/langgraph/create-react-agent-hitl
go run .
```
