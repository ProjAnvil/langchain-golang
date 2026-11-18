# tool-calling-errors

Go port of `langgraph/examples/tool-calling-errors.ipynb`.

The notebook raises a `ValueError` from a `get_weather` tool and shows how
`ToolNode` and `create_react_agent` handle it: default conversion into an error
`ToolMessage`, a custom `handle_tool_errors` message, disabling error handling,
and a ReAct agent that recovers on the next model turn. The Go version uses
`langchain/tools.NewToolNode` with `WithHandleToolErrors`, and a
`langchain/agents.CreateAgent` driven by an offline scripted model.

Error ToolMessages are flagged with `ResponseMetadata["status"] = "error"`,
the Go counterpart of Python's `ToolMessage.status = "error"`.

This example is fully offline. All models are deterministic in-process doubles;
no API key, network call, or external service is involved. The notebook's
later cells (a custom `call_tool` loop, model fallback via `should_fallback`,
and message removal with `RemoveMessage`) are represented by the error-recovery
loop above rather than reproduced cell by cell.

## Run

```sh
cd examples/langgraph/tool-calling-errors
go run .
```
