# pass-config-to-tools

Go port of `langgraph/examples/pass-config-to-tools.ipynb`.

The notebook defines three tools that read `config["configurable"]["user_id"]`
to update, delete and list a per-user favorites map, then runs a ReAct agent
three times with `{"configurable": {"user_id": "123"}}`.

## What this port demonstrates

- Passing run-scoped configuration with `runtime.ContextWithValues` and
  reading it inside a tool via `runtime.ValuesFromContext(ctx)`. The `ctx` a
  tool receives is the graph `runtime.Runtime`, so the configured values reach
  the tool unchanged (the Go analogue of `RunnableConfig` injection).
- Three tools operating on a shared `user_id -> pets` map: update, delete,
  list.
- A manual ReAct loop (`START -> agent -> action -> agent -> ...`) using
  `prebuilt.ToolNode`, with a conditional edge that stops when no tool call
  is pending.
- Three sequential runs reusing the same `user_id`, showing persisted state.

## Offline stub

The notebook uses `ChatAnthropic` plus `create_react_agent`. This port uses a
deterministic in-process model that emits a fixed tool call based on the user
message, then a final answer. Tools are single-string-input `NewSimple` tools
(`update` takes a comma-separated pet list; the others take a dummy input)
instead of the notebook's multi-argument structured tools. No API key,
network, or external service is used.

## Run

```sh
cd examples/langgraph/pass-config-to-tools
go run .
```
