# pass-run-time-values-to-tools

Go port of `langgraph/examples/pass-run-time-values-to-tools.ipynb`.

The notebook shows two ways to give a tool run-time context that is not part of
the tool's declared arguments:

1. `InjectedState` — a tool receives the graph state (here `docs`) and joins it.
2. `InjectedStore` plus `RunnableConfig` — a tool receives a cross-thread store
   and a `user_id` from config, then searches per-user documents.

## What this port demonstrates

- A `ToolCallWrapper` that injects `ToolCallRequest.State` and `Store` into the
  context passed to a tool. Go has no annotation-based `InjectedState` /
  `InjectedStore`, so this explicit wrapper+context pattern is the equivalent.
- Scenario 1: a `get_context` tool reads `docs` from the injected state.
- Scenario 2: a `get_context_store` tool reads `user_id` from the run config
  (via `runtime.ValuesFromContext`) and searches an `InMemoryStore` injected
  with `graph.WithStore` / `tools.WithToolNodeStore`. The same graph is run
  with `user_id` 1 and 2 to show isolated results.
- A manual ReAct loop (`START -> agent -> action -> agent -> ...`) using
  `prebuilt.ToolNode`.

## Offline stub

The notebook uses `ChatOpenAI`, `create_react_agent`, `MemorySaver` and
`InMemoryStore`. This port uses a deterministic in-process model and the
real in-memory store from the SDK. No API key, network, or external service is
used.

## Run

```sh
cd examples/langgraph/pass-run-time-values-to-tools
go run .
```
