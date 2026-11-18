# async

Go port of `langgraph/examples/async.ipynb`.

The notebook builds a ReAct-style agent/tool loop (`START -> agent -> action
-> agent -> ...`) and drives it with the asynchronous APIs `ainvoke`,
`astream` and `astream_log`. Go has no `async`/`await`: the same graph is
invoked synchronously with `Invoke`, and per-node output is observed through
`Stream` in `updates` mode. That is the closest analogue of the notebook's
`astream(stream_mode="updates")` loop.

## What this port demonstrates

- `messages` state key with `channels.MessagesReducer` (Python's
  `Annotated[list, add_messages]`).
- An agent/tool cycle: a conditional edge after `agent` routes either to the
  tool node (`continue`) or to `END` (`end`).
- `prebuilt.ToolNode` running a `search` tool registered with
  `langchain/tools`.
- Both `Invoke` (final state) and `Stream` (per-node updates).

## Offline stub

The notebook calls `ChatAnthropic` with an `ANTHROPIC_API_KEY`. This port uses a
deterministic in-process model double: the first `agent` turn emits an AI
message with a `search` tool call; after the tool runs, the second turn emits a
plain final answer. No API key, network, or external service is used.

## Run

```sh
cd examples/langgraph/async
go run .
```
