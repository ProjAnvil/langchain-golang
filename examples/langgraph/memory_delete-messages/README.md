# memory_delete-messages

Go port of `langgraph/examples/memory/delete-messages.ipynb`.

The example demonstrates message deletion in a tool-calling graph with a
MemorySaver checkpointer:

1. `UpdateState` applies a `messages.RemoveMessage` sentinel to delete one
   message from an existing thread (notebook cells 11-15).
2. A dedicated `delete_messages` node removes all but the three most recent
   messages during a run, routed to by the agent's conditional edge (cells
   17-21).

`channels.MessagesReducer` performs the ID-aware merge; message IDs are
assigned locally so removal commands can address them.

## Offline stub

`ChatAnthropic` and its `search` tool are replaced by a scripted model and an
in-process tool with deterministic output. No API key, network access, or
service is used.

## Run

```sh
cd examples/langgraph/memory_delete-messages && go run .
```
