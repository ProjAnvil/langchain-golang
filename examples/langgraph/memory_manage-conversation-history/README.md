# memory_manage-conversation-history

Go port of `langgraph/examples/memory/manage-conversation-history.ipynb`.

Two versions of the same ReAct-style graph are built with
`StateGraph`, `channels.MessagesReducer`, and a MemorySaver checkpointer:

1. the original agent passes the entire persisted `messages` list to the
   model, so it recalls earlier facts;
2. the modified agent applies a `filterMessages` helper (`messages[-1:]`)
   inside the node, so the model sees only the latest turn and cannot recall.

Both threads still persist the complete transcript; only the model input is
filtered.

## Offline stub

`ChatAnthropic` and its `search` tool are deterministic in-process doubles.
No API key, network access, or service is used.

## Run

```sh
cd examples/langgraph/memory_manage-conversation-history && go run .
```
