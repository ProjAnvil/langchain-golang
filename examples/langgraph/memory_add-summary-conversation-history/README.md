# memory_add-summary-conversation-history

Go port of `langgraph/examples/memory/add-summary-conversation-history.ipynb`.

The example builds a `StateGraph` with two channels (`messages` and `summary`)
and a MemorySaver checkpointer. The conversation node prepends a summary-based
system message, a conditional edge routes to the summarize node after the
thread grows beyond six messages, and the summarize node writes a new summary
while pruning old messages with `messages.RemoveMessage`. The final
`GetState` call shows the pruned transcript.

## Offline stub

`ChatAnthropic` is replaced by a deterministic in-process model that formulates
replies from the latest user text. No API key, network access, or service is
used; message IDs are assigned locally so that `RemoveMessage` can address
them.

## Run

```sh
cd examples/langgraph/memory_add-summary-conversation-history && go run .
```
