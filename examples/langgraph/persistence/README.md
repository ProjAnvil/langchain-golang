# persistence

Go port of `langgraph/examples/persistence.ipynb`.

Source notebook: `langgraph/examples/persistence.ipynb`.

What the Go port demonstrates:

1. **Thread-scoped checkpoint memory.** A `StateGraph` with a `messages`
   reducer is compiled with `checkpoint.NewMemorySaver()`. Two invocations on
   the same `graphpkg.Options{ThreadID: "1"}` share state, so the second turn
   sees the first turn's messages. `GetState` and `GetStateHistory` read the
   saved checkpoints back (the Go equivalents of `get_state` and
   `get_state_history`).
2. **Cross-thread long-term memory.** A second graph is compiled with
   `graphpkg.WithStore(store.NewInMemoryStore())`. Its node reads memories
   under the `("memories", user_id)` namespace from `runtime.Runtime.Store`,
   writes `User name is Bob` when the user asks to remember, and a different
   thread for the same `user_id` recalls it. This mirrors the notebook's
   `InMemoryStore` / `BaseStore` cell.

The chat model is replaced by a deterministic in-process function that echoes
the user's text and reports the visible history length; no API key or network
access is needed. All state lives in memory for the lifetime of the process.

## Run

```sh
cd examples/langgraph/persistence
go run .
```
