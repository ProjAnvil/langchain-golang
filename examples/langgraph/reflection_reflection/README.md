# reflection_reflection

Go port of `langgraph/examples/reflection/reflection.ipynb`.

The notebook builds a two-node reflection loop: a generator writes an essay,
a critic flips the message roles and returns critique as human feedback, and
`should_continue` stops the loop after three iterations. The Go port mirrors
that topology:

- `generate -> reflect -> generate` with a conditional edge on `generate`;
- a `messages` channel using `channels.MessagesReducer`, the Go form of
  Python's `add_messages`;
- a `MemorySaver` checkpointer and thread ID, as in the notebook's config;
- `types.END` for the notebook's `END`.

The essay generator and critic are deterministic offline stubs, so no API key
or network is required. The message-role flip is performed locally rather than
by calling an LLM.

## Run

```sh
cd examples/langgraph/reflection_reflection
go run .
```
