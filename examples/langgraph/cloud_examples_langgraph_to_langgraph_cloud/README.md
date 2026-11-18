# cloud_examples_langgraph_to_langgraph_cloud

Go port of
`langgraph/examples/cloud_examples/langgraph_to_langgraph_cloud.ipynb`.

The notebook runs the same weather ReAct agent locally and through the
LangGraph Cloud SDK. This port compares the same operations using an in-memory
client that wraps the local compiled graph:

- `ainvoke` vs `client.runs.wait`;
- `astream(stream_mode="values")` vs `client.runs.stream`;
- a local `MemorySaver` checkpointer with thread IDs vs `client.threads.create`
  plus thread-scoped runs and `threads.get_state`;
- `interrupt_before` on the tools node, paused and resumed both locally and via
  the client;
- `stream_mode="messages"` and chat-model stream events.

Every client call is recorded and printed at the end so the local/cloud API
correspondence is explicit.

## Offline stub

There is no LangGraph Cloud deployment, HTTP client, OpenAI key, or network
access. The cloud client is a local struct; the model and weather tool are
deterministic in-process doubles.

## Run

```sh
cd examples/langgraph/cloud_examples_langgraph_to_langgraph_cloud && go run .
```
