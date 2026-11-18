# stream-updates

Go port of `langgraph/examples/stream-updates.ipynb`.

The notebook streams a ReAct weather agent with `stream_mode="updates"` and
prints, for each streamed chunk, the node name and the values that node wrote.
The Go port uses `StreamUpdates`; each payload is a
`map[string]any{nodeName: {channel: value}}`, which is unpacked and printed per
node.

## Offline note

The Python notebook uses `ChatOpenAI` and an `OPENAI_API_KEY`. The Go port
replaces the model and the weather tool with deterministic local node functions.
No provider SDK, API key, or network is involved.

## Run

```sh
cd examples/langgraph/stream-updates
go run .
```
