# stream-multiple

Go port of `langgraph/examples/stream-multiple.ipynb`.

The notebook streams a ReAct weather agent with `stream_mode=["updates", "debug"]`
and prints the event family together with each payload. The Go port runs a
small agent/tools graph and consumes `StreamUpdates` and `StreamDebug` in one
`CompiledGraph.Stream` call, printing the mode label for every chunk.

## Offline note

The Python notebook uses `ChatOpenAI` and an `OPENAI_API_KEY`. This port uses a
deterministic in-process agent and tool (no provider SDK, no API key, no
network). All output comes from local state transitions.

## Run

```sh
cd examples/langgraph/stream-multiple
go run .
```
