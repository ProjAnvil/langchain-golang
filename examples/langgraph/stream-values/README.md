# stream-values

Go port of `langgraph/examples/stream-values.ipynb`.

The notebook streams `stream_mode="values"`, pretty-prints the last message of
every full-state snapshot, then captures the final snapshot in `final_result`
and prints its last message again. The Go port uses `StreamValues`; each
chunk's payload is the complete `map[string]any` state, from which the last
`messages.Message` is printed (role plus content, the Go equivalent of
`pretty_print()`).

## Offline note

The Python notebook uses `ChatOpenAI` and an `OPENAI_API_KEY`. The Go port
replaces the agent with two deterministic local nodes. No provider SDK, API
key, or network is involved.

## Run

```sh
cd examples/langgraph/stream-values
go run .
```
