# streaming-events-from-within-tools

Go port of `langgraph/examples/streaming-events-from-within-tools.ipynb`.

The notebook's tool writes chunks with `get_stream_writer()`, then shows a
model invoked inside a tool whose tokens are filtered by
`metadata["langgraph_node"] == "tools"`, and finally a tool-only graph whose
writer carries token payloads. The Go port maps these to `StreamWriter` custom
payloads, `StreamMessages` chunks with `MessageChunk.Metadata`, and a
tool-only graph with `StreamCustom`, respectively.

## Offline note

`ChatOpenAI` and `AsyncOpenAI` from the notebook are replaced by deterministic
in-process fakes. No API key, network, or external service is used.

## Run

```sh
cd examples/langgraph/streaming-events-from-within-tools
go run .
```
