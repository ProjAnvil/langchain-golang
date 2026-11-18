# streaming-content

Go port of `langgraph/examples/streaming-content.ipynb`.

The notebook has a node write a list of words through a `StreamWriter` and reads
them with `stream_mode="custom"`, then with `["custom", "updates"]`. A second
node variant dispatches named custom events and the notebook filters them with
`astream_events`. The Go port uses `graphpkg.StreamWriterFromContext` to emit
custom payloads and `StreamCustom` (plus `StreamUpdates`) to read them; named
payloads are filtered by the `name` field on the custom stream.

## Offline note

The notebook is already provider-free. The Go port is fully in-process: no API
key, no network, no external service.

## Run

```sh
cd examples/langgraph/streaming-content
go run .
```
