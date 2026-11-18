# streaming-subgraphs

Go port of `langgraph/examples/streaming-subgraphs.ipynb`.

The notebook defines a subgraph with two nodes and embeds it as a node of a
parent graph. It streams `updates` twice: once at the root only, and once with
`subgraphs=True`, where each chunk is a `(namespace, chunk)` tuple. The Go port
uses `AddSubgraph` and `StreamOptions.Subgraphs`; nested chunks carry their
path in `StreamChunk.Namespace` instead of a Python tuple.

## Offline note

The notebook is provider-free and uses only local state. The Go port is equally
self-contained: no API key, network, or external service.

## Run

```sh
cd examples/langgraph/streaming-subgraphs
go run .
```
