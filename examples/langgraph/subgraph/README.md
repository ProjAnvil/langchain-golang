# subgraph

Go port of `langgraph/examples/subgraph.ipynb`.

Source notebook: `langgraph/examples/subgraph.ipynb`.

What the Go port demonstrates:

1. A compiled subgraph added as a parent node with `StateGraph.AddSubgraph`.
   The shared state key `foo` passes from parent to child and back; `bar` is a
   subgraph-only key used to build the shared update. A `Stream` call with
   `Subgraphs: true` prints the nested update chunks with their namespaces,
   the Go counterpart of `stream(..., subgraphs=True)`.
2. A subgraph whose state keys are not shared with the parent, invoked from a
   normal node that translates state in and out with `CompiledGraph.Invoke`.

No API key, network or service is required.

Deviation from Python: `AddSubgraph` merges all of the child's final keys into
the parent state (the child-only `bar` key also appears in the parent's final
state). The notebook's typed schemas would drop it; the Go graph uses an
open `map[string]any`, so this extra key is visible. The shared `foo` value is
unchanged and is what the example asserts.

## Run

```sh
cd examples/langgraph/subgraph
go run .
```
