# subgraph-transform-state

Go port of `langgraph/examples/subgraph-transform-state.ipynb`.

Source notebook: `langgraph/examples/subgraph-transform-state.ipynb`.

What the Go port demonstrates: three nested graphs (grandchild -> child ->
parent) whose state schemas share no keys. Each wrapping node takes the
parent's key, builds a fresh input map for the compiled child graph, invokes it
with `CompiledGraph.Invoke`, and maps the child's output key back to the
parent's key. This is the notebook's "pass a function instead of a compiled
graph" pattern.

Deviation from the Python port: the Go example uses explicit state-key
translation calls rather than `StateGraph.AddSubgraph`, because `AddSubgraph`
merges the child's whole state into the parent (the right tool when keys are
shared, as in the `subgraph` example). Here no keys are shared, so explicit
transformation is the faithful port.

No API key, network or service is required.

## Run

```sh
cd examples/langgraph/subgraph-transform-state
go run .
```
