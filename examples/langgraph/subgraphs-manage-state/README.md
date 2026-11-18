# subgraphs-manage-state

Go port of `langgraph/examples/subgraphs-manage-state.ipynb`.

Source notebook: `langgraph/examples/subgraphs-manage-state.ipynb`.

What the Go port demonstrates:

* A weather subgraph compiled with `graphpkg.WithInterruptBefore("weather_node")`
  and embedded in a router graph via `StateGraph.AddSubgraph`, with a
  `checkpoint.NewMemorySaver()` checkpointer on the parent.
* Thread-based invocation: a non-weather query takes the normal branch; a
  weather query pauses inside the subgraph. The pause is surfaced as a parent
  interrupt and inspected with `GetState` (`next` and `tasks`), then continued
  with `InvokeWithOptions(..., Options{ThreadID, Resume: nil})`.
* `GetStateHistory` to read the thread's checkpoints, and `UpdateState` with
  `as_node="weather_graph"` to inject a final answer at the parent level and
  continue past the subgraph (the notebook's "skip the subgraph" pattern).
* A grandparent graph wrapping the router whose
  `GetGraph(WithXrayDepth(-1)).DrawMermaid()` renders the nested structure in
  Mermaid text (the notebook's `xray=1` drawing).

Offline note: the LLM/structured-output nodes and the weather tool are replaced
by deterministic keyword matching and plain functions. No API key, network or
service is required.

Deviation from Python: messages use Go `LastValue` channels with explicit
full-list updates instead of `MessagesReducer`, so a subgraph's returned
message list replaces the parent's rather than being appended twice at the
`AddSubgraph` boundary. Nested subgraph-state inspection uses the root
`GetState` snapshot (`Next`, `Tasks`, `Interrupts`) rather than Python's
per-task `subgraphs=True` state objects.

## Run

```sh
cd examples/langgraph/subgraphs-manage-state
go run .
```
