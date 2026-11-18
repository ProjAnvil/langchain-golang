# recursion-limit

Go port of `langgraph/examples/recursion-limit.ipynb`.

The notebook builds two cyclic graphs that keep appending to an `aggregate`
list until it reaches length 7, then shows that a low `recursion_limit` aborts
the run with `GraphRecursionError`.

## What this port demonstrates

- A conditional edge that loops between nodes until a state predicate is met.
- `channels.AppendSliceReducer` on `aggregate` (Python's
  `Annotated[list, operator.add]`).
- A second, branching loop using `AddJoinEdge` to join two parallel branches
  before re-entering the cycle.
- Overriding the recursion limit per run with `graphpkg.Options.RecursionLimit`
  (the Go form of the notebook's `{"recursion_limit": 4}` config) and catching
  `*types.GraphRecursionError`.

## Offline stub

No model is involved. Both graphs are deterministic counting loops. No API key,
network, or external service is used.

## Run

```sh
cd examples/langgraph/recursion-limit
go run .
```
