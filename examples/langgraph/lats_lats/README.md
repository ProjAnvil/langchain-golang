# lats_lats

Go port of `langgraph/examples/lats/lats.ipynb`.

The notebook implements Language Agent Tree Search: a `Node` type carrying a
reflection score, UCT-based selection, backpropagation, and a graph that loops
`start -> expand` until a solution is found or the tree exceeds height five.
The Go port preserves that structure:

- a Go `treeNode` type with `uct`, `backpropagate`, `select`, `height`, and
  `bestSolution` methods matching the notebook's `Node`;
- a `StateGraph` with `start` and `expand` nodes and a `should_loop`
  conditional edge routing to `expand` or `types.END`;
- tree state stored in a graph channel and returned by each node.

The LLM-based reflection and expansion are deterministic offline stubs: scores
rise monotonically and a candidate is marked solved at depth three, so the run
terminates without an API key, network, or external search tool.

## Run

```sh
cd examples/langgraph/lats_lats
go run .
```
