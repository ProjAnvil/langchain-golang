# branching

Go port of `langgraph/examples/branching.ipynb`.

Three graph shapes are built and invoked:

1. parallel branches that converge on a node with a multi-parent join edge;
2. the same shape using a plain fan-in that may fire the join more than once;
3. a conditional branch that dispatches either `b, c` or `c, d`.

The `aggregate` channel uses `channels.AppendSliceReducer`, the Go form of
Python's `Annotated[list, operator.add]`.

## Run

```sh
cd examples/langgraph/branching
go run .
```

No API key or external service required.
