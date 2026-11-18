# visualization

Go port of `langgraph/examples/visualization.ipynb`.

Source notebook: `langgraph/examples/visualization.ipynb`.

What the Go port demonstrates: building a `StateGraph` and exporting its
structure with `CompiledGraph.DrawMermaid()`, the Go equivalent of
`app.get_graph().draw_mermaid()`. It prints:

1. the default Mermaid flowchart (nodes, static edges, and a conditional edge
   registered with `AddConditionalEdgesWithPathMap`);
2. the same graph with Python-style drawing options (styles off, curve style,
   frontmatter theme, node colors) passed through to the Mermaid renderer;
3. a parent graph with a subgraph drawn first as a single node and then
   expanded with `GetGraph(graphpkg.WithXrayDepth(-1)).DrawMermaid()`, the Go
   equivalent of the notebook's `xray` PNG drawings.

Explicit offline note: only Mermaid **text** is emitted. The notebook's PNG
paths (`draw_mermaid_png` via the Mermaid API/pyppeteer and `draw_png` via
pygraphviz) are intentionally not reproduced, so there is no network call,
browser, PNG generation or API key. The random fractal tree is replaced by a
fixed graph so the output is deterministic.

## Run

```sh
cd examples/langgraph/visualization
go run .
```
