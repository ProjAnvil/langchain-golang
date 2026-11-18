# usaco_usaco

Go port of `examples/usaco/usaco.ipynb`.

## What the Go port demonstrates

Two graph shapes from the notebook are rebuilt with the Go `StateGraph` API:

1. basic: `START -> solver -> evaluate -> (END | solver)`
2. recall-augmented: `START -> draft -> retrieve -> solve -> evaluate
   -> (END | solve)`, compiled with a `checkpoint.NewMemorySaver()`

Both use the notebook's `writePython` tool-call protocol: the offline model
returns an AI message carrying a tool call, and `evaluate` reads the `code`
argument from it. `evaluate` appends a tool message on failure and sets
`status: success` when all tests pass; a conditional edge closes the retry
loop.

## Offline safe substitution

The notebook downloads the USACO dataset, calls ChatAnthropic, pulls prompts
from LangSmith Hub, uses a BM25 retriever, and runs generated Python in a
subprocess against test cases. This port replaces all of that:

- one fixed in-memory problem with three test cases
- a deterministic function-backed `ChatModel` double whose first submission
  is wrong and second is right
- a fixed two-item example corpus instead of BM25 retrieval
- **NO code execution and NO subprocess**: the evaluator reads a
  `# OUTPUT:` marker comment from the generated text and compares it to the
  expected output as a plain string, so no generated program is ever run and
  no shell command is ever invoked
- no dataset download, API key, network, or service is used

The notebook's later interrupt/resume and `update_state` walkthrough is not
reproduced; the port focuses on the solver/evaluate loop.

## Run

```sh
cd examples/langgraph/usaco_usaco
go run .
```
