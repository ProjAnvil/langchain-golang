# rewoo_rewoo

Go port of `langgraph/examples/rewoo/rewoo.ipynb`.

ReWOO (Reasoning WithOut Observation) plans all steps up front, then executes
them while binding evidence variables (`#E1`, `#E2`, ...). The Go port keeps
the notebook's graph `plan -> tool -> (loop) -> solve -> END`:

- a `plan` node that parses a canned plan with the same regex used in the
  notebook, producing `(description, #E name, tool, input)` tuples;
- a `tool` node executed one step at a time by a conditional edge, with a
  merging `results` channel holding the evidence variables;
- a `solve` node that substitutes the evidence into the plan and returns the
  final answer.

The LLM planner and solver are deterministic offline stubs. `Google` and `LLM`
tools are safe in-memory string lookups; there is no network access and no
code execution.

## Run

```sh
cd examples/langgraph/rewoo_rewoo
go run .
```
