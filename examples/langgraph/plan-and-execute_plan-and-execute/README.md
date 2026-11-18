# plan-and-execute_plan-and-execute

Go port of `langgraph/examples/plan-and-execute/plan-and-execute.ipynb`.

The notebook defines a three-node loop: a planner produces a step list, an
agent executes the first step, and a replanner either extends the plan or
returns the final response. The Go port mirrors it with `StateGraph`:

- `planner -> agent -> replan` edges plus a conditional edge on `replan` that
  routes back to `agent` or to `__end__` (the notebook's `should_end`);
- a `past_steps` channel registered with `channels.AppendSliceReducer`, the Go
  form of `Annotated[List[Tuple], operator.add]`;
- deterministic offline planner/replanner nodes and a canned executor step in
  place of the search-backed ReAct agent.

No API key or network is needed; all decisions and observations are fixed.
The `LLMCompiler` API is not used.

## Run

```sh
cd examples/langgraph/plan-and-execute_plan-and-execute
go run .
```
