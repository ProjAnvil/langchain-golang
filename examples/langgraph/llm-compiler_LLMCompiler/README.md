# llm-compiler_LLMCompiler

Go port of `langgraph/examples/llm-compiler/LLMCompiler.ipynb`.

The notebook pairs an LLM planner (which emits a DAG of tool tasks with `$N`
references to earlier results) with a scheduler that runs ready tasks in
parallel, and a joiner that returns the final answer or requests a replan.
The Go port keeps that structure using the SDK's control-flow primitives:

- a `StateGraph` with `planner`, `task`, and `join` nodes;
- conditional edges that dispatch each ready task with `types.Send` --
  dependent tasks wait for their `$N` observations -- and route to `join` when
  the DAG is drained (one task per superstep keeps the offline trace
  deterministic; the notebook's ThreadPoolExecutor parallelism is the main
  deviation);
- an `observations` channel with a merging reducer for concurrent writes;
- a joiner conditional edge that loops back to the planner on a replan signal
  or ends on a final AI message, matching the notebook's `should_continue`.

Planning, tools, and joining are deterministic offline stubs. Search returns
canned evidence and `calculate` uses a small in-process arithmetic parser; no
Python code is executed and no API key or network is required.

## Run

```sh
cd examples/langgraph/llm-compiler_LLMCompiler
go run .
```
