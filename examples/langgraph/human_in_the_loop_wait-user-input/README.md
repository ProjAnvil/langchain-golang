# human_in_the_loop_wait-user-input

Go port of `langgraph/examples/human_in_the_loop/wait-user-input.ipynb`.

Source notebook: `langgraph/examples/human_in_the_loop/wait-user-input.ipynb`.

What the Go port demonstrates:

1. A three-node graph (`step_1` -> `human_feedback` -> `step_3`) in which the
   `human_feedback` node calls `graphpkg.Interrupt(ctx, "Please provide
   feedback:")`. The first `InvokeWithOptions` on a thread pauses and returns
   the interrupt; a second invocation with
   `graphpkg.Options{ThreadID: ..., Resume: "go to step 3!"}` feeds the answer
   back into that exact `Interrupt` call and completes the run. `GetState`
   reads the final `user_feedback`.
2. A deterministic agent/tool loop where the agent emits an "AskHuman" style
   question, the `ask_human` node interrupts for a location, and the resumed
   run appends a tool message carrying the human's answer before returning to
   the agent for the final weather reply.

A `checkpoint.NewMemorySaver()` checkpointer makes both pauses durable and
resumable. No API key, network or service is required.

Deviation from Python: the Anthropic chat model, real `search` tool,
`ToolNode` and Pydantic `AskHuman` schema are replaced by deterministic nodes
and plain string messages; the graph topology and interrupt/resume semantics
are preserved. The notebook's PNG drawing is replaced by printed state only.

## Run

```sh
cd examples/langgraph/human_in_the_loop_wait-user-input
go run .
```
