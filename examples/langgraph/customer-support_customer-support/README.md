# customer-support_customer-support

Go port of the advanced (part 4) graph in
`langgraph/examples/customer-support/customer-support.ipynb`.

The Go graph keeps the notebook's multi-assistant architecture:

- a `primary_assistant` routes to four specialized assistants (flight, car
  rental, hotel, excursion) or runs tools directly;
- each skill has an entry node that pushes a dialog-state value, an assistant,
  safe tool and sensitive tool nodes, and a shared `leave_skill` node that pops
  the stack and returns control to the primary assistant;
- `dialog_state` uses a custom reducer implementing the notebook's
  push/pop `update_dialog_stack`;
- `WithInterruptBefore` gates every sensitive tool node, and a human approval
  is simulated by resuming the thread with nil input.

Two scenarios are run: a flight update that pauses for approval and a hotel
booking that completes without a sensitive tool.

## Offline stub

`ChatAnthropic`, the SQLite flight database, embeddings, policy retrieval,
Tavily search, and all booking APIs are replaced with deterministic in-memory
tools and a scripted model. No API key, network access, database, or service
is used.

## Run

```sh
cd examples/langgraph/customer-support_customer-support && go run .
```
