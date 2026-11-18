# persistence_postgres

Go port of `langgraph/examples/persistence_postgres.ipynb`.

Source notebook: `langgraph/examples/persistence_postgres.ipynb`.

What the Go port demonstrates: the notebook's checkpoint lifecycle around
`PostgresSaver` / `AsyncPostgresSaver` -- one-time `setup()`, several
`thread_id` invocations through a `create_react_agent`, and checkpoint reads
via `get(config)`, `get_tuple(config)` and `list(config)`. Here the agent is a
single deterministic weather node, and `CompiledGraph.GetState` /
`GetStateHistory` provide the Go equivalents of those checkpoint reads.

Explicit offline note: **no PostgreSQL service is contacted.** `PostgresSaver`,
its async variant, `ConnectionPool`, `DB_URI` and the `setup()` migration step
are replaced by a deterministic `checkpoint.NewMemorySaver()` stub. The
`ChatOpenAI` model and all network calls are intentionally not reproduced;
all checkpoints live only in process memory.

## Run

```sh
cd examples/langgraph/persistence_postgres
go run .
```
