# node-retries

Go port of `langgraph/examples/node-retries.ipynb`.

The notebook demonstrates per-node retry policies: a database-query node
retries only on `sqlite3.OperationalError`, and a model node uses a larger
`max_attempts` budget.

## What this port demonstrates

- `AddNodeWithPolicies` installing a `graphpkg.RetryPolicy` on individual nodes
  (the Go form of Python's `add_node(..., retry=RetryPolicy(...))`).
- A custom `RetryOn` predicate that retries only a simulated operational error
  (the analogue of Python's `retry_on=sqlite3.OperationalError`).
- A second node with `MaxAttempts: 5` (the analogue of
  `RetryPolicy(max_attempts=5)`).
- The topology `START -> model -> query_database -> END`, with a `messages`
  channel using `channels.MessagesReducer` (Python's
  `Annotated[Sequence[BaseMessage], operator.add]`).
- Printed attempt counters showing the retry actually happened.

## Offline stub

The notebook's `ChatAnthropic` model and in-memory SQLite `SQLDatabase` are
replaced with deterministic counters: `query_database` fails twice with a
simulated operational error before succeeding, and `model` succeeds on its
first attempt. Backoff is set to 10ms with no jitter so the demo stays fast.
No API key, database, network, or external service is used.

## Run

```sh
cd examples/langgraph/node-retries
go run .
```
