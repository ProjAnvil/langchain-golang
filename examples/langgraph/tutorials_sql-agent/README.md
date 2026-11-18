# tutorials_sql-agent

Go port of `examples/tutorials/sql-agent.ipynb`.

## What the Go port demonstrates

The notebook's hand-written SQL workflow is rebuilt as a Go `StateGraph`:

```
START -> list_tables -> call_get_schema -> get_schema -> generate_query
             -> (check_query -> run_query -> generate_query)* -> END
```

- `get_schema` and `run_query` use `langgraph/prebuilt.ToolNode` wrapping
  `langchain/tools.ToolNode`, exactly like the notebook's `ToolNode([...])`
  nodes
- `list_tables` builds a static tool call and invokes it through the same
  tool node, matching the notebook's predetermined-call example
- `generate_query` uses a `should_continue` conditional edge: a pending tool
  call routes to `check_query`, a plain answer routes to `END`
- a `checkpoint.NewMemorySaver()` is installed with a `ThreadID`
- `messages` uses `channels.MessagesReducer`

## Offline safe substitution

The notebook downloads a Chinook SQLite database, instantiates
`SQLDatabaseToolkit`/`SQLDatabase`, and executes generated SQL. This port:

- keeps a tiny in-memory schema description instead of a database
- implements the four tools (`sql_db_list_tables`, `sql_db_schema`,
  `sql_db_query`, query checking) as `core/tools` string transformations
- **never opens a database and never runs SQL**: queries are string-matched
  against canned result rows, and DML/DDL keywords are rejected
- uses a deterministic function-backed `ChatModel` double; no API key,
  network, or service is used

Note: the SDK's `langchain/toolkits/sqltoolkit` was consulted as the API
reference, but its real `NewSQLToolkit` requires a live `*sql.DB` and is
intentionally not used here.

## Run

```sh
cd examples/langgraph/tutorials_sql-agent
go run .
```
