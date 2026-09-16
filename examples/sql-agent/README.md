# sql-agent

A SQL analyst agent: in-memory SQLite seeded with products/orders, `sqltoolkit`'s four read-only tools (`sql_db_query`, `sql_db_schema`, `sql_db_list_tables`, `sql_db_query_checker`), and a `create_agent`-style loop. The toolkit's guardrails reject anything that is not a single read-only SELECT.

## Run

```sh
go run ./examples/sql-agent
```

Optional env: `OLLAMA_BASE_URL` (e.g. `http://localhost:11434`) replaces the offline scripted model with a local Ollama model (`llama3.1`).
