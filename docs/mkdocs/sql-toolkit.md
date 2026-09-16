# SQL toolkit

**Languages:** English | [简体中文](zh-CN/sql-toolkit.zh-CN.md)

`langchain/toolkits/sqltoolkit` adapts any `database/sql` connection into
four agent tools — the Go counterpart of langchain-community's
`SQLDatabaseToolkit`, composed the modern way (`get_tools()` +
`CreateAgent`, not the deprecated `create_sql_agent`). Runnable example:
[`examples/sql-agent`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/sql-agent).

## Installation

```bash
go get github.com/projanvil/langchain-golang
```

Bring your own driver: `_ "modernc.org/sqlite"` (pure Go) or
`_ "github.com/jackc/pgx/v5/stdlib"` for PostgreSQL.

## Construct the toolkit

```go
import "github.com/projanvil/langchain-golang/langchain/toolkits/sqltoolkit"

db, _ := sql.Open("sqlite", ":memory:") // any *sql.DB
toolkit, err := sqltoolkit.NewSQLToolkit(ctx, db, sqltoolkit.SQLite)
//                                     for Postgres: sqltoolkit.Postgres

toolList, err := toolkit.Tools(ctx)
```

The four tools mirror upstream names and descriptions verbatim:

| Tool | Purpose |
|---|---|
| `sql_db_query` | Execute one **read-only** SELECT and return rows (capped, default 30) |
| `sql_db_schema` | DDL + sample rows for the named tables (or all tables) |
| `sql_db_list_tables` | Tables and views available in the database |
| `sql_db_query_checker` | Validate a query — via the database planner (EXPLAIN), not an extra LLM call |

## An agent over the tools

```go
agent, err := agents.CreateAgent(model, toolList,
    agents.WithAgentSystemPrompt(
        "Answer database questions with the sql_db_* tools. Always inspect "+
            "the schema before querying."))
out, err := agent.Invoke(ctx, []messages.Message{
    messages.Human("Which hardware product is the most expensive?"),
})
```

## Read-only guardrails (deliberate divergence)

Upstream executes whatever SQL the model sends. This port rejects anything
that is not a single read-only statement — before it reaches the database:

- comments are stripped, then exactly **one** statement is allowed (no
  semicolon chaining);
- the statement must start with `SELECT` or `WITH`;
- write-capable keywords (`INTO`, `UPDATE`, `DELETE`) are rejected anywhere
  in the body.

Malformed queries return an `"Error: ..."` tool result (upstream
`run_no_throw` semantics) so the model can self-correct on the next turn,
while genuinely invalid SQL is caught by the checker's EXPLAIN pass.

## Tuning

```go
toolkit, _ := sqltoolkit.NewSQLToolkit(ctx, db, sqltoolkit.Postgres,
    sqltoolkit.WithSampleRows(3),        // sample rows in schema output
    sqltoolkit.WithMaxQueryRows(30),     // query result row cap
    sqltoolkit.WithMaxSchemaLength(12000), // rune cap on schema output
)
```

The caps keep agent prompts bounded: query results carry a truncation note
when capped, and schema output is length-limited the same way.

## Switching points

- **Dialect**: `sqltoolkit.SQLite` (via `sqlite_master`) or
  `sqltoolkit.Postgres` (via `information_schema`) — the introspection
  queries switch automatically.
- **Model**: any `language.ChatModel`; the example runs offline with a
  scripted double and switches to a local Ollama model via `OLLAMA_BASE_URL`.
- **Write access**: not available by design — point a writable helper at a
  restricted database role instead of loosening the guardrails.
