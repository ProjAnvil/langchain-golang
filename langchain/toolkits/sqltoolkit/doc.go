// Package sqltoolkit provides a read-only SQL agent toolkit over
// database/sql, the Go counterpart of langchain-community's
// SQLDatabaseToolkit.
//
// # Python parity
//
// Upstream lives in the frozen langchain-community package (there is no 1.x
// langchain equivalent; the toolkit does not exist in modern langchain-core),
// so this port targets:
//
//   - libs/community/langchain_community/agent_toolkits/sql/toolkit.py —
//     SQLDatabaseToolkit.get_tools returns [QuerySQLDatabaseTool,
//     InfoSQLDatabaseTool, ListSQLDatabaseTool, QuerySQLCheckerTool]; the
//     modern composition is `toolkit.get_tools()` + `create_agent`, not the
//     deprecated create_sql_agent. [Toolkit.Tools] mirrors that shape and
//     feeds agents.CreateAgent directly.
//   - libs/community/langchain_community/tools/sql_database/tool.py — tool
//     names (sql_db_query, sql_db_schema, sql_db_list_tables,
//     sql_db_query_checker), descriptions, and args schemas are copied
//     verbatim (with appended notes where Go behavior differs).
//   - libs/community/langchain_community/utilities/sql_database.py —
//     get_usable_table_names (tables ∪ views), get_table_info (DDL + "N rows
//     from T table:" sample block, sample_rows_in_table_info=3), and
//     run_no_throw ("Error: ..." results the model can self-correct on).
//
// # Go-side additions (deliberate divergences, see DIVERGENCES.md)
//
//   - Read-only guardrails on Query and CheckQuery: comments are stripped
//     before analysis, exactly one statement is allowed, the statement must
//     start with SELECT or WITH, and write-capable keywords (INTO, UPDATE,
//     DELETE) are rejected anywhere in the statement body. Upstream executes
//     whatever the model sends.
//   - QuerySQLCheckerTool validates via the database planner (EXPLAIN)
//     instead of an extra LLM call: deterministic, provider-independent, and
//     free.
//   - sql_db_schema accepts an empty table_names to describe every table.
//   - Query results are capped (default 30 rows) with a truncation note, and
//     schema output is length-capped (default 12000 runes), both to keep
//     agent prompts bounded. Upstream truncates query results at 300
//     characters but not schema output.
package sqltoolkit
