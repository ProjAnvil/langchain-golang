package sqltoolkit

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// Tool names and descriptions mirror langchain-community's SQL database
// tools verbatim (libs/community/langchain_community/tools/sql_database/tool.py);
// each tool below cites its upstream class. The toolkit composition mirrors
// SQLDatabaseToolkit.get_tools
// (libs/community/langchain_community/agent_toolkits/sql/toolkit.py).
const (
	// QueryToolName is the sql_db_query tool (upstream QuerySQLDatabaseTool).
	QueryToolName = "sql_db_query"
	// SchemaToolName is the sql_db_schema tool (upstream InfoSQLDatabaseTool).
	SchemaToolName = "sql_db_schema"
	// ListTablesToolName is the sql_db_list_tables tool (upstream
	// ListSQLDatabaseTool).
	ListTablesToolName = "sql_db_list_tables"
	// QueryCheckerToolName is the sql_db_query_checker tool (upstream
	// QuerySQLCheckerTool).
	QueryCheckerToolName = "sql_db_query_checker"
)

const (
	queryToolDescription = `Execute a SQL query against the database and get back the result..
If the query is not correct, an error message will be returned.
If an error is returned, rewrite the query, check the query, and try again.
Only a single read-only SELECT or WITH statement is allowed; comments are stripped before analysis, and write statements, multiple statements, and SELECT ... INTO are rejected with an error message.`

	schemaToolDescription = `Get the schema and sample rows for the specified SQL tables.
Leave table_names empty to return the schema for every table in the database.`

	listTablesToolDescription = `Input is an empty string, output is a comma-separated list of tables in the database.`

	queryCheckerToolDescription = `Use this tool to double check if your query is correct before executing it.
Always use this tool before executing a query with sql_db_query!
The query is validated with the database planner (EXPLAIN); the same read-only SELECT/WITH guardrail as sql_db_query applies. Returns OK, or the planner error.`
)

// Dialect selects the introspection strategy and guardrail flavor.
type Dialect string

const (
	// SQLite introspects through sqlite_master (modernc.org/sqlite registers
	// the database/sql driver name "sqlite").
	SQLite Dialect = "sqlite"
	// Postgres introspects through information_schema (any database/sql
	// postgres driver, e.g. pgx stdlib's "pgx").
	Postgres Dialect = "postgres"
)

const (
	defaultSampleRows    = 3
	defaultMaxQueryRows  = 30
	defaultMaxSchemaSize = 12000
)

// Toolkit is the SQL agent toolkit: a read-only set of database tools over an
// existing *sql.DB, the Go counterpart of langchain-community's
// SQLDatabaseToolkit.
type Toolkit struct {
	db              *sql.DB
	dialect         Dialect
	sampleRows      int
	maxQueryRows    int
	maxSchemaLength int
}

// Option configures a Toolkit at construction time.
type Option func(*Toolkit)

// WithSampleRows sets how many sample rows sql_db_schema includes per table
// (default 3, matching upstream sample_rows_in_table_info; 0 disables the
// sample block).
func WithSampleRows(n int) Option {
	return func(t *Toolkit) { t.sampleRows = n }
}

// WithMaxQueryRows sets the row cap for sql_db_query results (default 30).
// Additional rows are dropped and a truncation note is appended.
func WithMaxQueryRows(n int) Option {
	return func(t *Toolkit) { t.maxQueryRows = n }
}

// WithMaxSchemaLength caps the total sql_db_schema output length in runes
// (default 12000) so a wide database cannot blow up the agent prompt.
func WithMaxSchemaLength(n int) Option {
	return func(t *Toolkit) { t.maxSchemaLength = n }
}

// NewSQLToolkit creates a toolkit over db. The dialect selects the
// introspection strategy; ctx is used for a fail-fast connectivity check.
func NewSQLToolkit(ctx context.Context, db *sql.DB, dialect Dialect, opts ...Option) (*Toolkit, error) {
	if db == nil {
		return nil, fmt.Errorf("sqltoolkit: db is required")
	}
	switch dialect {
	case SQLite, Postgres:
	default:
		return nil, fmt.Errorf("sqltoolkit: unsupported dialect %q (supported: %q, %q)",
			dialect, SQLite, Postgres)
	}
	toolkit := &Toolkit{
		db:              db,
		dialect:         dialect,
		sampleRows:      defaultSampleRows,
		maxQueryRows:    defaultMaxQueryRows,
		maxSchemaLength: defaultMaxSchemaSize,
	}
	for _, opt := range opts {
		opt(toolkit)
	}
	if toolkit.sampleRows < 0 {
		return nil, fmt.Errorf("sqltoolkit: sample rows must be >= 0, got %d", toolkit.sampleRows)
	}
	if toolkit.maxQueryRows < 1 {
		return nil, fmt.Errorf("sqltoolkit: max query rows must be >= 1, got %d", toolkit.maxQueryRows)
	}
	if toolkit.maxSchemaLength < 1 {
		return nil, fmt.Errorf("sqltoolkit: max schema length must be >= 1, got %d", toolkit.maxSchemaLength)
	}
	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("sqltoolkit: ping db: %w", err)
	}
	return toolkit, nil
}

// Dialect returns the dialect the toolkit was built with; agents can splice
// it into SQL generation prompts like upstream's toolkit.get_context.
func (t *Toolkit) Dialect() Dialect { return t.dialect }

// Tools returns the four toolkit tools in upstream SQLDatabaseToolkit order.
// Pass them straight to agents.CreateAgent:
//
//	toolkit, _ := sqltoolkit.NewSQLToolkit(ctx, db, sqltoolkit.SQLite)
//	toolList, _ := toolkit.Tools(ctx)
//	agent, _ := agents.CreateAgent(model, toolList)
func (t *Toolkit) Tools(ctx context.Context) ([]tools.Tool, error) {
	queryTool, err := t.newQueryTool()
	if err != nil {
		return nil, err
	}
	schemaTool, err := t.newSchemaTool()
	if err != nil {
		return nil, err
	}
	listTool, err := t.newListTablesTool()
	if err != nil {
		return nil, err
	}
	checkerTool, err := t.newCheckerTool()
	if err != nil {
		return nil, err
	}
	return []tools.Tool{queryTool, schemaTool, listTool, checkerTool}, nil
}

// newQueryTool builds sql_db_query (upstream QuerySQLDatabaseTool). Errors
// surface as "Error: ..." tool content instead of tool failures so the model
// can rewrite the query and retry (upstream run_no_throw semantics).
func (t *Toolkit) newQueryTool() (tools.Func, error) {
	return tools.NewFunc(QueryToolName, queryToolDescription,
		schema.Object(map[string]schema.Schema{
			"query": schema.String("A detailed and correct SQL query."),
		}, "query"),
		func(ctx context.Context, input map[string]any) (tools.Result, error) {
			query, err := stringArg(input, "query")
			if err != nil {
				return tools.Result{Content: "Error: " + err.Error()}, nil
			}
			content, err := t.Query(ctx, query)
			if err != nil {
				return tools.Result{Content: "Error: " + err.Error()}, nil
			}
			return tools.Result{Content: content}, nil
		})
}

// newSchemaTool builds sql_db_schema (upstream InfoSQLDatabaseTool).
func (t *Toolkit) newSchemaTool() (tools.Func, error) {
	return tools.NewFunc(SchemaToolName, schemaToolDescription,
		schema.Object(map[string]schema.Schema{
			"table_names": schema.String(
				"A comma-separated list of the table names for which to return the schema. " +
					`Example input: 'table1, table2, table3'`),
		}, "table_names"),
		func(ctx context.Context, input map[string]any) (tools.Result, error) {
			tableNames, err := stringArg(input, "table_names")
			if err != nil {
				return tools.Result{Content: "Error: " + err.Error()}, nil
			}
			var tables []string
			for _, name := range strings.Split(tableNames, ",") {
				if name != "" {
					tables = append(tables, name)
				}
			}
			content, err := t.SchemaInfo(ctx, tables...)
			if err != nil {
				return tools.Result{Content: "Error: " + err.Error()}, nil
			}
			return tools.Result{Content: content}, nil
		})
}

// newListTablesTool builds sql_db_list_tables (upstream ListSQLDatabaseTool).
func (t *Toolkit) newListTablesTool() (tools.Func, error) {
	return tools.NewFunc(ListTablesToolName, listTablesToolDescription,
		schema.Object(map[string]schema.Schema{
			"tool_input": schema.String("An empty string"),
		}),
		func(ctx context.Context, _ map[string]any) (tools.Result, error) {
			tables, err := t.ListTables(ctx)
			if err != nil {
				return tools.Result{Content: "Error: " + err.Error()}, nil
			}
			return tools.Result{Content: strings.Join(tables, ", ")}, nil
		})
}

// newCheckerTool builds sql_db_query_checker (upstream QuerySQLCheckerTool,
// EXPLAIN-based instead of LLM-based).
func (t *Toolkit) newCheckerTool() (tools.Func, error) {
	return tools.NewFunc(QueryCheckerToolName, queryCheckerToolDescription,
		schema.Object(map[string]schema.Schema{
			"query": schema.String("A detailed SQL query to be checked."),
		}, "query"),
		func(ctx context.Context, input map[string]any) (tools.Result, error) {
			query, err := stringArg(input, "query")
			if err != nil {
				return tools.Result{Content: "Error: " + err.Error()}, nil
			}
			verdict, err := t.CheckQuery(ctx, query)
			if err != nil {
				return tools.Result{Content: "Error: " + err.Error()}, nil
			}
			return tools.Result{Content: verdict}, nil
		})
}

// stringArg extracts a string argument from a tool input map, tolerating a
// missing key (treated as the empty string) for optional inputs such as
// sql_db_schema's table_names.
func stringArg(input map[string]any, key string) (string, error) {
	value, ok := input[key]
	if !ok || value == nil {
		return "", nil
	}
	text, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("%s must be a string", key)
	}
	return text, nil
}
