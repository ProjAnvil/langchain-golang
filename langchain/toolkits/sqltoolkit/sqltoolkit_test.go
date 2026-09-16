package sqltoolkit

import (
	"context"
	"database/sql"
	"fmt"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	_ "modernc.org/sqlite" // database/sql driver name `sqlite` (pure Go, no cgo)

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
)

var testDBCounter atomic.Int64

// openTestDB opens an isolated in-memory sqlite database (shared-cache DSN so
// every pooled connection sees the same database) seeded with the two-table
// fixture used across the suite.
func openTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", fmt.Sprintf(
		"file:sqltoolkit_test_%d?mode=memory&cache=shared", testDBCounter.Add(1)))
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	statements := []string{
		`CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)`,
		`CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), amount_cents INTEGER NOT NULL)`,
		`INSERT INTO users (id, name, email) VALUES
			(1, 'alice', 'alice@example.com'),
			(2, 'bob', 'bob@example.com'),
			(3, 'carol', NULL),
			(4, 'dave', NULL),
			(5, 'erin', NULL)`,
		`INSERT INTO orders (id, user_id, amount_cents) VALUES
			(1, 1, 1000), (2, 1, 2500), (3, 2, 500),
			(4, 3, 750), (5, 2, 1200), (6, 4, 300), (7, 5, 900)`,
	}
	for _, statement := range statements {
		if _, err := db.Exec(statement); err != nil {
			t.Fatalf("seed fixture: %v", err)
		}
	}
	return db
}

// newTestToolkit returns a toolkit over a fresh fixture database.
func newTestToolkit(t *testing.T, opts ...Option) *Toolkit {
	t.Helper()
	toolkit, err := NewSQLToolkit(t.Context(), openTestDB(t), SQLite, opts...)
	if err != nil {
		t.Fatalf("NewSQLToolkit: %v", err)
	}
	return toolkit
}

// toolNamed fetches a tool from Tools() by exact name.
func toolNamed(t *testing.T, toolkit *Toolkit, name string) tools.Tool {
	t.Helper()
	toolList, err := toolkit.Tools(t.Context())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	for _, tool := range toolList {
		if tool.Name() == name {
			return tool
		}
	}
	t.Fatalf("tool %s not found", name)
	return nil
}

func TestToolkitToolsMatchUpstreamShape(t *testing.T) {
	toolkit := newTestToolkit(t)
	toolList, err := toolkit.Tools(t.Context())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	if len(toolList) != 4 {
		t.Fatalf("expected 4 tools, got %d", len(toolList))
	}
	gotNames := make([]string, len(toolList))
	for i, tool := range toolList {
		gotNames[i] = tool.Name()
		if tool.Description() == "" {
			t.Errorf("tool %s: empty description", tool.Name())
		}
		if tool.ArgsSchema() == nil {
			t.Errorf("tool %s: nil args schema", tool.Name())
		}
	}
	// Order and names mirror langchain-community SQLDatabaseToolkit.get_tools
	// (agent_toolkits/sql/toolkit.py) and tools/sql_database/tool.py.
	wantNames := []string{
		QueryToolName,        // QuerySQLDatabaseTool
		SchemaToolName,       // InfoSQLDatabaseTool
		ListTablesToolName,   // ListSQLDatabaseTool
		QueryCheckerToolName, // QuerySQLCheckerTool
	}
	if !slices.Equal(gotNames, wantNames) {
		t.Fatalf("tool names = %v, want %v", gotNames, wantNames)
	}

	for name, wantSubstring := range map[string]string{
		QueryToolName:        "Execute a SQL query against the database",
		SchemaToolName:       "Get the schema and sample rows",
		ListTablesToolName:   "comma-separated list of tables",
		QueryCheckerToolName: "before executing it",
	} {
		if tool := toolNamed(t, toolkit, name); !strings.Contains(tool.Description(), wantSubstring) {
			t.Errorf("tool %s description missing %q", name, wantSubstring)
		}
	}
}

func TestQueryToolReadOnlyGuardrails(t *testing.T) {
	toolkit := newTestToolkit(t)
	tool := toolNamed(t, toolkit, QueryToolName)

	cases := []struct {
		name        string
		query       string
		wantAllowed bool
	}{
		// Read-only statements pass.
		{name: "plain select", query: "SELECT * FROM users", wantAllowed: true},
		{name: "lowercase select", query: "select id from users where id = 1", wantAllowed: true},
		{name: "cte with", query: "WITH top AS (SELECT user_id FROM orders) SELECT COUNT(*) FROM top", wantAllowed: true},
		{name: "trailing semicolon and padding", query: "   SELECT 1;  ", wantAllowed: true},
		{name: "semicolon in string literal", query: "SELECT name FROM users WHERE name = 'a;b'", wantAllowed: true},
		{name: "write keyword inside string literal", query: "SELECT 'DELETE' AS word FROM users", wantAllowed: true},
		{name: "quoted identifier", query: `SELECT "name" FROM users`, wantAllowed: true},
		{name: "backtick identifier", query: "SELECT `name` FROM users", wantAllowed: true},
		{name: "leading block comment is stripped", query: "/* harmless */ SELECT COUNT(*) FROM users", wantAllowed: true},
		{name: "trailing line comment hides nothing executable", query: "SELECT 1 -- ; DELETE FROM users", wantAllowed: true},
		{name: "commented semicolon and statement", query: "SELECT 1 /* ; DELETE FROM users */", wantAllowed: true},
		{name: "redundant trailing semicolons", query: "SELECT 1;;", wantAllowed: true},

		// Non-SELECT leading statements are rejected.
		{name: "update", query: "UPDATE users SET name = 'x'", wantAllowed: false},
		{name: "delete", query: "DELETE FROM users", wantAllowed: false},
		{name: "insert", query: "INSERT INTO users (id, name) VALUES (9, 'x')", wantAllowed: false},
		{name: "drop", query: "DROP TABLE users", wantAllowed: false},
		{name: "create", query: "CREATE TABLE t (x INT)", wantAllowed: false},
		{name: "alter", query: "ALTER TABLE users ADD COLUMN z INT", wantAllowed: false},
		{name: "pragma", query: "PRAGMA table_info(users)", wantAllowed: false},
		{name: "vacuum", query: "VACUUM", wantAllowed: false},
		{name: "attach", query: "ATTACH DATABASE ':memory:' AS evil", wantAllowed: false},
		{name: "explain", query: "EXPLAIN SELECT 1", wantAllowed: false},
		{name: "empty", query: "", wantAllowed: false},
		{name: "comment only", query: "-- nothing", wantAllowed: false},

		// Multiple statements are rejected even when both are read-only.
		{name: "two selects", query: "SELECT 1; SELECT 2", wantAllowed: false},
		{name: "select then drop", query: "SELECT 1; DROP TABLE users", wantAllowed: false},
		{name: "second statement after newline", query: "SELECT 1\nDELETE FROM users", wantAllowed: false},

		// Writes hidden inside a SELECT/WITH statement are rejected.
		{name: "select into", query: "SELECT * INTO archive FROM users", wantAllowed: false},
		{name: "data modifying cte", query: "WITH del AS (DELETE FROM users RETURNING *) SELECT * FROM del", wantAllowed: false},
		{name: "update cte", query: "WITH u AS (UPDATE users SET name = 'x' RETURNING *) SELECT * FROM u", wantAllowed: false},
		{name: "for update lock", query: "SELECT * FROM users FOR UPDATE", wantAllowed: false},

		// Comments must not bypass the guards above.
		{name: "line comment disguises drop", query: "-- SELECT 1\nDROP TABLE users", wantAllowed: false},
		{name: "block comment before drop", query: "/* SELECT */ DROP TABLE users", wantAllowed: false},
		{name: "second statement hidden after comment", query: "SELECT 1; /* x */ DELETE FROM users", wantAllowed: false},
		{name: "unterminated block comment", query: "SELECT 1 /* oops", wantAllowed: false},
		{name: "unterminated string literal", query: "SELECT 'oops", wantAllowed: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := tool.Invoke(t.Context(), map[string]any{"query": tc.query})
			if err != nil {
				t.Fatalf("invoke: %v", err)
			}
			if tc.wantAllowed {
				if result.Content == "" || strings.HasPrefix(result.Content, "Error:") {
					t.Errorf("query %q unexpectedly rejected: %s", tc.query, result.Content)
				}
			} else {
				if !strings.HasPrefix(result.Content, "Error:") {
					t.Errorf("query %q unexpectedly allowed: %s", tc.query, result.Content)
				}
				if !strings.Contains(result.Content, "read-only guardrail") {
					t.Errorf("rejection for %q lacks guardrail explanation: %s", tc.query, result.Content)
				}
			}
		})
	}

	// The whole matrix must be side-effect free.
	var count int
	if err := toolkit.db.QueryRowContext(t.Context(), "SELECT COUNT(*) FROM users").Scan(&count); err != nil {
		t.Fatalf("count users: %v", err)
	}
	if count != 5 {
		t.Fatalf("users table mutated by guardrail matrix: got %d rows, want 5", count)
	}
}

func TestQueryToolResultFormat(t *testing.T) {
	toolkit := newTestToolkit(t)
	tool := toolNamed(t, toolkit, QueryToolName)

	t.Run("header and textualized rows", func(t *testing.T) {
		result, err := tool.Invoke(t.Context(), map[string]any{
			"query": "SELECT id, name, email FROM users ORDER BY id",
		})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		want := strings.Join([]string{
			"id\tname\temail",
			"1\talice\talice@example.com",
			"2\tbob\tbob@example.com",
			"3\tcarol\tNULL",
			"4\tdave\tNULL",
			"5\terin\tNULL",
		}, "\n")
		if result.Content != want {
			t.Fatalf("content =\n%q\nwant\n%q", result.Content, want)
		}
	})

	t.Run("empty result", func(t *testing.T) {
		result, err := tool.Invoke(t.Context(), map[string]any{
			"query": "SELECT * FROM users WHERE id = 999",
		})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if result.Content != "no rows returned" {
			t.Fatalf("content = %q, want %q", result.Content, "no rows returned")
		}
	})
}

func TestQueryToolRowLimit(t *testing.T) {
	setup := func(t *testing.T, opts ...Option) tools.Tool {
		t.Helper()
		toolkit := newTestToolkit(t, opts...)
		for i := 1; i <= 35; i++ {
			if _, err := toolkit.db.ExecContext(t.Context(),
				"INSERT INTO orders (id, user_id, amount_cents) VALUES (?, 1, ?)",
				100+i, i*10,
			); err != nil {
				t.Fatalf("seed orders: %v", err)
			}
		}
		return toolNamed(t, toolkit, QueryToolName)
	}

	t.Run("default limit 30", func(t *testing.T) {
		tool := setup(t)
		result, err := tool.Invoke(t.Context(), map[string]any{
			"query": "SELECT id FROM orders ORDER BY id",
		})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		lines := strings.Split(result.Content, "\n")
		if len(lines) != 32 { // header + 30 rows + truncation note
			t.Fatalf("got %d lines, want 32 (header, 30 rows, truncation note):\n%s", len(lines), result.Content)
		}
		if !strings.Contains(result.Content, "(result truncated at 30 rows)") {
			t.Fatalf("missing truncation note:\n%s", result.Content)
		}
	})

	t.Run("configured limit", func(t *testing.T) {
		tool := setup(t, WithMaxQueryRows(3))
		result, err := tool.Invoke(t.Context(), map[string]any{
			"query": "SELECT id FROM orders ORDER BY id",
		})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		lines := strings.Split(result.Content, "\n")
		if len(lines) != 5 { // header + 3 rows + truncation note
			t.Fatalf("got %d lines, want 5:\n%s", len(lines), result.Content)
		}
		if !strings.Contains(result.Content, "(result truncated at 3 rows)") {
			t.Fatalf("missing truncation note:\n%s", result.Content)
		}
	})
}

func TestQueryToolReturnsSQLErrorsAsContent(t *testing.T) {
	toolkit := newTestToolkit(t)
	tool := toolNamed(t, toolkit, QueryToolName)
	result, err := tool.Invoke(t.Context(), map[string]any{
		"query": "SELECT * FROM missing_table",
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !strings.HasPrefix(result.Content, "Error:") {
		t.Fatalf("content = %q, want an Error: prefix so the model can self-correct", result.Content)
	}
	if !strings.Contains(result.Content, "missing_table") {
		t.Fatalf("content does not mention the missing table: %q", result.Content)
	}
}

func TestListTablesTool(t *testing.T) {
	toolkit := newTestToolkit(t)
	tool := toolNamed(t, toolkit, ListTablesToolName)
	result, err := tool.Invoke(t.Context(), map[string]any{})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if result.Content != "orders, users" {
		t.Fatalf("content = %q, want %q", result.Content, "orders, users")
	}
}

func TestSchemaTool(t *testing.T) {
	toolkit := newTestToolkit(t)
	tool := toolNamed(t, toolkit, SchemaToolName)

	t.Run("single table with ddl and sample rows", func(t *testing.T) {
		result, err := tool.Invoke(t.Context(), map[string]any{"table_names": "users"})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		for _, want := range []string{
			"CREATE TABLE users",
			"3 rows from users table:",
			"id\tname\temail",
			"1\talice\talice@example.com",
			"NOT NULL",
		} {
			if !strings.Contains(result.Content, want) {
				t.Errorf("schema output missing %q:\n%s", want, result.Content)
			}
		}
		// Sample rows default to 3, so the 4th fixture row must be absent.
		if strings.Contains(result.Content, "dave") {
			t.Errorf("schema output shows more than 3 sample rows:\n%s", result.Content)
		}
	})

	t.Run("whitespace and multi-table input", func(t *testing.T) {
		result, err := tool.Invoke(t.Context(), map[string]any{"table_names": " users , orders "})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if !strings.Contains(result.Content, "CREATE TABLE users") ||
			!strings.Contains(result.Content, "CREATE TABLE orders") {
			t.Fatalf("expected both tables in output:\n%s", result.Content)
		}
	})

	t.Run("empty input lists every table", func(t *testing.T) {
		result, err := tool.Invoke(t.Context(), map[string]any{})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if !strings.Contains(result.Content, "CREATE TABLE users") ||
			!strings.Contains(result.Content, "CREATE TABLE orders") {
			t.Fatalf("expected both tables in output:\n%s", result.Content)
		}
	})

	t.Run("unknown table errors as content", func(t *testing.T) {
		result, err := tool.Invoke(t.Context(), map[string]any{"table_names": "nope"})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if !strings.HasPrefix(result.Content, "Error:") ||
			!strings.Contains(result.Content, "nope") ||
			!strings.Contains(result.Content, "not found in database") {
			t.Fatalf("content = %q, want an Error mentioning nope not found in database", result.Content)
		}
	})

	t.Run("sample rows disabled", func(t *testing.T) {
		disabled := newTestToolkit(t, WithSampleRows(0))
		tool := toolNamed(t, disabled, SchemaToolName)
		result, err := tool.Invoke(t.Context(), map[string]any{"table_names": "users"})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if !strings.Contains(result.Content, "CREATE TABLE users") {
			t.Fatalf("expected CREATE TABLE ddl:\n%s", result.Content)
		}
		if strings.Contains(result.Content, "rows from users table") {
			t.Fatalf("sample rows should be disabled:\n%s", result.Content)
		}
	})

	t.Run("schema length cap", func(t *testing.T) {
		capped := newTestToolkit(t, WithMaxSchemaLength(120))
		tool := toolNamed(t, capped, SchemaToolName)
		result, err := tool.Invoke(t.Context(), map[string]any{})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if !strings.Contains(result.Content, "schema info truncated") {
			t.Fatalf("expected truncation marker:\n%s", result.Content)
		}
		if len([]rune(result.Content)) > 120+100 { // cap + marker slack
			t.Fatalf("schema output far exceeds the cap: %d runes", len([]rune(result.Content)))
		}
	})
}

func TestQueryCheckerTool(t *testing.T) {
	toolkit := newTestToolkit(t)
	tool := toolNamed(t, toolkit, QueryCheckerToolName)

	t.Run("valid query returns OK", func(t *testing.T) {
		result, err := tool.Invoke(t.Context(), map[string]any{
			"query": "SELECT * FROM users WHERE id = 1",
		})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if result.Content != "OK" {
			t.Fatalf("content = %q, want %q", result.Content, "OK")
		}
	})

	t.Run("valid cte returns OK", func(t *testing.T) {
		result, err := tool.Invoke(t.Context(), map[string]any{
			"query": "WITH u AS (SELECT id FROM users) SELECT COUNT(*) FROM u",
		})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if result.Content != "OK" {
			t.Fatalf("content = %q, want %q", result.Content, "OK")
		}
	})

	t.Run("syntax error returns planner error", func(t *testing.T) {
		result, err := tool.Invoke(t.Context(), map[string]any{
			"query": "SELEC * FROM users",
		})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if !strings.HasPrefix(result.Content, "Error:") {
			t.Fatalf("content = %q, want an Error: prefix", result.Content)
		}
	})

	t.Run("unknown column returns planner error", func(t *testing.T) {
		result, err := tool.Invoke(t.Context(), map[string]any{
			"query": "SELECT nope FROM users",
		})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if !strings.HasPrefix(result.Content, "Error:") {
			t.Fatalf("content = %q, want an Error: prefix", result.Content)
		}
	})

	t.Run("write statements are guarded too", func(t *testing.T) {
		for _, query := range []string{
			"DROP TABLE users",
			"SELECT 1; DELETE FROM users",
			"SELECT * INTO archive FROM users",
		} {
			result, err := tool.Invoke(t.Context(), map[string]any{"query": query})
			if err != nil {
				t.Fatalf("invoke %q: %v", query, err)
			}
			if !strings.HasPrefix(result.Content, "Error:") ||
				!strings.Contains(result.Content, "read-only guardrail") {
				t.Fatalf("checker blessed a guarded query %q: %q", query, result.Content)
			}
		}
	})
}

func TestNewSQLToolkitValidation(t *testing.T) {
	t.Run("nil db", func(t *testing.T) {
		if _, err := NewSQLToolkit(t.Context(), nil, SQLite); err == nil {
			t.Fatal("expected error for nil db")
		}
	})
	t.Run("unsupported dialect", func(t *testing.T) {
		db := openTestDB(t)
		if _, err := NewSQLToolkit(t.Context(), db, Dialect("mysql")); err == nil {
			t.Fatal("expected error for unsupported dialect")
		}
		if _, err := NewSQLToolkit(t.Context(), db, Dialect("")); err == nil {
			t.Fatal("expected error for empty dialect")
		}
	})
	t.Run("negative options rejected", func(t *testing.T) {
		db := openTestDB(t)
		if _, err := NewSQLToolkit(t.Context(), db, SQLite, WithSampleRows(-1)); err == nil {
			t.Fatal("expected error for negative sample rows")
		}
	})
	t.Run("dialect accessor", func(t *testing.T) {
		toolkit := newTestToolkit(t)
		if got := toolkit.Dialect(); got != SQLite {
			t.Fatalf("Dialect() = %q, want %q", got, SQLite)
		}
	})
}

func TestDirectMethods(t *testing.T) {
	toolkit := newTestToolkit(t)

	tables, err := toolkit.ListTables(t.Context())
	if err != nil {
		t.Fatalf("ListTables: %v", err)
	}
	if !slices.Equal(tables, []string{"orders", "users"}) {
		t.Fatalf("ListTables = %v", tables)
	}

	info, err := toolkit.SchemaInfo(t.Context(), "users")
	if err != nil {
		t.Fatalf("SchemaInfo: %v", err)
	}
	if !strings.Contains(info, "CREATE TABLE users") {
		t.Fatalf("SchemaInfo missing DDL:\n%s", info)
	}

	rows, err := toolkit.Query(t.Context(), "SELECT COUNT(*) FROM users")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if !strings.Contains(rows, "5") {
		t.Fatalf("Query content = %q", rows)
	}

	if _, err := toolkit.Query(t.Context(), "DELETE FROM users"); err == nil {
		t.Fatal("Query should reject writes with an error for direct callers")
	}

	verdict, err := toolkit.CheckQuery(t.Context(), "SELECT 1")
	if err != nil {
		t.Fatalf("CheckQuery: %v", err)
	}
	if verdict != "OK" {
		t.Fatalf("CheckQuery = %q", verdict)
	}
}

// scriptedModel is a deterministic language.ChatModel that advances its
// response sequence across CreateAgent's per-iteration tool re-binding
// (FakeChatModel.BindTools returns a copy, so a multi-response sequence would
// restart on every bind and loop forever).
type scriptedModel struct {
	responses []messages.Message
	idx       int
}

func (m *scriptedModel) Invoke(
	_ context.Context, _ []messages.Message, _ ...runnables.Option,
) (messages.Message, error) {
	if m.idx >= len(m.responses) {
		return messages.Message{}, fmt.Errorf("scriptedModel: no more responses (call %d)", m.idx+1)
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp, nil
}

func (m *scriptedModel) Batch(
	ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option,
) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		resp, err := m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
		out[i] = resp
	}
	return out, nil
}

func (m *scriptedModel) Stream(
	_ context.Context, _ []messages.Message, _ ...runnables.Option,
) (runnables.Stream[messages.Message], error) {
	return nil, fmt.Errorf("scriptedModel: streaming not supported")
}

func (m *scriptedModel) InputSchema() schema.Schema { return schema.Object(nil) }

func (m *scriptedModel) OutputSchema() schema.Schema { return schema.Object(nil) }

func (m *scriptedModel) BindTools(_ []tools.Tool) (language.ChatModel, error) { return m, nil }

func (m *scriptedModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
}

// TestAgentComposition proves the toolkit plugs into the modern
// agents.CreateAgent flow (the Go equivalent of the upstream recommended
// `toolkit.get_tools()` + `create_agent` usage, not the deprecated
// create_sql_agent).
func TestAgentComposition(t *testing.T) {
	toolkit := newTestToolkit(t)
	toolList, err := toolkit.Tools(t.Context())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}

	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: ListTablesToolName, Args: map[string]any{}},
			},
		},
		messages.AI("found the tables"),
	}}
	agent, err := agents.CreateAgent(model, toolList)
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	out, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("which tables exist?")})
	if err != nil {
		t.Fatalf("agent invoke: %v", err)
	}

	var toolResult string
	for _, msg := range out {
		if msg.Role == messages.RoleTool {
			toolResult = msg.Content
		}
	}
	if toolResult != "orders, users" {
		t.Fatalf("agent tool result = %q, want %q", toolResult, "orders, users")
	}
	if last := out[len(out)-1]; last.Content != "found the tables" {
		t.Fatalf("final agent message = %q", last.Content)
	}
}
