package sqltoolkit

import (
	"database/sql"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	_ "github.com/jackc/pgx/v5/stdlib" // database/sql driver name `pgx`

	"github.com/projanvil/langchain-golang/core/tools"
)

// e2eDSN returns the connection string for the postgres e2e database, or ""
// when e2e tests should skip (CI runs without a server).
func e2eDSN() string {
	return strings.TrimSpace(os.Getenv("SQLTOOLKIT_PG_TEST_DSN"))
}

var e2eTableCounter atomic.Int64

// newPostgresToolkit opens the e2e postgres database, creates two uniquely
// named fixture tables, and returns a toolkit over them. The tables are
// dropped on cleanup.
func newPostgresToolkit(t *testing.T) (*Toolkit, string, string) {
	t.Helper()
	dsn := e2eDSN()
	if dsn == "" {
		t.Skip("SQLTOOLKIT_PG_TEST_DSN not set; skipping postgres e2e (requires a running postgres)")
	}
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		t.Fatalf("open pgx: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	suffix := e2eTableCounter.Add(1)
	users := fmt.Sprintf("sqltoolkit_users_%d", suffix)
	orders := fmt.Sprintf("sqltoolkit_orders_%d", suffix)
	ddl := fmt.Sprintf(`
		CREATE TABLE %[1]s (id integer PRIMARY KEY, name text NOT NULL, email text);
		CREATE TABLE %[2]s (id integer PRIMARY KEY, user_id integer NOT NULL REFERENCES %[1]s(id), amount_cents integer NOT NULL);
		INSERT INTO %[1]s (id, name, email) VALUES (1, 'alice', 'alice@example.com'), (2, 'bob', NULL);
		INSERT INTO %[2]s (id, user_id, amount_cents) VALUES (1, 1, 1000), (2, 2, 500);
	`, users, orders)
	if _, err := db.Exec(ddl); err != nil {
		t.Fatalf("create fixture: %v", err)
	}
	t.Cleanup(func() {
		_, _ = db.Exec(fmt.Sprintf("DROP TABLE IF EXISTS %[1]s, %[2]s", orders, users))
	})

	toolkit, err := NewSQLToolkit(t.Context(), db, Postgres)
	if err != nil {
		t.Fatalf("NewSQLToolkit: %v", err)
	}
	return toolkit, users, orders
}

func e2eToolNamed(t *testing.T, toolkit *Toolkit, name string) tools.Tool {
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

func TestE2EPostgresListTablesAndSchema(t *testing.T) {
	toolkit, users, orders := newPostgresToolkit(t)

	tables, err := toolkit.ListTables(t.Context())
	if err != nil {
		t.Fatalf("ListTables: %v", err)
	}
	if !contains(tables, users) || !contains(tables, orders) {
		t.Fatalf("ListTables missing fixture tables: %v", tables)
	}

	listTool := e2eToolNamed(t, toolkit, ListTablesToolName)
	result, err := listTool.Invoke(t.Context(), map[string]any{})
	if err != nil {
		t.Fatalf("invoke list tool: %v", err)
	}
	if !strings.Contains(result.Content, users) || !strings.Contains(result.Content, orders) {
		t.Fatalf("list tool content = %q", result.Content)
	}

	info, err := toolkit.SchemaInfo(t.Context(), users)
	if err != nil {
		t.Fatalf("SchemaInfo: %v", err)
	}
	for _, want := range []string{
		fmt.Sprintf("CREATE TABLE %q", users),
		`"id" integer`,
		`"name" text NOT NULL`,
		`"email" text`,
		"3 rows from " + users + " table:",
		"alice@example.com",
	} {
		if !strings.Contains(info, want) {
			t.Errorf("schema info missing %q:\n%s", want, info)
		}
	}
}

func contains(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}

func TestE2EPostgresQueryHappyPath(t *testing.T) {
	toolkit, users, _ := newPostgresToolkit(t)
	tool := e2eToolNamed(t, toolkit, QueryToolName)

	result, err := tool.Invoke(t.Context(), map[string]any{
		"query": fmt.Sprintf("SELECT id, name, email FROM %s ORDER BY id", users),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	want := strings.Join([]string{
		"id\tname\temail",
		"1\talice\talice@example.com",
		"2\tbob\tNULL",
	}, "\n")
	if result.Content != want {
		t.Fatalf("content =\n%q\nwant\n%q", result.Content, want)
	}
}

func TestE2EPostgresGuardrails(t *testing.T) {
	toolkit, users, _ := newPostgresToolkit(t)
	queryTool := e2eToolNamed(t, toolkit, QueryToolName)
	checkerTool := e2eToolNamed(t, toolkit, QueryCheckerToolName)

	rejected := []string{
		// Plain DML/DDL.
		fmt.Sprintf("UPDATE %s SET name = 'x'", users),
		fmt.Sprintf("DELETE FROM %s", users),
		// Multiple statements.
		fmt.Sprintf("SELECT id FROM %s; DROP TABLE %s", users, users),
		// SELECT ... INTO is a write on postgres even though it starts with SELECT.
		fmt.Sprintf("SELECT * INTO archive_%d FROM %s", e2eTableCounter.Load(), users),
		// Data-modifying CTEs.
		fmt.Sprintf("WITH d AS (DELETE FROM %s RETURNING *) SELECT * FROM d", users),
		fmt.Sprintf("WITH u AS (UPDATE %s SET name = 'x' RETURNING *) SELECT * FROM u", users),
		// Row locks are not read-only.
		fmt.Sprintf("SELECT * FROM %s FOR UPDATE", users),
		// Comment bypass.
		fmt.Sprintf("-- SELECT 1\nDROP TABLE %s", users),
	}
	for _, query := range rejected {
		result, err := queryTool.Invoke(t.Context(), map[string]any{"query": query})
		if err != nil {
			t.Fatalf("query tool %q: %v", query, err)
		}
		if !strings.HasPrefix(result.Content, "Error:") ||
			!strings.Contains(result.Content, "read-only guardrail") {
			t.Errorf("query tool allowed guarded query %q: %q", query, result.Content)
		}
		check, err := checkerTool.Invoke(t.Context(), map[string]any{"query": query})
		if err != nil {
			t.Fatalf("checker %q: %v", query, err)
		}
		if !strings.HasPrefix(check.Content, "Error:") {
			t.Errorf("checker blessed guarded query %q: %q", query, check.Content)
		}
	}

	// Postgres dollar-quoted strings must not be mistaken for statement
	// separators (the ';' inside $$...$$ is literal text).
	result, err := queryTool.Invoke(t.Context(), map[string]any{
		"query": "SELECT length($tag$ab;cd$tag$) AS len",
	})
	if err != nil {
		t.Fatalf("dollar-quoted query: %v", err)
	}
	if result.Content != "len\n5" {
		t.Fatalf("dollar-quoted content = %q, want %q", result.Content, "len\n5")
	}

	// The matrix must have been side-effect free: fixture rows intact and no
	// archive table created.
	var count int
	if err := toolkit.db.QueryRowContext(t.Context(),
		fmt.Sprintf("SELECT COUNT(*) FROM %s", users)).Scan(&count); err != nil {
		t.Fatalf("count users: %v", err)
	}
	if count != 2 {
		t.Fatalf("fixture table mutated: %d rows, want 2", count)
	}
	var archive string
	err = toolkit.db.QueryRowContext(t.Context(), `
		SELECT table_name FROM information_schema.tables
		WHERE table_schema = current_schema() AND table_name LIKE 'archive_%' LIMIT 1`,
	).Scan(&archive)
	if err == nil {
		t.Fatalf("guardrail matrix created table %q", archive)
	}
	if !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("check archive table: %v", err)
	}
}

func TestE2EPostgresChecker(t *testing.T) {
	toolkit, users, _ := newPostgresToolkit(t)
	tool := e2eToolNamed(t, toolkit, QueryCheckerToolName)

	result, err := tool.Invoke(t.Context(), map[string]any{
		"query": fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE email IS NOT NULL", users),
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if result.Content != "OK" {
		t.Fatalf("valid query verdict = %q", result.Content)
	}

	result, err = tool.Invoke(t.Context(), map[string]any{
		"query": "SELEC 1",
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !strings.HasPrefix(result.Content, "Error:") {
		t.Fatalf("invalid query verdict = %q", result.Content)
	}
}
