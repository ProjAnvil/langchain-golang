package sqltoolkit

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

// The introspection paths that talk to postgres (information_schema queries)
// or that only fail when the database breaks mid-flight cannot be exercised
// against the sqlite fixture, so this file drives them through a scripted
// database/sql driver: the handler answers every query in-process, with no
// server and no network.

// fakeRows serves scripted rows to database/sql, then io.EOF (or a scripted
// terminal error that surfaces through Rows.Err).
type fakeRows struct {
	columns []string
	rows    [][]driver.Value
	nextErr error // returned by Next once rows is exhausted, instead of io.EOF
	idx     int
}

func (r *fakeRows) Columns() []string { return r.columns }
func (r *fakeRows) Close() error      { return nil }

func (r *fakeRows) Next(dest []driver.Value) error {
	if r.idx >= len(r.rows) {
		if r.nextErr != nil {
			return r.nextErr
		}
		return io.EOF
	}
	copy(dest, r.rows[r.idx])
	r.idx++
	return nil
}

// fakeConn routes every query to a scripted handler. Only the query surface
// is implemented (no Prepare/Begin); the toolkit never uses the rest.
type fakeConn struct {
	handler func(query string, args []driver.NamedValue) (driver.Rows, error)
}

func (c *fakeConn) Prepare(string) (driver.Stmt, error) {
	return nil, errors.New("fakeConn: prepared statements not supported")
}
func (c *fakeConn) Close() error { return nil }
func (c *fakeConn) Begin() (driver.Tx, error) {
	return nil, errors.New("fakeConn: transactions not supported")
}
func (c *fakeConn) Ping(context.Context) error { return nil }
func (c *fakeConn) QueryContext(_ context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	return c.handler(query, args)
}

type fakeConnector struct {
	handler func(query string, args []driver.NamedValue) (driver.Rows, error)
}

func (c *fakeConnector) Connect(context.Context) (driver.Conn, error) {
	return &fakeConn{handler: c.handler}, nil
}
func (c *fakeConnector) Driver() driver.Driver { return fakeDrv{} }

type fakeDrv struct{}

func (fakeDrv) Open(string) (driver.Conn, error) {
	return nil, errors.New("fakeDrv: use sql.OpenDB with fakeConnector")
}

// staticRows builds a finite fakeRows result set.
func staticRows(columns []string, rows ...[]driver.Value) *fakeRows {
	return &fakeRows{columns: columns, rows: rows}
}

// errScripted is the sentinel terminal error served by scripted result sets.
var errScripted = errors.New("scripted rows failure")

// newScriptedToolkit opens a Toolkit over the scripted driver. The handler
// receives every database/sql query the toolkit issues, including the $1/?
// bound arguments, so tests can assert on both.
func newScriptedToolkit(
	t *testing.T, dialect Dialect,
	handler func(query string, args []driver.NamedValue) (driver.Rows, error),
	opts ...Option,
) *Toolkit {
	t.Helper()
	db := sql.OpenDB(&fakeConnector{handler: handler})
	t.Cleanup(func() { _ = db.Close() })
	toolkit, err := NewSQLToolkit(t.Context(), db, dialect, opts...)
	if err != nil {
		t.Fatalf("NewSQLToolkit: %v", err)
	}
	return toolkit
}

// openBlankToolkit opens a real in-memory sqlite database with no tables.
func openBlankToolkit(t *testing.T) *Toolkit {
	t.Helper()
	db, err := sql.Open("sqlite", "file:sqltoolkit_blank?mode=memory&cache=shared")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	toolkit, err := NewSQLToolkit(t.Context(), db, SQLite)
	if err != nil {
		t.Fatalf("NewSQLToolkit: %v", err)
	}
	return toolkit
}

func TestFormatSQLValue(t *testing.T) {
	ts := time.Date(2026, 9, 16, 10, 30, 5, 500_000_000, time.UTC)
	cases := []struct {
		value any
		want  string
	}{
		{nil, "NULL"},
		{[]byte("blob"), "blob"},
		{ts, "2026-09-16T10:30:05.5Z"},
		{42, "42"},
		{"plain", "plain"},
		{3.5, "3.5"},
	}
	for _, tc := range cases {
		if got := formatSQLValue(tc.value); got != tc.want {
			t.Errorf("formatSQLValue(%v) = %q, want %q", tc.value, got, tc.want)
		}
	}
}

func TestQuoteIdentifier(t *testing.T) {
	if got := quoteIdentifier(`we"ird`); got != `"we""ird"` {
		t.Errorf(`quoteIdentifier("we\"ird") = %q, want %q`, got, `"we""ird"`)
	}
}

func TestListTablesPostgresScripted(t *testing.T) {
	t.Run("information schema happy path", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, Postgres, func(query string, _ []driver.NamedValue) (driver.Rows, error) {
			if !strings.Contains(query, "information_schema.tables") {
				return nil, errors.New("unexpected query: " + query)
			}
			return staticRows([]string{"table_name"},
				[]driver.Value{"orders"}, []driver.Value{"users"}), nil
		})
		tables, err := toolkit.ListTables(t.Context())
		if err != nil {
			t.Fatalf("ListTables: %v", err)
		}
		if strings.Join(tables, ",") != "orders,users" {
			t.Errorf("ListTables = %v, want [orders users]", tables)
		}
	})

	t.Run("query failure", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, Postgres, func(string, []driver.NamedValue) (driver.Rows, error) {
			return nil, errors.New("connection refused")
		})
		if _, err := toolkit.ListTables(t.Context()); err == nil {
			t.Fatal("ListTables succeeded, want error")
		}
	})

	t.Run("scan failure", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, Postgres, func(string, []driver.NamedValue) (driver.Rows, error) {
			// struct{} is not a scannable driver.Value, forcing rows.Scan to fail.
			return staticRows([]string{"table_name"}, []driver.Value{struct{}{}}), nil
		})
		if _, err := toolkit.ListTables(t.Context()); err == nil {
			t.Fatal("ListTables succeeded, want scan error")
		}
	})
}

func TestListTablesUnsupportedDialect(t *testing.T) {
	toolkit := &Toolkit{dialect: Dialect("mysql")}
	_, err := toolkit.ListTables(t.Context())
	if err == nil || !strings.Contains(err.Error(), "unsupported dialect") {
		t.Fatalf("err = %v, want unsupported dialect error", err)
	}
}

func TestTableDDLSqliteBranches(t *testing.T) {
	router := func(ddl driver.Rows, ddlErr error) func(string, []driver.NamedValue) (driver.Rows, error) {
		return func(query string, _ []driver.NamedValue) (driver.Rows, error) {
			if !strings.Contains(query, "SELECT sql FROM sqlite_master") {
				return nil, errors.New("unexpected query: " + query)
			}
			return ddl, ddlErr
		}
	}

	t.Run("appends missing terminator", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, SQLite, router(staticRows([]string{"sql"},
			[]driver.Value{"CREATE TABLE users (id INTEGER)"}), nil))
		ddl, err := toolkit.tableDDL(t.Context(), "users")
		if err != nil {
			t.Fatalf("tableDDL: %v", err)
		}
		if !strings.HasSuffix(ddl, ";") || strings.Contains(ddl, ";;") {
			t.Errorf("ddl = %q, want exactly one trailing semicolon", ddl)
		}
	})

	t.Run("keeps stored terminator", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, SQLite, router(staticRows([]string{"sql"},
			[]driver.Value{"  CREATE TABLE users (id INTEGER);  "}), nil))
		ddl, err := toolkit.tableDDL(t.Context(), "users")
		if err != nil {
			t.Fatalf("tableDDL: %v", err)
		}
		if got := strings.TrimSpace(ddl); got != "CREATE TABLE users (id INTEGER);" {
			t.Errorf("ddl = %q, want the stored statement verbatim", ddl)
		}
	})

	t.Run("blank stored ddl yields placeholder", func(t *testing.T) {
		for _, blank := range []driver.Value{"", nil} {
			toolkit := newScriptedToolkit(t, SQLite, router(staticRows([]string{"sql"},
				[]driver.Value{blank}), nil))
			ddl, err := toolkit.tableDDL(t.Context(), "users")
			if err != nil {
				t.Fatalf("tableDDL(%v): %v", blank, err)
			}
			if ddl != "-- no stored DDL for users" {
				t.Errorf("tableDDL(%v) = %q, want placeholder", blank, ddl)
			}
		}
	})

	t.Run("missing row errors", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, SQLite, router(staticRows([]string{"sql"}), nil))
		if _, err := toolkit.tableDDL(t.Context(), "ghost"); !errors.Is(err, sql.ErrNoRows) {
			t.Fatalf("err = %v, want sql.ErrNoRows", err)
		}
	})

	t.Run("query failure", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, SQLite, router(nil, errors.New("disk i/o error")))
		if _, err := toolkit.tableDDL(t.Context(), "users"); err == nil {
			t.Fatal("tableDDL succeeded, want error")
		}
	})
}

func TestPostgresTableDDLScripted(t *testing.T) {
	router := func(rows driver.Rows, err error) func(string, []driver.NamedValue) (driver.Rows, error) {
		return func(query string, _ []driver.NamedValue) (driver.Rows, error) {
			if !strings.Contains(query, "information_schema.columns") {
				return nil, errors.New("unexpected query: " + query)
			}
			return rows, err
		}
	}

	t.Run("synthesizes create table", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, Postgres, router(staticRows([]string{
			"column_name", "data_type", "character_maximum_length", "is_nullable", "column_default",
		},
			[]driver.Value{"id", "integer", nil, "NO", nil},
			[]driver.Value{"name", "character varying", int64(80), "YES", nil},
			[]driver.Value{"note", "text", int64(0), "YES", "'unknown'::text"},
		), nil))
		ddl, err := toolkit.tableDDL(t.Context(), "users")
		if err != nil {
			t.Fatalf("tableDDL: %v", err)
		}
		for _, want := range []string{
			`CREATE TABLE "users" (`,
			"\n\t" + `"id" integer NOT NULL,`,
			"\n\t" + `"name" character varying(80),`,
			// A zero character_maximum_length is not rendered as varchar(0).
			"\n\t" + `"note" text DEFAULT 'unknown'::text,`,
			"\n);",
		} {
			if !strings.Contains(ddl, want) {
				t.Errorf("ddl missing %q:\n%s", want, ddl)
			}
		}
	})

	t.Run("unknown table errors", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, Postgres, router(staticRows([]string{
			"column_name", "data_type", "character_maximum_length", "is_nullable", "column_default",
		}), nil))
		_, err := toolkit.tableDDL(t.Context(), "ghost")
		if err == nil || !strings.Contains(err.Error(), `table "ghost" not found`) {
			t.Fatalf("err = %v, want table not found", err)
		}
	})

	t.Run("query failure", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, Postgres, router(nil, errors.New("connection reset")))
		if _, err := toolkit.tableDDL(t.Context(), "users"); err == nil {
			t.Fatal("tableDDL succeeded, want error")
		}
	})

	t.Run("scan failure", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, Postgres, router(staticRows([]string{
			"column_name", "data_type", "character_maximum_length", "is_nullable", "column_default",
		},
			[]driver.Value{struct{}{}, "integer", nil, "NO", nil},
		), nil))
		if _, err := toolkit.tableDDL(t.Context(), "users"); err == nil {
			t.Fatal("tableDDL succeeded, want scan error")
		}
	})

	t.Run("rows error after first column", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, Postgres, router(&fakeRows{
			columns: []string{
				"column_name", "data_type", "character_maximum_length", "is_nullable", "column_default",
			},
			rows: [][]driver.Value{{"id", "integer", nil, "NO", nil}},
			// Next returns this once the good row is consumed, so the loop
			// exits with rows.Err() reporting it.
			nextErr: errScripted,
		}, nil))
		if _, err := toolkit.tableDDL(t.Context(), "users"); !errors.Is(err, errScripted) {
			t.Fatalf("err = %v, want scripted rows error", err)
		}
	})
}

// sqliteScriptRoutes every introspection query a SchemaInfo call makes.
type sqliteScript struct {
	listTables driver.Rows
	ddl        driver.Rows
	sample     driver.Rows
	listErr    error
	ddlErr     error
}

func (s sqliteScript) handler(query string, _ []driver.NamedValue) (driver.Rows, error) {
	switch {
	case strings.Contains(query, "NOT LIKE 'sqlite"):
		return s.listTables, s.listErr
	case strings.Contains(query, "SELECT sql FROM sqlite_master"):
		return s.ddl, s.ddlErr
	case strings.Contains(query, "SELECT * FROM"):
		if s.sample == nil {
			return staticRows(nil), nil // an empty-but-valid result set
		}
		return s.sample, nil
	default:
		return nil, errors.New("unexpected query: " + query)
	}
}

// freshFixture returns a new single-table script; fakeRows is stateful, so
// every subtest needs its own instance.
func freshFixture() sqliteScript {
	return sqliteScript{
		listTables: staticRows([]string{"name"}, []driver.Value{"users"}),
		ddl:        staticRows([]string{"sql"}, []driver.Value{"CREATE TABLE users (id INTEGER)"}),
	}
}

func TestSampleRowsBlockFailurePaths(t *testing.T) {
	t.Run("query failure becomes embedded error", func(t *testing.T) {
		fixture := freshFixture()
		toolkit := newScriptedToolkit(t, SQLite, func(query string, args []driver.NamedValue) (driver.Rows, error) {
			if strings.Contains(query, "SELECT * FROM") {
				return nil, errors.New("no such table: users")
			}
			return fixture.handler(query, args)
		})
		info, err := toolkit.SchemaInfo(t.Context())
		if err != nil {
			t.Fatalf("SchemaInfo: %v (sample-row failures must not fail the whole report)", err)
		}
		if !strings.Contains(info, "3 rows from users table:") || !strings.Contains(info, "Error: no such table") {
			t.Errorf("info missing embedded sample error:\n%s", info)
		}
	})

	t.Run("rows failure becomes embedded error", func(t *testing.T) {
		script := freshFixture()
		script.sample = &fakeRows{
			columns: []string{"id"},
			rows:    [][]driver.Value{{int64(1)}},
			nextErr: errScripted,
		}
		toolkit := newScriptedToolkit(t, SQLite, script.handler)
		info, err := toolkit.SchemaInfo(t.Context())
		if err != nil {
			t.Fatalf("SchemaInfo: %v", err)
		}
		if !strings.Contains(info, "Error: "+errScripted.Error()) {
			t.Errorf("info missing embedded rows error:\n%s", info)
		}
	})
}

func TestQueryFailurePathsScripted(t *testing.T) {
	t.Run("rows failure returns error", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, SQLite, func(string, []driver.NamedValue) (driver.Rows, error) {
			return &fakeRows{
				columns: []string{"id"},
				rows:    [][]driver.Value{{int64(1)}},
				nextErr: errScripted,
			}, nil
		})
		if _, err := toolkit.Query(t.Context(), "SELECT * FROM users"); !errors.Is(err, errScripted) {
			t.Fatalf("err = %v, want scripted rows error", err)
		}
	})

	t.Run("query failure returns error", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, SQLite, func(string, []driver.NamedValue) (driver.Rows, error) {
			return nil, errors.New("malformed database image")
		})
		if _, err := toolkit.Query(t.Context(), "SELECT * FROM users"); err == nil {
			t.Fatal("Query succeeded, want error")
		}
	})
}

func TestSchemaInfoFailureBranches(t *testing.T) {
	t.Run("list tables failure", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, SQLite, func(string, []driver.NamedValue) (driver.Rows, error) {
			return nil, errors.New("corrupt catalog")
		})
		if _, err := toolkit.SchemaInfo(t.Context(), "users"); err == nil {
			t.Fatal("SchemaInfo succeeded, want error")
		}
	})

	t.Run("empty database reports no tables", func(t *testing.T) {
		toolkit := openBlankToolkit(t)
		info, err := toolkit.SchemaInfo(t.Context())
		if err != nil {
			t.Fatalf("SchemaInfo: %v", err)
		}
		if info != "no tables found in the database" {
			t.Errorf("info = %q, want the no-tables notice", info)
		}
	})

	t.Run("ddl failure surfaces through the per-table loop", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, SQLite, func(query string, _ []driver.NamedValue) (driver.Rows, error) {
			if strings.Contains(query, "SELECT sql FROM sqlite_master") {
				return nil, errors.New("catalog read failed")
			}
			return staticRows([]string{"name"}, []driver.Value{"users"}), nil
		})
		if _, err := toolkit.SchemaInfo(t.Context(), "users"); err == nil {
			t.Fatal("SchemaInfo succeeded, want error")
		}
	})

	t.Run("blank names fall back to every usable table", func(t *testing.T) {
		toolkit := newScriptedToolkit(t, SQLite, sqliteScript{
			listTables: staticRows([]string{"name"}, []driver.Value{"users"}),
			ddl:        staticRows([]string{"sql"}, []driver.Value{"CREATE TABLE users (id INTEGER)"}),
		}.handler)
		info, err := toolkit.SchemaInfo(t.Context(), "  ", "")
		if err != nil {
			t.Fatalf("SchemaInfo: %v", err)
		}
		if !strings.Contains(info, "CREATE TABLE users") {
			t.Errorf("info = %q, want the usable table's DDL", info)
		}
	})
}

// TestClosedDatabaseFailurePaths drives the toolkit against a real sqlite
// handle that has been closed underneath it (the driver-failure path a
// process hits when the database shuts down mid-session).
func TestClosedDatabaseFailurePaths(t *testing.T) {
	db, err := sql.Open("sqlite", "file:sqltoolkit_closed?mode=memory&cache=shared")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	toolkit, err := NewSQLToolkit(t.Context(), db, SQLite)
	if err != nil {
		t.Fatalf("NewSQLToolkit: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	if _, err := toolkit.ListTables(t.Context()); err == nil {
		t.Error("ListTables on closed db succeeded, want error")
	}
	if _, err := toolkit.SchemaInfo(t.Context(), "users"); err == nil {
		t.Error("SchemaInfo on closed db succeeded, want error")
	}
	if _, err := toolkit.Query(t.Context(), "SELECT 1"); err == nil {
		t.Error("Query on closed db succeeded, want error")
	}
	if _, err := toolkit.CheckQuery(t.Context(), "SELECT 1"); err == nil {
		t.Error("CheckQuery on closed db succeeded, want error")
	}
}
