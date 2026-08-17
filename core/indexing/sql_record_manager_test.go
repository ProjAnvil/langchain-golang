package indexing

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

func TestSQLRecordManagerLifecycle(t *testing.T) {
	ctx := context.Background()
	db := openTestSQLDB(t)
	manager, err := NewSQLRecordManager("unit", db, WithSQLRecordNowQuery("SELECT NOW"))
	if err != nil {
		t.Fatalf("new manager: %v", err)
	}
	if err := manager.CreateSchema(ctx); err != nil {
		t.Fatalf("create schema: %v", err)
	}

	if err := manager.Update(ctx, []string{"a", "b"}, []string{"one", "two"}, time.Time{}); err != nil {
		t.Fatalf("update: %v", err)
	}
	exists, err := manager.Exists(ctx, []string{"a", "missing", "b"})
	if err != nil {
		t.Fatalf("exists: %v", err)
	}
	if fmt.Sprint(exists) != "[true false true]" {
		t.Fatalf("exists: %#v", exists)
	}
	keys, err := manager.ListKeys(ctx, []string{"one"}, time.Time{}, 10)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(keys) != 1 || keys[0] != "a" {
		t.Fatalf("keys: %#v", keys)
	}
	if err := manager.DeleteKeys(ctx, []string{"a"}); err != nil {
		t.Fatalf("delete: %v", err)
	}
	exists, err = manager.Exists(ctx, []string{"a", "b"})
	if err != nil {
		t.Fatalf("exists after delete: %v", err)
	}
	if fmt.Sprint(exists) != "[false true]" {
		t.Fatalf("exists after delete: %#v", exists)
	}
}

func TestSQLRecordManagerValidation(t *testing.T) {
	db := openTestSQLDB(t)
	if _, err := NewSQLRecordManager("unit", nil); err == nil {
		t.Fatal("expected nil db error")
	}
	if _, err := NewSQLRecordManager("unit", db, WithSQLRecordTable("bad-name")); err == nil {
		t.Fatal("expected table validation error")
	}
	manager, err := NewSQLRecordManager("unit", db, WithSQLRecordNowQuery("SELECT NOW"))
	if err != nil {
		t.Fatalf("new manager: %v", err)
	}
	if err := manager.Update(context.Background(), []string{"a"}, []string{"one", "two"}, time.Time{}); err == nil {
		t.Fatal("expected group length error")
	}
	if err := manager.Update(context.Background(), []string{"a"}, nil, time.Now().Add(time.Hour)); err == nil {
		t.Fatal("expected future time error")
	}
}

func TestIndexDocumentsWithSQLRecordManagerFullCleanup(t *testing.T) {
	ctx := context.Background()
	db := openTestSQLDB(t)
	manager, err := NewSQLRecordManager("unit", db, WithSQLRecordNowQuery("SELECT NOW"))
	if err != nil {
		t.Fatalf("new manager: %v", err)
	}
	if err := manager.CreateSchema(ctx); err != nil {
		t.Fatalf("create schema: %v", err)
	}
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	original := []documents.Document{
		documents.New("old a", map[string]any{"source": "a"}),
		documents.New("keep b", map[string]any{"source": "b"}),
	}
	if _, err := IndexDocuments(ctx, original, manager, store, Options{SourceIDKey: "source"}); err != nil {
		t.Fatalf("index original: %v", err)
	}
	oldKey, err := HashDocument(original[0])
	if err != nil {
		t.Fatalf("hash: %v", err)
	}
	bKey, err := HashDocument(original[1])
	if err != nil {
		t.Fatalf("hash b: %v", err)
	}
	testSQLAdvanceTime(db, time.Second)

	replacement := []documents.Document{documents.New("new a", map[string]any{"source": "a"})}
	got, err := IndexDocuments(ctx, replacement, manager, store, Options{
		SourceIDKey: "source",
		Cleanup:     CleanupFull,
	})
	if err != nil {
		t.Fatalf("index replacement: %v", err)
	}
	if got.NumAdded != 1 || got.NumDeleted != 2 {
		t.Fatalf("result: %+v", got)
	}
	if docs, err := store.GetByIDs(ctx, []string{oldKey}); err != nil || len(docs) != 0 {
		t.Fatalf("old doc should be deleted: docs=%#v err=%v", docs, err)
	}
	if docs, err := store.GetByIDs(ctx, []string{bKey}); err != nil || len(docs) != 0 {
		t.Fatalf("other stale source should be deleted by full cleanup: docs=%#v err=%v", docs, err)
	}
}

type testSQLRecord struct {
	namespace string
	key       string
	groupID   string
	updatedAt time.Time
}

type testSQLState struct {
	mu      sync.Mutex
	now     time.Time
	records map[string]testSQLRecord

	// Failure-injection knobs used by error-path tests.
	nowValue   driver.Value // overrides now in GetTime when non-nil
	nowErr     error        // fails the now query
	execErrOn  string       // uppercase substring; matching exec queries fail
	queryErrOn string       // uppercase substring; matching queries fail
	beginErr   error        // fails BeginTx
	commitErr  error        // fails tx Commit
	badRow     bool         // record_key queries return a non-string row value
	rowsErr    error        // returned by rows after all values are consumed
}

var testSQLStates sync.Map
var testSQLDBStates sync.Map

func openTestSQLDB(t *testing.T) *sql.DB {
	t.Helper()
	registerTestSQLDriver()
	name := fmt.Sprintf("%s-%d", t.Name(), time.Now().UnixNano())
	state := &testSQLState{
		now:     time.Now().UTC().Truncate(time.Microsecond),
		records: map[string]testSQLRecord{},
	}
	testSQLStates.Store(name, state)
	db, err := sql.Open("lc_indexing_test", name)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	testSQLDBStates.Store(db, state)
	t.Cleanup(func() {
		db.Close()
		testSQLStates.Delete(name)
		testSQLDBStates.Delete(db)
	})
	return db
}

func testSQLAdvanceTime(db *sql.DB, d time.Duration) {
	testSQLMutateState(db, func(state *testSQLState) {
		state.now = state.now.Add(d)
	})
}

func testSQLMutateState(db *sql.DB, fn func(*testSQLState)) {
	raw, ok := testSQLDBStates.Load(db)
	if !ok {
		return
	}
	state := raw.(*testSQLState)
	state.mu.Lock()
	defer state.mu.Unlock()
	fn(state)
}

var registerTestSQLDriverOnce sync.Once

func registerTestSQLDriver() {
	registerTestSQLDriverOnce.Do(func() {
		sql.Register("lc_indexing_test", testSQLDriver{})
	})
}

type testSQLDriver struct{}

func (d testSQLDriver) Open(name string) (driver.Conn, error) {
	raw, ok := testSQLStates.Load(name)
	if !ok {
		return nil, fmt.Errorf("unknown test sql database %q", name)
	}
	return &testSQLConn{state: raw.(*testSQLState)}, nil
}

type testSQLConn struct {
	state *testSQLState
}

func (c *testSQLConn) Prepare(string) (driver.Stmt, error) {
	return nil, fmt.Errorf("prepare unsupported")
}
func (c *testSQLConn) Close() error { return nil }
func (c *testSQLConn) Begin() (driver.Tx, error) {
	return testSQLTx{state: c.state}, nil
}

func (c *testSQLConn) BeginTx(context.Context, driver.TxOptions) (driver.Tx, error) {
	if c.state.beginErr != nil {
		return nil, c.state.beginErr
	}
	return testSQLTx{state: c.state}, nil
}

func (c *testSQLConn) ExecContext(_ context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	upper := strings.ToUpper(strings.TrimSpace(query))
	if c.state.execErrOn != "" && strings.Contains(upper, c.state.execErrOn) {
		return nil, fmt.Errorf("injected exec error for %q", c.state.execErrOn)
	}
	switch {
	case strings.HasPrefix(upper, "CREATE TABLE"):
		return driver.RowsAffected(0), nil
	case strings.HasPrefix(upper, "DELETE FROM") && strings.Contains(upper, " IN "):
		namespace := argString(args[0].Value)
		count := int64(0)
		for _, arg := range args[1:] {
			key := argString(arg.Value)
			if _, ok := c.state.records[recordMapKey(namespace, key)]; ok {
				delete(c.state.records, recordMapKey(namespace, key))
				count++
			}
		}
		return driver.RowsAffected(count), nil
	case strings.HasPrefix(upper, "DELETE FROM"):
		namespace := argString(args[0].Value)
		key := argString(args[1].Value)
		delete(c.state.records, recordMapKey(namespace, key))
		return driver.RowsAffected(1), nil
	case strings.HasPrefix(upper, "INSERT INTO"):
		namespace := argString(args[0].Value)
		key := argString(args[1].Value)
		groupID := argString(args[2].Value)
		updatedAt := args[3].Value.(time.Time)
		c.state.records[recordMapKey(namespace, key)] = testSQLRecord{
			namespace: namespace,
			key:       key,
			groupID:   groupID,
			updatedAt: updatedAt,
		}
		return driver.RowsAffected(1), nil
	default:
		return nil, fmt.Errorf("unsupported exec query %q", query)
	}
}

func (c *testSQLConn) QueryContext(_ context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	upper := strings.ToUpper(strings.TrimSpace(query))
	if upper == "SELECT NOW" {
		if c.state.nowErr != nil {
			return nil, c.state.nowErr
		}
		value := driver.Value(c.state.now)
		if c.state.nowValue != nil {
			value = c.state.nowValue
		}
		return &testSQLRows{columns: []string{"now"}, values: [][]driver.Value{{value}}}, nil
	}
	if c.state.queryErrOn != "" && strings.Contains(upper, c.state.queryErrOn) {
		return nil, fmt.Errorf("injected query error for %q", c.state.queryErrOn)
	}
	switch {
	case strings.HasPrefix(upper, "SELECT RECORD_KEY") && strings.Contains(upper, "RECORD_KEY IN"):
		namespace := argString(args[0].Value)
		values := [][]driver.Value{}
		for _, arg := range args[1:] {
			key := argString(arg.Value)
			if _, ok := c.state.records[recordMapKey(namespace, key)]; ok {
				values = append(values, []driver.Value{key})
			}
		}
		if c.state.badRow {
			values = append(values, []driver.Value{nil})
		}
		return &testSQLRows{columns: []string{"record_key"}, values: values, err: c.state.rowsErr}, nil
	case strings.HasPrefix(upper, "SELECT RECORD_KEY"):
		namespace := argString(args[0].Value)
		argIndex := 1
		before := time.Time{}
		if strings.Contains(upper, "UPDATED_AT <") {
			before = args[argIndex].Value.(time.Time)
			argIndex++
		}
		groups := map[string]bool{}
		if strings.Contains(upper, "GROUP_ID IN") {
			groupCount := strings.Count(query[strings.Index(strings.ToUpper(query), "GROUP_ID IN"):], "?")
			if strings.Contains(upper, " LIMIT ") {
				groupCount--
			}
			for i := 0; i < groupCount; i++ {
				groups[argString(args[argIndex].Value)] = true
				argIndex++
			}
		}
		limit := 0
		if strings.Contains(upper, " LIMIT ") {
			limit = int(args[argIndex].Value.(int64))
		}
		records := make([]testSQLRecord, 0, len(c.state.records))
		for _, record := range c.state.records {
			if record.namespace != namespace {
				continue
			}
			if !before.IsZero() && !record.updatedAt.Before(before) {
				continue
			}
			if len(groups) > 0 && !groups[record.groupID] {
				continue
			}
			records = append(records, record)
		}
		sort.Slice(records, func(i, j int) bool { return records[i].key < records[j].key })
		if limit > 0 && len(records) > limit {
			records = records[:limit]
		}
		values := make([][]driver.Value, len(records))
		for i, record := range records {
			values[i] = []driver.Value{record.key}
		}
		if c.state.badRow {
			values = append(values, []driver.Value{nil})
		}
		return &testSQLRows{columns: []string{"record_key"}, values: values, err: c.state.rowsErr}, nil
	default:
		return nil, fmt.Errorf("unsupported query %q", query)
	}
}

type testSQLTx struct{ state *testSQLState }

func (t testSQLTx) Commit() error {
	if t.state != nil && t.state.commitErr != nil {
		return t.state.commitErr
	}
	return nil
}
func (t testSQLTx) Rollback() error { return nil }

type testSQLRows struct {
	columns []string
	values  [][]driver.Value
	index   int
	err     error
}

func (r *testSQLRows) Columns() []string { return r.columns }
func (r *testSQLRows) Close() error      { return nil }

func (r *testSQLRows) Next(dest []driver.Value) error {
	if r.index >= len(r.values) {
		if r.err != nil {
			err := r.err
			r.err = nil
			return err
		}
		return io.EOF
	}
	copy(dest, r.values[r.index])
	r.index++
	return nil
}

func argString(value any) string {
	if value == nil {
		return ""
	}
	return fmt.Sprint(value)
}

func recordMapKey(namespace string, key string) string {
	return namespace + "\x00" + key
}

func mustSQLManager(t *testing.T, db *sql.DB) *SQLRecordManager {
	t.Helper()
	manager, err := NewSQLRecordManager("unit", db, WithSQLRecordNowQuery("SELECT NOW"))
	if err != nil {
		t.Fatalf("new manager: %v", err)
	}
	return manager
}

func TestSQLRecordManagerGetTimeValueTypes(t *testing.T) {
	ctx := context.Background()
	refTime := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

	tests := []struct {
		name    string
		value   driver.Value
		wantErr bool
		check   func(got time.Time) bool
	}{
		{
			name:  "time.Time",
			value: refTime,
			check: func(got time.Time) bool { return got.Equal(refTime) },
		},
		{
			name:  "string RFC3339",
			value: refTime.Format(time.RFC3339Nano),
			check: func(got time.Time) bool { return got.Equal(refTime) },
		},
		{
			name:  "string datetime",
			value: "2026-01-02 03:04:05",
			check: func(got time.Time) bool { return got.UTC().Equal(refTime) },
		},
		{
			name:    "string invalid",
			value:   "not-a-time",
			wantErr: true,
		},
		{
			name:  "bytes RFC3339",
			value: []byte(refTime.Format(time.RFC3339Nano)),
			check: func(got time.Time) bool { return got.Equal(refTime) },
		},
		{
			name:  "bytes datetime",
			value: []byte("2026-01-02 03:04:05"),
			check: func(got time.Time) bool { return got.UTC().Equal(refTime) },
		},
		{
			name:    "bytes invalid",
			value:   []byte("nope"),
			wantErr: true,
		},
		{
			name:  "int64 unix",
			value: int64(1700000000),
			check: func(got time.Time) bool { return got.Unix() == 1700000000 },
		},
		{
			name:  "float64 unix",
			value: float64(1700000000.5),
			check: func(got time.Time) bool { return got.Unix() == 1700000000 && got.Nanosecond() == 500000000 },
		},
		{
			name:    "unsupported type",
			value:   true,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			db := openTestSQLDB(t)
			testSQLMutateState(db, func(state *testSQLState) {
				state.nowValue = tt.value
			})
			manager := mustSQLManager(t, db)

			got, err := manager.GetTime(ctx)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got time %v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("get time: %v", err)
			}
			if tt.check != nil && !tt.check(got) {
				t.Fatalf("unexpected time %v", got)
			}
		})
	}
}

func TestSQLRecordManagerGetTimeQueryError(t *testing.T) {
	db := openTestSQLDB(t)
	testSQLMutateState(db, func(state *testSQLState) {
		state.nowErr = errTest
	})
	manager := mustSQLManager(t, db)

	if _, err := manager.GetTime(context.Background()); !errors.Is(err, errTest) {
		t.Fatalf("expected query error, got %v", err)
	}
}

func TestSQLRecordManagerUpdateErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("get time", func(t *testing.T) {
		db := openTestSQLDB(t)
		testSQLMutateState(db, func(state *testSQLState) { state.nowErr = errTest })
		manager := mustSQLManager(t, db)
		if err := manager.Update(ctx, []string{"a"}, nil, time.Time{}); !errors.Is(err, errTest) {
			t.Fatalf("expected get time error, got %v", err)
		}
	})

	t.Run("begin", func(t *testing.T) {
		db := openTestSQLDB(t)
		testSQLMutateState(db, func(state *testSQLState) { state.beginErr = errTest })
		manager := mustSQLManager(t, db)
		if err := manager.Update(ctx, []string{"a"}, nil, time.Time{}); !errors.Is(err, errTest) {
			t.Fatalf("expected begin error, got %v", err)
		}
	})

	t.Run("delete exec", func(t *testing.T) {
		db := openTestSQLDB(t)
		testSQLMutateState(db, func(state *testSQLState) { state.execErrOn = "DELETE" })
		manager := mustSQLManager(t, db)
		if err := manager.Update(ctx, []string{"a"}, nil, time.Time{}); err == nil {
			t.Fatal("expected delete exec error")
		}
	})

	t.Run("insert exec", func(t *testing.T) {
		db := openTestSQLDB(t)
		testSQLMutateState(db, func(state *testSQLState) { state.execErrOn = "INSERT" })
		manager := mustSQLManager(t, db)
		if err := manager.Update(ctx, []string{"a"}, nil, time.Time{}); err == nil {
			t.Fatal("expected insert exec error")
		}
	})

	t.Run("commit", func(t *testing.T) {
		db := openTestSQLDB(t)
		testSQLMutateState(db, func(state *testSQLState) { state.commitErr = errTest })
		manager := mustSQLManager(t, db)
		if err := manager.Update(ctx, []string{"a"}, nil, time.Time{}); !errors.Is(err, errTest) {
			t.Fatalf("expected commit error, got %v", err)
		}
	})
}

func TestSQLRecordManagerExistsErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("query", func(t *testing.T) {
		db := openTestSQLDB(t)
		testSQLMutateState(db, func(state *testSQLState) { state.queryErrOn = "RECORD_KEY IN" })
		manager := mustSQLManager(t, db)
		if _, err := manager.Exists(ctx, []string{"a"}); err == nil {
			t.Fatal("expected query error")
		}
	})

	t.Run("scan", func(t *testing.T) {
		db := openTestSQLDB(t)
		testSQLMutateState(db, func(state *testSQLState) { state.badRow = true })
		manager := mustSQLManager(t, db)
		if _, err := manager.Exists(ctx, []string{"a"}); err == nil {
			t.Fatal("expected scan error")
		}
	})

	t.Run("rows", func(t *testing.T) {
		db := openTestSQLDB(t)
		testSQLMutateState(db, func(state *testSQLState) { state.rowsErr = errTest })
		manager := mustSQLManager(t, db)
		if _, err := manager.Exists(ctx, []string{"a"}); !errors.Is(err, errTest) {
			t.Fatalf("expected rows error, got %v", err)
		}
	})

	t.Run("empty keys", func(t *testing.T) {
		db := openTestSQLDB(t)
		manager := mustSQLManager(t, db)
		exists, err := manager.Exists(ctx, nil)
		if err != nil {
			t.Fatalf("exists with no keys: %v", err)
		}
		if len(exists) != 0 {
			t.Fatalf("exists with no keys: %#v", exists)
		}
	})
}

func TestSQLRecordManagerDeleteKeysErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("empty keys", func(t *testing.T) {
		db := openTestSQLDB(t)
		manager := mustSQLManager(t, db)
		if err := manager.DeleteKeys(ctx, nil); err != nil {
			t.Fatalf("delete with no keys: %v", err)
		}
	})

	t.Run("exec", func(t *testing.T) {
		db := openTestSQLDB(t)
		testSQLMutateState(db, func(state *testSQLState) { state.execErrOn = "DELETE FROM" })
		manager := mustSQLManager(t, db)
		if err := manager.DeleteKeys(ctx, []string{"a"}); err == nil {
			t.Fatal("expected exec error")
		}
	})
}

func TestSQLRecordManagerListKeysErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("query", func(t *testing.T) {
		db := openTestSQLDB(t)
		testSQLMutateState(db, func(state *testSQLState) { state.queryErrOn = "SELECT RECORD_KEY" })
		manager := mustSQLManager(t, db)
		if _, err := manager.ListKeys(ctx, nil, time.Time{}, 0); err == nil {
			t.Fatal("expected query error")
		}
	})

	t.Run("scan", func(t *testing.T) {
		db := openTestSQLDB(t)
		manager := mustSQLManager(t, db)
		if err := manager.Update(ctx, []string{"a"}, nil, time.Time{}); err != nil {
			t.Fatalf("update: %v", err)
		}
		testSQLMutateState(db, func(state *testSQLState) { state.badRow = true })
		if _, err := manager.ListKeys(ctx, nil, time.Time{}, 0); err == nil {
			t.Fatal("expected scan error")
		}
	})

	t.Run("rows", func(t *testing.T) {
		db := openTestSQLDB(t)
		testSQLMutateState(db, func(state *testSQLState) { state.rowsErr = errTest })
		manager := mustSQLManager(t, db)
		if _, err := manager.ListKeys(ctx, nil, time.Time{}, 0); !errors.Is(err, errTest) {
			t.Fatalf("expected rows error, got %v", err)
		}
	})

	t.Run("before and group filters", func(t *testing.T) {
		db := openTestSQLDB(t)
		manager := mustSQLManager(t, db)
		if err := manager.Update(ctx, []string{"a", "b"}, []string{"one", "two"}, time.Time{}); err != nil {
			t.Fatalf("update: %v", err)
		}
		keys, err := manager.ListKeys(ctx, []string{"one"}, time.Now().Add(time.Hour), 0)
		if err != nil {
			t.Fatalf("list keys: %v", err)
		}
		if len(keys) != 1 || keys[0] != "a" {
			t.Fatalf("keys: %#v", keys)
		}
		keys, err = manager.ListKeys(ctx, nil, time.Now().Add(-time.Hour), 0)
		if err != nil {
			t.Fatalf("list keys: %v", err)
		}
		if len(keys) != 0 {
			t.Fatalf("before filter should exclude all keys, got %#v", keys)
		}
	})
}

func TestSQLRecordManagerCreateSchemaError(t *testing.T) {
	db := openTestSQLDB(t)
	testSQLMutateState(db, func(state *testSQLState) { state.execErrOn = "CREATE TABLE" })
	manager := mustSQLManager(t, db)
	if err := manager.CreateSchema(context.Background()); err == nil {
		t.Fatal("expected exec error")
	}
}

func TestPlaceholders(t *testing.T) {
	if got := placeholders(0); got != "" {
		t.Fatalf("placeholders(0) = %q, want empty", got)
	}
	if got := placeholders(-1); got != "" {
		t.Fatalf("placeholders(-1) = %q, want empty", got)
	}
	if got := placeholders(3); got != "?, ?, ?" {
		t.Fatalf("placeholders(3) = %q, want %q", got, "?, ?, ?")
	}
}
