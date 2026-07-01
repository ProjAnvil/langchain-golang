package indexing

import (
	"context"
	"database/sql"
	"database/sql/driver"
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
	raw, ok := testSQLDBStates.Load(db)
	if !ok {
		return
	}
	state := raw.(*testSQLState)
	state.mu.Lock()
	defer state.mu.Unlock()
	state.now = state.now.Add(d)
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
func (c *testSQLConn) Close() error              { return nil }
func (c *testSQLConn) Begin() (driver.Tx, error) { return testSQLTx{}, nil }

func (c *testSQLConn) BeginTx(context.Context, driver.TxOptions) (driver.Tx, error) {
	return testSQLTx{}, nil
}

func (c *testSQLConn) ExecContext(_ context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	upper := strings.ToUpper(strings.TrimSpace(query))
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
	switch {
	case upper == "SELECT NOW":
		return &testSQLRows{columns: []string{"now"}, values: [][]driver.Value{{c.state.now}}}, nil
	case strings.HasPrefix(upper, "SELECT RECORD_KEY") && strings.Contains(upper, "RECORD_KEY IN"):
		namespace := argString(args[0].Value)
		values := [][]driver.Value{}
		for _, arg := range args[1:] {
			key := argString(arg.Value)
			if _, ok := c.state.records[recordMapKey(namespace, key)]; ok {
				values = append(values, []driver.Value{key})
			}
		}
		return &testSQLRows{columns: []string{"record_key"}, values: values}, nil
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
		return &testSQLRows{columns: []string{"record_key"}, values: values}, nil
	default:
		return nil, fmt.Errorf("unsupported query %q", query)
	}
}

type testSQLTx struct{}

func (t testSQLTx) Commit() error   { return nil }
func (t testSQLTx) Rollback() error { return nil }

type testSQLRows struct {
	columns []string
	values  [][]driver.Value
	index   int
}

func (r *testSQLRows) Columns() []string { return r.columns }
func (r *testSQLRows) Close() error      { return nil }

func (r *testSQLRows) Next(dest []driver.Value) error {
	if r.index >= len(r.values) {
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
