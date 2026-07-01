package indexing

import (
	"context"
	"database/sql"
	"fmt"
	"regexp"
	"strings"
	"time"
)

const defaultSQLRecordTable = "langchain_index_records"

// SQLRecordManager stores indexing records in a SQL database. It uses portable
// SQL and avoids driver-specific upsert syntax by doing delete+insert inside a
// transaction for updates.
type SQLRecordManager struct {
	Namespace string
	DB        *sql.DB
	Table     string
	NowQuery  string
}

// SQLRecordManagerOption configures a SQLRecordManager.
type SQLRecordManagerOption func(*SQLRecordManager)

// WithSQLRecordTable sets the records table name.
func WithSQLRecordTable(table string) SQLRecordManagerOption {
	return func(m *SQLRecordManager) {
		m.Table = table
	}
}

// WithSQLRecordNowQuery sets the query used by GetTime. The query must return
// either a time.Time or a Unix timestamp in seconds.
func WithSQLRecordNowQuery(query string) SQLRecordManagerOption {
	return func(m *SQLRecordManager) {
		m.NowQuery = query
	}
}

// NewSQLRecordManager creates a SQL-backed record manager.
func NewSQLRecordManager(namespace string, db *sql.DB, opts ...SQLRecordManagerOption) (*SQLRecordManager, error) {
	if db == nil {
		return nil, fmt.Errorf("db is required")
	}
	manager := &SQLRecordManager{
		Namespace: namespace,
		DB:        db,
		Table:     defaultSQLRecordTable,
		NowQuery:  "SELECT CURRENT_TIMESTAMP",
	}
	for _, opt := range opts {
		opt(manager)
	}
	if !validSQLIdentifier(manager.Table) {
		return nil, fmt.Errorf("invalid SQL record table name %q", manager.Table)
	}
	return manager, nil
}

// CreateSchema creates the record table if it does not exist.
func (m *SQLRecordManager) CreateSchema(ctx context.Context) error {
	_, err := m.DB.ExecContext(ctx, fmt.Sprintf(
		`CREATE TABLE IF NOT EXISTS %s (
namespace TEXT NOT NULL,
record_key TEXT NOT NULL,
group_id TEXT,
updated_at TIMESTAMP NOT NULL,
PRIMARY KEY (namespace, record_key)
)`,
		m.Table,
	))
	return err
}

// GetTime returns the database server time.
func (m *SQLRecordManager) GetTime(ctx context.Context) (time.Time, error) {
	var value any
	if err := m.DB.QueryRowContext(ctx, m.NowQuery).Scan(&value); err != nil {
		return time.Time{}, err
	}
	switch typed := value.(type) {
	case time.Time:
		return typed, nil
	case string:
		parsed, err := time.Parse(time.RFC3339Nano, typed)
		if err == nil {
			return parsed, nil
		}
		parsed, err = time.Parse("2006-01-02 15:04:05", typed)
		if err != nil {
			return time.Time{}, fmt.Errorf("parse SQL time %q: %w", typed, err)
		}
		return parsed, nil
	case []byte:
		return parseSQLTimeBytes(typed)
	case int64:
		return time.Unix(typed, 0), nil
	case float64:
		sec := int64(typed)
		nsec := int64((typed - float64(sec)) * 1e9)
		return time.Unix(sec, nsec), nil
	default:
		return time.Time{}, fmt.Errorf("unsupported SQL time type %T", value)
	}
}

// Update upserts record keys with optional group IDs.
func (m *SQLRecordManager) Update(ctx context.Context, keys []string, groupIDs []string, timeAtLeast time.Time) error {
	if len(groupIDs) > 0 && len(keys) != len(groupIDs) {
		return fmt.Errorf("length of keys must match length of groupIDs")
	}
	now, err := m.GetTime(ctx)
	if err != nil {
		return err
	}
	if !timeAtLeast.IsZero() && now.Before(timeAtLeast) {
		return fmt.Errorf("timeAtLeast must be in the past")
	}
	tx, err := m.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for i, key := range keys {
		groupID := ""
		if len(groupIDs) > 0 {
			groupID = groupIDs[i]
		}
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s WHERE namespace = ? AND record_key = ?`, m.Table), m.Namespace, key); err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`INSERT INTO %s (namespace, record_key, group_id, updated_at) VALUES (?, ?, ?, ?)`, m.Table), m.Namespace, key, groupID, now); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// Exists reports which keys have records.
func (m *SQLRecordManager) Exists(ctx context.Context, keys []string) ([]bool, error) {
	out := make([]bool, len(keys))
	if len(keys) == 0 {
		return out, nil
	}
	query := fmt.Sprintf(`SELECT record_key FROM %s WHERE namespace = ? AND record_key IN (%s)`, m.Table, placeholders(len(keys)))
	args := make([]any, 0, len(keys)+1)
	args = append(args, m.Namespace)
	for _, key := range keys {
		args = append(args, key)
	}
	rows, err := m.DB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	found := map[string]bool{}
	for rows.Next() {
		var key string
		if err := rows.Scan(&key); err != nil {
			return nil, err
		}
		found[key] = true
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for i, key := range keys {
		out[i] = found[key]
	}
	return out, nil
}

// DeleteKeys deletes records by key.
func (m *SQLRecordManager) DeleteKeys(ctx context.Context, keys []string) error {
	if len(keys) == 0 {
		return nil
	}
	query := fmt.Sprintf(`DELETE FROM %s WHERE namespace = ? AND record_key IN (%s)`, m.Table, placeholders(len(keys)))
	args := make([]any, 0, len(keys)+1)
	args = append(args, m.Namespace)
	for _, key := range keys {
		args = append(args, key)
	}
	_, err := m.DB.ExecContext(ctx, query, args...)
	return err
}

// ListKeys lists record keys, optionally filtered by group ID and updated time.
func (m *SQLRecordManager) ListKeys(ctx context.Context, groupIDs []string, before time.Time, limit int) ([]string, error) {
	query := strings.Builder{}
	query.WriteString(fmt.Sprintf(`SELECT record_key FROM %s WHERE namespace = ?`, m.Table))
	args := []any{m.Namespace}
	if !before.IsZero() {
		query.WriteString(` AND updated_at < ?`)
		args = append(args, before)
	}
	if len(groupIDs) > 0 {
		query.WriteString(` AND group_id IN (` + placeholders(len(groupIDs)) + `)`)
		for _, groupID := range groupIDs {
			args = append(args, groupID)
		}
	}
	if limit > 0 {
		query.WriteString(` LIMIT ?`)
		args = append(args, limit)
	}
	rows, err := m.DB.QueryContext(ctx, query.String(), args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	keys := []string{}
	for rows.Next() {
		var key string
		if err := rows.Scan(&key); err != nil {
			return nil, err
		}
		keys = append(keys, key)
	}
	return keys, rows.Err()
}

func placeholders(count int) string {
	if count <= 0 {
		return ""
	}
	parts := make([]string, count)
	for i := range parts {
		parts[i] = "?"
	}
	return strings.Join(parts, ", ")
}

func parseSQLTimeBytes(value []byte) (time.Time, error) {
	text := string(value)
	if parsed, err := time.Parse(time.RFC3339Nano, text); err == nil {
		return parsed, nil
	}
	parsed, err := time.Parse("2006-01-02 15:04:05", text)
	if err != nil {
		return time.Time{}, fmt.Errorf("parse SQL time %q: %w", text, err)
	}
	return parsed, nil
}

var sqlIdentifierPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

func validSQLIdentifier(value string) bool {
	return sqlIdentifierPattern.MatchString(value)
}
