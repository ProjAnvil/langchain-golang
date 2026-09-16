package sqltoolkit

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"
)

// formatSQLValue renders one database value as text: NULL for nil, raw text
// for byte slices, RFC 3339 timestamps, and fmt's default rendering
// otherwise.
func formatSQLValue(value any) string {
	switch typed := value.(type) {
	case nil:
		return "NULL"
	case []byte:
		return string(typed)
	case time.Time:
		return typed.Format(time.RFC3339Nano)
	default:
		return fmt.Sprintf("%v", typed)
	}
}

// quoteIdentifier wraps an identifier in the dialect's quoting characters
// (double quotes for both supported dialects; sqlite additionally accepts
// them for its identifiers).
func quoteIdentifier(name string) string {
	return `"` + strings.ReplaceAll(name, `"`, `""`) + `"`
}

// ListTables returns the sorted usable table and view names, mirroring
// SQLDatabase.get_usable_table_names (tables ∪ views).
func (t *Toolkit) ListTables(ctx context.Context) ([]string, error) {
	var (
		rows *sql.Rows
		err  error
	)
	switch t.dialect {
	case SQLite:
		rows, err = t.db.QueryContext(ctx, `
			SELECT name FROM sqlite_master
			WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite\_%' ESCAPE '\'
			ORDER BY name`)
	case Postgres:
		rows, err = t.db.QueryContext(ctx, `
			SELECT table_name FROM information_schema.tables
			WHERE table_schema = current_schema()
			  AND table_type IN ('BASE TABLE', 'VIEW')
			ORDER BY table_name`)
	default:
		return nil, fmt.Errorf("sqltoolkit: unsupported dialect %q", t.dialect)
	}
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var tables []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return nil, err
		}
		tables = append(tables, name)
	}
	return tables, rows.Err()
}

// tableDDL returns the CREATE TABLE (or CREATE VIEW) definition for table.
// sqlite stores the original DDL in sqlite_master, so it is returned verbatim;
// postgres has no stored DDL for user tables, so a CREATE TABLE statement is
// synthesized from information_schema (the spec's "DDL or column list" shape).
func (t *Toolkit) tableDDL(ctx context.Context, table string) (string, error) {
	if t.dialect == SQLite {
		var ddl sql.NullString
		err := t.db.QueryRowContext(ctx,
			`SELECT sql FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?`, table,
		).Scan(&ddl)
		if err != nil {
			return "", err
		}
		if !ddl.Valid || strings.TrimSpace(ddl.String) == "" {
			return fmt.Sprintf("-- no stored DDL for %s", table), nil
		}
		if !strings.HasSuffix(strings.TrimSpace(ddl.String), ";") {
			return strings.TrimSpace(ddl.String) + ";", nil
		}
		return ddl.String, nil
	}
	return t.postgresTableDDL(ctx, table)
}

// postgresTableDDL synthesizes a CREATE TABLE statement from
// information_schema.columns for the current schema.
func (t *Toolkit) postgresTableDDL(ctx context.Context, table string) (string, error) {
	rows, err := t.db.QueryContext(ctx, `
		SELECT column_name, data_type, character_maximum_length, is_nullable, column_default
		FROM information_schema.columns
		WHERE table_schema = current_schema() AND table_name = $1
		ORDER BY ordinal_position`, table)
	if err != nil {
		return "", err
	}
	defer func() { _ = rows.Close() }()

	lines := []string{"CREATE TABLE " + quoteIdentifier(table) + " ("}
	for rows.Next() {
		var name, dataType, isNullable string
		var maxLen sql.NullInt64
		var columnDefault sql.NullString
		if err := rows.Scan(&name, &dataType, &maxLen, &isNullable, &columnDefault); err != nil {
			return "", err
		}
		fullType := dataType
		if maxLen.Valid && maxLen.Int64 > 0 {
			fullType = fmt.Sprintf("%s(%d)", dataType, maxLen.Int64)
		}
		line := fmt.Sprintf("\t%s %s", quoteIdentifier(name), fullType)
		if isNullable == "NO" {
			line += " NOT NULL"
		}
		if columnDefault.Valid && columnDefault.String != "" {
			line += " DEFAULT " + columnDefault.String
		}
		lines = append(lines, line+",")
	}
	if err := rows.Err(); err != nil {
		return "", err
	}
	if len(lines) == 1 {
		return "", fmt.Errorf("table %q not found in database", table)
	}
	lines = append(lines, ");")
	return strings.Join(lines, "\n"), nil
}

// sampleRows renders the sample-rows block used by sql_db_schema, following
// the upstream SQLDatabase._get_sample_rows shape:
//
//	{sampleRows} rows from {table} table:
//	{columns joined by tabs}
//	{rows, one per line, values joined by tabs}
//
// Upstream comma-joins row values while tab-joining headers; this port uses
// tabs for both for consistency with the query tool's output format.
func (t *Toolkit) sampleRowsBlock(ctx context.Context, table string) string {
	prefix := fmt.Sprintf("%d rows from %s table:", t.sampleRows, table)
	rows, err := t.db.QueryContext(ctx,
		fmt.Sprintf("SELECT * FROM %s LIMIT %d", quoteIdentifier(table), t.sampleRows))
	if err != nil {
		return prefix + "\nError: " + err.Error()
	}
	defer func() { _ = rows.Close() }()

	columns, err := rows.Columns()
	if err != nil {
		return prefix + "\nError: " + err.Error()
	}
	lines := []string{prefix, strings.Join(columns, "\t")}
	values := make([]any, len(columns))
	scans := make([]any, len(columns))
	for i := range values {
		scans[i] = &values[i]
	}
	for rows.Next() {
		if err := rows.Scan(scans...); err != nil {
			return prefix + "\nError: " + err.Error()
		}
		parts := make([]string, len(values))
		for i, value := range values {
			parts[i] = formatSQLValue(value)
		}
		lines = append(lines, strings.Join(parts, "\t"))
	}
	if err := rows.Err(); err != nil {
		return prefix + "\nError: " + err.Error()
	}
	return strings.Join(lines, "\n")
}

// SchemaInfo returns DDL plus sample rows for the given tables. An empty
// table list returns the information for every usable table (a Go-side
// convenience; upstream requires explicit table names). Unknown tables
// produce an error mirroring get_table_info_no_throw's
// "Tables [...] not found in database".
func (t *Toolkit) SchemaInfo(ctx context.Context, tables ...string) (string, error) {
	names := make([]string, 0, len(tables))
	for _, table := range tables {
		if trimmed := strings.TrimSpace(table); trimmed != "" {
			names = append(names, trimmed)
		}
	}
	usable, err := t.ListTables(ctx)
	if err != nil {
		return "", err
	}
	if len(names) == 0 {
		names = usable
	}
	if len(names) == 0 {
		return "no tables found in the database", nil
	}

	usableSet := make(map[string]struct{}, len(usable))
	for _, table := range usable {
		usableSet[table] = struct{}{}
	}
	var missing []string
	for _, name := range names {
		if _, ok := usableSet[name]; !ok {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		return "", fmt.Errorf("tables %v not found in database", missing)
	}

	blocks := make([]string, 0, len(names))
	for _, name := range names {
		ddl, err := t.tableDDL(ctx, name)
		if err != nil {
			return "", err
		}
		if t.sampleRows > 0 {
			ddl += "\n" + t.sampleRowsBlock(ctx, name)
		}
		blocks = append(blocks, ddl)
	}
	info := strings.Join(blocks, "\n\n")

	// Cap the payload so large schemas cannot blow up the agent prompt.
	if limit := t.maxSchemaLength; limit > 0 && len([]rune(info)) > limit {
		info = string([]rune(info)[:limit]) +
			fmt.Sprintf("\n\n(schema info truncated at %d characters)", limit)
	}
	return info, nil
}

// Query executes a read-only query after passing the read-only guardrails
// and returns up to maxQueryRows rows, prefixed by a tab-joined header row.
// SQL errors are returned as ordinary errors; the sql_db_query tool renders
// them as "Error: ..." content so the model can rewrite and retry (upstream
// run_no_throw behavior).
func (t *Toolkit) Query(ctx context.Context, query string) (string, error) {
	statement, err := guardReadOnly(t.dialect, query)
	if err != nil {
		return "", err
	}
	rows, err := t.db.QueryContext(ctx, statement)
	if err != nil {
		return "", err
	}
	defer func() { _ = rows.Close() }()

	columns, err := rows.Columns()
	if err != nil {
		return "", err
	}
	lines := []string{strings.Join(columns, "\t")}
	values := make([]any, len(columns))
	scans := make([]any, len(columns))
	for i := range values {
		scans[i] = &values[i]
	}
	returned := 0
	for rows.Next() {
		if returned >= t.maxQueryRows {
			lines = append(lines, fmt.Sprintf("(result truncated at %d rows)", t.maxQueryRows))
			return strings.Join(lines, "\n"), nil
		}
		if err := rows.Scan(scans...); err != nil {
			return "", err
		}
		parts := make([]string, len(values))
		for i, value := range values {
			parts[i] = formatSQLValue(value)
		}
		lines = append(lines, strings.Join(parts, "\t"))
		returned++
	}
	if err := rows.Err(); err != nil {
		return "", err
	}
	if returned == 0 {
		return "no rows returned", nil
	}
	return strings.Join(lines, "\n"), nil
}

// CheckQuery validates a query with the database planner (EXPLAIN) without
// executing it, returning "OK" when the query compiles or the planner error
// otherwise. The same read-only guardrail as Query applies, so the checker
// never blesses a query the query tool would refuse. Upstream's
// QuerySQLCheckerTool validates through an LLM prompt; this port uses the
// database itself, which is deterministic and needs no extra model call.
func (t *Toolkit) CheckQuery(ctx context.Context, query string) (string, error) {
	statement, err := guardReadOnly(t.dialect, query)
	if err != nil {
		return "", err
	}
	rows, err := t.db.QueryContext(ctx, "EXPLAIN "+statement)
	if err != nil {
		return "", err
	}
	defer func() { _ = rows.Close() }()
	return "OK", nil
}
