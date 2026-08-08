// Package sqlite implements checkpoint.Saver on top of SQLite, the Go
// equivalent of Python's `langgraph.checkpoint.sqlite.SqliteSaver`: durable
// versioned checkpoints and pending writes in a single file (or `:memory:`),
// in WAL mode, with the same two-table schema.
//
// This package is its own Go module — it is the repo's only third-party
// dependency (`modernc.org/sqlite`, a pure-Go SQLite, no cgo) and the root
// module deliberately stays dependency-free. Consumers add it with:
//
//	go get github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite
//
// Run this module's tests from its own directory (`make test-sqlite` from the
// repo root does the same): the root module's `go test ./...` does not cross
// the nested-module boundary.
//
// # Pre-1.0 behavior change: __tasks__ write idx
//
// Earlier revisions stored `__tasks__` (Command.Goto routing) writes at the
// reserved idx -2; they now use the positional idx, matching Python and
// MemorySaver, so multiple `__tasks__` writes in one batch all survive.
// Databases written by the older revision remain fully readable — old -2
// rows and new positional rows coexist under the same (task_id, idx) primary
// key and both come back in PendingWrites — but a re-invocation no longer
// overwrites an older revision's -2 row in place.
package sqlite
