// Package savertest runs the shared checkpoint.Saver contract suite,
// ported from Python's saver contract tests
// (libs/checkpoint-postgres/tests/test_sync.py and the saver-facing cases
// in libs/langgraph/tests/). Every Saver implementation — MemorySaver,
// sqlite, postgres — runs the same suite.
//
// The suite follows the philosophy of Python's standardtests packages: one
// behavioral contract, every backend. Divergences forced by the Go port are
// documented at the subtests that encode them — see the list_filter subtest
// for the per-thread, exact-namespace List semantics (Python's
// list(None, filter=...) searches across threads and namespaces; the Go
// Config has no such wildcard form).
package savertest
