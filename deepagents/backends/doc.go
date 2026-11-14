// Package backends ports the deepagents backends subsystem to Go. It mirrors
// the Python reference implementation under deepagents/backends/ in the
// upstream deepagents repository (libs/deepagents/deepagents/backends/).
//
// The core abstraction is the [Backend] interface (upstream
// BackendProtocol, protocol.py), a virtual filesystem with ls, read, write,
// edit, glob, and grep operations over virtual '/'-rooted paths, plus the
// optional [Deleter] capability. Two implementations are provided:
// FilesystemBackend (real disk under a root directory, filesystem.py) and
// StateBackend (in-memory files state, state.go). Supporting pieces are the
// glob/grep matchers and virtual-path normalization in match.go and utils.go,
// standardized file-operation error codes, and the permission rule types the
// middleware layer evaluates.
package backends
