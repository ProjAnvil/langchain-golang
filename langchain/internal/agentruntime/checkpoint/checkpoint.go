// Package checkpoint is a thin alias layer over
// `github.com/projanvil/langchain-golang/langgraph/checkpoint`; see that
// package for documentation. New code should import langgraph/checkpoint
// directly.
//
// The shim remains for in-repo compatibility only. Note that the Saver
// interface broke twice pre-1.0: in M2 (the versioned
// GetTuple/List/Put/PutWrites/DeleteThread contract keyed by Config replaced
// the M1 Get/Put/Delete methods) and in M5 (ListOptions gained Filter,
// Write gained TaskPath, and PutWrites gained a taskPath parameter); the
// type aliases still resolve unchanged.
package checkpoint

import "github.com/projanvil/langchain-golang/langgraph/checkpoint"

type Config = checkpoint.Config
type Checkpoint = checkpoint.Checkpoint
type PlannedTask = checkpoint.PlannedTask
type Metadata = checkpoint.Metadata
type Write = checkpoint.Write
type Tuple = checkpoint.Tuple
type ListOptions = checkpoint.ListOptions
type Saver = checkpoint.Saver
type MemorySaver = checkpoint.MemorySaver

var NewMemorySaver = checkpoint.NewMemorySaver
