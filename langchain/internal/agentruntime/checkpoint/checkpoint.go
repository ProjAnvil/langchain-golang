// Package checkpoint is a thin alias layer over
// `github.com/projanvil/langchain-golang/langgraph/checkpoint`; see that
// package for documentation. New code should import langgraph/checkpoint
// directly.
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
