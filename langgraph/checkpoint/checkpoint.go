// Package checkpoint implements the checkpoint persistence contract of the Go
// port of Python's langgraph: the versioned Checkpoint snapshot type, the
// Saver read/write interface, and the in-memory MemorySaver, mirroring
// `langgraph.checkpoint.base` / `langgraph.checkpoint.memory`.
//
// Scope note: checkpoints are versioned and retained per thread (full
// history, listable and addressable by ID, so time-travel/forking is
// possible), but storage is in memory only; persistent backends and
// pluggable serializers are later milestones.
package checkpoint

import (
	"context"
	"time"
)

// Config identifies a checkpoint position within a thread, mirroring the
// checkpoint-relevant fields of Python's `RunnableConfig`
// (`configurable.thread_id` / `checkpoint_ns` / `checkpoint_id`).
type Config struct {
	// ThreadID identifies the conversation/run the checkpoint belongs to.
	ThreadID string
	// CheckpointNS namespaces checkpoints within a thread: "" is the root
	// graph; subgraph runs use "node" or "a/b"-style paths.
	CheckpointNS string
	// CheckpointID selects a specific checkpoint; empty means "the latest".
	CheckpointID string
}

// Checkpoint is a versioned snapshot of a graph run's channel state at one
// superstep boundary, mirroring Python's `langgraph.checkpoint.base.Checkpoint`.
type Checkpoint struct {
	// V is the Go checkpoint format version, always 1.
	V int
	// ID identifies this checkpoint. NewID(step) produces IDs whose
	// lexicographic order matches chronological order.
	ID string
	// TS is the wall-clock time the checkpoint was created.
	TS time.Time
	// ChannelValues is the value of each channel (state key) at this point.
	ChannelValues map[string]any
	// ChannelVersions is the last version written to each channel.
	ChannelVersions map[string]int64
	// VersionsSeen records, per node, the channel versions that node had
	// observed when it last ran: node -> channel -> version.
	VersionsSeen map[string]map[string]int64
	// Next lists the tasks scheduled for the superstep after this checkpoint.
	Next []PlannedTask
}

// PlannedTask is a node invocation scheduled for a future superstep,
// mirroring what Python persists in a checkpoint's `pending_sends` / Pregel
// task descriptions.
type PlannedTask struct {
	// ID is the deterministic task identity, TaskID(cpID, step, node, arg)
	// (assigned by the versioned executor in M2 Task 3).
	ID string
	// Node is the name of the node to invoke.
	Node string
	// Arg is the per-invocation input; non-nil only for Send-driven
	// invocations (nil means "use the shared graph state").
	Arg map[string]any
}

// Metadata describes how a checkpoint was produced, mirroring Python's
// `CheckpointMetadata`.
type Metadata struct {
	// Source is one of "input" (initial/new-turn input), "loop" (a superstep
	// boundary), "update" (a manual state update), or "fork" (a time-travel
	// fork).
	Source string
	// Step is the superstep number: -1 for the "input" checkpoint, 0 for the
	// first "loop" checkpoint.
	Step int
	// Parents maps checkpoint namespace -> parent checkpoint ID.
	Parents map[string]string
}

// Reserved channel names used in Write.Channel for control-plane writes that
// are not user state keys, mirroring Python's reserved channels
// (`langgraph.constants.INTERRUPT` / `TASKS` / `ERROR`).
const (
	// ReservedInterrupt persists a raised interrupt; Value is types.Interrupt.
	ReservedInterrupt = "__interrupt__"
	// ReservedTasks persists a scheduled Send; Value is types.Send (plain
	// node names are normalized to Send{Node: name}, per D4).
	ReservedTasks = "__tasks__"
	// ReservedError persists a task error; Value is a string.
	ReservedError = "__error__"
)

// Write is a single pending write recorded against a checkpoint by a task,
// mirroring Python's `PendingWrite` (task_id, channel, value).
type Write struct {
	// TaskID identifies the task that produced the write (stamped by
	// Saver.PutWrites).
	TaskID string
	// Channel is the state key written, or one of the Reserved* constants.
	Channel string
	// Value is the written value.
	Value any
}

// Tuple bundles a checkpoint with everything needed to resume from it,
// mirroring Python's `CheckpointTuple`.
type Tuple struct {
	Config     Config
	Checkpoint Checkpoint
	Metadata   Metadata
	// ParentConfig is the put-time position of the parent checkpoint (D3),
	// nil for a thread's first checkpoint.
	ParentConfig *Config
	// PendingWrites holds the writes recorded against this checkpoint via
	// PutWrites, in insertion order.
	PendingWrites []Write
}

// ListOptions filters Saver.List results.
type ListOptions struct {
	// Before, when non-nil with a non-empty CheckpointID, restricts results
	// to checkpoints strictly before that one (older IDs, same thread and
	// namespace).
	Before *Config
	// Limit caps the number of returned tuples; 0 means no limit.
	Limit int
}

// Saver persists versioned checkpoints and their pending writes, mirroring
// the read/write contract of Python's `BaseCheckpointSaver`.
//
// This is a breaking redesign of the M1 Saver (single latest checkpoint per
// thread, no context, no history): checkpoints are now immutable,
// ID-addressable snapshots retained per thread until DeleteThread.
type Saver interface {
	// GetTuple returns the tuple for cfg — the checkpoint identified by
	// cfg.CheckpointID, or the latest when it is empty — or (nil, nil) when
	// no matching checkpoint exists.
	GetTuple(ctx context.Context, cfg Config) (*Tuple, error)
	// List returns the checkpoint history for cfg.ThreadID/cfg.CheckpointNS,
	// newest first, filtered by opts.
	List(ctx context.Context, cfg Config, opts ListOptions) ([]Tuple, error)
	// Put stores cp as a new checkpoint and returns the Config identifying
	// it. cfg carries the caller's current position: when cfg.CheckpointID
	// is non-empty it is recorded as the new checkpoint's parent link (D3).
	// newVersions, when non-nil, is merged into the stored ChannelVersions.
	Put(ctx context.Context, cfg Config, cp Checkpoint, md Metadata, newVersions map[string]int64) (Config, error)
	// PutWrites appends writes (each stamped with taskID) to the pending
	// writes of the checkpoint identified by cfg.
	PutWrites(ctx context.Context, cfg Config, writes []Write, taskID string) error
	// DeleteThread removes all checkpoints and pending writes for threadID.
	DeleteThread(ctx context.Context, threadID string) error
}
