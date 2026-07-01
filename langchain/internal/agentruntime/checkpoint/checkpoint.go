package checkpoint

import (
	"sync"

	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
)

// Checkpoint is a snapshot of a graph run's state, saved when execution
// pauses on an Interrupt so it can later be resumed, mirroring (a small
// subset of) Python's checkpoint concept from `langgraph.checkpoint.base`.
//
// Scope note: Python's checkpoints are versioned, support time-travel
// (listing/forking historical checkpoints), and are persisted through
// pluggable serializers to Postgres/SQLite/etc. This port keeps exactly one
// checkpoint per thread ID (the most recent), in memory only, which is
// sufficient for the "pause on interrupt, resume with Command.Resume"
// human-in-the-loop pattern CompiledGraph.Invoke implements.
type Checkpoint struct {
	// Values is the full channel state at the time of the pause.
	Values map[string]any
	// Next is the node scheduled to re-execute when resumed. Matching
	// Python's `interrupt()` semantics, the node is re-executed from its
	// start; it is not resumed mid-function.
	Next string
	// PendingInterrupts holds the interrupts raised the last time Next was
	// executed, in call order, so a Command.Resume value can be routed to
	// the correct interrupt() call when the node is re-executed.
	PendingInterrupts []agentruntime.Interrupt
}

// Saver persists and retrieves the single most recent Checkpoint for a
// thread, mirroring the essential read/write contract of Python's
// `BaseCheckpointSaver` (minus versioning, listing, and pending-writes
// tracking).
type Saver interface {
	// Get returns the most recent checkpoint for threadID, or ok=false if
	// none exists.
	Get(threadID string) (cp Checkpoint, ok bool)
	// Put saves cp as the most recent checkpoint for threadID.
	Put(threadID string, cp Checkpoint)
	// Delete removes any saved checkpoint for threadID (e.g. once a run
	// completes without pausing).
	Delete(threadID string)
}

// MemorySaver is an in-memory Saver, the Go equivalent of Python's
// `InMemorySaver` scoped to this port's single-checkpoint-per-thread model.
// The zero value is ready to use.
type MemorySaver struct {
	mu          sync.Mutex
	checkpoints map[string]Checkpoint
}

// NewMemorySaver constructs an empty MemorySaver.
func NewMemorySaver() *MemorySaver {
	return &MemorySaver{checkpoints: map[string]Checkpoint{}}
}

// Get implements Saver.
func (s *MemorySaver) Get(threadID string) (Checkpoint, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.checkpoints == nil {
		return Checkpoint{}, false
	}
	cp, ok := s.checkpoints[threadID]
	return cp, ok
}

// Put implements Saver.
func (s *MemorySaver) Put(threadID string, cp Checkpoint) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.checkpoints == nil {
		s.checkpoints = map[string]Checkpoint{}
	}
	s.checkpoints[threadID] = cp
}

// Delete implements Saver.
func (s *MemorySaver) Delete(threadID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.checkpoints, threadID)
}
