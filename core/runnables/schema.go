package runnables

// StreamEventData holds the data payload associated with a streaming event.
// It is the Go equivalent of Python's EventData TypedDict in
// langchain_core.runnables.schema.
type StreamEventData struct {
	// Input is the input passed to the Runnable at start, or nil when not yet
	// known (e.g., when the Runnable itself streams its input).
	Input any
	// Output is available at the END of the Runnable's execution.
	Output any
	// Chunk is a streaming chunk from the output. Chunks support addition:
	// summing all chunks equals the final Output.
	Chunk any
	// Error is non-nil when the Runnable raised an exception (on_*_error
	// events only).
	Error error
	// ToolCallID links a tool-execution error event to the originating call.
	ToolCallID string
}

// StreamEvent is the Go equivalent of Python's StandardStreamEvent TypedDict
// from langchain_core.runnables.schema. Every streaming event produced by
// astream_events (or its Go equivalent) conforms to this shape.
//
// Event names follow the pattern: on_[runnable_type]_(start|stream|end).
// Runnable types include: llm, chat_model, prompt, tool, chain.
type StreamEvent struct {
	// Event is the event name, e.g. "on_chat_model_start".
	Event string
	// RunID is a unique identifier for this runnable invocation.
	RunID string
	// Name is the name of the Runnable that generated the event.
	Name string
	// Tags are inherited from parent Runnables and passed via config.
	Tags []string
	// Metadata is inherited from parent Runnables and passed via config.
	Metadata map[string]any
	// ParentIDs is the ordered list of ancestor run IDs, root-first.
	ParentIDs []string
	// Data is the event-specific payload.
	Data StreamEventData
}

// CustomStreamEvent is the Go equivalent of Python's CustomStreamEvent
// TypedDict. It is emitted when a Runnable dispatches an "on_custom_event".
// Data is free-form and defined by the emitting Runnable.
type CustomStreamEvent struct {
	// Event is always "on_custom_event".
	Event string
	// RunID is a unique identifier for this invocation.
	RunID string
	// Name is the user-defined event name.
	Name string
	// Tags are inherited from parent Runnables.
	Tags []string
	// Metadata is inherited from parent Runnables.
	Metadata map[string]any
	// ParentIDs is the ordered list of ancestor run IDs.
	ParentIDs []string
	// Data is the free-form event payload defined by the emitter.
	Data any
}

// ConfigurableFieldSpec describes one configurable field on a Runnable.
// It is the Go equivalent of Python's ConfigurableFieldSpec from
// langchain_core.runnables.utils, used by configurable_fields and
// configurable_alternatives to expose runtime configuration options.
type ConfigurableFieldSpec struct {
	// ID is the unique field identifier (typically lowercase with underscores).
	ID string
	// Annotation is a human-readable type hint for the field.
	Annotation string
	// Name is an optional human-readable display name.
	Name string
	// Description is an optional human-readable description.
	Description string
	// Default is the default value when the field is not set at runtime.
	Default any
	// IsShared indicates whether the field is shared across all instances
	// of the Runnable within a single chain. Shared fields appear once in
	// the merged config schema.
	IsShared bool
	// Dependencies lists IDs of other configurable fields that must be set
	// before this one (for ordered initialization).
	Dependencies []string
}
