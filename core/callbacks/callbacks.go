package callbacks

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/outputs"
)

// EventKind identifies a callback lifecycle event.
type EventKind string

const (
	EventChatModelStart    EventKind = "chat_model_start"
	EventChatModelStream   EventKind = "chat_model_stream"
	EventChatModelProtocol EventKind = "chat_model_protocol"
	EventChatModelEnd      EventKind = "chat_model_end"
	EventChatModelError    EventKind = "chat_model_error"
	EventLLMStart          EventKind = "llm_start"
	EventLLMStream         EventKind = "llm_stream"
	EventLLMEnd            EventKind = "llm_end"
	EventLLMError          EventKind = "llm_error"
	EventToolStart         EventKind = "tool_start"
	EventToolEnd           EventKind = "tool_end"
	EventToolError         EventKind = "tool_error"
	EventRetrieverStart    EventKind = "retriever_start"
	EventRetrieverEnd      EventKind = "retriever_end"
	EventRetrieverError    EventKind = "retriever_error"
)

// Event is the normalized tracing payload emitted by runnables, models, tools,
// retrievers, and vector stores.
type Event struct {
	Kind      EventKind      `json:"kind"`
	Name      string         `json:"name,omitempty"`
	RunID     string         `json:"run_id,omitempty"`
	ParentID  string         `json:"parent_id,omitempty"`
	Tags      []string       `json:"tags,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
	Input     any            `json:"input,omitempty"`
	Output    any            `json:"output,omitempty"`
	Chunk     any            `json:"chunk,omitempty"`
	Error     string         `json:"error,omitempty"`
	Timestamp time.Time      `json:"timestamp"`
}

// Handler receives callback events.
type Handler interface {
	HandleEvent(ctx context.Context, event Event) error
}

// Manager fans callback events out to handlers.
type Manager struct {
	handlers []Handler
	tags     []string
	metadata map[string]any
	parentID string
}

// NewManager creates a callback manager.
func NewManager(handlers ...Handler) Manager {
	return Manager{handlers: append([]Handler(nil), handlers...)}
}

// WithTags returns a manager that appends inherited tags to every event.
func (m Manager) WithTags(tags ...string) Manager {
	m.tags = append(append([]string(nil), m.tags...), tags...)
	return m
}

// WithMetadata returns a manager that merges inherited metadata into every
// event. Event metadata wins on key conflicts.
func (m Manager) WithMetadata(metadata map[string]any) Manager {
	if len(metadata) == 0 {
		return m
	}
	merged := make(map[string]any, len(m.metadata)+len(metadata))
	for key, value := range m.metadata {
		merged[key] = value
	}
	for key, value := range metadata {
		merged[key] = value
	}
	m.metadata = merged
	return m
}

// WithParentRunID returns a manager that fills missing event parent IDs.
func (m Manager) WithParentRunID(parentID string) Manager {
	m.parentID = parentID
	return m
}

// Child returns a manager for child runs that inherits handlers, tags, and
// metadata while assigning a default parent run ID.
func (m Manager) Child(parentID string) Manager {
	return m.WithParentRunID(parentID)
}

// Emit sends an event to all handlers.
func (m Manager) Emit(ctx context.Context, event Event) error {
	event = m.prepareEvent(event)
	for _, handler := range m.handlers {
		if err := handler.HandleEvent(ctx, event); err != nil {
			return err
		}
	}
	return nil
}

// HandleEvent lets a Manager be nested inside another Manager.
func (m Manager) HandleEvent(ctx context.Context, event Event) error {
	return m.Emit(ctx, event)
}

// Empty reports whether the manager has handlers.
func (m Manager) Empty() bool {
	return len(m.handlers) == 0
}

func (m Manager) prepareEvent(event Event) Event {
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now().UTC()
	}
	if event.ParentID == "" {
		event.ParentID = m.parentID
	}
	if len(m.tags) > 0 || len(event.Tags) > 0 {
		tags := make([]string, 0, len(m.tags)+len(event.Tags))
		tags = append(tags, m.tags...)
		tags = append(tags, event.Tags...)
		event.Tags = tags
	}
	if len(m.metadata) > 0 || len(event.Metadata) > 0 {
		metadata := make(map[string]any, len(m.metadata)+len(event.Metadata))
		for key, value := range m.metadata {
			metadata[key] = value
		}
		for key, value := range event.Metadata {
			metadata[key] = value
		}
		event.Metadata = metadata
	}
	return event
}

// StdOutHandler writes human-readable lifecycle events to an io.Writer.
type StdOutHandler struct {
	Writer io.Writer
}

// NewStdOutHandler creates a stdout callback handler.
func NewStdOutHandler(writer io.Writer) StdOutHandler {
	if writer == nil {
		writer = os.Stdout
	}
	return StdOutHandler{Writer: writer}
}

// HandleEvent writes selected lifecycle events.
func (h StdOutHandler) HandleEvent(_ context.Context, event Event) error {
	writer := h.Writer
	if writer == nil {
		writer = os.Stdout
	}
	switch event.Kind {
	case EventChatModelStart, EventLLMStart, EventToolStart, EventRetrieverStart:
		_, err := fmt.Fprintf(writer, "\n> Entering %s %s...\n", eventLabel(event.Kind), event.Name)
		return err
	case EventChatModelEnd, EventLLMEnd, EventToolEnd, EventRetrieverEnd:
		_, err := fmt.Fprintf(writer, "> Finished %s %s.\n", eventLabel(event.Kind), event.Name)
		return err
	case EventChatModelError, EventLLMError, EventToolError, EventRetrieverError:
		_, err := fmt.Fprintf(writer, "> %s %s error: %s\n", eventLabel(event.Kind), event.Name, event.Error)
		return err
	default:
		return nil
	}
}

// StreamingStdOutHandler writes stream chunks to an io.Writer.
type StreamingStdOutHandler struct {
	Writer io.Writer
}

func NewStreamingStdOutHandler(writer io.Writer) StreamingStdOutHandler {
	if writer == nil {
		writer = os.Stdout
	}
	return StreamingStdOutHandler{Writer: writer}
}

func (h StreamingStdOutHandler) HandleEvent(_ context.Context, event Event) error {
	if event.Kind != EventChatModelStream && event.Kind != EventLLMStream {
		return nil
	}
	writer := h.Writer
	if writer == nil {
		writer = os.Stdout
	}
	_, err := io.WriteString(writer, eventText(event.Chunk))
	return err
}

// FileHandler writes callback output to an opened file.
type FileHandler struct {
	file *os.File
}

// NewFileHandler opens a callback file. The caller should call Close.
func NewFileHandler(filename string, appendMode bool) (*FileHandler, error) {
	flag := os.O_CREATE | os.O_WRONLY
	if appendMode {
		flag |= os.O_APPEND
	} else {
		flag |= os.O_TRUNC
	}
	file, err := os.OpenFile(filename, flag, 0o644)
	if err != nil {
		return nil, err
	}
	return &FileHandler{file: file}, nil
}

func (h *FileHandler) HandleEvent(ctx context.Context, event Event) error {
	if h == nil || h.file == nil {
		return fmt.Errorf("callback file is not open")
	}
	return NewStdOutHandler(h.file).HandleEvent(ctx, event)
}

func (h *FileHandler) Close() error {
	if h == nil || h.file == nil {
		return nil
	}
	err := h.file.Close()
	h.file = nil
	return err
}

// UsageMetadataHandler aggregates AI message usage by model name.
type UsageMetadataHandler struct {
	mu    sync.Mutex
	usage map[string]messages.UsageMetadata
}

func NewUsageMetadataHandler() *UsageMetadataHandler {
	return &UsageMetadataHandler{usage: map[string]messages.UsageMetadata{}}
}

func (h *UsageMetadataHandler) HandleEvent(_ context.Context, event Event) error {
	if event.Kind != EventChatModelEnd {
		return nil
	}
	for _, message := range outputMessages(event.Output) {
		if message.Role != messages.RoleAI {
			continue
		}
		modelName, _ := message.ResponseMetadata["model_name"].(string)
		if modelName == "" {
			modelName, _ = message.ResponseMetadata["model"].(string)
		}
		if modelName == "" {
			modelName, _ = event.Metadata["model_name"].(string)
		}
		if modelName == "" || message.UsageMetadata == (messages.UsageMetadata{}) {
			continue
		}
		h.add(modelName, message.UsageMetadata)
	}
	return nil
}

// Usage returns a defensive copy of usage aggregated by model name.
func (h *UsageMetadataHandler) Usage() map[string]messages.UsageMetadata {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make(map[string]messages.UsageMetadata, len(h.usage))
	for key, value := range h.usage {
		out[key] = value
	}
	return out
}

func (h *UsageMetadataHandler) add(modelName string, usage messages.UsageMetadata) {
	h.mu.Lock()
	defer h.mu.Unlock()
	current := h.usage[modelName]
	current.InputTokens += usage.InputTokens
	current.OutputTokens += usage.OutputTokens
	current.TotalTokens += usage.TotalTokens
	h.usage[modelName] = current
}

// Recorder is an in-memory callback handler for tests.
type Recorder struct {
	mu     sync.Mutex
	events []Event
}

// NewRecorder creates an event recorder.
func NewRecorder() *Recorder {
	return &Recorder{}
}

// HandleEvent records an event.
func (r *Recorder) HandleEvent(_ context.Context, event Event) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, cloneEvent(event))
	return nil
}

// Events returns recorded events.
func (r *Recorder) Events() []Event {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Event, len(r.events))
	for i, event := range r.events {
		out[i] = cloneEvent(event)
	}
	return out
}

func cloneEvent(event Event) Event {
	event.Tags = append([]string(nil), event.Tags...)
	if event.Metadata != nil {
		metadata := make(map[string]any, len(event.Metadata))
		for key, value := range event.Metadata {
			metadata[key] = value
		}
		event.Metadata = metadata
	}
	return event
}

func eventLabel(kind EventKind) string {
	value := string(kind)
	value = strings.TrimSuffix(value, "_start")
	value = strings.TrimSuffix(value, "_stream")
	value = strings.TrimSuffix(value, "_end")
	value = strings.TrimSuffix(value, "_error")
	return strings.ReplaceAll(value, "_", " ")
}

func eventText(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	case messages.Message:
		return messages.Text(v)
	case fmt.Stringer:
		return v.String()
	default:
		return fmt.Sprint(v)
	}
}

func outputMessages(value any) []messages.Message {
	switch v := value.(type) {
	case messages.Message:
		return []messages.Message{v}
	case []messages.Message:
		return v
	case outputs.ChatGeneration:
		return []messages.Message{v.Message}
	case outputs.ChatResult:
		out := make([]messages.Message, len(v.Generations))
		for i, generation := range v.Generations {
			out[i] = generation.Message
		}
		return out
	default:
		return nil
	}
}
