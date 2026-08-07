package graph

import (
	"context"
	"sync"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/outputs"
)

// MessageChunk is the StreamMessages payload: one streamed message chunk
// together with the metadata Python attaches to every messages tuple
// (`pregel/_messages.py`).
type MessageChunk struct {
	// Message is the streamed token chunk (or, for a non-streaming model, the
	// final message). Legacy LLM string tokens are wrapped as AI messages.
	Message messages.Message
	// Metadata carries the emitting node's identity: `langgraph_node`,
	// `langgraph_step`, and `langgraph_checkpoint_ns` ("" for the root graph,
	// the subgraph node path otherwise).
	Metadata map[string]any
}

// messagesBridge is the callbacks.Handler the executor installs into each
// node's context (via callbacks.ContextWithManager) while StreamMessages is
// active. It maps model stream events to messages chunks on the run's
// emitter. Node code opts in by pulling the manager with
// callbacks.ManagerFromContext and fanning it into its model configs
// (runnables.WithCallbacks).
type messagesBridge struct {
	emitter  *streamEmitter
	metadata map[string]any

	mu   sync.Mutex
	seen map[string]bool
}

func newMessagesBridge(emitter *streamEmitter, node string, step int) *messagesBridge {
	return &messagesBridge{
		emitter: emitter,
		metadata: map[string]any{
			"langgraph_node":          node,
			"langgraph_step":          step,
			"langgraph_checkpoint_ns": emitter.ns,
		},
		seen: map[string]bool{},
	}
}

// HandleEvent maps EventChatModelStream/EventLLMStream chunks (Chunk =
// messages.Message for chat models, string tokens for legacy LLMs) to
// messages chunks, and EventChatModelEnd's final message likewise — deduped
// by message ID (Python parity: `StreamMessagesHandler` emits the end message
// with dedupe=True, so a final message already seen as stream chunks is not
// delivered twice; one with an unseen or empty ID is, which is how
// non-streaming models surface in messages mode).
func (h *messagesBridge) HandleEvent(_ context.Context, event callbacks.Event) error {
	switch event.Kind {
	case callbacks.EventChatModelStream, callbacks.EventLLMStream:
		chunk, ok := messageChunkMessage(event.Chunk)
		if !ok {
			return nil
		}
		h.markSeen(chunk.ID)
		h.emit(chunk)
	case callbacks.EventChatModelEnd:
		for _, message := range eventOutputMessages(event.Output) {
			if h.alreadySeen(message.ID) {
				continue
			}
			h.emit(message)
		}
	}
	return nil
}

// emit delivers one messages chunk with a fresh metadata copy (Python builds
// a new dict per emit; copying keeps later handler state out of delivered
// chunks and shields them from consumer mutation).
func (h *messagesBridge) emit(message messages.Message) {
	metadata := make(map[string]any, len(h.metadata))
	for key, value := range h.metadata {
		metadata[key] = value
	}
	h.emitter.emit(StreamMessages, MessageChunk{Message: message, Metadata: metadata})
}

func (h *messagesBridge) markSeen(id string) {
	if id == "" {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.seen[id] = true
}

// alreadySeen reports whether id was streamed before; empty IDs never dedupe.
func (h *messagesBridge) alreadySeen(id string) bool {
	if id == "" {
		return false
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.seen[id]
}

// messageChunkMessage normalizes a stream-event chunk: chat models emit
// messages.Message values; legacy LLMs emit plain string tokens, wrapped as
// AI messages.
func messageChunkMessage(chunk any) (messages.Message, bool) {
	switch v := chunk.(type) {
	case messages.Message:
		return v, true
	case string:
		return messages.AI(v), true
	default:
		return messages.Message{}, false
	}
}

// eventOutputMessages extracts the message(s) an EventChatModelEnd carries in
// its Output, covering the shapes chat models emit (message, message list,
// generation, chat result).
func eventOutputMessages(value any) []messages.Message {
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
