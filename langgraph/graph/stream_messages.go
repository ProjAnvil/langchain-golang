package graph

import (
	"context"
	"maps"
	"sort"
	"sync"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/outputs"
	"github.com/projanvil/langchain-golang/langgraph/channels"
)

// messageSeen is the run-scoped message-ID ledger messages mode dedupes
// against, mirroring the single `seen` set on Python's StreamMessagesHandler
// (one handler per stream, shared by every node of the run and its
// subgraphs): model chunks, node outputs, and node inputs all mark through
// it, so each message surfaces exactly once per stream. Safe for concurrent
// use (sibling nodes of one superstep mark concurrently).
type messageSeen struct {
	mu  sync.Mutex
	ids map[string]bool
}

// mark records id. Empty IDs never dedupe (mirroring the bridge's existing
// rule for messages without an ID) and are not recorded.
func (s *messageSeen) mark(id string) {
	if id == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ids == nil {
		s.ids = map[string]bool{}
	}
	s.ids[id] = true
}

// seen reports whether id was recorded. Empty IDs report false.
func (s *messageSeen) seen(id string) bool {
	if id == "" {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ids[id]
}

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
//
// The bridge is a raw-payload handler: a node's trace_policy must not rewrite
// (or drop) the payloads messages mode delivers to the caller — upstream
// langgraph #9486 scopes payload policies to tracers only.
//
// Node-RETURNED messages (#9463) are not emitted through the bridge: the run
// loop scans each task's outcome directly (see emitNodeMessages), against the
// same run-scoped seen ledger the bridge marks its model chunks in.
type messagesBridge struct {
	emitter  *streamEmitter
	metadata map[string]any

	seen *messageSeen
}

// messagesMetadata builds the per-node metadata Python attaches to every
// messages chunk (`pregel/_messages.py`): the emitting node's identity, the
// superstep it runs in, and its checkpoint namespace ("" for the root graph,
// the subgraph node path otherwise). Shared by the chat-model bridge and the
// node-output emission path so both stamp identical metadata.
func messagesMetadata(ns, node string, step int) map[string]any {
	return map[string]any{
		"langgraph_node":          node,
		"langgraph_step":          step,
		"langgraph_checkpoint_ns": ns,
	}
}

func newMessagesBridge(emitter *streamEmitter, node string, step int) *messagesBridge {
	return &messagesBridge{
		emitter:  emitter,
		metadata: messagesMetadata(emitter.ns, node, step),
		seen:     emitter.msgSeen,
	}
}

// RawPayloads opts the bridge out of payload-policy transforms
// (callbacks.RawPayloadHandler): messages mode delivers the node's real
// payloads to the caller, which a trace_policy — kept to shrink tracer
// ingest — must not scrub (upstream #9486).
func (h *messagesBridge) RawPayloads() bool { return true }

// HandleEvent maps EventChatModelStream/EventLLMStream chunks (Chunk =
// messages.Message for chat models, string tokens for legacy LLMs) to
// messages chunks, and EventChatModelEnd's final message likewise — deduped
// by message ID (Python parity: `StreamMessagesHandler` emits the end message
// with dedupe=True, so a final message already seen as stream chunks is not
// delivered twice; one with an unseen or empty ID is, which is how
// non-streaming models surface in messages mode).
func (h *messagesBridge) HandleEvent(ctx context.Context, event callbacks.Event) error {
	// Middleware-internal model calls are kept out of the messages
	// projection (Python langchain #39252 InternalCallTransformer; Go
	// marks them with a context token instead of metadata dicts).
	if callbacks.IsInternalCall(ctx) {
		return nil
	}
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
	h.emitter.emit(StreamMessages, MessageChunk{Message: message, Metadata: copyMessagesMetadata(h.metadata)})
}

func (h *messagesBridge) markSeen(id string) {
	h.seen.mark(id)
}

// alreadySeen reports whether id was streamed before; empty IDs never dedupe.
func (h *messagesBridge) alreadySeen(id string) bool {
	return h.seen.seen(id)
}

// copyMessagesMetadata returns a shallow copy of metadata for one delivered
// chunk (see messagesBridge.emit).
func copyMessagesMetadata(metadata map[string]any) map[string]any {
	out := make(map[string]any, len(metadata))
	maps.Copy(out, metadata)
	return out
}

// markMessagesSeen records the IDs of every message a node's input carries
// (#9463), mirroring Python's on_chain_start input scan: a node that echoes
// its input's messages back does not re-emit them, because their IDs were
// marked before the node ran. A nil emitter (messages mode inactive, or a
// nested run whose inherited carriers were stripped, #9458) is a no-op.
func (e *streamEmitter) markMessagesSeen(input map[string]any) {
	if e == nil || !e.modes[StreamMessages] || len(input) == 0 {
		return
	}
	for _, m := range nodeResultMessages(input) {
		e.msgSeen.mark(m.ID)
	}
}

// emitNodeMessages emits the messages a node's RETURN value carries through
// the run's messages stream (#9463), mirroring Python's on_chain_end ->
// `_find_and_emit_messages`: messages already streamed (model chunks, earlier
// node outputs) or marked from a node input are deduped by ID; the chunks
// carry the node's own metadata (see messagesMetadata). Called by the run
// loop per executed task outcome, before the task's updates chunk.
func (e *streamEmitter) emitNodeMessages(node string, step int, update map[string]any) {
	if e == nil || !e.modes[StreamMessages] || len(update) == 0 {
		return
	}
	metadata := messagesMetadata(e.ns, node, step)
	for _, m := range nodeResultMessages(update) {
		if e.msgSeen.seen(m.ID) {
			continue
		}
		e.msgSeen.mark(m.ID)
		e.emit(StreamMessages, MessageChunk{Message: m, Metadata: copyMessagesMetadata(metadata)})
	}
}

// nodeResultMessages extracts the messages a node result carries, mirroring
// `_find_and_emit_messages` over a state dict: every top-level value that is
// a message, or a sequence whose elements are messages. Keys are visited in
// sorted order (Go maps are unordered; Python's dicts preserve the node's
// insertion order) so emission order is deterministic. RemoveMessage
// sentinels are not messages and are skipped.
func nodeResultMessages(update map[string]any) []messages.Message {
	if len(update) == 0 {
		return nil
	}
	keys := make([]string, 0, len(update))
	for k := range update {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var out []messages.Message
	for _, k := range keys {
		out = append(out, valueMessages(update[k])...)
	}
	return out
}

// valueMessages extracts the messages from one node-result value: a message,
// a message/message-update list, or a channels.Overwrite wrapping either.
func valueMessages(v any) []messages.Message {
	switch val := v.(type) {
	case messages.Message:
		return []messages.Message{val}
	case []messages.Message:
		return val
	case []messages.MessageUpdate:
		out := make([]messages.Message, 0, len(val))
		for _, el := range val {
			if m, ok := el.(messages.Message); ok {
				out = append(out, m)
			}
		}
		return out
	case []any:
		out := make([]messages.Message, 0, len(val))
		for _, el := range val {
			if m, ok := el.(messages.Message); ok {
				out = append(out, m)
			}
		}
		return out
	default:
		if inner, ok := channels.AsOverwrite(v); ok {
			return valueMessages(inner)
		}
		return nil
	}
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
