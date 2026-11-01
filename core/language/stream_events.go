package language

import (
	"context"
	"strconv"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/streamevents"
)

// StreamEvents runs a chat model stream and projects v3 protocol events.
//
// Providers that natively emit EventChatModelProtocol callbacks feed the
// projection directly. Models that only emit legacy message chunks are bridged
// into the v3 content-block protocol so callers still observe typed text,
// reasoning, tool-call, and output projections.
func StreamEvents(
	ctx context.Context,
	model ChatModel,
	input []messages.Message,
	opts ...runnables.Option,
) (*streamevents.ChatModelStream, error) {
	projection := streamevents.NewChatModelStream()
	collector := &protocolCollector{stream: projection}

	cfg := runnables.NewConfig(opts...)
	manager := callbacks.NewManager(collector)
	if !cfg.Callbacks.Empty() {
		manager = callbacks.NewManager(cfg.Callbacks, collector)
	}

	stream, err := model.Stream(ctx, input, append(opts, runnables.WithCallbacks(manager))...)
	if err != nil {
		projection.Fail(err)
		return projection, err
	}
	defer func() { _ = stream.Close() }()

	bridge := newChunkProtocolBridge(projection, collector)
	for {
		chunk, ok, err := stream.Next(ctx)
		if err != nil {
			projection.Fail(err)
			return projection, err
		}
		if !ok {
			break
		}
		bridge.Push(chunk)
	}
	bridge.Finish()
	return projection, nil
}

type protocolCollector struct {
	stream *streamevents.ChatModelStream
	seen   bool
}

func (h *protocolCollector) HandleEvent(_ context.Context, event callbacks.Event) error {
	if event.Kind != callbacks.EventChatModelProtocol {
		return nil
	}
	protocolEvent, ok := event.Chunk.(streamevents.Event)
	if !ok {
		return nil
	}
	h.seen = true
	h.stream.Dispatch(protocolEvent)
	return nil
}

type chunkProtocolBridge struct {
	stream      *streamevents.ChatModelStream
	collector   *protocolCollector
	started     bool
	textStarted bool
	text        string
	// wireIndices maps each streaming block's key to its stable wire index
	// (#41257): blocks keyed by their source-side index (or, lacking one,
	// their type and position) keep one index for every slice instead of the
	// old len(events) fallback that drifted per event.
	wireIndices map[string]int
	nextWire    int
}

func newChunkProtocolBridge(
	stream *streamevents.ChatModelStream,
	collector *protocolCollector,
) *chunkProtocolBridge {
	return &chunkProtocolBridge{
		stream:      stream,
		collector:   collector,
		wireIndices: make(map[string]int),
		nextWire:    1, // 0 is reserved for the text block
	}
}

func (b *chunkProtocolBridge) Push(chunk messages.Message) {
	if b.collector.seen {
		return
	}
	b.ensureStarted()
	if chunk.Content != "" {
		b.ensureTextStarted()
		b.text += chunk.Content
		b.stream.Dispatch(streamevents.Event{
			Event: streamevents.EventContentBlockDelta,
			Index: 0,
			Delta: messages.NonStandardContentBlock{
				Type:  "text-delta",
				Value: map[string]any{"text": chunk.Content},
			},
		})
	}
	for i, block := range chunk.ContentBlocks {
		b.pushBlock(block, i)
	}
	for _, call := range chunk.ToolCalls {
		b.stream.Dispatch(streamevents.Event{
			Event: streamevents.EventContentBlockFinish,
			Index: b.freshIndex(),
			Content: messages.ToolCallBlock{
				ID:   call.ID,
				Name: call.Name,
				Args: call.Args,
			},
		})
	}
	for _, call := range chunk.InvalidToolCalls {
		b.stream.Dispatch(streamevents.Event{
			Event: streamevents.EventContentBlockFinish,
			Index: b.freshIndex(),
			Content: messages.NonStandardContentBlock{
				Type: "invalid_tool_call",
				Value: map[string]any{
					"id":   call.ID,
					"name": call.Name,
					"args": call.Args,
				},
			},
		})
	}
}

func (b *chunkProtocolBridge) Finish() {
	if b.stream.Done() {
		return
	}
	if b.collector.seen {
		return
	}
	if b.started && b.text != "" {
		b.stream.Dispatch(streamevents.Event{
			Event: streamevents.EventContentBlockFinish,
			Index: 0,
			Content: messages.TextBlock{
				Text: b.text,
			},
		})
	}
	b.stream.Dispatch(streamevents.Event{
		Event:  streamevents.EventMessageFinish,
		Output: messages.AI(b.text),
	})
}

func (b *chunkProtocolBridge) ensureStarted() {
	if b.started {
		return
	}
	b.started = true
	b.stream.Dispatch(streamevents.Event{Event: streamevents.EventMessageStart})
}

func (b *chunkProtocolBridge) ensureTextStarted() {
	if b.textStarted {
		return
	}
	b.textStarted = true
	b.stream.Dispatch(streamevents.Event{
		Event: streamevents.EventContentBlockStart,
		Index: 0,
		Content: messages.TextBlock{
			Text: "",
		},
	})
}

func (b *chunkProtocolBridge) pushBlock(block messages.ContentBlock, position int) {
	m := messages.BlockToMap(block)
	blockType, _ := m["type"].(string)
	switch blockType {
	case "text":
		if text, _ := m["text"].(string); text != "" {
			b.ensureTextStarted()
			b.text += text
			b.stream.Dispatch(streamevents.Event{
				Event: streamevents.EventContentBlockDelta,
				Index: 0,
				Delta: messages.NonStandardContentBlock{
					Type:  "text-delta",
					Value: map[string]any{"text": text},
				},
			})
		}
	case "reasoning":
		if reasoning, _ := m["reasoning"].(string); reasoning != "" {
			index := b.wireIndex(m, position)
			b.stream.Dispatch(streamevents.Event{
				Event: streamevents.EventContentBlockDelta,
				Index: index,
				Delta: messages.NonStandardContentBlock{
					Type:  "reasoning-delta",
					Value: map[string]any{"reasoning": reasoning},
				},
			})
			b.stream.Dispatch(streamevents.Event{
				Event: streamevents.EventContentBlockFinish,
				Index: index,
				Content: messages.ReasoningBlock{
					Reasoning: reasoning,
				},
			})
		}
	case "tool_call_chunk":
		id, _ := m["id"].(string)
		name, _ := m["name"].(string)
		args, _ := m["args"].(string)
		index, first := b.wireIndexAlloc(m, position)
		if first {
			// The call's first slice emits the start skeleton carrying its
			// id/name (and source index when present); later slices are
			// tool-call-delta events carrying only args (#41257).
			b.stream.Dispatch(streamevents.Event{
				Event: streamevents.EventContentBlockStart,
				Index: index,
				Content: messages.ToolCallChunkBlock{
					ID:    id,
					Name:  name,
					Args:  "",
					Index: m["index"],
				},
			})
		}
		if args != "" {
			b.stream.Dispatch(streamevents.Event{
				Event: streamevents.EventContentBlockDelta,
				Index: index,
				Delta: messages.NonStandardContentBlock{
					Type:  streamevents.ToolCallDeltaType,
					Value: map[string]any{"args": args},
				},
			})
		}
	default:
		if blockType != "" {
			b.stream.Dispatch(streamevents.Event{
				Event:   streamevents.EventContentBlockFinish,
				Index:   b.wireIndex(m, position),
				Content: block,
			})
		}
	}
}

// wireIndex returns the stable wire index for a streaming block, allocating
// one on first observation. wireIndexAlloc additionally reports whether the
// index was just allocated (the caller's first observation of the block).
func (b *chunkProtocolBridge) wireIndex(m map[string]any, position int) int {
	index, _ := b.wireIndexAlloc(m, position)
	return index
}

func (b *chunkProtocolBridge) wireIndexAlloc(m map[string]any, position int) (int, bool) {
	key := blockWireKey(m, position)
	if index, ok := b.wireIndices[key]; ok {
		return index, false
	}
	index, ok := blockSourceIndex(m)
	if !ok {
		index = b.nextWire
		b.nextWire++
	}
	b.wireIndices[key] = index
	return index, true
}

// freshIndex returns a never-reused wire index for one-shot blocks that are
// finalized as they arrive (already-complete tool calls on a chunk): each
// observation is a distinct block, so it gets a distinct index.
func (b *chunkProtocolBridge) freshIndex() int {
	index := b.nextWire
	b.nextWire++
	return index
}

// blockWireKey identifies a streaming block across chunks. Blocks carrying a
// source-side index (int, float, or string) are keyed by it; blocks without
// one are bucketed by (type, position), mirroring Python's compat bridge, so
// every chunk of the same streamed block shares one key.
func blockWireKey(m map[string]any, position int) string {
	switch v := m["index"].(type) {
	case int:
		return "idx:" + strconv.Itoa(v)
	case float64:
		return "idx:" + strconv.Itoa(int(v))
	case string:
		return "idx:" + v
	default:
		blockType, _ := m["type"].(string)
		return "pos:" + blockType + ":" + strconv.Itoa(position)
	}
}

// blockSourceIndex extracts an int wire index from a block's source-side
// index field, if it carries a numeric one.
func blockSourceIndex(m map[string]any) (int, bool) {
	switch v := m["index"].(type) {
	case int:
		return v, true
	case float64:
		return int(v), true
	default:
		return 0, false
	}
}
