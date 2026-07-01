package language

import (
	"context"

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
	defer stream.Close()

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
}

func newChunkProtocolBridge(
	stream *streamevents.ChatModelStream,
	collector *protocolCollector,
) *chunkProtocolBridge {
	return &chunkProtocolBridge{
		stream:    stream,
		collector: collector,
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
			Delta: messages.ContentBlock{
				"type": "text-delta",
				"text": chunk.Content,
			},
		})
	}
	for _, block := range chunk.ContentBlocks {
		b.pushBlock(block)
	}
	for _, call := range chunk.ToolCalls {
		b.stream.Dispatch(streamevents.Event{
			Event: streamevents.EventContentBlockFinish,
			Index: len(b.stream.Events()),
			Content: messages.ContentBlock{
				"type": "tool_call",
				"id":   call.ID,
				"name": call.Name,
				"args": call.Args,
			},
		})
	}
	for _, call := range chunk.InvalidToolCalls {
		b.stream.Dispatch(streamevents.Event{
			Event: streamevents.EventContentBlockFinish,
			Index: len(b.stream.Events()),
			Content: messages.ContentBlock{
				"type": "invalid_tool_call",
				"id":   call.ID,
				"name": call.Name,
				"args": call.Args,
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
			Content: messages.ContentBlock{
				"type": "text",
				"text": b.text,
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
		Content: messages.ContentBlock{
			"type": "text",
			"text": "",
		},
	})
}

func (b *chunkProtocolBridge) pushBlock(block messages.ContentBlock) {
	blockType, _ := block["type"].(string)
	switch blockType {
	case "text":
		if text, _ := block["text"].(string); text != "" {
			b.ensureTextStarted()
			b.text += text
			b.stream.Dispatch(streamevents.Event{
				Event: streamevents.EventContentBlockDelta,
				Index: 0,
				Delta: messages.ContentBlock{
					"type": "text-delta",
					"text": text,
				},
			})
		}
	case "reasoning":
		if reasoning, _ := block["reasoning"].(string); reasoning != "" {
			index := blockIndex(block, len(b.stream.Events()))
			b.stream.Dispatch(streamevents.Event{
				Event: streamevents.EventContentBlockDelta,
				Index: index,
				Delta: messages.ContentBlock{
					"type":      "reasoning-delta",
					"reasoning": reasoning,
				},
			})
			b.stream.Dispatch(streamevents.Event{
				Event: streamevents.EventContentBlockFinish,
				Index: index,
				Content: messages.ContentBlock{
					"type":      "reasoning",
					"reasoning": reasoning,
				},
			})
		}
	case "tool_call_chunk":
		b.stream.Dispatch(streamevents.Event{
			Event: streamevents.EventContentBlockDelta,
			Index: blockIndex(block, len(b.stream.Events())),
			Delta: messages.ContentBlock{
				"type": "tool_call_chunk",
				"id":   block["id"],
				"name": block["name"],
				"args": block["args"],
			},
		})
	default:
		if blockType != "" {
			b.stream.Dispatch(streamevents.Event{
				Event:   streamevents.EventContentBlockFinish,
				Index:   blockIndex(block, len(b.stream.Events())),
				Content: block,
			})
		}
	}
}

func blockIndex(block messages.ContentBlock, fallback int) int {
	switch index := block["index"].(type) {
	case int:
		return index
	case float64:
		return int(index)
	default:
		return fallback
	}
}
