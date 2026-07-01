package agents

// Streaming events layer for CreateAgent (Phase 1, "events" mode).
//
// This file adds the public StreamEvent type plus the streaming entry point
// Agent.StreamEvents, and provides the helpers that wire the model node and
// tools node into the graph's event sink (graph.NodeEventSink, obtained from
// the run context via graph.EventSinkFromContext).
//
// Design (see migration_plan/streaming-design.md):
//
//   - StreamEvent reuses core/streamevents.Event as the model-chunk carrier
//     (the v3 content-block protocol) and adds agent-level fields (node/tool
//     lifecycle, the Text convenience for SSE-style callers, the assembled
//     Message for model_end, the final state for the terminal event).
//   - The graph emits only graph-internal RawEvents (node_start/node_end).
//     CreateAgent's eventSink maps those onto public StreamEvents. The
//     model/tool nodes emit their domain events directly through the same
//     sink via the typed emit methods below. This keeps the graph package's
//     public surface minimal and avoids any import cycle (graph is under
//     internal/agentruntime and cannot reach back to agents).
//   - Final-result recovery: StreamEvents emits a single terminal event of
//     Type==StreamEnd carrying the final state map (Event.State) and the
//     final assembled AI message (Event.Message). It is always the last
//     event before the stream closes.

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/streamevents"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/graph"
)

// StreamEventType identifies the kind of a StreamEvent.
type StreamEventType string

const (
	// StreamNodeStart is emitted just before a graph node is dispatched. Node
	// is set to the node name ("model", "tools", "before_agent",
	// "after_agent").
	StreamNodeStart StreamEventType = "node_start"
	// StreamNodeEnd is emitted just after a node returns. Start/end pairs are
	// always balanced per node invocation (even on error/interrupt paths).
	StreamNodeEnd StreamEventType = "node_end"
	// StreamModelDelta is emitted per model chunk. Delta carries the raw v3
	// content-block protocol event; Text is a convenience holding the text
	// delta string (empty for non-text deltas). Node is "model".
	StreamModelDelta StreamEventType = "model_delta"
	// StreamModelEnd is emitted once per model call with the fully assembled
	// AI message (text + tool calls accumulated from the stream). Message is
	// set. Node is "model".
	StreamModelEnd StreamEventType = "model_end"
	// StreamToolStart is emitted before each tool dispatch. ToolName and
	// ToolArgs are set. Node is "tools".
	StreamToolStart StreamEventType = "tool_start"
	// StreamToolEnd is emitted after each tool dispatch. ToolName and
	// ToolResult are set. Node is "tools".
	StreamToolEnd StreamEventType = "tool_end"
	// StreamEnd is the terminal event: it is emitted exactly once, as the
	// last event before the stream closes, carrying the final agent state
	// (State) and the final assembled AI message (Message). An interrupted or
	// errored run instead terminates with an event whose Err is set.
	StreamEnd StreamEventType = "end"
)

// StreamEvent is one event in an Agent.StreamEvents stream.
//
// Which fields are populated depends on Type:
//
//   - node_start / node_end:        Node
//   - model_delta:                  Node, Delta, Text
//   - model_end:                    Node, Message
//   - tool_start:                   Node, ToolName, ToolArgs
//   - tool_end:                     Node, ToolName, ToolResult
//   - end:                          State, Message (or Err on failure)
//
// Unpopulated fields retain their zero value.
type StreamEvent struct {
	// Type is the event kind (see the StreamEventType constants).
	Type StreamEventType
	// Node is the originating graph node name.
	Node string
	// Delta is the raw v3 content-block protocol event for a model_delta. Use
	// this when you need reasoning/tool_call deltas (not just text).
	Delta *streamevents.Event
	// Text is the convenience text-delta string for a model_delta (empty for
	// non-text deltas). Provided so SSE-style callers don't need to dig into
	// Delta.
	Text string
	// Message is the assembled AI message for a model_end, and the final AI
	// message for the terminal StreamEnd event.
	Message *messages.Message
	// ToolName is set for tool_start / tool_end.
	ToolName string
	// ToolArgs is set for tool_start.
	ToolArgs map[string]any
	// ToolResult is set for tool_end (the tool's structured result, if any).
	ToolResult map[string]any
	// State is the final agent state, set on the terminal StreamEnd event.
	State map[string]any
	// Err is set on a terminal event when the run ended in error or was
	// interrupted (in which case it carries the surfaced error/interrupt).
	Err error
}

// StreamEvents runs the agent over msgs and returns a pull-based stream of
// StreamEvents that the caller drains via Next until it returns (_, false, nil).
//
// Event ordering (per the design spec):
//
//  1. node_start/node_end pairs around every dispatched node (before_agent,
//     model, tools, after_agent), balanced per invocation.
//  2. Within a "model" node: zero or more model_delta events (one per model
//     chunk), then exactly one model_end with the assembled AI message.
//  3. Within a "tools" node: one tool_start/tool_end pair per dispatched tool.
//  4. Exactly one terminal event (Type==StreamEnd) as the last event,
//     carrying the final state and final AI message (or Err).
//
// Final-result recovery: drain the stream until Next returns ok==false or you
// observe the StreamEnd event. The final state map is under Event.State; the
// final assembled AI message (the last AI message in State["messages"]) is
// under Event.Message for convenience. On an interrupted/errored run, the
// terminal event carries Err instead.
//
// Goroutine lifecycle: StreamEvents spawns one goroutine that drives
// graph.InvokeStream and pushes events into a channel-backed stream. Close()
// cancels the internal context (the graph's run loop unwinds via the ctx
// plumbed through runNode) and the producer goroutine then closes the channel;
// an Interrupt or parent ctx cancellation likewise ends the stream cleanly.
// Close is idempotent and safe to defer and to call from any goroutine.
func (a *Agent) StreamEvents(ctx context.Context, msgs []messages.Message) (runnables.Stream[StreamEvent], error) {
	if a == nil || a.Graph == nil {
		return nil, fmt.Errorf("agents: StreamEvents requires a compiled graph")
	}

	runCtx := a.withRunTags(ctx)
	if a.debug {
		slog.Info("agents: stream_events start",
			slog.String("agent_name", a.Name),
			slog.Int("input_messages", len(msgs)))
	}

	events := make(chan StreamEvent, 16)
	streamCtx, cancel := context.WithCancel(runCtx)

	// done is closed by the producer goroutine immediately before it closes
	// `events`, so emitters can select against it to avoid blocking on a send
	// whose consumer has gone away (Close / ctx cancel).
	done := make(chan struct{})
	sink := &eventSink{ch: events, done: done}

	go func() {
		defer close(events)
		defer close(done)
		result, err := a.Graph.InvokeStream(streamCtx, map[string]any{"messages": msgs}, graph.Options{}, sink)
		if streamCtx.Err() != nil {
			// Context cancelled (caller called Close, or parent ctx expired).
			return
		}
		if err != nil {
			sink.send(StreamEvent{Type: StreamEnd, Err: err})
			return
		}
		if len(result.Interrupts) > 0 {
			// Mirror InvokeWithState's "interrupted is an error" contract so
			// callers see the pause as a terminal failure (resume still goes
			// through Agent.Graph.InvokeWithOptions directly).
			interruptErr := fmt.Errorf("agents: run interrupted (%d pending interrupt(s)); use Agent.Graph directly with a checkpointer to resume", len(result.Interrupts))
			sink.send(StreamEvent{Type: StreamEnd, State: result.Values, Err: interruptErr})
			return
		}
		terminal := StreamEvent{Type: StreamEnd, State: result.Values}
		if outMsgs, _ := result.Values["messages"].([]messages.Message); len(outMsgs) > 0 {
			last := outMsgs[len(outMsgs)-1]
			terminal.Message = &last
		}
		if a.debug {
			outMsgs, _ := result.Values["messages"].([]messages.Message)
			slog.Info("agents: stream_events done",
				slog.String("agent_name", a.Name),
				slog.Int("output_messages", len(outMsgs)))
		}
		sink.send(terminal)
	}()

	return &eventStream{ch: events, cancel: cancel}, nil
}

// eventStream adapts a <-chan StreamEvent into a runnables.Stream[StreamEvent].
//
// Lifecycle: the producer goroutine (in StreamEvents) drives graph.InvokeStream
// and pushes events into the channel; on return it closes the channel. Close
// cancels the internal context, which makes InvokeStream return (the graph's
// run loop checks ctx via the node functions and the context plumbed through
// runNode), after which the producer closes the channel. Close is idempotent
// (Cancel is safe to call repeatedly) and safe to call from any goroutine;
// calling it after the stream has ended naturally is a no-op.
type eventStream struct {
	ch     <-chan StreamEvent
	cancel context.CancelFunc
}

func (s *eventStream) Next(ctx context.Context) (StreamEvent, bool, error) {
	select {
	case ev, ok := <-s.ch:
		if !ok {
			return StreamEvent{}, false, nil
		}
		return ev, true, nil
	case <-ctx.Done():
		return StreamEvent{}, false, ctx.Err()
	}
}

func (s *eventStream) Close() error {
	s.cancel()
	return nil
}

// eventSink implements graph.NodeEventSink by mapping graph-internal RawEvents
// (node_start/node_end) onto public StreamEvents, and additionally offers the
// typed emit methods (emitModelDelta/emitModelEnd/emitToolStart/emitToolEnd)
// the model/tool node builders call directly for their domain events.
//
// All sends select on a `done` channel that the producer goroutine closes
// immediately before closing the events channel on exit, so a node emitting an
// event after the consumer has gone away (Close / ctx cancel) never blocks and
// never panics on send-to-closed-channel. The events channel is buffered
// (cap 16) so under normal streaming the consumer drains it fast enough that
// the done-path is never hit.
type eventSink struct {
	ch   chan<- StreamEvent
	done <-chan struct{}
}

// EmitRawEvent implements graph.NodeEventSink, mapping the graph's internal
// node lifecycle events onto public StreamEvents.
func (s *eventSink) EmitRawEvent(raw graph.RawEvent) {
	var t StreamEventType
	switch raw.Kind {
	case graph.RawNodeStart:
		t = StreamNodeStart
	case graph.RawNodeEnd:
		t = StreamNodeEnd
	default:
		return
	}
	s.send(StreamEvent{Type: t, Node: raw.Node})
}

// emitModelDelta emits a model_delta event, deriving the Text convenience
// from a text-delta block when present.
func (s *eventSink) emitModelDelta(ev streamevents.Event) {
	if s == nil {
		return
	}
	text := ""
	if ev.Delta != nil {
		if t, ok := ev.Delta["text"].(string); ok {
			text = t
		}
	}
	delta := ev
	s.send(StreamEvent{Type: StreamModelDelta, Node: ModelNodeName, Delta: &delta, Text: text})
}

// emitModelEnd emits a model_end event with the assembled message.
func (s *eventSink) emitModelEnd(msg messages.Message) {
	if s == nil {
		return
	}
	s.send(StreamEvent{Type: StreamModelEnd, Node: ModelNodeName, Message: &msg})
}

// emitToolStart emits a tool_start event for a single tool call.
func (s *eventSink) emitToolStart(call messages.ToolCall) {
	if s == nil {
		return
	}
	s.send(StreamEvent{Type: StreamToolStart, Node: ToolsNodeName, ToolName: call.Name, ToolArgs: call.Args})
}

// emitToolEnd emits a tool_end event for a completed tool call.
func (s *eventSink) emitToolEnd(call messages.ToolCall, result map[string]any) {
	if s == nil {
		return
	}
	s.send(StreamEvent{Type: StreamToolEnd, Node: ToolsNodeName, ToolName: call.Name, ToolResult: result})
}

// send pushes ev into the channel, or returns without blocking once the
// producer has signalled shutdown (closed `done`).
func (s *eventSink) send(ev StreamEvent) {
	select {
	case s.ch <- ev:
	case <-s.done:
	}
}

// sinkFromContext returns the active *eventSink installed by StreamEvents, or
// nil when streaming is not active. The model/tool node builders use this to
// decide between the streaming and non-streaming code paths (nil → non-stream,
// zero overhead).
func sinkFromContext(ctx context.Context) *eventSink {
	sink := graph.EventSinkFromContext(ctx)
	if sink == nil {
		return nil
	}
	if es, ok := sink.(*eventSink); ok {
		return es
	}
	return nil
}

// streamChunkBridge projects legacy message-chunk streams (the
// messages.Message-per-chunk shape emitted by FakeChatModel and any partner
// model that hasn't adopted the v3 protocol callbacks) into v3 content-block
// protocol events, invoking dispatch for each so model_delta events are
// surfaced live. It mirrors core/language.chunkProtocolBridge, kept local so
// the model node can emit deltas while also accumulating via
// streamevents.ChatModelStream.
type streamChunkBridge struct {
	dispatch     func(streamevents.Event)
	started      bool
	textStarted  bool
	text         string
	eventCounter int
}

func (b *streamChunkBridge) push(chunk messages.Message) {
	b.ensureStarted()
	if chunk.Content != "" {
		b.ensureTextStarted()
		b.text += chunk.Content
		b.dispatch(streamevents.Event{
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
		idx := b.nextIndex()
		b.dispatch(streamevents.Event{
			Event: streamevents.EventContentBlockFinish,
			Index: idx,
			Content: messages.ContentBlock{
				"type": "tool_call",
				"id":   call.ID,
				"name": call.Name,
				"args": call.Args,
			},
		})
	}
	for _, call := range chunk.InvalidToolCalls {
		idx := b.nextIndex()
		b.dispatch(streamevents.Event{
			Event: streamevents.EventContentBlockFinish,
			Index: idx,
			Content: messages.ContentBlock{
				"type": "invalid_tool_call",
				"id":   call.ID,
				"name": call.Name,
				"args": call.Args,
			},
		})
	}
}

func (b *streamChunkBridge) finish() {
	if b.started && b.text != "" {
		b.dispatch(streamevents.Event{
			Event:   streamevents.EventContentBlockFinish,
			Index:   0,
			Content: messages.ContentBlock{"type": "text", "text": b.text},
		})
	}
	b.dispatch(streamevents.Event{
		Event:  streamevents.EventMessageFinish,
		Output: messages.AI(b.text),
	})
}

func (b *streamChunkBridge) ensureStarted() {
	if b.started {
		return
	}
	b.started = true
	b.dispatch(streamevents.Event{Event: streamevents.EventMessageStart})
}

func (b *streamChunkBridge) ensureTextStarted() {
	if b.textStarted {
		return
	}
	b.textStarted = true
	b.dispatch(streamevents.Event{
		Event:   streamevents.EventContentBlockStart,
		Index:   0,
		Content: messages.ContentBlock{"type": "text", "text": ""},
	})
}

func (b *streamChunkBridge) pushBlock(block messages.ContentBlock) {
	blockType, _ := block["type"].(string)
	switch blockType {
	case "text":
		if text, _ := block["text"].(string); text != "" {
			b.ensureTextStarted()
			b.text += text
			b.dispatch(streamevents.Event{
				Event: streamevents.EventContentBlockDelta,
				Index: 0,
				Delta: messages.ContentBlock{"type": "text-delta", "text": text},
			})
		}
	case "tool_call_chunk":
		b.dispatch(streamevents.Event{
			Event: streamevents.EventContentBlockDelta,
			Index: b.nextIndex(),
			Delta: messages.ContentBlock{
				"type": "tool_call_chunk",
				"id":   block["id"],
				"name": block["name"],
				"args": block["args"],
			},
		})
	default:
		if blockType != "" {
			b.dispatch(streamevents.Event{
				Event:   streamevents.EventContentBlockFinish,
				Index:   b.nextIndex(),
				Content: block,
			})
		}
	}
}

// nextIndex returns a monotonically increasing block index for non-text
// blocks, mirroring core/language's fallback behavior.
func (b *streamChunkBridge) nextIndex() int {
	b.eventCounter++
	return b.eventCounter
}
