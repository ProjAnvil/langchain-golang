package agents

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/graph"
)

// drainStream reads all events from s until it closes, asserting the stream
// does not block forever. It returns the collected events.
func drainStream(t *testing.T, s runnables.Stream[StreamEvent]) []StreamEvent {
	t.Helper()
	var out []StreamEvent
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		ev, ok, err := s.Next(ctx)
		cancel()
		if err != nil {
			t.Fatalf("stream Next error: %v", err)
		}
		if !ok {
			return out
		}
		out = append(out, ev)
	}
}

// eventTypes extracts the Type sequence from a slice of events, eliding the
// terminal StreamEnd (which every stream ends with) so callers can assert the
// meaningful prefix.
func eventTypes(events []StreamEvent) []StreamEventType {
	out := make([]StreamEventType, 0, len(events))
	for _, ev := range events {
		out = append(out, ev.Type)
	}
	return out
}

// requireTerminalEnd finds the single StreamEnd event and returns it, failing
// the test if absent or duplicated.
func requireTerminalEnd(t *testing.T, events []StreamEvent) StreamEvent {
	t.Helper()
	var end StreamEvent
	count := 0
	for _, ev := range events {
		if ev.Type == StreamEnd {
			count++
			end = ev
		}
	}
	if count != 1 {
		t.Fatalf("expected exactly one StreamEnd terminal event, got %d in %v", count, eventTypes(events))
	}
	return end
}

// streamSequenceModel is a language.ChatModel test double that supports Stream.
// It returns Responses in order, and for Stream it emits the configured chunk
// slice for the current response index (one chunk slice per response). Like
// sequenceModel it returns itself from BindTools so call state is shared across
// the repeated bind+invoke cycles the model node performs.
type streamSequenceModel struct {
	mu           sync.Mutex
	responses    []messages.Message
	streamChunks [][]messages.Message // per-response chunks; len must equal len(responses)
	idx          int
	boundTools   []coretools.Tool
	invocations  [][]messages.Message
}

func (m *streamSequenceModel) Invoke(ctx context.Context, input []messages.Message, opts ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.invocations = append(m.invocations, append([]messages.Message(nil), input...))
	if m.idx >= len(m.responses) {
		return messages.Message{}, fmt.Errorf("streamSequenceModel: no more responses (call %d)", m.idx+1)
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp, nil
}

func (m *streamSequenceModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		var err error
		out[i], err = m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (m *streamSequenceModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	m.mu.Lock()
	m.invocations = append(m.invocations, append([]messages.Message(nil), input...))
	if m.idx >= len(m.responses) {
		m.mu.Unlock()
		return nil, fmt.Errorf("streamSequenceModel: no more responses (call %d)", m.idx+1)
	}
	chunks := m.streamChunks[m.idx]
	resp := m.responses[m.idx]
	m.idx++
	m.mu.Unlock()
	if len(chunks) > 0 {
		return runnables.NewSliceStream(append([]messages.Message(nil), chunks...)), nil
	}
	return runnables.NewSliceStream([]messages.Message{resp}), nil
}

func (m *streamSequenceModel) InputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *streamSequenceModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *streamSequenceModel) BindTools(boundTools []coretools.Tool) (language.ChatModel, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.boundTools = boundTools
	return m, nil
}

func (m *streamSequenceModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true, Streaming: true}
}

// newStreamingEchoTool builds the same echo tool used elsewhere in these tests.
func newStreamingEchoTool(t *testing.T) coretools.Tool {
	t.Helper()
	tool, err := coretools.NewSimple("echo", "echoes its input", func(_ context.Context, input string) (coretools.Result, error) {
		return coretools.Result{Content: "echo:" + input}, nil
	})
	if err != nil {
		t.Fatalf("new echo tool: %v", err)
	}
	return tool
}

// --- Spec test 1: model_delta sequence + assembled message (text == concat) ---

func TestStreamEventsModelDeltaSequenceAndAssembledMessage(t *testing.T) {
	// A streaming model that emits three text chunks then ends.
	model := &streamSequenceModel{
		responses: []messages.Message{messages.AI("Hi there")},
		streamChunks: [][]messages.Message{
			{messages.AI("Hi"), messages.AI(" there"), messages.AI("") /* final empty chunk */ },
		},
	}
	agent, err := CreateAgent(model, nil)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	stream, err := agent.StreamEvents(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}
	defer stream.Close()

	events := drainStream(t, stream)

	// Collect text deltas; they should concatenate to the assembled message.
	var textDeltas []string
	var modelEnds []*messages.Message
	for _, ev := range events {
		switch ev.Type {
		case StreamModelDelta:
			textDeltas = append(textDeltas, ev.Text)
		case StreamModelEnd:
			modelEnds = append(modelEnds, ev.Message)
		}
	}

	if len(textDeltas) < 1 {
		t.Fatalf("expected at least one model_delta, got %d (events: %v)", len(textDeltas), eventTypes(events))
	}
	// The bridge emits a text-delta per non-empty chunk ("Hi", " there"); the
	// empty final chunk produces no text-delta. Concatenation must equal the
	// assembled message text.
	if got := strings.Join(textDeltas, ""); got != "Hi there" {
		t.Fatalf("concatenated deltas = %q, want %q", got, "Hi there")
	}
	if len(modelEnds) != 1 {
		t.Fatalf("expected exactly one model_end, got %d", len(modelEnds))
	}
	if modelEnds[0].Content != "Hi there" {
		t.Fatalf("assembled message content = %q, want %q", modelEnds[0].Content, "Hi there")
	}
	if modelEnds[0].Role != messages.RoleAI {
		t.Fatalf("assembled message role = %q, want %q", modelEnds[0].Role, messages.RoleAI)
	}

	end := requireTerminalEnd(t, events)
	if end.Message == nil || end.Message.Content != "Hi there" {
		t.Fatalf("terminal StreamEnd message = %#v, want content %q", end.Message, "Hi there")
	}
}

// --- Spec test 2: tool-call loop event order + stream/non-stream equivalence ---

func TestStreamEventsToolLoopEventOrderAndEquivalence(t *testing.T) {
	// Two model calls: first requests a tool, second is the final answer.
	// Each is streamed as chunks.
	model := &streamSequenceModel{
		responses: []messages.Message{
			{
				Role: messages.RoleAI,
				ToolCalls: []messages.ToolCall{
					{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
				},
			},
			messages.AI("done"),
		},
		streamChunks: [][]messages.Message{
			// First call: a single chunk carrying the tool call.
			{{
				Role:      messages.RoleAI,
				ToolCalls: []messages.ToolCall{{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}}},
			}},
			// Second call: streamed text.
			{messages.AI("do"), messages.AI("ne")},
		},
	}

	agentStream, err := CreateAgent(model, []coretools.Tool{newStreamingEchoTool(t)})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	stream, err := agentStream.StreamEvents(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}
	defer stream.Close()
	events := drainStream(t, stream)

	// Build the type sequence and assert the expected ordering.
	types := eventTypes(events)

	// Locate indices of the key event kinds to assert relative ordering.
	idxOf := func(t StreamEventType) int {
		for i, ty := range types {
			if ty == t {
				return i
			}
		}
		return -1
	}

	// Expected skeleton:
	//   node_start(model) ... model_end ... node_end(model)
	//   node_start(tools) tool_start tool_end node_end(tools)
	//   node_start(model) model_delta* model_end node_end(model)
	//   StreamEnd
	ns1 := idxOf(StreamNodeStart)
	me1 := idxOf(StreamModelEnd)
	ne1 := idxOf(StreamNodeEnd)
	if ns1 < 0 || me1 < 0 || ne1 < 0 {
		t.Fatalf("missing first model node lifecycle events: %v", types)
	}
	if !(ns1 < me1 && me1 <= ne1) {
		t.Fatalf("first model node ordering wrong: node_start(%d) < model_end(%d) <= node_end(%d), types=%v", ns1, me1, ne1, types)
	}

	ts := idxOf(StreamToolStart)
	te := idxOf(StreamToolEnd)
	if ts < 0 || te < 0 || !(ts < te) {
		t.Fatalf("tool_start/tool_end ordering wrong: start(%d) end(%d), types=%v", ts, te, types)
	}
	// tool_end must come before the third node_start (the post-tool model
	// call): occurrence 0 = first model, 1 = tools, 2 = second model.
	secondModelStart := idxFrom(types, StreamNodeStart, 2)
	if secondModelStart < 0 {
		t.Fatalf("expected a third node_start (the post-tool model call), types=%v", types)
	}
	if te >= secondModelStart {
		t.Fatalf("tool_end (%d) must precede the post-tool model node_start (%d), types=%v", te, secondModelStart, types)
	}

	// Exactly one tool_start/tool_end (one tool call).
	if countType(events, StreamToolStart) != 1 || countType(events, StreamToolEnd) != 1 {
		t.Fatalf("expected exactly one tool_start/tool_end pair, got starts=%d ends=%d",
			countType(events, StreamToolStart), countType(events, StreamToolEnd))
	}

	// tool_start must carry the tool name and args.
	var toolStartEv StreamEvent
	for _, ev := range events {
		if ev.Type == StreamToolStart {
			toolStartEv = ev
		}
	}
	if toolStartEv.ToolName != "echo" {
		t.Fatalf("tool_start name = %q, want echo", toolStartEv.ToolName)
	}
	if toolStartEv.ToolArgs["tool_input"] != "hi" {
		t.Fatalf("tool_start args = %#v, want tool_input=hi", toolStartEv.ToolArgs)
	}

	// tool_end must carry the result content.
	var toolEndEv StreamEvent
	for _, ev := range events {
		if ev.Type == StreamToolEnd {
			toolEndEv = ev
		}
	}
	if got := toolEndEv.ToolResult["content"]; got != "echo:hi" {
		t.Fatalf("tool_end result content = %v, want echo:hi", got)
	}

	// Two model_end events (one per model call) and the second's deltas must
	// concatenate to "done".
	var modelEndCount int
	var lastModelEnd *messages.Message
	var secondCallDeltas []string
	modelEndSeen := false
	for _, ev := range events {
		if ev.Type == StreamModelEnd {
			modelEndCount++
			lastModelEnd = ev.Message
			modelEndSeen = true
		}
		if ev.Type == StreamModelDelta && modelEndSeen {
			secondCallDeltas = append(secondCallDeltas, ev.Text)
		}
	}
	if modelEndCount != 2 {
		t.Fatalf("expected 2 model_end events (one per model call), got %d", modelEndCount)
	}
	if lastModelEnd == nil || lastModelEnd.Content != "done" {
		t.Fatalf("second model_end content = %#v, want done", lastModelEnd)
	}
	if got := strings.Join(secondCallDeltas, ""); got != "done" {
		t.Fatalf("second-call delta concat = %q, want done", got)
	}

	// Stream/non-stream equivalence: re-run with a fresh model of the same
	// shape via InvokeWithState and compare the final message history.
	modelInvoke := &streamSequenceModel{
		responses: []messages.Message{
			{
				Role: messages.RoleAI,
				ToolCalls: []messages.ToolCall{
					{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
				},
			},
			messages.AI("done"),
		},
		streamChunks: [][]messages.Message{
			{{
				Role:      messages.RoleAI,
				ToolCalls: []messages.ToolCall{{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}}},
			}},
			{messages.AI("done")},
		},
	}
	agentInvoke, err := CreateAgent(modelInvoke, []coretools.Tool{newStreamingEchoTool(t)})
	if err != nil {
		t.Fatalf("create invoke agent: %v", err)
	}
	invokeState, err := agentInvoke.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke with state: %v", err)
	}

	end := requireTerminalEnd(t, events)
	streamMsgs, _ := end.State["messages"].([]messages.Message)
	invokeMsgs, _ := invokeState["messages"].([]messages.Message)
	if len(streamMsgs) != len(invokeMsgs) {
		t.Fatalf("stream/non-stream message count mismatch: stream=%d invoke=%d", len(streamMsgs), len(invokeMsgs))
	}
	for i := range streamMsgs {
		if streamMsgs[i].Role != invokeMsgs[i].Role {
			t.Fatalf("message %d role mismatch: stream=%q invoke=%q", i, streamMsgs[i].Role, invokeMsgs[i].Role)
		}
		if streamMsgs[i].Content != invokeMsgs[i].Content {
			t.Fatalf("message %d content mismatch: stream=%q invoke=%q", i, streamMsgs[i].Content, invokeMsgs[i].Content)
		}
		if len(streamMsgs[i].ToolCalls) != len(invokeMsgs[i].ToolCalls) {
			t.Fatalf("message %d tool call count mismatch: stream=%d invoke=%d", i, len(streamMsgs[i].ToolCalls), len(invokeMsgs[i].ToolCalls))
		}
	}
}

// --- Spec test 3: node_start/node_end for before_agent / model / tools / after_agent ---

func TestStreamEventsNodeLifecycleBeforeAfterAgent(t *testing.T) {
	model := &streamSequenceModel{
		responses: []messages.Message{
			{
				Role: messages.RoleAI,
				ToolCalls: []messages.ToolCall{
					{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
				},
			},
			messages.AI("done"),
		},
		streamChunks: [][]messages.Message{
			{{
				Role:      messages.RoleAI,
				ToolCalls: []messages.ToolCall{{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}}},
			}},
			{messages.AI("done")},
		},
	}
	var log []string
	lifecycle := &recordingAgentLifecycleMiddleware{tag: "L", log: &log}
	agent, err := CreateAgent(model, []coretools.Tool{newStreamingEchoTool(t)}, WithAgentMiddleware(lifecycle))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	stream, err := agent.StreamEvents(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}
	defer stream.Close()
	events := drainStream(t, stream)

	// Collect node names seen via node_start / node_end.
	startNodes := map[string]int{}
	endNodes := map[string]int{}
	for _, ev := range events {
		switch ev.Type {
		case StreamNodeStart:
			startNodes[ev.Node]++
		case StreamNodeEnd:
			endNodes[ev.Node]++
		}
	}
	for _, name := range []string{BeforeAgentNodeName, ModelNodeName, ToolsNodeName, AfterAgentNodeName} {
		if startNodes[name] == 0 {
			t.Fatalf("expected node_start for %q, got start map %#v", name, startNodes)
		}
		if startNodes[name] != endNodes[name] {
			t.Fatalf("unbalanced node lifecycle for %q: starts=%d ends=%d", name, startNodes[name], endNodes[name])
		}
	}
	// before_agent / after_agent run exactly once each.
	if startNodes[BeforeAgentNodeName] != 1 {
		t.Fatalf("before_agent should start exactly once, got %d", startNodes[BeforeAgentNodeName])
	}
	if startNodes[AfterAgentNodeName] != 1 {
		t.Fatalf("after_agent should start exactly once, got %d", startNodes[AfterAgentNodeName])
	}
}

// --- Spec test 4: concurrent fan-out (Send) — events interleave but each
// node's start/end pair is balanced ---

func TestStreamEventsConcurrentFanOutBalanced(t *testing.T) {
	// Build a raw graph (not via CreateAgent) with a fan-out node using Send,
	// so we can exercise concurrent node execution + interleaved events.
	g := graph.NewStateGraph()
	g.AddReducer("out", appendStringReducer)

	var concurrentNow int32
	var maxConcurrent int32
	g.AddNode("fanout", func(_ context.Context, _ map[string]any) (any, error) {
		return nil, nil
	})
	g.AddNode("worker", func(_ context.Context, state map[string]any) (any, error) {
		n := atomic.AddInt32(&concurrentNow, 1)
		for {
			old := atomic.LoadInt32(&maxConcurrent)
			if n <= old || atomic.CompareAndSwapInt32(&maxConcurrent, old, n) {
				break
			}
		}
		defer atomic.AddInt32(&concurrentNow, -1)
		// Hold long enough to guarantee overlap under the scheduler.
		time.Sleep(10 * time.Millisecond)
		return map[string]any{"out": []string{state["subject"].(string)}}, nil
	})
	g.AddEdge(agentruntime.START, "fanout")
	g.AddConditionalEdges("fanout", func(_ context.Context, state map[string]any) ([]any, error) {
		subjects := state["subjects"].([]string)
		dests := make([]any, len(subjects))
		for i, s := range subjects {
			dests[i] = &agentruntime.Send{Node: "worker", Arg: map[string]any{"subject": s}}
		}
		return dests, nil
	})
	g.AddEdge("worker", agentruntime.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	events := make(chan graph.RawEvent, 64)
	sink := &rawEventChanSink{ch: events}
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = cg.InvokeStream(context.Background(), map[string]any{"subjects": []string{"a", "b", "c"}}, graph.Options{}, sink)
	}()

	var collected []graph.RawEvent
collect:
	for {
		select {
		case ev, ok := <-events:
			if !ok {
				break collect
			}
			collected = append(collected, ev)
		case <-done:
			// drain remaining buffered
			for {
				select {
				case ev := <-events:
					collected = append(collected, ev)
				default:
					break collect
				}
			}
		}
	}

	if maxConcurrent < 2 {
		t.Fatalf("expected concurrent worker execution (maxConcurrent>=2), got %d", maxConcurrent)
	}
	// Balance check: per-node start count == end count.
	starts := map[string]int{}
	ends := map[string]int{}
	for _, ev := range collected {
		switch ev.Kind {
		case graph.RawNodeStart:
			starts[ev.Node]++
		case graph.RawNodeEnd:
			ends[ev.Node]++
		}
	}
	for node, s := range starts {
		if ends[node] != s {
			t.Fatalf("unbalanced lifecycle for node %q: starts=%d ends=%d", node, s, ends[node])
		}
	}
	// 3 worker invocations expected.
	if starts["worker"] != 3 {
		t.Fatalf("expected 3 worker starts, got %d", starts["worker"])
	}
}

// rawEventChanSink adapts a chan graph.RawEvent to graph.NodeEventSink, for
// the raw-graph fan-out test.
type rawEventChanSink struct {
	ch chan<- graph.RawEvent
}

func (s *rawEventChanSink) EmitRawEvent(event graph.RawEvent) {
	select {
	case s.ch <- event:
	default:
	}
}

// --- Spec test 5: interrupt mid-stream ends cleanly, no leak ---

func TestStreamEventsInterruptEndsCleanly(t *testing.T) {
	model := &streamSequenceModel{
		responses: []messages.Message{messages.AI("done")},
		streamChunks: [][]messages.Message{
			{messages.AI("done")},
		},
	}
	// A BeforeModel middleware that interrupts (no checkpointer: the graph
	// surfaces the interrupt as a paused Result, which StreamEvents turns into
	// a terminal Err event).
	mw := interruptBeforeModelStreamMW{}

	agent, err := CreateAgent(model, nil, WithAgentMiddleware(mw))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	stream, err := agent.StreamEvents(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}

	events := drainStream(t, stream)
	end := requireTerminalEnd(t, events)
	if end.Err == nil {
		t.Fatalf("expected terminal Err for an interrupted run, got %#v", end)
	}
	if !strings.Contains(end.Err.Error(), "interrupted") {
		t.Fatalf("expected interrupt error, got %v", end.Err)
	}

	// Confirm Close is safe to call after the stream already ended.
	if err := stream.Close(); err != nil {
		t.Fatalf("Close after end returned error: %v", err)
	}
}

// interruptBeforeModelStreamMW interrupts once on BeforeModel.
type interruptBeforeModelStreamMW struct{}

func (interruptBeforeModelStreamMW) BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	if confirmed, _ := state["confirmed"].(bool); confirmed {
		return nil, nil
	}
	graph.Interrupt(ctx, "proceed?")
	return nil, nil
}

// --- Spec test 6: Close mid-stream stops the producer (no goroutine leak) ---

func TestStreamEventsCloseStopsProducer(t *testing.T) {
	// A model whose Stream blocks until the context is cancelled, so the only
	// way the producer goroutine exits is via Close.
	model := &blockingStreamModel{}
	agent, err := CreateAgent(model, nil)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	stream, err := agent.StreamEvents(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}

	// Give the producer a moment to start, then Close.
	time.Sleep(20 * time.Millisecond)
	if err := stream.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Next after Close must return (false, nil) once the channel drains/closes,
	// and must not block forever. Use a timeout to detect a leaked goroutine.
	done := make(chan struct{})
	go func() {
		defer close(done)
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		for {
			_, ok, err := stream.Next(ctx)
			if err != nil {
				return
			}
			if !ok {
				return
			}
		}
	}()
	select {
	case <-done:
		// good: stream ended after Close.
	case <-time.After(5 * time.Second):
		t.Fatal("stream did not end within 5s of Close — producer goroutine likely leaked")
	}
}

// blockingStreamModel blocks on Stream until ctx is cancelled, then returns an
// error (so InvokeStream returns and the producer unwinds).
type blockingStreamModel struct{}

func (m *blockingStreamModel) Invoke(ctx context.Context, input []messages.Message, opts ...runnables.Option) (messages.Message, error) {
	<-ctx.Done()
	return messages.Message{}, ctx.Err()
}
func (m *blockingStreamModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	return nil, ctx.Err()
}
func (m *blockingStreamModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	<-ctx.Done()
	return nil, ctx.Err()
}
func (m *blockingStreamModel) InputSchema() schema.Schema  { return schema.Object(map[string]schema.Schema{}) }
func (m *blockingStreamModel) OutputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *blockingStreamModel) BindTools(boundTools []coretools.Tool) (language.ChatModel, error) {
	return m, nil
}
func (m *blockingStreamModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{Streaming: true}
}

// --- helpers ---

func countType(events []StreamEvent, t StreamEventType) int {
	n := 0
	for _, ev := range events {
		if ev.Type == t {
			n++
		}
	}
	return n
}

// idxFrom returns the index of the nth (0-based) occurrence of t in types, or
// -1 if absent.
func idxFrom(types []StreamEventType, t StreamEventType, occurrence int) int {
	seen := 0
	for i, ty := range types {
		if ty == t {
			if seen == occurrence {
				return i
			}
			seen++
		}
	}
	return -1
}

func appendStringReducer(existing any, next any) (any, error) {
	var out []string
	if e, ok := existing.([]string); ok {
		out = append(out, e...)
	}
	if n, ok := next.([]string); ok {
		out = append(out, n...)
	}
	sort.Strings(out)
	return out, nil
}

// --- Spec test: WrapModelStreamHook middleware observes/rewrites each delta ---

// deltaSpy is a WrapModelStreamHook middleware that records every delta it sees
// and rewrites the text via onDelta. It deliberately ignores the inner
// transform (the redaction it performs is the terminal transform in this test).
type deltaSpy struct {
	onDelta func(string) string
}

// TransformModelStream implements middleware.WrapModelStreamHook.
func (d deltaSpy) TransformModelStream(_ middleware.DeltaTransform) middleware.DeltaTransform {
	return func(s string) string {
		return d.onDelta(s)
	}
}

// TestStreamEvents_MiddlewareTransformsDelta verifies that a middleware
// implementing WrapModelStreamHook sees each streaming model delta and can
// rewrite it — here, redacting a secret end-to-end so it never reaches the
// consumer. This is the foundation for PII streaming-delta redaction. It also
// asserts (per the task brief) that the assembled model_end text reflects the
// same transform, so model_end is consistent with the streamed deltas, AND
// that no raw surface a consumer can read (notably the content-block-finish
// event's fully-assembled ev.Delta.Content["text"], which emitModelDelta
// passes through verbatim) leaks the un-redacted text.
func TestStreamEvents_MiddlewareTransformsDelta(t *testing.T) {
	model := language.NewFakeChatModel(language.WithStreamChunks(
		messages.AI("secret-"),
		messages.AI("1234"),
	))
	var saw []string
	xform := deltaSpy{onDelta: func(s string) string {
		saw = append(saw, s)
		return strings.ReplaceAll(s, "secret", "REDACTED")
	}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(xform))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	stream, err := agent.StreamEvents(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("StreamEvents: %v", err)
	}
	defer stream.Close()

	var events []StreamEvent
	var deltaText string
	var modelEndText string
	for {
		ev, ok, err := stream.Next(context.Background())
		if err != nil || !ok {
			break
		}
		events = append(events, ev)
		switch ev.Type {
		case StreamModelDelta:
			deltaText += ev.Text
		case StreamModelEnd:
			if ev.Message != nil {
				modelEndText = ev.Message.Content
			}
		}
	}
	if strings.Contains(deltaText, "secret") {
		t.Errorf("delta text not redacted: %q", deltaText)
	}
	if got, want := deltaText, "REDACTED-1234"; got != want {
		t.Errorf("delta text = %q, want %q", got, want)
	}
	if len(saw) == 0 {
		t.Errorf("middleware never observed deltas")
	}
	// The assembled model_end message must reflect the same transform so
	// downstream consumers (state, after_model hooks) see consistent text.
	if strings.Contains(modelEndText, "secret") {
		t.Errorf("model_end text not redacted: %q", modelEndText)
	}
	if got, want := modelEndText, "REDACTED-1234"; got != want {
		t.Errorf("model_end text = %q, want %q", got, want)
	}
	// Leak guard: the legacy-bridge finish() stashes the fully-assembled text
	// in the content-block-finish event's Content["text"], and emitModelDelta
	// surfaces that raw event as StreamEvent.Delta. A consumer reading
	// StreamEvent.Delta.Content["text"] must NOT see the secret.
	for _, ev := range events {
		if ev.Type != StreamModelDelta || ev.Delta == nil || ev.Delta.Content == nil {
			continue
		}
		if ev.Delta.Content["type"] != "text" {
			continue
		}
		text, ok := ev.Delta.Content["text"].(string)
		if !ok {
			continue
		}
		if strings.Contains(text, "secret") {
			t.Errorf("raw Delta.Content[\"text\"] on finish event not redacted: %q", text)
		}
	}
}

// prefixStreamMW is a WrapModelStreamHook that COMPOSES with the inner
// transform, prepending marker to each delta. Stacking two of these reveals
// composition order: the outermost middleware's marker ends up leftmost in the
// emitted text.
type prefixStreamMW struct {
	marker string
}

// TransformModelStream implements middleware.WrapModelStreamHook.
func (m prefixStreamMW) TransformModelStream(inner middleware.DeltaTransform) middleware.DeltaTransform {
	return func(s string) string {
		return m.marker + inner(s)
	}
}

// TestStreamEvents_MiddlewareTransformOrder verifies that stacked
// WrapModelStreamHook middleware compose in WrapModelCall order — mws[0] is
// outermost at execution, matching how the same middleware list orders under
// WrapModelCall. With A (marker "A:") first and B (marker "B:") second, input
// "x" must surface as "A:B:x" (A's marker outside B's). The old forward
// composition loop produced "B:A:x" (mws[1] outermost); this test fails on
// that loop and passes on the reversed loop.
func TestStreamEvents_MiddlewareTransformOrder(t *testing.T) {
	model := language.NewFakeChatModel(language.WithStreamChunks(
		messages.AI("x"),
	))
	a := prefixStreamMW{marker: "A:"}
	b := prefixStreamMW{marker: "B:"}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(a, b))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	stream, err := agent.StreamEvents(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("StreamEvents: %v", err)
	}
	defer stream.Close()

	var deltaText string
	for {
		ev, ok, err := stream.Next(context.Background())
		if err != nil || !ok {
			break
		}
		if ev.Type == StreamModelDelta && ev.Text != "" {
			deltaText += ev.Text
		}
	}
	if got, want := deltaText, "A:B:x"; got != want {
		t.Errorf("delta text = %q, want %q (mws[0] must be outermost, matching WrapModelCall)", got, want)
	}
}

// TestStreamEvents_PIIStreamTransformer_BoundaryStraddle is the end-to-end
// check for Task 3.2: a real CreateAgent wired with a PII stream transformer
// (implementing WrapModelStreamHook) is driven through StreamEvents with a
// FakeChatModel that splits a PII pattern across chunk boundaries. Asserts:
//   - no StreamModelDelta event's Text contains the raw pattern (the per-delta
//     lookback buffer must redact across the boundary);
//   - the assembled model_end message is correctly redacted (contains the
//     redaction token, NOT the raw pattern), AND is not corrupted (no
//     duplicated leading fragment from the multi-call full-text path);
//   - the raw content-block-finish Delta.Content["text"] (which emitModelDelta
//     passes through verbatim) does not leak the raw pattern either.
func TestStreamEvents_PIIStreamTransformer_BoundaryStraddle(t *testing.T) {
	model := language.NewFakeChatModel(language.WithStreamChunks(
		messages.AI("lead TOKEN-"),
		messages.AI("ABCDEFGHIJKLMNOPQRST trail"),
	))
	xform := middleware.NewPIIStreamTransformer([]string{`TOKEN-[A-Z]{20}`})
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(xform))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	stream, err := agent.StreamEvents(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("StreamEvents: %v", err)
	}
	defer stream.Close()

	var deltaText string
	var modelEndText string
	var rawFinishTexts []string
	const rawPat = "TOKEN-ABCDEFGHIJKLMNOPQRST"
	for {
		ev, ok, err := stream.Next(context.Background())
		if err != nil || !ok {
			break
		}
		switch ev.Type {
		case StreamModelDelta:
			if ev.Text != "" {
				deltaText += ev.Text
				if strings.Contains(ev.Text, rawPat) {
					t.Errorf("delta leaked raw PII: %q", ev.Text)
				}
			}
			if ev.Delta != nil && ev.Delta.Content != nil && ev.Delta.Content["type"] == "text" {
				if text, ok := ev.Delta.Content["text"].(string); ok {
					rawFinishTexts = append(rawFinishTexts, text)
					if strings.Contains(text, rawPat) {
						t.Errorf("raw Delta.Content[text] leaked PII: %q", text)
					}
				}
			}
		case StreamModelEnd:
			if ev.Message != nil {
				modelEndText = ev.Message.Content
			}
		}
	}

	// Delta stream: the per-delta lookback buffer must withhold the raw
	// pattern. With the pattern source length as the lookback window, the
	// redacted text typically surfaces incrementally; for small windows
	// the entire redacted segment may stay in the held tail until the
	// model_end full-text reset flushes it, so we only assert the raw
	// pattern never leaks — not that a redaction token appears in deltas.
	if strings.Contains(deltaText, rawPat) {
		t.Errorf("delta stream leaked raw PII: %q", deltaText)
	}

	// model_end: must be redacted AND not corrupted. The full assembled text
	// is "lead TOKEN-ABCDEFGHIJKLMNOPQRST trail" — redacted it becomes
	// "lead [REDACTED] trail". A naive append-only buffer would corrupt this
	// to something like "lead [REDACTED][REDACTED] trail" (held-tail from
	// deltas appended to fresh full text) or "lead lead [REDACTED] trail"
	// (prefix duplicated). Guards:
	if strings.Contains(modelEndText, rawPat) {
		t.Errorf("model_end leaked raw PII: %q", modelEndText)
	}
	if !strings.Contains(modelEndText, "[REDACTED") {
		t.Errorf("model_end missing redaction token: %q", modelEndText)
	}
	if c := strings.Count(modelEndText, "lead"); c != 1 {
		t.Errorf("model_end prefix corrupted (count(lead)=%d): %q", c, modelEndText)
	}
	if c := strings.Count(modelEndText, "[REDACTED"); c != 1 {
		t.Errorf("model_end redaction token corrupted (count=%d): %q", c, modelEndText)
	}
	if !strings.HasSuffix(modelEndText, "trail") {
		t.Errorf("model_end truncated: %q", modelEndText)
	}
}
