package graph

import (
	"context"
	"errors"
	"iter"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// collectStream drains a stream iterator, returning the chunks and the first
// error yielded (if any).
func collectStream(t *testing.T, seq iter.Seq2[StreamChunk, error]) ([]StreamChunk, error) {
	t.Helper()
	var chunks []StreamChunk
	for c, err := range seq {
		if err != nil {
			return chunks, err
		}
		chunks = append(chunks, c)
	}
	return chunks, nil
}

// streamLinearGraph builds n1 -> n2 -> n3 where n1 and n3 increment v and n2
// writes nothing (a no-change superstep).
func streamLinearGraph(t *testing.T, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("n1", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"v": state["v"].(int) + 1}, nil
	})
	g.AddNode("n2", func(_ context.Context, _ map[string]any) (any, error) {
		return nil, nil
	})
	g.AddNode("n3", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"v": state["v"].(int) + 1}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", "n2")
	g.AddEdge("n2", "n3")
	g.AddEdge("n3", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

func TestStreamValuesLinear(t *testing.T) {
	cg := streamLinearGraph(t)
	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0},
		StreamOptions{Modes: []StreamMode{StreamValues}}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	want := []map[string]any{{"v": 0}, {"v": 1}, {"v": 2}} // none for n2's no-change superstep
	if len(chunks) != len(want) {
		t.Fatalf("got %d values chunks, want %d: %+v", len(chunks), len(want), chunks)
	}
	for i, c := range chunks {
		if c.Mode != StreamValues {
			t.Fatalf("chunk %d mode = %q, want values", i, c.Mode)
		}
		if c.Namespace != "" {
			t.Fatalf("chunk %d namespace = %q, want root", i, c.Namespace)
		}
		if !reflect.DeepEqual(c.Payload, want[i]) {
			t.Fatalf("chunk %d payload = %+v, want %+v", i, c.Payload, want[i])
		}
	}
}

func TestStreamUpdatesLinear(t *testing.T) {
	cg := streamLinearGraph(t)
	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0},
		StreamOptions{Modes: []StreamMode{StreamUpdates}}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	want := []map[string]any{
		{"n1": map[string]any{"v": 1}},
		{"n3": map[string]any{"v": 2}},
	}
	if len(chunks) != len(want) {
		t.Fatalf("got %d updates chunks, want %d: %+v", len(chunks), len(want), chunks)
	}
	for i, c := range chunks {
		if c.Mode != StreamUpdates {
			t.Fatalf("chunk %d mode = %q, want updates", i, c.Mode)
		}
		if !reflect.DeepEqual(c.Payload, want[i]) {
			t.Fatalf("chunk %d payload = %+v, want %+v", i, c.Payload, want[i])
		}
	}
}

// debugSeq is a compact (type, step, name) projection of a debug chunk.
type debugSeq struct {
	typ  string
	step int
	name string
}

func projectDebug(t *testing.T, chunks []StreamChunk) []debugSeq {
	t.Helper()
	var out []debugSeq
	for _, c := range chunks {
		if c.Mode != StreamDebug {
			t.Fatalf("chunk mode = %q, want debug", c.Mode)
		}
		w, ok := c.Payload.(map[string]any)
		if !ok {
			t.Fatalf("debug payload is %T, want map[string]any", c.Payload)
		}
		d := debugSeq{typ: w["type"].(string), step: w["step"].(int)}
		if w["timestamp"].(string) == "" {
			t.Fatalf("debug chunk missing timestamp: %+v", w)
		}
		switch d.typ {
		case "task", "task_result":
			d.name = w["payload"].(map[string]any)["name"].(string)
		case "checkpoint":
			md := w["payload"].(map[string]any)["metadata"].(checkpoint.Metadata)
			if md.Step != d.step {
				t.Fatalf("checkpoint wrapper step %d != metadata step %d", d.step, md.Step)
			}
		default:
			t.Fatalf("unknown debug type %q", d.typ)
		}
		out = append(out, d)
	}
	return out
}

func TestStreamDebugLinear(t *testing.T) {
	cg := streamLinearGraph(t, WithCheckpointer(checkpoint.NewMemorySaver()))
	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0},
		StreamOptions{Options: Options{ThreadID: "t"}, Modes: []StreamMode{StreamDebug}}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	want := []debugSeq{
		{"checkpoint", -1, ""},
		{"task", 0, "n1"},
		{"task_result", 0, "n1"},
		{"checkpoint", 0, ""},
		{"task", 1, "n2"},
		{"task_result", 1, "n2"},
		{"checkpoint", 1, ""},
		{"task", 2, "n3"},
		{"task_result", 2, "n3"},
		{"checkpoint", 2, ""},
	}
	got := projectDebug(t, chunks)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("debug sequence = %+v, want %+v", got, want)
	}

	// Spot-check the documented payload fields.
	for _, c := range chunks {
		w := c.Payload.(map[string]any)
		p := w["payload"].(map[string]any)
		switch w["type"] {
		case "task":
			if !reflect.DeepEqual(p["triggers"], []string{p["name"].(string)}) {
				t.Fatalf("task triggers = %+v, want [name]", p["triggers"])
			}
			if p["input"] == nil {
				t.Fatalf("task %v input is nil", p["name"])
			}
		case "task_result":
			if p["name"] == "n1" {
				if !reflect.DeepEqual(p["result"], map[string]any{"v": 1}) {
					t.Fatalf("task_result n1 result = %+v", p["result"])
				}
				if p["error"] != nil || p["interrupts"] != nil {
					t.Fatalf("task_result n1 error/interrupts = %v/%v", p["error"], p["interrupts"])
				}
			}
		case "checkpoint":
			for _, k := range []string{"config", "parent_config", "values", "metadata", "next"} {
				if _, ok := p[k]; !ok {
					t.Fatalf("checkpoint payload missing key %q: %+v", k, p)
				}
			}
		}
	}
}

// modeSeq is a compact projection used to assert the exact interleaving of a
// multi-mode stream.
func modeSeq(chunks []StreamChunk) []string {
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		s := string(c.Mode)
		if c.Mode == StreamDebug {
			s += ":" + c.Payload.(map[string]any)["type"].(string)
		}
		out = append(out, s)
	}
	return out
}

func TestStreamMultiMode(t *testing.T) {
	cg := streamLinearGraph(t, WithCheckpointer(checkpoint.NewMemorySaver()))
	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0},
		StreamOptions{Options: Options{ThreadID: "t"}, Modes: []StreamMode{StreamDebug, StreamValues, StreamUpdates}}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	// updates chunks bunch AFTER the task's debug task_result (post-wg.Wait
	// deterministic-order emission) instead of interleaving with node-time
	// events as in Python — this sequence pins that documented divergence.
	want := []string{
		"values", "debug:checkpoint",
		"debug:task", "debug:task_result", "updates", "values", "debug:checkpoint",
		"debug:task", "debug:task_result", "debug:checkpoint",
		"debug:task", "debug:task_result", "updates", "values", "debug:checkpoint",
	}
	if got := modeSeq(chunks); !reflect.DeepEqual(got, want) {
		t.Fatalf("mode sequence = %v, want %v", got, want)
	}
}

// streamFanoutInterruptGraph builds start -> {worker, pause} where worker
// completes normally and pause raises an in-node interrupt, so the paused
// checkpoint carries a completed sibling's writes to replay on resume.
func streamFanoutInterruptGraph(t *testing.T) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("start", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"v": state["v"].(int) + 1}, nil
	})
	g.AddNode("worker", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"w": "done"}, nil
	})
	g.AddNode("pause", func(ctx context.Context, _ map[string]any) (any, error) {
		v := Interrupt(ctx, "need input")
		return map[string]any{"p": v}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddEdge("start", "worker")
	g.AddEdge("start", "pause")
	g.AddEdge("worker", types.END)
	g.AddEdge("pause", types.END)
	cg, err := g.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

func TestStreamPauseInterruptChunks(t *testing.T) {
	cg := streamFanoutInterruptGraph(t)
	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0},
		StreamOptions{Options: Options{ThreadID: "t"}, Modes: []StreamMode{StreamUpdates, StreamValues}}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}

	var interruptUpdates, interruptValues bool
	for _, c := range chunks {
		p, ok := c.Payload.(map[string]any)
		if !ok {
			t.Fatalf("chunk %v payload is %T", c.Mode, c.Payload)
		}
		intr, has := p["__interrupt__"].([]types.Interrupt)
		if !has {
			continue
		}
		if len(intr) != 1 || intr[0].Value != "need input" {
			t.Fatalf("interrupts = %+v, want one 'need input'", intr)
		}
		switch c.Mode {
		case StreamUpdates:
			interruptUpdates = true
		case StreamValues:
			interruptValues = true
		}
	}
	if !interruptUpdates {
		t.Fatal("no updates chunk carrying __interrupt__")
	}
	if !interruptValues {
		t.Fatal("no values chunk carrying __interrupt__")
	}
}

func TestStreamResumeReplaysUpdates(t *testing.T) {
	cg := streamFanoutInterruptGraph(t)
	ctx := context.Background()
	if _, err := collectStream(t, cg.Stream(ctx, map[string]any{"v": 0},
		StreamOptions{Options: Options{ThreadID: "t"}, Modes: []StreamMode{StreamUpdates}})); err != nil {
		t.Fatalf("initial Stream() error = %v", err)
	}

	chunks, err := collectStream(t, cg.Stream(ctx, nil,
		StreamOptions{Options: Options{ThreadID: "t", Resume: "go"}, Modes: []StreamMode{StreamUpdates, StreamValues}}))
	if err != nil {
		t.Fatalf("resume Stream() error = %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("resume stream yielded no chunks")
	}
	// The completed sibling's writes are re-emitted as updates before the
	// resumed task's own updates.
	first := chunks[0]
	if first.Mode != StreamUpdates {
		t.Fatalf("first resume chunk mode = %q, want updates", first.Mode)
	}
	if want := (map[string]any{"worker": map[string]any{"w": "done"}}); !reflect.DeepEqual(first.Payload, want) {
		t.Fatalf("first resume chunk = %+v, want replayed %+v", first.Payload, want)
	}
	var sawPause, sawFinalValues bool
	for _, c := range chunks[1:] {
		switch c.Mode {
		case StreamUpdates:
			if reflect.DeepEqual(c.Payload, map[string]any{"pause": map[string]any{"p": "go"}}) {
				sawPause = true
			}
		case StreamValues:
			if reflect.DeepEqual(c.Payload, map[string]any{"v": 1, "w": "done", "p": "go"}) {
				sawFinalValues = true
			}
		}
	}
	if !sawPause {
		t.Fatal("no updates chunk for the resumed pause node")
	}
	if !sawFinalValues {
		t.Fatal("no final values chunk after resume")
	}
}

func streamSubgraphParent(t *testing.T, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	child := NewStateGraph()
	child.AddNode("c1", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"x": "child"}, nil
	})
	child.AddEdge(types.START, "c1")
	child.AddEdge("c1", types.END)
	childCg, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	parent := NewStateGraph()
	parent.AddNode("p1", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"v": state["v"].(int) + 1}, nil
	})
	parent.AddSubgraph("sub", childCg)
	parent.AddEdge(types.START, "p1")
	parent.AddEdge("p1", "sub")
	parent.AddEdge("sub", types.END)
	cg, err := parent.Compile(opts...)
	if err != nil {
		t.Fatalf("parent Compile() error = %v", err)
	}
	return cg
}

func TestStreamSubgraphsDroppedByDefault(t *testing.T) {
	cg := streamSubgraphParent(t)
	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0},
		StreamOptions{Modes: []StreamMode{StreamUpdates, StreamValues}}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("no chunks")
	}
	for _, c := range chunks {
		if c.Namespace != "" {
			t.Fatalf("Subgraphs:false stream yielded subgraph chunk: %+v", c)
		}
	}
}

func testStreamSubgraphNamespaces(t *testing.T, cg *CompiledGraph, opts Options) {
	t.Helper()
	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0},
		StreamOptions{Options: opts, Modes: []StreamMode{StreamUpdates, StreamValues}, Subgraphs: true}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	var childUpdates, childValues bool
	for _, c := range chunks {
		if c.Namespace == "" {
			continue
		}
		if c.Namespace != "sub" {
			t.Fatalf("subgraph chunk namespace = %q, want %q", c.Namespace, "sub")
		}
		switch c.Mode {
		case StreamUpdates:
			if reflect.DeepEqual(c.Payload, map[string]any{"c1": map[string]any{"x": "child"}}) {
				childUpdates = true
			}
		case StreamValues:
			childValues = true
		}
	}
	if !childUpdates {
		t.Fatal("no subgraph updates chunk with node-path namespace")
	}
	if !childValues {
		t.Fatal("no subgraph values chunk with node-path namespace")
	}
}

func TestStreamSubgraphNamespacesNoCheckpointer(t *testing.T) {
	testStreamSubgraphNamespaces(t, streamSubgraphParent(t), Options{})
}

func TestStreamSubgraphNamespacesWithCheckpointer(t *testing.T) {
	cg := streamSubgraphParent(t, WithCheckpointer(checkpoint.NewMemorySaver()))
	testStreamSubgraphNamespaces(t, cg, Options{ThreadID: "t"})
}

func TestStreamEarlyBreakCancels(t *testing.T) {
	g := NewStateGraph()
	var calls atomic.Int64
	g.AddNode("loop", func(_ context.Context, state map[string]any) (any, error) {
		n := calls.Add(1)
		return map[string]any{"n": n}, nil
	})
	g.AddEdge(types.START, "loop")
	g.AddEdge("loop", "loop")
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		count := 0
		for _, err := range cg.Stream(context.Background(), map[string]any{"n": int64(0)},
			StreamOptions{Modes: []StreamMode{StreamValues}}) {
			if err != nil {
				return
			}
			count++
			if count == 1 {
				break
			}
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("stream did not shut down after early break (goroutine leak)")
	}
	// After the iterator returns, the run goroutine is finished: the node must
	// not run again.
	stopped := calls.Load()
	time.Sleep(50 * time.Millisecond)
	if got := calls.Load(); got != stopped {
		t.Fatalf("node kept running after early break: %d -> %d calls", stopped, got)
	}
}

func TestStreamEmptyModes(t *testing.T) {
	cg := streamLinearGraph(t)
	_, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0}, StreamOptions{}))
	if err == nil {
		t.Fatal("expected error for empty Modes")
	}
}

func TestStreamNodeError(t *testing.T) {
	g := NewStateGraph()
	want := errors.New("boom")
	g.AddNode("bad", func(_ context.Context, _ map[string]any) (any, error) {
		return nil, want
	})
	g.AddEdge(types.START, "bad")
	g.AddEdge("bad", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	_, err = collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0},
		StreamOptions{Modes: []StreamMode{StreamValues}}))
	if !errors.Is(err, want) {
		t.Fatalf("Stream() error = %v, want %v", err, want)
	}
}
