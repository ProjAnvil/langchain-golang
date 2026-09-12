// Tests for combinator chain-event instrumentation (design t18 PR1): every
// combinator emits chain_start/chain_end (or chain_error) for its own run and
// mints child run IDs via childOptions when callbacks are active. Golden
// sequences follow .superpowers/sdd/2026-09-12-parity-catchup/t18-design.md
// section 5.
package runnables

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/schema"
)

// chainRecorder builds a recorder-backed callback manager plus the Option
// installing it.
func chainRecorder() (*callbacks.Recorder, Option) {
	recorder := callbacks.NewRecorder()
	return recorder, WithCallbacks(callbacks.NewManager(recorder))
}

// chainKindNames renders events as "kind:name" pairs for golden-sequence
// comparison. Minted run IDs are random, so identity is asserted separately.
func chainKindNames(events []callbacks.Event) []string {
	out := make([]string, len(events))
	for i, event := range events {
		out[i] = string(event.Kind) + ":" + event.Name
	}
	return out
}

// chainFind returns the first event with the given kind and name.
func chainFind(t *testing.T, events []callbacks.Event, kind callbacks.EventKind, name string) callbacks.Event {
	t.Helper()
	for _, event := range events {
		if event.Kind == kind && event.Name == name {
			return event
		}
	}
	t.Fatalf("no %s:%s event in %v", kind, name, chainKindNames(events))
	return callbacks.Event{}
}

// chainAssertSequence compares kind:name pairs against want.
func chainAssertSequence(t *testing.T, events []callbacks.Event, want ...string) {
	t.Helper()
	got := chainKindNames(events)
	if len(got) != len(want) {
		t.Fatalf("event sequence:\ngot  %v\nwant %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("event %d:\ngot  %v\nwant %v", i, got, want)
		}
	}
}

// chainAssertParenting checks that every event's ParentID resolves to the
// RunID of an earlier chain_start event (or the given root parent).
func chainAssertParenting(t *testing.T, events []callbacks.Event, rootParent string) {
	t.Helper()
	seen := map[string]bool{rootParent: true}
	for i, event := range events {
		if !seen[event.ParentID] {
			t.Fatalf("event %d (%s:%s) parent %q not seen before; events: %v",
				i, event.Kind, event.Name, event.ParentID, chainKindNames(events))
		}
		if event.Kind == callbacks.EventChainStart {
			seen[event.RunID] = true
		}
	}
}

func chainUpper() Func[string, string] {
	return NewFunc(
		func(_ context.Context, input string, _ ...Option) (string, error) {
			return strings.ToUpper(input), nil
		},
		schema.String(""), schema.String(""),
	)
}

func TestNewRunIDFormatsUUIDv4(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 1000; i++ {
		id := NewRunID()
		if len(id) != 36 {
			t.Fatalf("id %q: length %d, want 36", id, len(id))
		}
		parts := strings.Split(id, "-")
		if len(parts) != 5 {
			t.Fatalf("id %q: want 5 dash-separated groups", id)
		}
		for j, want := range []int{8, 4, 4, 4, 12} {
			if len(parts[j]) != want {
				t.Fatalf("id %q: group %d length %d, want %d", id, j, len(parts[j]), want)
			}
		}
		if id != strings.ToLower(id) {
			t.Fatalf("id %q: want lowercase hex", id)
		}
		if parts[2][0] != '4' {
			t.Fatalf("id %q: version nibble %c, want 4", id, parts[2][0])
		}
		variant := parts[3][0]
		if variant != '8' && variant != '9' && variant != 'a' && variant != 'b' {
			t.Fatalf("id %q: variant nibble %c, want 8/9/a/b", id, variant)
		}
		if seen[id] {
			t.Fatalf("duplicate id %q", id)
		}
		seen[id] = true
	}
}

func TestFuncChainEventsInvoke(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	fn := NewFunc(
		func(_ context.Context, input string, _ ...Option) (string, error) {
			return input + "!", nil
		},
		schema.String(""), schema.String(""),
	)
	output, err := fn.Invoke(context.Background(), "go", withCallbacks)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if output != "go!" {
		t.Fatalf("output %q", output)
	}
	events := recorder.Events()
	chainAssertSequence(t, events, "chain_start:func", "chain_end:func")
	start := events[0]
	end := events[1]
	if start.RunID == "" || start.RunID != end.RunID {
		t.Fatalf("run ids: start=%q end=%q", start.RunID, end.RunID)
	}
	if start.ParentID != "" || end.ParentID != "" {
		t.Fatalf("parent ids: start=%q end=%q", start.ParentID, end.ParentID)
	}
	if start.Input != "go" {
		t.Fatalf("start input %#v", start.Input)
	}
	if end.Output != "go!" {
		t.Fatalf("end output %#v", end.Output)
	}
}

func TestFuncChainEventsError(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	boom := errors.New("boom")
	fn := NewFunc(
		func(_ context.Context, _ string, _ ...Option) (string, error) {
			return "", boom
		},
		schema.String(""), schema.String(""),
	)
	if _, err := fn.Invoke(context.Background(), "x", withCallbacks); !errors.Is(err, boom) {
		t.Fatalf("invoke err %v", err)
	}
	events := recorder.Events()
	chainAssertSequence(t, events, "chain_start:func", "chain_error:func")
	if !strings.Contains(events[1].Error, "boom") {
		t.Fatalf("error payload %q", events[1].Error)
	}
}

func TestFuncChainEventsRespectUserRunID(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	_, err := chainUpper().Invoke(context.Background(), "x", withCallbacks, WithRunID("root"))
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	events := recorder.Events()
	for i, event := range events {
		if event.RunID != "root" {
			t.Fatalf("event %d run id %q, want root", i, event.RunID)
		}
	}
}

func TestFuncInstallsManagerInContext(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	var ok bool
	fn := NewFunc(
		func(ctx context.Context, _ string, _ ...Option) (string, error) {
			_, ok = callbacks.ManagerFromContext(ctx)
			return "", nil
		},
		schema.String(""), schema.String(""),
	)
	if _, err := fn.Invoke(context.Background(), "x", withCallbacks); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !ok {
		t.Fatal("ManagerFromContext not available inside Func body")
	}
	// Without callbacks the context is left untouched.
	ok = true
	fn2 := NewFunc(
		func(ctx context.Context, _ string, _ ...Option) (string, error) {
			_, ok = callbacks.ManagerFromContext(ctx)
			return "", nil
		},
		schema.String(""), schema.String(""),
	)
	if _, err := fn2.Invoke(context.Background(), "x"); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if ok {
		t.Fatal("ManagerFromContext must stay absent without callbacks")
	}
	if len(recorder.Events()) != 2 {
		t.Fatalf("recorder saw %d events, want 2", len(recorder.Events()))
	}
}

func TestFuncBatchMintsPerElementRuns(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	outputs, err := chainUpper().Batch(context.Background(), []string{"a", "b"}, withCallbacks)
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if outputs[0] != "A" || outputs[1] != "B" {
		t.Fatalf("outputs %v", outputs)
	}
	events := recorder.Events()
	chainAssertSequence(t, events,
		"chain_start:func", "chain_end:func",
		"chain_start:func", "chain_end:func",
	)
	if events[0].RunID == events[2].RunID {
		t.Fatal("parallel elements must get distinct run ids")
	}
	if events[1].Output != "A" || events[3].Output != "B" {
		t.Fatalf("element outputs: %v %v", events[1].Output, events[3].Output)
	}
}

// chainNestedModel is the universal in-library pattern of a chat model invoked
// inside a Func body — model.Invoke(ctx, input, opts...) — reading its run
// identity straight off the received config, exactly like the partner adapters'
// emit helpers (partners/openai/chatmodel.go et al.).
type chainNestedModel struct{}

func (chainNestedModel) Invoke(ctx context.Context, input string, opts ...Option) (string, error) {
	cfg := NewConfig(opts...)
	if cfg.Callbacks.Empty() {
		return input + "-model", nil
	}
	start := callbacks.Event{
		Kind:     callbacks.EventChatModelStart,
		Name:     cfg.Name,
		RunID:    cfg.RunID,
		ParentID: cfg.ParentID,
		Tags:     cfg.Tags,
		Metadata: cfg.Metadata,
		Input:    input,
	}
	if err := cfg.Callbacks.Emit(ctx, start); err != nil {
		return "", err
	}
	end := start
	end.Kind = callbacks.EventChatModelEnd
	end.Output = input + "-model"
	if err := cfg.Callbacks.Emit(ctx, end); err != nil {
		return "", err
	}
	return input + "-model", nil
}

// TestFuncInvokeForwardsChildConfigToFn pins the Python RunnableLambda
// semantics for what a Func body receives: a CHILD config, not the caller's
// opts. Python's Runnable._call_with_config pops run_id (it belongs to the
// Func's own run) and swaps the function's callbacks for
// run_manager.get_child(), so a model invoked inside the lambda starts its own
// run as a child of the lambda run. Forwarding the raw opts instead would make
// chat_model_start share the Func run's RunID — duplicate run IDs in one
// LangSmith batch and broken parent_ids in StreamEvents.
func TestFuncInvokeForwardsChildConfigToFn(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	fn := NewFunc(
		func(ctx context.Context, input string, opts ...Option) (string, error) {
			return chainNestedModel{}.Invoke(ctx, input, opts...)
		},
		schema.String(""), schema.String(""),
	)
	output, err := fn.Invoke(context.Background(), "go", withCallbacks)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if output != "go-model" {
		t.Fatalf("output %q", output)
	}
	events := recorder.Events()
	chainAssertSequence(t, events,
		"chain_start:func",
		"chat_model_start:", "chat_model_end:",
		"chain_end:func",
	)
	funcStart := events[0]
	modelStart, modelEnd := events[1], events[2]
	if funcStart.RunID == "" {
		t.Fatal("func run id must be minted")
	}
	if modelStart.RunID == "" || modelStart.RunID == funcStart.RunID {
		t.Fatalf("model run id %q must be its own, not the func run's %q",
			modelStart.RunID, funcStart.RunID)
	}
	if modelStart.ParentID != funcStart.RunID {
		t.Fatalf("model run parent %q, want the func run %q",
			modelStart.ParentID, funcStart.RunID)
	}
	if modelEnd.RunID != modelStart.RunID {
		t.Fatalf("model end run id %q, want %q (pairing)", modelEnd.RunID, modelStart.RunID)
	}
}

// TestFuncInvokeFnConfigWithoutCallbacks pins the no-callbacks derivation: the
// Func run consumes the incoming RunID (Python's config.pop("run_id")), so the
// function sees it shifted to ParentID and no fresh mint.
func TestFuncInvokeFnConfigWithoutCallbacks(t *testing.T) {
	var seen Config
	fn := NewFunc(
		func(_ context.Context, _ string, opts ...Option) (string, error) {
			seen = NewConfig(opts...)
			return "ok", nil
		},
		schema.String(""), schema.String(""),
	)
	if _, err := fn.Invoke(context.Background(), "x", WithRunID("root"), WithTags("t"), WithMetadata("k", "v")); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if seen.RunID != "" {
		t.Fatalf("fn saw RunID %q; the Func run must have consumed it", seen.RunID)
	}
	if seen.ParentID != "root" {
		t.Fatalf("fn ParentID %q, want the consumed run id root", seen.ParentID)
	}
	if seen.Tags[0] != "t" || seen.Metadata["k"] != "v" {
		t.Fatalf("tags/metadata must pass through unchanged: %#v", seen)
	}
}

func TestPipeChainEventsInvoke(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	double := NewFunc(
		func(_ context.Context, input string, _ ...Option) (string, error) {
			return input + input, nil
		},
		schema.String(""), schema.String(""),
	)
	chain := Pipe(double, chainUpper())
	output, err := chain.Invoke(context.Background(), "go", withCallbacks)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if output != "GOGO" {
		t.Fatalf("output %q", output)
	}
	events := recorder.Events()
	chainAssertSequence(t, events,
		"chain_start:sequence",
		"chain_start:seq:step:1", "chain_end:seq:step:1",
		"chain_start:seq:step:2", "chain_end:seq:step:2",
		"chain_end:sequence",
	)
	chainAssertParenting(t, events, "")
	self := events[0]
	if self.RunID == "" {
		t.Fatal("sequence run id must be minted")
	}
	if self.Input != "go" {
		t.Fatalf("self input %#v", self.Input)
	}
	if events[5].Output != "GOGO" {
		t.Fatalf("self output %#v", events[5].Output)
	}
	step1 := events[1]
	if step1.ParentID != self.RunID {
		t.Fatalf("step 1 parent %q, want %q", step1.ParentID, self.RunID)
	}
	step2 := events[3]
	if step2.ParentID != self.RunID {
		t.Fatalf("step 2 parent %q, want %q", step2.ParentID, self.RunID)
	}
	if step1.RunID == step2.RunID {
		t.Fatal("steps must get distinct run ids")
	}
}

func TestPipeChainEventsRespectUserRunID(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	chain := Pipe(chainUpper(), chainUpper())
	if _, err := chain.Invoke(context.Background(), "x", withCallbacks, WithRunID("root")); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	events := recorder.Events()
	self := chainFind(t, events, callbacks.EventChainStart, "sequence")
	if self.RunID != "root" {
		t.Fatalf("self run id %q, want root", self.RunID)
	}
	for _, name := range []string{"seq:step:1", "seq:step:2"} {
		child := chainFind(t, events, callbacks.EventChainStart, name)
		if child.ParentID != "root" {
			t.Fatalf("%s parent %q, want root", name, child.ParentID)
		}
	}
}

// chainChunkSource is a runnable that streams fixed chunks; it is not itself
// instrumented, so any chain events observed while streaming it belong to the
// surrounding combinator.
type chainChunkSource struct{ chunks []string }

func (r chainChunkSource) Invoke(_ context.Context, input string, _ ...Option) (string, error) {
	return input, nil
}

func (r chainChunkSource) Batch(_ context.Context, inputs []string, _ ...Option) ([]string, error) {
	out := make([]string, len(inputs))
	copy(out, inputs)
	return out, nil
}

func (r chainChunkSource) Stream(_ context.Context, _ string, _ ...Option) (Stream[string], error) {
	return NewSliceStream(append([]string(nil), r.chunks...)), nil
}

func (r chainChunkSource) InputSchema() schema.Schema  { return schema.String("") }
func (r chainChunkSource) OutputSchema() schema.Schema { return schema.String("") }

func TestSeqStreamEmitsChainStreamChunks(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	chain := Pipe(chainChunkSource{chunks: []string{"a", "b"}}, chainUpper())
	stream, err := chain.Stream(context.Background(), "ignored", withCallbacks)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	var got []string
	for {
		value, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		got = append(got, value)
	}
	if strings.Join(got, ",") != "A,B" {
		t.Fatalf("stream values %v", got)
	}
	events := recorder.Events()
	chainAssertSequence(t, events,
		"chain_start:sequence",
		"chain_start:seq:step:2", "chain_end:seq:step:2", "chain_stream:sequence",
		"chain_start:seq:step:2", "chain_end:seq:step:2", "chain_stream:sequence",
		"chain_end:sequence",
	)
	chainAssertParenting(t, events, "")
	chunks := []callbacks.Event{}
	for _, event := range events {
		if event.Kind == callbacks.EventChainStream {
			chunks = append(chunks, event)
		}
	}
	if len(chunks) != 2 || chunks[0].Chunk != "A" || chunks[1].Chunk != "B" {
		t.Fatalf("chain_stream chunks: %+v", chunks)
	}
	self := events[0]
	for _, chunk := range chunks {
		if chunk.RunID != self.RunID {
			t.Fatalf("chunk run id %q, want %q", chunk.RunID, self.RunID)
		}
	}
}

func TestEachChainEventsPairPerElement(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	each, err := NewEach(chainUpper())
	if err != nil {
		t.Fatalf("new each: %v", err)
	}
	outputs, err := each.Invoke(context.Background(), []string{"a", "b", "c"}, withCallbacks)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if strings.Join(outputs, ",") != "A,B,C" {
		t.Fatalf("outputs %v", outputs)
	}
	events := recorder.Events()
	if events[0].Kind != callbacks.EventChainStart || events[0].Name != "each" {
		t.Fatalf("first event %s:%s", events[0].Kind, events[0].Name)
	}
	if last := events[len(events)-1]; last.Kind != callbacks.EventChainEnd || last.Name != "each" {
		t.Fatalf("last event %s:%s", last.Kind, last.Name)
	}
	self := events[0]
	if self.Input == nil {
		t.Fatal("self start must carry the slice input")
	}
	// The middle events are three child runs (start/end pairs) in any
	// interleaving; each must have a distinct run id parented to the self run
	// with start before end.
	middle := events[1 : len(events)-1]
	if len(middle) != 6 {
		t.Fatalf("middle events %v", chainKindNames(middle))
	}
	seenStart := map[string]bool{}
	runIDs := map[string]bool{}
	for _, event := range middle {
		if event.Name != "each" {
			t.Fatalf("child event name %q", event.Name)
		}
		if event.ParentID != self.RunID {
			t.Fatalf("child parent %q, want self %q", event.ParentID, self.RunID)
		}
		switch event.Kind {
		case callbacks.EventChainStart:
			if seenStart[event.RunID] {
				t.Fatalf("duplicate start for run %q", event.RunID)
			}
			seenStart[event.RunID] = true
			runIDs[event.RunID] = true
		case callbacks.EventChainEnd:
			if !seenStart[event.RunID] {
				t.Fatalf("end before start for run %q", event.RunID)
			}
		default:
			t.Fatalf("unexpected child event %s", event.Kind)
		}
	}
	if len(runIDs) != 3 {
		t.Fatalf("want 3 distinct child runs, got %d", len(runIDs))
	}
}

func TestRetryChainEventsSuccessAfterFailure(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	attempts := 0
	flaky := NewFunc(
		func(_ context.Context, input string, _ ...Option) (string, error) {
			attempts++
			if attempts == 1 {
				return "", errors.New("transient")
			}
			return input, nil
		},
		schema.String(""), schema.String(""),
	)
	retryable, err := NewRetry(flaky, 3)
	if err != nil {
		t.Fatalf("new retry: %v", err)
	}
	output, err := retryable.Invoke(context.Background(), "x", withCallbacks)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if output != "x" {
		t.Fatalf("output %q", output)
	}
	events := recorder.Events()
	chainAssertSequence(t, events,
		"chain_start:retry",
		"chain_start:retry:attempt:1", "chain_error:retry:attempt:1",
		"chain_start:retry:attempt:2", "chain_end:retry:attempt:2",
		"chain_end:retry",
	)
	chainAssertParenting(t, events, "")
}

func TestRetryChainEventsExhausted(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	failing := NewFunc(
		func(_ context.Context, _ string, _ ...Option) (string, error) {
			return "", errors.New("always")
		},
		schema.String(""), schema.String(""),
	)
	retryable, err := NewRetry(failing, 2)
	if err != nil {
		t.Fatalf("new retry: %v", err)
	}
	if _, err := retryable.Invoke(context.Background(), "x", withCallbacks); err == nil {
		t.Fatal("invoke must fail")
	}
	events := recorder.Events()
	chainAssertSequence(t, events,
		"chain_start:retry",
		"chain_start:retry:attempt:1", "chain_error:retry:attempt:1",
		"chain_start:retry:attempt:2", "chain_error:retry:attempt:2",
		"chain_error:retry",
	)
	chainAssertParenting(t, events, "")
}

func TestWithFallbacksChainEvents(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	primary := NewFunc(
		func(_ context.Context, _ string, _ ...Option) (string, error) {
			return "", errors.New("primary down")
		},
		schema.String(""), schema.String(""),
	)
	fallback := chainUpper()
	fb, err := NewWithFallbacks(primary, fallback)
	if err != nil {
		t.Fatalf("new fallbacks: %v", err)
	}
	output, err := fb.Invoke(context.Background(), "x", withCallbacks)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if output != "X" {
		t.Fatalf("output %q", output)
	}
	events := recorder.Events()
	chainAssertSequence(t, events,
		"chain_start:with_fallbacks",
		"chain_start:fallback:primary", "chain_error:fallback:primary",
		"chain_start:fallback:1", "chain_end:fallback:1",
		"chain_end:with_fallbacks",
	)
	chainAssertParenting(t, events, "")
}

func TestPickChainEvents(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	source := NewFunc(
		func(_ context.Context, _ string, _ ...Option) (map[string]any, error) {
			return map[string]any{"k": "v", "drop": 1}, nil
		},
		schema.String(""), schema.Object(nil),
	)
	pick, err := NewPick[string, any](source, "k")
	if err != nil {
		t.Fatalf("new pick: %v", err)
	}
	output, err := pick.Invoke(context.Background(), "x", withCallbacks)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if output != "v" {
		t.Fatalf("output %#v", output)
	}
	events := recorder.Events()
	// Both the Pick self run and the wrapped runnable's child run are named
	// "pick"; the self run is the first start, the child is parented to it.
	chainAssertSequence(t, events,
		"chain_start:pick",
		"chain_start:pick", "chain_end:pick",
		"chain_end:pick",
	)
	self := events[0]
	if events[1].ParentID != self.RunID {
		t.Fatalf("inner parent %q, want %q", events[1].ParentID, self.RunID)
	}
	if last := events[3]; last.Output != "v" {
		t.Fatalf("self output %#v", last.Output)
	}
}

func TestRouterChainEvents(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	router := NewRouter(map[string]Runnable[string, string]{"a": chainUpper()})
	output, err := router.Invoke(
		context.Background(),
		RouterInput[string]{Key: "a", Input: "x"},
		withCallbacks,
	)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if output != "X" {
		t.Fatalf("output %q", output)
	}
	events := recorder.Events()
	chainAssertSequence(t, events,
		"chain_start:router",
		"chain_start:route:a", "chain_end:route:a",
		"chain_end:router",
	)
	chainAssertParenting(t, events, "")
	self := events[0]
	input, ok := self.Input.(RouterInput[string])
	if !ok || input.Key != "a" || input.Input != "x" {
		t.Fatalf("router start input %#v", self.Input)
	}
}

func TestRouterChainEventsUnknownKey(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	router := NewRouter(map[string]Runnable[string, string]{})
	if _, err := router.Invoke(
		context.Background(),
		RouterInput[string]{Key: "missing", Input: "x"},
		withCallbacks,
	); err == nil {
		t.Fatal("invoke must fail")
	}
	chainAssertSequence(t, recorder.Events(), "chain_start:router", "chain_error:router")
}

func TestBranchChainEvents(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	yes := NewFunc(
		func(_ context.Context, _ string, _ ...Option) (bool, error) { return true, nil },
		schema.String(""), schema.Boolean(""),
	)
	branch, err := NewBranch([]BranchCase[string, string]{{Condition: yes, Runnable: chainUpper()}}, chainUpper())
	if err != nil {
		t.Fatalf("new branch: %v", err)
	}
	output, err := branch.Invoke(context.Background(), "x", withCallbacks)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if output != "X" {
		t.Fatalf("output %q", output)
	}
	events := recorder.Events()
	chainAssertSequence(t, events,
		"chain_start:branch",
		"chain_start:condition:1", "chain_end:condition:1",
		"chain_start:branch:1", "chain_end:branch:1",
		"chain_end:branch",
	)
	chainAssertParenting(t, events, "")
}

func TestBindEmitsNoSelfEvents(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	bound, err := Bind(chainUpper(), withCallbacks)
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	output, err := bound.Invoke(context.Background(), "x")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if output != "X" {
		t.Fatalf("output %q", output)
	}
	events := recorder.Events()
	chainAssertSequence(t, events, "chain_start:func", "chain_end:func")
}

func TestAssignChainEvents(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	assign := NewAssign(map[string]Runnable[map[string]any, any]{
		"k": NewFunc(
			func(_ context.Context, input map[string]any, _ ...Option) (any, error) {
				return fmt.Sprint(input["x"]), nil
			},
			schema.Object(nil), schema.String(""),
		),
	})
	output, err := assign.Invoke(
		context.Background(),
		map[string]any{"x": 1},
		withCallbacks,
	)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	events := recorder.Events()
	chainAssertSequence(t, events,
		"chain_start:assign",
		"chain_start:assign:key:k", "chain_end:assign:key:k",
		"chain_end:assign",
	)
	chainAssertParenting(t, events, "")
	if output["k"] != "1" {
		t.Fatalf("output %#v", output)
	}
	self := events[0]
	if events[3].RunID != self.RunID {
		t.Fatal("assign end must close the self run")
	}
}

func TestPassthroughChainEvents(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	pass := NewPassthrough[string](schema.String(""))
	output, err := pass.Invoke(context.Background(), "through", withCallbacks)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if output != "through" {
		t.Fatalf("output %q", output)
	}
	events := recorder.Events()
	chainAssertSequence(t, events, "chain_start:passthrough", "chain_end:passthrough")
	if events[1].Output != "through" {
		t.Fatalf("passthrough output %#v", events[1].Output)
	}
}

func TestNoCallbacksLeavesChildRunIDEmpty(t *testing.T) {
	seen := []Config{}
	capture := configCaptureRunnable[string, string]{output: "ok", seen: &seen}
	chain := Pipe(capture, capture)
	if _, err := chain.Invoke(context.Background(), "x", WithRunID("root")); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	for i, cfg := range seen {
		if cfg.RunID != "" || cfg.ParentID != "root" {
			t.Fatalf("child %d config: run=%q parent=%q; minting must be callbacks-gated",
				i, cfg.RunID, cfg.ParentID)
		}
	}
}

func TestChildRunIDMatchesChildEvents(t *testing.T) {
	recorder, withCallbacks := chainRecorder()
	var childConfig Config
	first := NewFunc(
		func(_ context.Context, _ string, opts ...Option) (string, error) {
			childConfig = NewConfig(opts...)
			return "ok", nil
		},
		schema.String(""), schema.String(""),
	)
	chain := Pipe(first, chainUpper())
	if _, err := chain.Invoke(context.Background(), "x", withCallbacks); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	events := recorder.Events()
	// The run ID childOptions minted into the step's config is consumed by the
	// step Func as its own run ID; the fn body then receives a DERIVED config
	// (fnChildOptions) whose ParentID is exactly that ID and whose RunID is a
	// fresh mint.
	start := chainFind(t, events, callbacks.EventChainStart, "seq:step:1")
	end := chainFind(t, events, callbacks.EventChainEnd, "seq:step:1")
	if childConfig.RunID == "" {
		t.Fatal("fn config run id must be minted when callbacks are active")
	}
	if start.RunID != end.RunID {
		t.Fatalf("step events carry %q/%q", start.RunID, end.RunID)
	}
	if childConfig.ParentID != start.RunID {
		t.Fatalf("fn config parent %q, want the step run id %q",
			childConfig.ParentID, start.RunID)
	}
	if childConfig.RunID == start.RunID {
		t.Fatalf("fn config run id %q must be a fresh mint, not the step run's",
			childConfig.RunID)
	}
}
