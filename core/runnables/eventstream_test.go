// Tests for the Runnable-level StreamEvents driver (design t18 PR2, section
// 7.1-7.4 and 7.6): golden event sequences for the instrumented combinators,
// include/exclude filter semantics, iterator-break lifecycle (no goroutine
// leaks), the chat-model legacy-chunk and v3 protocol projection paths, and
// concurrent Each pairing. The package is external (runnables_test) so the
// chat-model tests can import core/language without an import cycle.
package runnables_test

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/streamevents"
)

// ---- helpers -------------------------------------------------------------

// esCollect drains one StreamEvents iterator, failing the test on an
// unexpected final error (wantErr toggles the expectation).
func esCollect(t *testing.T, seq func(yield func(runnables.StreamEvent, error) bool), wantErr string) []runnables.StreamEvent {
	t.Helper()
	var events []runnables.StreamEvent
	var finalErr error
	for event, err := range seq {
		if err != nil {
			finalErr = err
			continue
		}
		if event.Event == "" {
			t.Fatalf("projector leaked an empty event name at index %d", len(events))
		}
		events = append(events, event)
	}
	if wantErr == "" && finalErr != nil {
		t.Fatalf("unexpected final error: %v", finalErr)
	}
	if wantErr != "" {
		if finalErr == nil || !strings.Contains(finalErr.Error(), wantErr) {
			t.Fatalf("final error %v, want it to contain %q", finalErr, wantErr)
		}
	}
	return events
}

// esNames renders events as "event:name" pairs for golden-sequence asserts.
func esNames(events []runnables.StreamEvent) []string {
	out := make([]string, len(events))
	for i, event := range events {
		out[i] = event.Event + ":" + event.Name
	}
	return out
}

func esAssertNames(t *testing.T, events []runnables.StreamEvent, want ...string) {
	t.Helper()
	got := esNames(events)
	if len(got) != len(want) {
		t.Fatalf("event sequence:\ngot  %v\nwant %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("event %d:\ngot  %v\nwant %v", i, got[i], want[i])
		}
	}
}

func esUpper() runnables.Func[string, string] {
	return runnables.NewFunc(
		func(_ context.Context, input string, _ ...runnables.Option) (string, error) {
			return strings.ToUpper(input), nil
		},
		schema.String(""), schema.String(""),
	)
}

// esDouble concatenates the input with itself, so Pipe(double, upper) turns
// "go" into "GOGO" and each step's payload is distinguishable.
func esDouble() runnables.Func[string, string] {
	return runnables.NewFunc(
		func(_ context.Context, input string, _ ...runnables.Option) (string, error) {
			return input + input, nil
		},
		schema.String(""), schema.String(""),
	)
}

func esFlakyFunc(failFirst int, msg string) runnables.Func[string, string] {
	calls := 0
	return runnables.NewFunc(
		func(_ context.Context, input string, _ ...runnables.Option) (string, error) {
			calls++
			if calls <= failFirst {
				return "", errors.New(msg)
			}
			return strings.ToUpper(input), nil
		},
		schema.String(""), schema.String(""),
	)
}

// ---- 7.1 golden sequences -------------------------------------------------

func TestStreamEventsFuncGoldenSequence(t *testing.T) {
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), esUpper(), "go", runnables.StreamEventOptions{},
	), "")
	esAssertNames(t, events, "on_chain_start:func", "on_chain_end:func")
	start, end := events[0], events[1]
	if start.RunID == "" {
		t.Fatal("root run id must be minted")
	}
	if start.RunID != end.RunID {
		t.Fatalf("run ids: start=%q end=%q", start.RunID, end.RunID)
	}
	if start.ParentIDs != nil || end.ParentIDs != nil {
		t.Fatalf("root parent ids must be nil: %v %v", start.ParentIDs, end.ParentIDs)
	}
	if start.Data.Input != "go" {
		t.Fatalf("start input %#v", start.Data.Input)
	}
	if end.Data.Output != "GO" {
		t.Fatalf("end output %#v", end.Data.Output)
	}
}

func TestStreamEventsPipeGoldenSequence(t *testing.T) {
	chain := runnables.Pipe(esDouble(), esUpper())
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), chain, "go", runnables.StreamEventOptions{},
	), "")
	esAssertNames(t, events,
		"on_chain_start:sequence",
		"on_chain_start:seq:step:1", "on_chain_end:seq:step:1",
		"on_chain_start:seq:step:2", "on_chain_end:seq:step:2",
		"on_chain_stream:sequence",
		"on_chain_end:sequence",
	)
	root := events[0]
	if root.ParentIDs != nil {
		t.Fatalf("root parent ids %v", root.ParentIDs)
	}
	if root.Data.Input != "go" {
		t.Fatalf("root start input %#v", root.Data.Input)
	}
	rootID := root.RunID
	for _, event := range events {
		switch event.Name {
		case "sequence":
			if event.ParentIDs != nil {
				t.Fatalf("sequence event parent ids %v", event.ParentIDs)
			}
			if event.RunID != rootID {
				t.Fatalf("sequence event run id %q, want %q", event.RunID, rootID)
			}
		case "seq:step:1", "seq:step:2":
			if fmt.Sprint(event.ParentIDs) != fmt.Sprint([]string{rootID}) {
				t.Fatalf("step %s parent ids %v, want [%s]", event.Name, event.ParentIDs, rootID)
			}
		}
	}
	step1, step2 := events[1], events[3]
	if step1.RunID == "" || step1.RunID != events[2].RunID {
		t.Fatalf("step 1 run ids %q/%q", step1.RunID, events[2].RunID)
	}
	if step2.RunID == "" || step2.RunID != events[4].RunID {
		t.Fatalf("step 2 run ids %q/%q", step2.RunID, events[4].RunID)
	}
	if step1.RunID == step2.RunID {
		t.Fatal("steps must get distinct run ids")
	}
	if events[1].Data.Input != "go" || events[2].Data.Output != "gogo" {
		t.Fatalf("step 1 payloads: in=%#v out=%#v", events[1].Data.Input, events[2].Data.Output)
	}
	if events[3].Data.Input != "gogo" || events[4].Data.Output != "GOGO" {
		t.Fatalf("step 2 payloads: in=%#v out=%#v", events[3].Data.Input, events[4].Data.Output)
	}
	if chunk := events[5]; chunk.Data.Chunk != "GOGO" {
		t.Fatalf("stream chunk %#v", chunk.Data.Chunk)
	}
}

func TestStreamEventsRetryGoldenSequence(t *testing.T) {
	retryable, err := runnables.NewRetry(esFlakyFunc(1, "transient"), 3)
	if err != nil {
		t.Fatalf("new retry: %v", err)
	}
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), retryable, "x", runnables.StreamEventOptions{},
	), "")
	esAssertNames(t, events,
		"on_chain_start:retry",
		"on_chain_start:retry:attempt:1", "on_chain_error:retry:attempt:1",
		"on_chain_start:retry:attempt:2", "on_chain_end:retry:attempt:2",
		"on_chain_stream:retry",
		"on_chain_end:retry",
	)
	root := events[0].RunID
	for _, event := range events[1:] {
		if event.Name != "retry" && fmt.Sprint(event.ParentIDs) != fmt.Sprint([]string{root}) {
			t.Fatalf("%s parent ids %v, want [%s]", event.Name, event.ParentIDs, root)
		}
	}
	errEvent := events[2]
	if errEvent.Data.Error == nil || !strings.Contains(errEvent.Data.Error.Error(), "transient") {
		t.Fatalf("error payload %#v", errEvent.Data.Error)
	}
}

func TestStreamEventsWithFallbacksGoldenSequence(t *testing.T) {
	fb, err := runnables.NewWithFallbacks(esFlakyFunc(99, "primary down"), esUpper())
	if err != nil {
		t.Fatalf("new fallbacks: %v", err)
	}
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), fb, "x", runnables.StreamEventOptions{},
	), "")
	esAssertNames(t, events,
		"on_chain_start:with_fallbacks",
		"on_chain_start:fallback:primary", "on_chain_error:fallback:primary",
		"on_chain_start:fallback:1", "on_chain_end:fallback:1",
		"on_chain_stream:with_fallbacks",
		"on_chain_end:with_fallbacks",
	)
	root := events[0].RunID
	if events[1].RunID == root || events[3].RunID == root || events[1].RunID == events[3].RunID {
		t.Fatal("attempt runs must be distinct children of the root run")
	}
}

func TestStreamEventsPickGoldenSequence(t *testing.T) {
	source := runnables.NewFunc(
		func(_ context.Context, _ string, _ ...runnables.Option) (map[string]any, error) {
			return map[string]any{"k": "v"}, nil
		},
		schema.String(""), schema.Object(nil),
	)
	pick, err := runnables.NewPick[string, any](source, "k")
	if err != nil {
		t.Fatalf("new pick: %v", err)
	}
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), pick, "x", runnables.StreamEventOptions{},
	), "")
	esAssertNames(t, events,
		"on_chain_start:pick",
		"on_chain_start:pick", "on_chain_end:pick",
		"on_chain_end:pick",
	)
	if last := events[3]; last.Data.Output != "v" {
		t.Fatalf("pick output %#v", last.Data.Output)
	}
}

func TestStreamEventsRouterGoldenSequence(t *testing.T) {
	router := runnables.NewRouter(map[string]runnables.Runnable[string, string]{"a": esUpper()})
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), router,
		runnables.RouterInput[string]{Key: "a", Input: "x"},
		runnables.StreamEventOptions{},
	), "")
	esAssertNames(t, events,
		"on_chain_start:router",
		"on_chain_start:route:a", "on_chain_end:route:a",
		"on_chain_stream:router",
		"on_chain_end:router",
	)
	root := events[0].RunID
	if fmt.Sprint(events[1].ParentIDs) != fmt.Sprint([]string{root}) {
		t.Fatalf("route child parent ids %v", events[1].ParentIDs)
	}
}

func TestStreamEventsBindIsTransparent(t *testing.T) {
	bound, err := runnables.Bind(esUpper())
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), bound, "x", runnables.StreamEventOptions{},
	), "")
	esAssertNames(t, events, "on_chain_start:func", "on_chain_end:func")
}

func TestStreamEventsFuncErrorYieldsTerminalError(t *testing.T) {
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), esFlakyFunc(99, "boom"), "x", runnables.StreamEventOptions{},
	), "boom")
	esAssertNames(t, events, "on_chain_start:func", "on_chain_error:func")
}

// esFailingSource streams one chunk, then fails.
type esFailingSource struct{ err error }

func (r esFailingSource) Invoke(_ context.Context, input string, _ ...runnables.Option) (string, error) {
	return input, nil
}

func (r esFailingSource) Batch(_ context.Context, inputs []string, _ ...runnables.Option) ([]string, error) {
	out := make([]string, len(inputs))
	copy(out, inputs)
	return out, nil
}

func (r esFailingSource) Stream(_ context.Context, _ string, _ ...runnables.Option) (runnables.Stream[string], error) {
	return &esFailStream{chunks: []string{"ok"}, err: r.err}, nil
}

func (r esFailingSource) InputSchema() schema.Schema  { return schema.String("") }
func (r esFailingSource) OutputSchema() schema.Schema { return schema.String("") }

type esFailStream struct {
	chunks []string
	err    error
	pulled bool
}

func (s *esFailStream) Next(_ context.Context) (string, bool, error) {
	if len(s.chunks) > 0 {
		value := s.chunks[0]
		s.chunks = nil
		return value, true, nil
	}
	if s.pulled {
		return "", false, nil
	}
	s.pulled = true
	return "", false, s.err
}

func (s *esFailStream) Close() error { return nil }

func TestStreamEventsMidStreamErrorYieldsTerminalError(t *testing.T) {
	chain := runnables.Pipe(esFailingSource{err: errors.New("wire cut")}, esUpper())
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), chain, "ignored", runnables.StreamEventOptions{},
	), "wire cut")
	esAssertNames(t, events,
		"on_chain_start:sequence",
		"on_chain_start:seq:step:2", "on_chain_end:seq:step:2",
		"on_chain_stream:sequence",
		"on_chain_error:sequence",
	)
	errEvent := events[len(events)-1]
	if errEvent.Data.Error == nil || !strings.Contains(errEvent.Data.Error.Error(), "wire cut") {
		t.Fatalf("error payload %#v", errEvent.Data.Error)
	}
}

// ---- 7.6 concurrent Each --------------------------------------------------

func TestStreamEventsEachConcurrentPairing(t *testing.T) {
	slow := runnables.NewFunc(
		func(_ context.Context, input string, _ ...runnables.Option) (string, error) {
			time.Sleep(5 * time.Millisecond)
			return strings.ToUpper(input), nil
		},
		schema.String(""), schema.String(""),
	)
	each, err := runnables.NewEach(slow)
	if err != nil {
		t.Fatalf("new each: %v", err)
	}
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), each, []string{"a", "b", "c"}, runnables.StreamEventOptions{},
	), "")
	if first := events[0]; first.Event != "on_chain_start" || first.Name != "each" {
		t.Fatalf("first event %s:%s", first.Event, first.Name)
	}
	if last := events[len(events)-1]; last.Event != "on_chain_end" || last.Name != "each" {
		t.Fatalf("last event %s:%s", last.Event, last.Name)
	}
	started := map[string]bool{}
	runIDs := map[string]bool{}
	for _, event := range events[1 : len(events)-1] {
		if event.Name != "each" {
			t.Fatalf("child event name %q", event.Name)
		}
		if len(event.ParentIDs) == 0 {
			t.Fatalf("child run %q has empty parent ids", event.RunID)
		}
		switch event.Event {
		case "on_chain_start":
			if started[event.RunID] {
				t.Fatalf("duplicate start for run %q", event.RunID)
			}
			started[event.RunID] = true
			runIDs[event.RunID] = true
		case "on_chain_end":
			if !started[event.RunID] {
				t.Fatalf("end before start for run %q (interleaving broke pairing)", event.RunID)
			}
		default:
			t.Fatalf("unexpected child event %s", event.Event)
		}
	}
	if len(runIDs) != 3 {
		t.Fatalf("want 3 distinct element runs, got %d", len(runIDs))
	}
	// All elements share one intermediate parent (the minted each-child run).
	firstParent := events[1].ParentIDs
	for _, event := range events[1 : len(events)-1] {
		if fmt.Sprint(event.ParentIDs) != fmt.Sprint(firstParent) {
			t.Fatalf("element parent ids differ: %v vs %v", event.ParentIDs, firstParent)
		}
	}
}

// ---- 7.2 filter semantics -------------------------------------------------

func TestStreamEventsNoFiltersIncludesEverything(t *testing.T) {
	chain := runnables.Pipe(esUpper(), esUpper())
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), chain, "x", runnables.StreamEventOptions{},
	), "")
	if len(events) != 7 {
		t.Fatalf("unfiltered stream yielded %d events, want 7: %v", len(events), esNames(events))
	}
}

func TestStreamEventsIncludeNamesOR(t *testing.T) {
	chain := runnables.Pipe(esUpper(), esUpper())
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), chain, "x",
		runnables.StreamEventOptions{IncludeNames: []string{"seq:step:1", "seq:step:2"}},
	), "")
	esAssertNames(t, events,
		"on_chain_start:seq:step:1", "on_chain_end:seq:step:1",
		"on_chain_start:seq:step:2", "on_chain_end:seq:step:2",
	)
}

func TestStreamEventsExcludeNames(t *testing.T) {
	chain := runnables.Pipe(esUpper(), esUpper())
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), chain, "x",
		runnables.StreamEventOptions{ExcludeNames: []string{"sequence"}},
	), "")
	esAssertNames(t, events,
		"on_chain_start:seq:step:1", "on_chain_end:seq:step:1",
		"on_chain_start:seq:step:2", "on_chain_end:seq:step:2",
	)
}

func TestStreamEventsIncludeTypesByRunType(t *testing.T) {
	model := language.NewFakeChatModel(language.WithStreamChunks(
		messages.AI("Hi"), messages.AI("!"),
	))
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), model,
		[]messages.Message{messages.Human("hello")},
		runnables.StreamEventOptions{IncludeTypes: []string{"chat_model"}},
	), "")
	esAssertNames(t, events,
		"on_chat_model_start:", "on_chat_model_stream:", "on_chat_model_stream:", "on_chat_model_end:",
	)
	// The chat model at the root emits no chain runs, so a chain-only filter
	// sees nothing.
	chainOnly := esCollect(t, runnables.StreamEvents(
		t.Context(), model,
		[]messages.Message{messages.Human("hello")},
		runnables.StreamEventOptions{IncludeTypes: []string{"chain"}},
	), "")
	if len(chainOnly) != 0 {
		t.Fatalf("chain-only filter yielded %v", esNames(chainOnly))
	}
}

func TestStreamEventsExcludeTypes(t *testing.T) {
	chain := runnables.Pipe(esUpper(), esUpper())
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), chain, "x",
		runnables.StreamEventOptions{ExcludeTypes: []string{"chain"}},
	), "")
	if len(events) != 0 {
		t.Fatalf("exclude chain yielded %v", esNames(events))
	}
}

func TestStreamEventsIncludeExcludeTags(t *testing.T) {
	chain := runnables.Pipe(esUpper(), esUpper())
	all := esCollect(t, runnables.StreamEvents(
		t.Context(), chain, "x", runnables.StreamEventOptions{},
		runnables.WithTags("red"),
	), "")
	for _, event := range all {
		want := fmt.Sprintf("event %s tags %v to contain red", event.Event, event.Tags)
		found := false
		for _, tag := range event.Tags {
			if tag == "red" {
				found = true
			}
		}
		if !found {
			t.Fatal(want)
		}
	}
	kept := esCollect(t, runnables.StreamEvents(
		t.Context(), chain, "x",
		runnables.StreamEventOptions{IncludeTags: []string{"red"}},
		runnables.WithTags("red"),
	), "")
	if len(kept) != len(all) {
		t.Fatalf("include red kept %d of %d events", len(kept), len(all))
	}
	dropped := esCollect(t, runnables.StreamEvents(
		t.Context(), chain, "x",
		runnables.StreamEventOptions{ExcludeTags: []string{"red"}},
		runnables.WithTags("red"),
	), "")
	if len(dropped) != 0 {
		t.Fatalf("exclude red yielded %v", esNames(dropped))
	}
	missed := esCollect(t, runnables.StreamEvents(
		t.Context(), chain, "x",
		runnables.StreamEventOptions{IncludeTags: []string{"blue"}},
		runnables.WithTags("red"),
	), "")
	if len(missed) != 0 {
		t.Fatalf("include blue yielded %v", esNames(missed))
	}
}

func TestStreamEventsFilteredRunKeepsSiblingParentIDs(t *testing.T) {
	retryable, err := runnables.NewRetry(esFlakyFunc(1, "transient"), 3)
	if err != nil {
		t.Fatalf("new retry: %v", err)
	}
	chain := runnables.Pipe(retryable, esUpper())
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), chain, "x",
		runnables.StreamEventOptions{ExcludeNames: []string{"retry"}},
	), "")
	// The retry run's own start/end are filtered out, but its attempts still
	// resolve their parent chain through the filtered run's id.
	for _, event := range events {
		if event.Name == "retry" {
			t.Fatalf("filtered run leaked event %s", event.Event)
		}
	}
	attempts := 0
	var wantParent string
	for _, event := range events {
		if strings.HasPrefix(event.Name, "retry:attempt") {
			attempts++
			if len(event.ParentIDs) != 2 {
				t.Fatalf("attempt %s parent ids %v, want [root, retry-run]", event.Name, event.ParentIDs)
			}
			if wantParent == "" {
				wantParent = event.ParentIDs[1]
			}
			if event.ParentIDs[1] != wantParent {
				t.Fatalf("attempt parent ids %v, want retry run %q", event.ParentIDs, wantParent)
			}
		}
	}
	if attempts == 0 {
		t.Fatal("no attempt events survived the filter")
	}
}

// ---- 7.3 iterator break lifecycle ------------------------------------------

// esBlockingSource blocks in Next until ctx is cancelled, recording Close.
type esBlockingSource struct {
	closed atomic.Bool
}

func (r *esBlockingSource) Invoke(ctx context.Context, input string, _ ...runnables.Option) (string, error) {
	stream, err := r.Stream(ctx, input)
	if err != nil {
		return "", err
	}
	value, _, err := stream.Next(ctx)
	return value, err
}

func (r *esBlockingSource) Batch(_ context.Context, inputs []string, _ ...runnables.Option) ([]string, error) {
	out := make([]string, len(inputs))
	copy(out, inputs)
	return out, nil
}

func (r *esBlockingSource) Stream(_ context.Context, _ string, _ ...runnables.Option) (runnables.Stream[string], error) {
	return &esBlockStream{source: r}, nil
}

func (r *esBlockingSource) InputSchema() schema.Schema  { return schema.String("") }
func (r *esBlockingSource) OutputSchema() schema.Schema { return schema.String("") }

type esBlockStream struct {
	source *esBlockingSource
	done   bool
}

func (s *esBlockStream) Next(ctx context.Context) (string, bool, error) {
	if s.done {
		return "", false, nil
	}
	s.done = true
	<-ctx.Done()
	return "", false, ctx.Err()
}

func (s *esBlockStream) Close() error {
	s.source.closed.Store(true)
	return nil
}

func TestStreamEventsBreakJoinsWithoutLeak(t *testing.T) {
	source := &esBlockingSource{}
	chain := runnables.Pipe(
		runnables.Runnable[string, string](source),
		esUpper(),
	)
	before := runtime.NumGoroutine()
	sawEvent := make(chan runnables.StreamEvent, 1)
	for event := range runnables.StreamEvents(
		t.Context(), chain, "ignored", runnables.StreamEventOptions{},
	) {
		sawEvent <- event
		break // consumer abandons the iterator after the first event
	}
	// The iterator must not return until the producer goroutine exited; the
	// stream Close (deferred before the done-channel close) is therefore
	// already observable — no sleep needed.
	if !source.closed.Load() {
		t.Fatal("iterator returned before the producer goroutine closed the stream (goroutine leak)")
	}
	select {
	case event := <-sawEvent:
		if event.Event != "on_chain_start" {
			t.Fatalf("first event %s:%s", event.Event, event.Name)
		}
	default:
		t.Fatal("no event yielded before break")
	}
	// Give scheduler a beat, then confirm the goroutine count settled back.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		runtime.Gosched()
		if runtime.NumGoroutine() <= before {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if after := runtime.NumGoroutine(); after > before {
		t.Fatalf("goroutine count %d after break, want <= %d before", after, before)
	}
	// The driver is reusable after an abandoned iteration.
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), esUpper(), "again", runnables.StreamEventOptions{},
	), "")
	esAssertNames(t, events, "on_chain_start:func", "on_chain_end:func")
}

func TestStreamEventsHonorsCallerContextCancellation(t *testing.T) {
	source := &esBlockingSource{}
	chain := runnables.Pipe(runnables.Runnable[string, string](source), esUpper())
	ctx, cancel := context.WithCancel(t.Context())
	events := make([]runnables.StreamEvent, 0, 1)
	var wg sync.WaitGroup
	wg.Go(func() {
		for event := range runnables.StreamEvents(ctx, chain, "ignored", runnables.StreamEventOptions{}) {
			if event.Event == "" {
				continue
			}
			events = append(events, event)
			cancel()
			// Stop consuming: a racing chain_error emission may or may not be
			// delivered after cancellation, so keep the assertion to exactly
			// the pre-cancel events.
			break
		}
	})
	wg.Wait()
	if !source.closed.Load() {
		t.Fatal("external cancellation must join the producer goroutine")
	}
	if len(events) != 1 {
		t.Fatalf("events %v", esNames(events))
	}
}

// ---- 7.4 chat-model projection ---------------------------------------------

func TestStreamEventsChatModelLegacyChunks(t *testing.T) {
	model := language.NewFakeChatModel(language.WithStreamChunks(
		messages.AI("Hello"), messages.AI(" world"),
	))
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), model,
		[]messages.Message{messages.Human("hi")},
		runnables.StreamEventOptions{},
	), "")
	esAssertNames(t, events,
		"on_chat_model_start:",
		"on_chat_model_stream:", "on_chat_model_stream:",
		"on_chat_model_end:",
	)
	start, end := events[0], events[3]
	if start.RunID == "" || start.RunID != end.RunID {
		t.Fatalf("model run ids %q/%q", start.RunID, end.RunID)
	}
	if start.ParentIDs != nil {
		t.Fatalf("root model parent ids %v", start.ParentIDs)
	}
	if start.Data.Input == nil {
		t.Fatal("start must carry the input messages")
	}
	chunks := []any{events[1].Data.Chunk, events[2].Data.Chunk}
	for i, want := range []string{"Hello", " world"} {
		chunk, ok := chunks[i].(messages.Message)
		if !ok {
			t.Fatalf("chunk %d type %T, want messages.Message (legacy passthrough)", i, chunks[i])
		}
		if chunk.Content != want {
			t.Fatalf("chunk %d content %q, want %q", i, chunk.Content, want)
		}
	}
	output, ok := end.Data.Output.(messages.Message)
	if !ok {
		t.Fatalf("end output type %T, want messages.Message", end.Data.Output)
	}
	if output.Content != "Hello world" {
		t.Fatalf("aggregated end output %q, want %q", output.Content, "Hello world")
	}
}

func TestStreamEventsChatModelInvokeOutputPassesThrough(t *testing.T) {
	model := language.NewFakeChatModel()
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), model,
		[]messages.Message{messages.Human("hi")},
		runnables.StreamEventOptions{},
	), "")
	esAssertNames(t, events, "on_chat_model_start:", "on_chat_model_end:")
	end := events[1]
	output, ok := end.Data.Output.(messages.Message)
	if !ok {
		t.Fatalf("end output type %T", end.Data.Output)
	}
	if output.Content != "fake response: hi" {
		t.Fatalf("end output %q", output.Content)
	}
}

// esV3Model is a fake chat model that mirrors a v3 provider: it emits
// chat_model_start, v3 protocol events (streamevents.Event chunks) during
// pulls, a legacy chunk alongside each delta (to prove suppression), and a
// final chat_model_end. Offline by construction: no HTTP infrastructure, the
// protocol events are dispatched straight onto the callback manager exactly
// like providerutil.EmitProtocol does.
type esV3Model struct{}

func (m esV3Model) Invoke(ctx context.Context, input []messages.Message, opts ...runnables.Option) (messages.Message, error) {
	cfg := runnables.NewConfig(opts...)
	_ = cfg.Callbacks.Emit(ctx, callbacks.Event{Kind: callbacks.EventChatModelStart, Input: input})
	output := messages.AI("Hello")
	_ = cfg.Callbacks.Emit(ctx, callbacks.Event{Kind: callbacks.EventChatModelEnd, Output: nil})
	return output, nil
}

func (m esV3Model) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, input := range inputs {
		output, err := m.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, err
		}
		out[i] = output
	}
	return out, nil
}

func (m esV3Model) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	cfg := runnables.NewConfig(opts...)
	_ = cfg.Callbacks.Emit(ctx, callbacks.Event{Kind: callbacks.EventChatModelStart, Input: input})
	return &esV3Stream{cfg: cfg, ctx: ctx}, nil
}

func (m esV3Model) InputSchema() schema.Schema {
	return schema.Schema{"type": "array", "description": "chat messages"}
}

func (m esV3Model) OutputSchema() schema.Schema { return schema.Object(nil) }

type esV3Stream struct {
	cfg  runnables.Config
	ctx  context.Context
	step int
}

func (s *esV3Stream) Next(ctx context.Context) (messages.Message, bool, error) {
	protocol := func(event streamevents.Event) {
		_ = s.cfg.Callbacks.Emit(ctx, callbacks.Event{
			Kind:  callbacks.EventChatModelProtocol,
			Chunk: event,
		})
	}
	legacy := func(chunk messages.Message) {
		_ = s.cfg.Callbacks.Emit(ctx, callbacks.Event{
			Kind:  callbacks.EventChatModelStream,
			Chunk: chunk,
		})
	}
	switch s.step {
	case 0:
		s.step = 1
		protocol(streamevents.Event{Event: streamevents.EventMessageStart})
		protocol(streamevents.Event{
			Event:   streamevents.EventContentBlockStart,
			Index:   0,
			Content: messages.TextBlock{Text: ""},
		})
		protocol(streamevents.Event{
			Event: streamevents.EventContentBlockDelta,
			Index: 0,
			Delta: messages.NonStandardContentBlock{
				Type:  "text-delta",
				Value: map[string]any{"text": "Hel"},
			},
		})
		legacy(messages.AI("Hel"))
		return messages.AI("Hel"), true, nil
	case 1:
		s.step = 2
		protocol(streamevents.Event{
			Event: streamevents.EventContentBlockDelta,
			Index: 0,
			Delta: messages.NonStandardContentBlock{
				Type:  "text-delta",
				Value: map[string]any{"text": "lo"},
			},
		})
		legacy(messages.AI("lo"))
		return messages.AI("lo"), true, nil
	case 2:
		s.step = 3
		protocol(streamevents.Event{
			Event:   streamevents.EventContentBlockFinish,
			Index:   0,
			Content: messages.TextBlock{Text: "Hello"},
		})
		protocol(streamevents.Event{
			Event:  streamevents.EventMessageFinish,
			Output: messages.AI("Hello"),
		})
		_ = s.cfg.Callbacks.Emit(ctx, callbacks.Event{Kind: callbacks.EventChatModelEnd})
		return messages.Message{}, false, nil
	default:
		return messages.Message{}, false, nil
	}
}

func (s *esV3Stream) Close() error { return nil }

func TestStreamEventsChatModelV3Protocol(t *testing.T) {
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), esV3Model{},
		[]messages.Message{messages.Human("hi")},
		runnables.StreamEventOptions{},
	), "")
	esAssertNames(t, events,
		"on_chat_model_start:",
		"on_chat_model_stream:", "on_chat_model_stream:",
		"on_chat_model_end:",
	)
	// v3 deltas win: each on_chat_model_stream chunk is the protocol delta
	// event, and the interleaved legacy chunks are suppressed.
	for i, event := range events[1:3] {
		delta, ok := event.Data.Chunk.(streamevents.Event)
		if !ok {
			t.Fatalf("stream chunk %d type %T, want streamevents.Event (v3 delta)", i, event.Data.Chunk)
		}
		if delta.Event != streamevents.EventContentBlockDelta {
			t.Fatalf("stream chunk %d protocol event %q", i, delta.Event)
		}
		m := messages.BlockToMap(delta.Delta)
		if text, _ := m["text"].(string); text != []string{"Hel", "lo"}[i] {
			t.Fatalf("stream chunk %d delta text %q", i, text)
		}
	}
	// message-start/finish fold: no extra events, and the end output comes
	// from the aggregator's assembled message.
	end := events[3]
	output, ok := end.Data.Output.(messages.Message)
	if !ok {
		t.Fatalf("end output type %T, want messages.Message", end.Data.Output)
	}
	if output.Content != "Hello" {
		t.Fatalf("aggregated end output %q, want %q", output.Content, "Hello")
	}
	for _, event := range events {
		if event.RunID != events[0].RunID {
			t.Fatalf("model run id drifted: %q vs %q", event.RunID, events[0].RunID)
		}
	}
}

func TestStreamEventsCallerCallbacksSeeEventsToo(t *testing.T) {
	recorder := callbacks.NewRecorder()
	events := esCollect(t, runnables.StreamEvents(
		t.Context(), esUpper(), "x", runnables.StreamEventOptions{},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	), "")
	esAssertNames(t, events, "on_chain_start:func", "on_chain_end:func")
	if len(recorder.Events()) != 2 {
		t.Fatalf("caller recorder saw %d events, want 2", len(recorder.Events()))
	}
}
