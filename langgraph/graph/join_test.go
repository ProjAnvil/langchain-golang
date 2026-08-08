package graph

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func joinNoop(_ context.Context, _ map[string]any) (any, error) { return nil, nil }

// joinBuilderBase returns a builder with nodes a, b, c registered and a as
// entry point, so AddJoinEdge validation failures are the only Compile errors.
func joinBuilderBase() *StateGraph {
	g := NewStateGraph()
	g.AddNode("a", joinNoop)
	g.AddNode("b", joinNoop)
	g.AddNode("c", joinNoop)
	g.AddEdge(types.START, "a")
	return g
}

// TestAddJoinEdgeValidation covers the call-time (count/dup/reserved-name)
// and Compile-time (node existence, duplicate join channel) checks. Python
// accepts a single-element start tuple and silently set-dedups (state.py:956);
// Go deliberately tightens both to errors (spec M6 documented divergence), and
// rejects END as a join child where Python allows it (state.py:963-964).
func TestAddJoinEdgeValidation(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(g *StateGraph)
		wantErr string
	}{
		{"zero parents", func(g *StateGraph) { g.AddJoinEdge(nil, "c") }, "at least 2"},
		{"single parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a"}, "c") }, "at least 2"},
		{"duplicate parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "a"}, "c") }, "duplicate join parent"},
		{"START parent", func(g *StateGraph) { g.AddJoinEdge([]string{types.START, "a"}, "c") }, "invalid join parent"},
		{"END parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a", types.END}, "c") }, "invalid join parent"},
		{"START child", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "b"}, types.START) }, "invalid join child"},
		{"END child", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "b"}, types.END) }, "invalid join child"},
		{"unknown parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "zzz"}, "c") }, "not a registered node"},
		{"unknown child", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "b"}, "zzz") }, "not a registered node"},
		{"duplicate join edge", func(g *StateGraph) {
			g.AddJoinEdge([]string{"a", "b"}, "c")
			g.AddJoinEdge([]string{"a", "b"}, "c")
		}, "duplicate join channel"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := joinBuilderBase()
			tc.mutate(g)
			_, err := g.Compile()
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Compile() error = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}

// TestCompileJoinRegistersBarrierChannel: Compile registers a Barrier
// prototype under "join:a+b:c" and the parent->barrier index, without
// polluting the builder's own channelProtos (Compile stays re-entrant).
func TestCompileJoinRegistersBarrierChannel(t *testing.T) {
	g := joinBuilderBase()
	g.AddJoinEdge([]string{"a", "b"}, "c")
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	proto, ok := cg.channelProtos["join:a+b:c"]
	if !ok {
		t.Fatal("channelProtos missing join:a+b:c")
	}
	b, ok := proto.(*channels.Barrier)
	if !ok {
		t.Fatalf("join:a+b:c proto = %T, want *channels.Barrier", proto)
	}
	if got := b.Names(); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("Names() = %v, want [a b]", got)
	}
	if got := cg.joinsByParent["a"]; len(got) != 1 || got[0] != "join:a+b:c" {
		t.Fatalf("joinsByParent[a] = %v", got)
	}
	if got := cg.joinsByParent["b"]; len(got) != 1 || got[0] != "join:a+b:c" {
		t.Fatalf("joinsByParent[b] = %v", got)
	}
	if _, polluted := g.channelProtos["join:a+b:c"]; polluted {
		t.Fatal("builder channelProtos polluted by Compile")
	}
	if _, err := g.Compile(); err != nil {
		t.Fatalf("second Compile() error = %v (join registration must be re-entrant)", err)
	}
}

// TestJoinBasicTrigger: two parents in the same superstep fill the barrier;
// the child runs exactly once; no join key leaks into the result.
func TestJoinBasicTrigger(t *testing.T) {
	var childCalls int
	g := NewStateGraph()
	g.AddNode("entry", joinNoop)
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"a_done": true}, nil
	})
	g.AddNode("b", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"b_done": true}, nil
	})
	g.AddNode("c", func(_ context.Context, state map[string]any) (any, error) {
		childCalls++
		if _, leaked := state["join:a+b:c"]; leaked {
			t.Error("join key leaked into node input")
		}
		return map[string]any{"c_done": true}, nil
	})
	g.AddEdge(types.START, "entry")
	g.AddEdge("entry", "a")
	g.AddEdge("entry", "b")
	g.AddJoinEdge([]string{"a", "b"}, "c")
	g.AddEdge("c", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if childCalls != 1 {
		t.Fatalf("childCalls = %d, want 1 (barrier triggers exactly once)", childCalls)
	}
	for _, k := range []string{"a_done", "b_done", "c_done"} {
		if res.Values[k] != true {
			t.Fatalf("Values[%q] = %v, want true (Values = %v)", k, res.Values[k], res.Values)
		}
	}
	for k := range res.Values {
		if strings.HasPrefix(k, "join:") {
			t.Fatalf("join key %q leaked into Result.Values", k)
		}
	}
}

// sortedAddReducer mirrors the sorted_add reducer used by the Python
// waiting-edge tests (test_pregel.py:1956-1961): append then sort, so final
// accumulated values match Python's assertions exactly.
func sortedAddReducer(existing, update any) (any, error) {
	out, err := channels.AppendSliceReducer(existing, update)
	if err != nil {
		return nil, err
	}
	s, _ := out.([]string)
	sort.Strings(s)
	return s, nil
}

// newWaitingEdgeGraph builds the fan-out graph of test_pregel.py:1953:
// rewrite_query -> analyzer_one -> retriever_one, rewrite_query ->
// retriever_two, [retriever_one, retriever_two] -> qa -> END.
func newWaitingEdgeGraph(qaCalls *int) *StateGraph {
	g := NewStateGraph()
	g.AddReducer("docs", sortedAddReducer)
	g.AddNode("rewrite_query", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"query": "query: " + state["query"].(string)}, nil
	})
	g.AddNode("analyzer_one", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"query": "analyzed: " + state["query"].(string)}, nil
	})
	g.AddNode("retriever_one", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc1", "doc2"}}, nil
	})
	g.AddNode("retriever_two", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc3", "doc4"}}, nil
	})
	g.AddNode("qa", func(_ context.Context, state map[string]any) (any, error) {
		*qaCalls++
		docs, _ := state["docs"].([]string)
		return map[string]any{"answer": strings.Join(docs, ",")}, nil
	})
	g.AddEdge(types.START, "rewrite_query")
	g.AddEdge("rewrite_query", "analyzer_one")
	g.AddEdge("analyzer_one", "retriever_one")
	g.AddEdge("rewrite_query", "retriever_two")
	g.AddJoinEdge([]string{"retriever_one", "retriever_two"}, "qa")
	g.AddEdge("qa", types.END)
	return g
}

var joinWantValues = map[string]any{
	"query":  "analyzed: query: what is weather in sf",
	"docs":   []string{"doc1", "doc2", "doc3", "doc4"},
	"answer": "doc1,doc2,doc3,doc4",
}

// TestJoinSameSuperstepExactlyOnce ports test_simple_multi_edge
// (test_pregel.py:3059): up and side complete in the same superstep; down
// (join child of [up, side]) still runs exactly once.
func TestJoinSameSuperstepExactlyOnce(t *testing.T) {
	var downCalls int
	g := NewStateGraph()
	g.AddReducer("my_key", func(existing, update any) (any, error) {
		ex, _ := existing.(string)
		u, _ := update.(string)
		return ex + u, nil
	})
	g.AddNode("up", joinNoop)
	g.AddNode("side", joinNoop)
	g.AddNode("other", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"my_key": "_more"}, nil
	})
	g.AddNode("down", func(_ context.Context, _ map[string]any) (any, error) {
		downCalls++
		return nil, nil
	})
	g.AddEdge(types.START, "up")
	g.AddEdge("up", "side")
	// Python's `other` is an implicit finish point (test_pregel.py:3077);
	// Go's staticNext requires an explicit outgoing edge, so wire it to END
	// (its write still lands in the final state, as Python asserts).
	g.AddEdge("up", "other")
	g.AddEdge("other", types.END)
	g.AddJoinEdge([]string{"up", "side"}, "down")
	g.AddEdge("down", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), map[string]any{"my_key": "hello"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Values["my_key"] != "hello_more" {
		t.Fatalf("my_key = %v, want %q", res.Values["my_key"], "hello_more")
	}
	if downCalls != 1 {
		t.Fatalf("downCalls = %d, want 1 (same-superstep parents trigger once)", downCalls)
	}
}

// TestJoinFanOutWaitingEdge ports test_pregel.py:1953: the parents complete
// in DIFFERENT supersteps (retriever_two one step before retriever_one); qa
// waits for both, runs once, and sees the fully accumulated docs. The
// interrupt_after subtest mirrors :2018-2036.
func TestJoinFanOutWaitingEdge(t *testing.T) {
	var qaCalls int
	g := newWaitingEdgeGraph(&qaCalls)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), map[string]any{"query": "what is weather in sf"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if !reflect.DeepEqual(res.Values, joinWantValues) {
		t.Fatalf("Values = %v, want %v", res.Values, joinWantValues)
	}
	if qaCalls != 1 {
		t.Fatalf("qaCalls = %d, want 1", qaCalls)
	}

	// updates chunks in Go's deterministic task order (documented M3
	// divergence from Python's as-they-finish timing).
	var updates []any
	for c, err := range cg.Stream(context.Background(), map[string]any{"query": "what is weather in sf"},
		StreamOptions{Modes: []StreamMode{StreamUpdates}}) {
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		updates = append(updates, c.Payload)
	}
	wantUpdates := []any{
		map[string]any{"rewrite_query": map[string]any{"query": "query: what is weather in sf"}},
		map[string]any{"analyzer_one": map[string]any{"query": "analyzed: query: what is weather in sf"}},
		map[string]any{"retriever_two": map[string]any{"docs": []string{"doc3", "doc4"}}},
		map[string]any{"retriever_one": map[string]any{"docs": []string{"doc1", "doc2"}}},
		map[string]any{"qa": map[string]any{"answer": "doc1,doc2,doc3,doc4"}},
	}
	if !reflect.DeepEqual(updates, wantUpdates) {
		t.Fatalf("updates = %v, want %v", updates, wantUpdates)
	}

	// interrupt_after(retriever_one): pause after the second parent commits,
	// resume runs qa exactly once (test_pregel.py:2018-2036).
	var qaCalls2 int
	g2 := newWaitingEdgeGraph(&qaCalls2)
	cg2, err := g2.Compile(WithCheckpointer(checkpoint.NewMemorySaver()), WithInterruptAfter("retriever_one"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()
	paused, err := cg2.InvokeWithOptions(ctx, map[string]any{"query": "what is weather in sf"}, Options{ThreadID: "1"})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("run1 Interrupts = %v, want 1", paused.Interrupts)
	}
	if qaCalls2 != 0 {
		t.Fatalf("qaCalls after pause = %d, want 0", qaCalls2)
	}
	done, err := cg2.InvokeWithOptions(ctx, nil, Options{ThreadID: "1"})
	if err != nil {
		t.Fatalf("resume error = %v", err)
	}
	if !reflect.DeepEqual(done.Values, joinWantValues) {
		t.Fatalf("resume Values = %v, want %v", done.Values, joinWantValues)
	}
	if qaCalls2 != 1 {
		t.Fatalf("qaCalls after resume = %d, want 1", qaCalls2)
	}
}

// TestJoinWaitingEdgePlusRegularEdge ports test_pregel.py:2710: an extra
// plain edge rewrite_query -> qa bypasses the barrier (OR semantics) — qa
// runs once early with empty docs and once via the barrier; "having been
// triggered before doesn't break the semantics of the named barrier".
func TestJoinWaitingEdgePlusRegularEdge(t *testing.T) {
	var qaCalls int
	var answers []string
	g := newWaitingEdgeGraph(&qaCalls)
	// Wrap qa to record every invocation's answer, in run order.
	qaFn := g.nodes["qa"]
	g.nodes["qa"] = func(ctx context.Context, state map[string]any) (any, error) {
		out, err := qaFn(ctx, state)
		if m, ok := out.(map[string]any); ok {
			answers = append(answers, m["answer"].(string))
		}
		return out, err
	}
	g.AddEdge("rewrite_query", "qa") // the Python test's "silly edge" (:2759)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), map[string]any{"query": "what is weather in sf"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if qaCalls != 2 {
		t.Fatalf("qaCalls = %d, want 2 (plain edge + barrier)", qaCalls)
	}
	if !reflect.DeepEqual(answers, []string{"", "doc1,doc2,doc3,doc4"}) {
		t.Fatalf("answers = %v, want [\"\" \"doc1,doc2,doc3,doc4\"]", answers)
	}
	if !reflect.DeepEqual(res.Values, joinWantValues) {
		t.Fatalf("Values = %v, want %v", res.Values, joinWantValues)
	}
}

// TestJoinLoopReset ports test_pregel.py:2808 (waiting_edge_multiple): the
// join sits inside a decider loop; after each trigger the barrier is consumed
// and re-arms, so round 2 re-triggers exactly once. The withCache variant
// mirrors Python's cache parametrization and additionally puts a cache policy
// on retriever_one (a join PARENT) so the cache-hit injection path
// (graph.go: arrival write appended to a cache-injected outcome) is covered.
func TestJoinLoopReset(t *testing.T) {
	for _, withCache := range []bool{false, true} {
		t.Run(fmt.Sprintf("withCache=%v", withCache), func(t *testing.T) {
			var rewriteCalls int
			g := NewStateGraph()
			g.AddReducer("docs", sortedAddReducer)
			rewrite := func(_ context.Context, state map[string]any) (any, error) {
				rewriteCalls++
				return map[string]any{"query": "query: " + state["query"].(string)}, nil
			}
			cachePolicy := NodePolicies{}
			if withCache {
				cachePolicy = NodePolicies{Cache: &CachePolicy{}}
			}
			g.AddNodeWithPolicies("rewrite_query", rewrite, cachePolicy)
			g.AddNode("analyzer_one", func(_ context.Context, state map[string]any) (any, error) {
				return map[string]any{"query": "analyzed: " + state["query"].(string)}, nil
			})
			g.AddNodeWithPolicies("retriever_one", func(_ context.Context, _ map[string]any) (any, error) {
				return map[string]any{"docs": []string{"doc1", "doc2"}}, nil
			}, cachePolicy)
			g.AddNode("retriever_two", func(_ context.Context, _ map[string]any) (any, error) {
				return map[string]any{"docs": []string{"doc3", "doc4"}}, nil
			})
			g.AddNode("decider", joinNoop)
			g.AddNode("qa", func(_ context.Context, state map[string]any) (any, error) {
				docs, _ := state["docs"].([]string)
				return map[string]any{"answer": strings.Join(docs, ",")}, nil
			})
			g.AddEdge(types.START, "rewrite_query")
			g.AddEdge("rewrite_query", "analyzer_one")
			g.AddEdge("analyzer_one", "retriever_one")
			g.AddEdge("rewrite_query", "retriever_two")
			g.AddJoinEdge([]string{"retriever_one", "retriever_two"}, "decider")
			g.AddConditionalEdges("decider", func(_ context.Context, state map[string]any) ([]any, error) {
				if strings.Count(state["query"].(string), "analyzed") > 1 {
					return To("qa"), nil
				}
				return To("rewrite_query"), nil
			})
			g.AddEdge("qa", types.END)

			var cg *CompiledGraph
			var err error
			if withCache {
				cg, err = g.Compile(WithCache(checkpoint.NewInMemoryCache()))
			} else {
				cg, err = g.Compile()
			}
			if err != nil {
				t.Fatalf("Compile() error = %v", err)
			}
			want := map[string]any{
				"query":  "analyzed: query: analyzed: query: what is weather in sf",
				"docs":   []string{"doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"},
				"answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
			}
			// Two full runs, mirroring Python's invoke+stream count
			// assertions (rewrite_query_count == 4 uncached, 2 cached).
			for run := 0; run < 2; run++ {
				res, err := cg.Invoke(context.Background(), map[string]any{"query": "what is weather in sf"})
				if err != nil {
					t.Fatalf("run %d Invoke() error = %v", run, err)
				}
				if !reflect.DeepEqual(res.Values, want) {
					t.Fatalf("run %d Values = %v, want %v", run, res.Values, want)
				}
			}
			wantCalls := 4
			if withCache {
				wantCalls = 2
			}
			if rewriteCalls != wantCalls {
				t.Fatalf("rewriteCalls = %d, want %d", rewriteCalls, wantCalls)
			}
		})
	}
}

// TestJoinSendBypassesBarrier: a types.Send to the join child bypasses the
// barrier (Python OR semantics). The Send PUSH task (arg input) and the
// barrier task (shared state) are two legitimate independent dispatches —
// the barrier trigger must NOT dedup against the Send.
func TestJoinSendBypassesBarrier(t *testing.T) {
	var qaCalls int
	var answers []string
	g := NewStateGraph()
	g.AddReducer("docs", sortedAddReducer)
	g.AddNode("entry", joinNoop)
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc1"}}, nil
	})
	g.AddNode("b", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc2"}}, nil
	})
	g.AddNode("qa", func(_ context.Context, state map[string]any) (any, error) {
		qaCalls++
		docs, _ := state["docs"].([]string)
		answers = append(answers, strings.Join(docs, ","))
		return map[string]any{"answer": strings.Join(docs, ",")}, nil
	})
	g.AddEdge(types.START, "entry")
	g.AddConditionalEdges("entry", func(_ context.Context, _ map[string]any) ([]any, error) {
		return []any{&types.Send{Node: "qa", Arg: map[string]any{"docs": []string{"sent"}}}, "a", "b"}, nil
	})
	g.AddJoinEdge([]string{"a", "b"}, "qa")
	g.AddEdge("qa", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if qaCalls != 2 {
		t.Fatalf("qaCalls = %d, want 2 (Send + barrier)", qaCalls)
	}
	// Superstep 2 runs the Send task (arg input, answer "sent"); the parents
	// fill the barrier in the same superstep, so superstep 3 runs the barrier
	// task on shared state.
	if !reflect.DeepEqual(answers, []string{"sent", "doc1,doc2"}) {
		t.Fatalf("answers = %v, want [sent doc1,doc2]", answers)
	}
	if res.Values["answer"] != "doc1,doc2" {
		t.Fatalf("final answer = %v, want doc1,doc2", res.Values["answer"])
	}
}

// TestJoinThreeParents (Go extension; Python has no three-parent waiting-edge
// case): join:a+b+c fires only after all three arrive.
func TestJoinThreeParents(t *testing.T) {
	var dCalls int
	g := NewStateGraph()
	g.AddNode("entry", joinNoop)
	g.AddEdge(types.START, "entry")
	for _, n := range []string{"a", "b", "c"} {
		g.AddNode(n, joinNoop)
		g.AddEdge("entry", n)
	}
	g.AddNode("d", func(_ context.Context, _ map[string]any) (any, error) {
		dCalls++
		return nil, nil
	})
	g.AddJoinEdge([]string{"a", "b", "c"}, "d")
	g.AddEdge("d", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, ok := cg.channelProtos["join:a+b+c:d"].(*channels.Barrier); !ok {
		t.Fatal("join:a+b+c:d barrier not registered")
	}
	if _, err := cg.Invoke(context.Background(), nil); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if dCalls != 1 {
		t.Fatalf("dCalls = %d, want 1", dCalls)
	}
}

// TestJoinParentInterruptResume (Go-new): parent a completes and parent b
// interrupts in the same superstep; a's barrier arrival is persisted as a
// pending write with its task batch; resume replays it (a does NOT re-run),
// b re-runs with the resume value, its arrival fills the barrier, and c runs
// exactly once.
func TestJoinParentInterruptResume(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var aCalls, bCalls, cCalls int
	g := NewStateGraph()
	g.AddNode("entry", joinNoop)
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		aCalls++
		return map[string]any{"a_done": true}, nil
	})
	g.AddNode("b", func(ctx context.Context, _ map[string]any) (any, error) {
		bCalls++
		Interrupt(ctx, "b needs input") // panics on run 1; returns "ok" on resume
		return map[string]any{"b_done": true}, nil
	})
	g.AddNode("c", func(_ context.Context, _ map[string]any) (any, error) {
		cCalls++
		return map[string]any{"c_done": true}, nil
	})
	g.AddEdge(types.START, "entry")
	g.AddEdge("entry", "a")
	g.AddEdge("entry", "b")
	g.AddJoinEdge([]string{"a", "b"}, "c")
	g.AddEdge("c", types.END)

	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()
	paused, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("run1 Interrupts = %v, want 1", paused.Interrupts)
	}
	if aCalls != 1 || bCalls != 1 || cCalls != 0 {
		t.Fatalf("after pause: a=%d b=%d c=%d, want 1/1/0", aCalls, bCalls, cCalls)
	}

	// a's arrival rode its task batch into the pause checkpoint's pending
	// writes (the interrupt-path completedTaskWrites closure).
	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple() = %v, %v", tup, err)
	}
	foundArrival := false
	for _, w := range tup.PendingWrites {
		if w.Channel == "join:a+b:c" && w.Value == "a" {
			foundArrival = true
		}
	}
	if !foundArrival {
		t.Fatal("pause checkpoint pending writes missing {join:a+b:c: a}")
	}

	res, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "ok"})
	if err != nil {
		t.Fatalf("resume error = %v", err)
	}
	if aCalls != 1 {
		t.Fatalf("aCalls = %d after resume, want 1 (replayed, not re-run)", aCalls)
	}
	if bCalls != 2 || cCalls != 1 {
		t.Fatalf("after resume: b=%d c=%d, want 2/1", bCalls, cCalls)
	}
	for _, k := range []string{"a_done", "b_done", "c_done"} {
		if res.Values[k] != true {
			t.Fatalf("Values[%q] = %v, want true", k, res.Values[k])
		}
	}
}

// TestJoinCheckpointPartialArrival (Go-new): with parents in DIFFERENT
// supersteps (a -> b), a's arrival is committed channel state; b's interrupt
// pauses with the partial barrier ([a]) inside the checkpoint, GetState
// filters the join key, and resume completes the barrier exactly once.
func TestJoinCheckpointPartialArrival(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var aCalls, bCalls, cCalls int
	g := NewStateGraph()
	g.AddNode("entry", joinNoop)
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		aCalls++
		return map[string]any{"a_done": true}, nil
	})
	g.AddNode("b", func(ctx context.Context, _ map[string]any) (any, error) {
		bCalls++
		Interrupt(ctx, "b needs input")
		return map[string]any{"b_done": true}, nil
	})
	g.AddNode("c", func(_ context.Context, _ map[string]any) (any, error) {
		cCalls++
		return nil, nil
	})
	g.AddEdge(types.START, "entry")
	g.AddEdge("entry", "a")
	g.AddEdge("a", "b")
	g.AddJoinEdge([]string{"a", "b"}, "c")
	g.AddEdge("c", types.END)

	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()
	if _, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("run1 error = %v", err)
	}

	// The pause checkpoint carries the partial barrier as committed channel
	// state (b's superstep never committed, so a's [a] arrival survives).
	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple() = %v, %v", tup, err)
	}
	if got := tup.Checkpoint.ChannelValues["join:a+b:c"]; !reflect.DeepEqual(got, []string{"a"}) {
		t.Fatalf("checkpoint join:a+b:c = %v, want []string{\"a\"} (partial arrival persisted)", got)
	}
	// ... while the user-visible snapshot filters it.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if _, leaked := snap.Values["join:a+b:c"]; leaked {
		t.Fatal("join key leaked into GetState Values")
	}
	if snap.Values["a_done"] != true {
		t.Fatalf("GetState Values = %v, want a_done committed", snap.Values)
	}

	if _, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "ok"}); err != nil {
		t.Fatalf("resume error = %v", err)
	}
	if aCalls != 1 || bCalls != 2 || cCalls != 1 {
		t.Fatalf("after resume: a=%d b=%d c=%d, want 1/2/1", aCalls, bCalls, cCalls)
	}
}

// TestJoinKeyNotLeaked (Go-new, spec risk item): the join channel is
// control-plane — it must not appear in any node's input, any stream chunk
// payload (values/updates/debug), Result.Values, or GetState/GetStateHistory.
func TestJoinKeyNotLeaked(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var mu sync.Mutex
	seenInputKeys := map[string]bool{}
	var qaCalls int

	g := NewStateGraph()
	g.AddReducer("docs", sortedAddReducer)
	wrap := func(fn NodeFunc) NodeFunc {
		return func(ctx context.Context, state map[string]any) (any, error) {
			mu.Lock()
			for k := range state {
				seenInputKeys[k] = true
			}
			mu.Unlock()
			return fn(ctx, state)
		}
	}
	g.AddNode("rewrite_query", wrap(func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"query": "query: " + state["query"].(string)}, nil
	}))
	g.AddNode("analyzer_one", wrap(func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"query": "analyzed: " + state["query"].(string)}, nil
	}))
	g.AddNode("retriever_one", wrap(func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc1", "doc2"}}, nil
	}))
	g.AddNode("retriever_two", wrap(func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc3", "doc4"}}, nil
	}))
	g.AddNode("qa", wrap(func(_ context.Context, state map[string]any) (any, error) {
		qaCalls++
		docs, _ := state["docs"].([]string)
		return map[string]any{"answer": strings.Join(docs, ",")}, nil
	}))
	g.AddEdge(types.START, "rewrite_query")
	g.AddEdge("rewrite_query", "analyzer_one")
	g.AddEdge("analyzer_one", "retriever_one")
	g.AddEdge("rewrite_query", "retriever_two")
	g.AddJoinEdge([]string{"retriever_one", "retriever_two"}, "qa")
	g.AddEdge("qa", types.END)

	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()
	for c, err := range cg.Stream(ctx, map[string]any{"query": "q"},
		StreamOptions{
			Options: Options{ThreadID: "t"},
			Modes:   []StreamMode{StreamValues, StreamUpdates, StreamDebug},
		}) {
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		payload, _ := json.Marshal(c.Payload)
		if strings.Contains(string(payload), "join:") {
			t.Fatalf("join key leaked into %s chunk: %s", c.Mode, payload)
		}
	}
	if qaCalls != 1 {
		t.Fatalf("qaCalls = %d, want 1", qaCalls)
	}
	for k := range seenInputKeys {
		if strings.HasPrefix(k, "join:") {
			t.Fatalf("join key %q leaked into a node input", k)
		}
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if _, leaked := snap.Values["join:retriever_one+retriever_two:qa"]; leaked {
		t.Fatal("join key leaked into GetState Values")
	}
	hist, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: "t"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("GetStateHistory() error = %v", err)
	}
	for i, s := range hist {
		for k := range s.Values {
			if strings.HasPrefix(k, "join:") {
				t.Fatalf("join key %q leaked into history snapshot %d", k, i)
			}
		}
	}
}
