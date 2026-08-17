package graph

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// --- test fakes -----------------------------------------------------------

var errSaverBoom = errors.New("saver boom")

// getTupleErrSaver fails every GetTuple call.
type getTupleErrSaver struct{ checkpoint.Saver }

func (s *getTupleErrSaver) GetTuple(context.Context, checkpoint.Config) (*checkpoint.Tuple, error) {
	return nil, errSaverBoom
}

// putErrSaver fails every Put call.
type putErrSaver struct{ checkpoint.Saver }

func (s *putErrSaver) Put(context.Context, checkpoint.Config, checkpoint.Checkpoint, checkpoint.Metadata, map[string]int64) (checkpoint.Config, error) {
	return checkpoint.Config{}, errSaverBoom
}

// putWritesErrSaver fails every PutWrites call.
type putWritesErrSaver struct{ checkpoint.Saver }

func (s *putWritesErrSaver) PutWrites(context.Context, checkpoint.Config, []checkpoint.Write, string, string) error {
	return errSaverBoom
}

// failNthPutSaver fails the n-th Put call (1-based), delegating the rest.
type failNthPutSaver struct {
	checkpoint.Saver
	n     int64
	calls atomic.Int64
}

func (s *failNthPutSaver) Put(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, nv map[string]int64) (checkpoint.Config, error) {
	if s.calls.Add(1) == s.n {
		return checkpoint.Config{}, errSaverBoom
	}
	return s.Saver.Put(ctx, cfg, cp, md, nv)
}

// failNthPutWritesSaver fails the n-th PutWrites call (1-based).
type failNthPutWritesSaver struct {
	checkpoint.Saver
	n     int64
	calls atomic.Int64
}

func (s *failNthPutWritesSaver) PutWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID, taskPath string) error {
	if s.calls.Add(1) == s.n {
		return errSaverBoom
	}
	return s.Saver.PutWrites(ctx, cfg, writes, taskID, taskPath)
}

// cacheErrCache is a checkpoint.Cache whose Get/Set fail on demand; a nil
// error means Get reports a miss and Set succeeds.
type cacheErrCache struct {
	getErr error
	setErr error
}

func (c *cacheErrCache) Get(context.Context, string, string) ([]checkpoint.Write, bool, error) {
	if c.getErr != nil {
		return nil, false, c.getErr
	}
	return nil, false, nil
}

func (c *cacheErrCache) Set(context.Context, string, string, []checkpoint.Write, time.Duration) error {
	return c.setErr
}

func (c *cacheErrCache) Clear(context.Context, string) error { return nil }

// errUpdateChannel is a test channels.Channel whose Update fails: on every
// call when failAll is set, otherwise only on the empty step-boundary
// notification (see runState.applyWrites).
type errUpdateChannel struct {
	failAll bool
	value   any
	set     bool
}

func (c *errUpdateChannel) Update(values []any) (bool, error) {
	if c.failAll || len(values) == 0 {
		return false, errSaverBoom
	}
	c.value, c.set = values[0], true
	return true, nil
}

func (c *errUpdateChannel) Get() (any, error) {
	if !c.set {
		return nil, channels.ErrEmptyChannel
	}
	return c.value, nil
}

func (c *errUpdateChannel) IsAvailable() bool { return c.set }

func (c *errUpdateChannel) Checkpoint() (any, bool) { return c.value, c.set }

func (c *errUpdateChannel) FromCheckpoint(value any) channels.Channel {
	if value == nil {
		return &errUpdateChannel{failAll: c.failAll}
	}
	return &errUpdateChannel{failAll: c.failAll, value: value, set: true}
}

func noopNode(runtime.Runtime, map[string]any) (any, error) { return nil, nil }

// compileLinear tersely compiles a single-node graph a -> END.
func compileLinear(t *testing.T, fn NodeFunc, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("a", fn)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// --- builder validation ---------------------------------------------------

func TestBuilderValidationErrors(t *testing.T) {
	t.Run("AddNode invalid names", func(t *testing.T) {
		for _, name := range []string{"", types.START, types.END} {
			g := NewStateGraph()
			g.AddNode(name, noopNode)
			if _, err := g.Compile(); err == nil {
				t.Fatalf("Compile() error = nil, want invalid node name %q error", name)
			}
		}
	})
	t.Run("AddNode nil function", func(t *testing.T) {
		g := NewStateGraph()
		g.AddNode("a", nil)
		if _, err := g.Compile(); err == nil {
			t.Fatal("Compile() error = nil, want nil function error")
		}
	})
	t.Run("AddChannel nil prototype", func(t *testing.T) {
		g := NewStateGraph()
		g.AddChannel("k", nil)
		if _, err := g.Compile(); err == nil {
			t.Fatal("Compile() error = nil, want nil channel prototype error")
		}
	})
	t.Run("AddConditionalEdges nil router", func(t *testing.T) {
		g := NewStateGraph()
		g.AddConditionalEdges("a", nil)
		if _, err := g.Compile(); err == nil {
			t.Fatal("Compile() error = nil, want nil router error")
		}
	})
	t.Run("AddConditionalEdges duplicate", func(t *testing.T) {
		g := NewStateGraph()
		router := func(runtime.Runtime, map[string]any) ([]any, error) { return To(types.END), nil }
		g.AddConditionalEdges("a", router)
		g.AddConditionalEdges("a", router)
		if _, err := g.Compile(); err == nil {
			t.Fatal("Compile() error = nil, want duplicate conditional edge error")
		}
	})
	t.Run("SetEntryPoint twice", func(t *testing.T) {
		g := NewStateGraph()
		g.AddNode("a", noopNode)
		g.AddNode("b", noopNode)
		g.SetEntryPoint("a")
		g.SetEntryPoint("b")
		if _, err := g.Compile(); err == nil {
			t.Fatal("Compile() error = nil, want duplicate entry point error")
		}
	})
	t.Run("AddSubgraph nil child", func(t *testing.T) {
		g := NewStateGraph()
		g.AddSubgraph("sub", nil)
		if _, err := g.Compile(); err == nil {
			t.Fatal("Compile() error = nil, want nil subgraph error")
		}
	})
}

func TestCompileReferenceErrors(t *testing.T) {
	t.Run("entry point not a registered node", func(t *testing.T) {
		g := NewStateGraph()
		g.AddNode("a", noopNode)
		g.SetEntryPoint("ghost")
		if _, err := g.Compile(); err == nil || !strings.Contains(err.Error(), "ghost") {
			t.Fatalf("Compile() error = %v, want it to name the unregistered entry point", err)
		}
	})
	t.Run("edge source not a registered node", func(t *testing.T) {
		g := NewStateGraph()
		g.AddNode("a", noopNode)
		g.AddEdge(types.START, "a")
		g.AddEdge("ghost", "a")
		if _, err := g.Compile(); err == nil || !strings.Contains(err.Error(), "ghost") {
			t.Fatalf("Compile() error = %v, want it to name the unregistered edge source", err)
		}
	})
	t.Run("conditional edge source not a registered node", func(t *testing.T) {
		g := NewStateGraph()
		g.AddNode("a", noopNode)
		g.AddEdge(types.START, "a")
		g.AddEdge("a", types.END)
		g.AddConditionalEdges("ghost", func(runtime.Runtime, map[string]any) ([]any, error) { return To(types.END), nil })
		if _, err := g.Compile(); err == nil || !strings.Contains(err.Error(), "ghost") {
			t.Fatalf("Compile() error = %v, want it to name the unregistered router source", err)
		}
	})
}

// --- node result / routing errors -----------------------------------------

func TestNodeUnsupportedResultTypeErrors(t *testing.T) {
	cg := compileLinear(t, func(runtime.Runtime, map[string]any) (any, error) {
		return "not a map", nil
	})
	if _, err := cg.Invoke(context.Background(), map[string]any{}); err == nil ||
		!strings.Contains(err.Error(), "unsupported type") {
		t.Fatalf("Invoke() error = %v, want an unsupported-node-result error", err)
	}
}

func TestCommandInvalidGotoDestinationErrors(t *testing.T) {
	cg := compileLinear(t, func(runtime.Runtime, map[string]any) (any, error) {
		return &types.Command{Goto: []any{42}}, nil
	})
	if _, err := cg.Invoke(context.Background(), map[string]any{}); err == nil ||
		!strings.Contains(err.Error(), "unsupported routing destination") {
		t.Fatalf("Invoke() error = %v, want an unsupported routing destination error", err)
	}
}

func TestRouterErrorPropagates(t *testing.T) {
	want := errors.New("router boom")
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddConditionalEdges("a", func(runtime.Runtime, map[string]any) ([]any, error) { return nil, want })
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.Invoke(context.Background(), map[string]any{}); !errors.Is(err, want) {
		t.Fatalf("Invoke() error = %v, want %v", err, want)
	}
}

func TestInvokeStreamSurfacesRunError(t *testing.T) {
	want := errors.New("node boom")
	cg := compileLinear(t, func(runtime.Runtime, map[string]any) (any, error) {
		return nil, want
	})
	if _, err := cg.InvokeStream(context.Background(), map[string]any{}, Options{}, nil); !errors.Is(err, want) {
		t.Fatalf("InvokeStream() error = %v, want %v", err, want)
	}
}

// --- options / checkpoint-loading errors ----------------------------------

func TestCheckpointIDOptionErrors(t *testing.T) {
	t.Run("requires a checkpointer", func(t *testing.T) {
		cg := compileLinear(t, noopNode)
		if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{CheckpointID: "cp"}); err == nil ||
			!strings.Contains(err.Error(), "requires a checkpointer") {
			t.Fatalf("InvokeWithOptions() error = %v, want a checkpointer requirement error", err)
		}
	})
	t.Run("requires ThreadID", func(t *testing.T) {
		cg := compileLinear(t, noopNode, WithCheckpointer(checkpoint.NewMemorySaver()))
		if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{CheckpointID: "cp"}); err == nil ||
			!strings.Contains(err.Error(), "requires ThreadID") {
			t.Fatalf("InvokeWithOptions() error = %v, want a ThreadID requirement error", err)
		}
	})
}

func TestResumeOptionErrors(t *testing.T) {
	t.Run("requires ThreadID", func(t *testing.T) {
		cg := compileLinear(t, noopNode, WithCheckpointer(checkpoint.NewMemorySaver()))
		if _, err := cg.InvokeWithOptions(context.Background(), nil, Options{Resume: "v"}); err == nil ||
			!strings.Contains(err.Error(), "requires ThreadID") {
			t.Fatalf("InvokeWithOptions() error = %v, want a ThreadID requirement error", err)
		}
	})
	t.Run("no checkpoint for thread", func(t *testing.T) {
		cg := compileLinear(t, noopNode, WithCheckpointer(checkpoint.NewMemorySaver()))
		if _, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t", Resume: "v"}); err == nil ||
			!strings.Contains(err.Error(), "no checkpoint found") {
			t.Fatalf("InvokeWithOptions() error = %v, want a no-checkpoint error", err)
		}
	})
	t.Run("scalar resume with multiple pending interrupts", func(t *testing.T) {
		saver := checkpoint.NewMemorySaver()
		g := NewStateGraph()
		g.AddNode("entry", noopNode)
		g.AddNode("i1", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
			Interrupt(ctx, "q1")
			return nil, nil
		})
		g.AddNode("i2", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
			Interrupt(ctx, "q2")
			return nil, nil
		})
		g.AddEdge(types.START, "entry")
		g.AddConditionalEdges("entry", func(runtime.Runtime, map[string]any) ([]any, error) {
			return To("i1", "i2"), nil
		})
		g.AddEdge("i1", types.END)
		g.AddEdge("i2", types.END)
		cg, err := g.Compile(WithCheckpointer(saver))
		if err != nil {
			t.Fatalf("Compile() error = %v", err)
		}
		res, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"})
		if err != nil {
			t.Fatalf("InvokeWithOptions() error = %v", err)
		}
		if len(res.Interrupts) != 2 {
			t.Fatalf("Interrupts = %+v, want 2 pending interrupts", res.Interrupts)
		}
		if _, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t", Resume: "scalar"}); err == nil ||
			!strings.Contains(err.Error(), "map[string]any") {
			t.Fatalf("resume error = %v, want the scalar-resume-with-multiple-interrupts error", err)
		}
	})
}

func TestRunCheckpointLoadError(t *testing.T) {
	cg := compileLinear(t, noopNode, WithCheckpointer(&getTupleErrSaver{Saver: checkpoint.NewMemorySaver()}))
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

// --- checkpoint save / write flush errors ---------------------------------

func TestRunInputCheckpointSaveError(t *testing.T) {
	cg := compileLinear(t, noopNode, WithCheckpointer(&putErrSaver{Saver: checkpoint.NewMemorySaver()}))
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"x": 1}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

func TestRunLoopCheckpointSaveError(t *testing.T) {
	// First Put (input checkpoint) succeeds, second (loop checkpoint) fails.
	saver := &failNthPutSaver{Saver: checkpoint.NewMemorySaver(), n: 2}
	cg := compileLinear(t, func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"x": 1}, nil
	}, WithCheckpointer(saver))
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

func TestRunTaskWritesPersistError(t *testing.T) {
	cg := compileLinear(t, func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"x": 1}, nil
	}, WithCheckpointer(&putWritesErrSaver{Saver: checkpoint.NewMemorySaver()}))
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

func TestRunAsyncFlushErrorSurfaces(t *testing.T) {
	// Async durability defers saver errors to the exit flush; a background Put
	// failure must surface as the invoke's error.
	cg := compileLinear(t, func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"x": 1}, nil
	}, WithCheckpointer(&putErrSaver{Saver: checkpoint.NewMemorySaver()}), WithDurability(DurabilityAsync))
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

// newDeltaItemsGraph builds a one-node graph with a delta-channeled "items"
// key, used to exercise delta input-write persistence.
func newDeltaItemsGraph(t *testing.T, saver checkpoint.Saver) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("items", channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, 1))
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

func TestRunDeltaInputWritesPersistError(t *testing.T) {
	t.Run("fresh turn", func(t *testing.T) {
		cg := newDeltaItemsGraph(t, &putWritesErrSaver{Saver: checkpoint.NewMemorySaver()})
		_, err := cg.InvokeWithOptions(context.Background(), map[string]any{"items": []int{1}}, Options{ThreadID: "t"})
		if !errors.Is(err, errSaverBoom) {
			t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
		}
	})
	t.Run("new turn on existing checkpoint", func(t *testing.T) {
		saver := &failNthPutWritesSaver{Saver: checkpoint.NewMemorySaver(), n: 1}
		cg := newDeltaItemsGraph(t, saver)
		if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"x": 1}, Options{ThreadID: "t"}); err != nil {
			t.Fatalf("first InvokeWithOptions() error = %v", err)
		}
		_, err := cg.InvokeWithOptions(context.Background(), map[string]any{"items": []int{2}}, Options{ThreadID: "t"})
		if !errors.Is(err, errSaverBoom) {
			t.Fatalf("second InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
		}
	})
}

// --- input batch applyWrites errors ---------------------------------------

// newErrUpdateChannelGraph builds a one-node graph whose "bad" key is backed
// by a channel whose Update always errors.
func newErrUpdateChannelGraph(t *testing.T, saver checkpoint.Saver) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("bad", &errUpdateChannel{failAll: true})
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	var opts []CompileOption
	if saver != nil {
		opts = append(opts, WithCheckpointer(saver))
	}
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

func TestRunInputApplyWritesError(t *testing.T) {
	t.Run("fresh turn", func(t *testing.T) {
		cg := newErrUpdateChannelGraph(t, nil)
		if _, err := cg.Invoke(context.Background(), map[string]any{"bad": 1}); !errors.Is(err, errSaverBoom) {
			t.Fatalf("Invoke() error = %v, want %v", err, errSaverBoom)
		}
	})
	t.Run("new turn on existing checkpoint", func(t *testing.T) {
		cg := newErrUpdateChannelGraph(t, checkpoint.NewMemorySaver())
		if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"x": 1}, Options{ThreadID: "t"}); err != nil {
			t.Fatalf("first InvokeWithOptions() error = %v", err)
		}
		if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"bad": 1}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
			t.Fatalf("second InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
		}
	})
}

// --- interrupt pause persistence errors -----------------------------------

// compileInterruptGraph builds a graph whose single node interrupts in-node.
func compileInterruptGraph(t *testing.T, saver checkpoint.Saver) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("ask", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		Interrupt(ctx, "q")
		return nil, nil
	})
	g.AddEdge(types.START, "ask")
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

func TestInterruptBeforeSaveError(t *testing.T) {
	saver := &failNthPutSaver{Saver: checkpoint.NewMemorySaver(), n: 2}
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithInterruptBefore("a"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

func TestInterruptBeforePersistInterruptsError(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(&putWritesErrSaver{Saver: checkpoint.NewMemorySaver()}), WithInterruptBefore("a"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

func TestInterruptAfterSaveError(t *testing.T) {
	saver := &failNthPutSaver{Saver: checkpoint.NewMemorySaver(), n: 2}
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithInterruptAfter("a"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

func TestInterruptAfterPersistInterruptsError(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(&putWritesErrSaver{Saver: checkpoint.NewMemorySaver()}), WithInterruptAfter("a"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

func TestInNodeInterruptPauseSaveError(t *testing.T) {
	saver := &failNthPutSaver{Saver: checkpoint.NewMemorySaver(), n: 2}
	cg := compileInterruptGraph(t, saver)
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

func TestInNodeInterruptPersistError(t *testing.T) {
	cg := compileInterruptGraph(t, &putWritesErrSaver{Saver: checkpoint.NewMemorySaver()})
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

// compileSiblingInterruptGraph builds a graph whose router fans out to a
// sibling node (siblingFn) and an interrupting node in one superstep.
func compileSiblingInterruptGraph(t *testing.T, saver checkpoint.Saver, siblingFn NodeFunc) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("entry", noopNode)
	g.AddNode("sibling", siblingFn)
	g.AddNode("ask", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		Interrupt(ctx, "q")
		return nil, nil
	})
	g.AddEdge(types.START, "entry")
	g.AddConditionalEdges("entry", func(runtime.Runtime, map[string]any) ([]any, error) {
		return To("sibling", "ask"), nil
	})
	g.AddEdge("sibling", types.END)
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

func TestPauseCompletedSiblingBadGotoErrors(t *testing.T) {
	cg := compileSiblingInterruptGraph(t, checkpoint.NewMemorySaver(), func(runtime.Runtime, map[string]any) (any, error) {
		return &types.Command{Goto: []any{42}}, nil
	})
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); err == nil ||
		!strings.Contains(err.Error(), "unsupported routing destination") {
		t.Fatalf("InvokeWithOptions() error = %v, want an unsupported routing destination error", err)
	}
}

func TestPauseCompletedSiblingWritesPersistError(t *testing.T) {
	cg := compileSiblingInterruptGraph(t, &putWritesErrSaver{Saver: checkpoint.NewMemorySaver()}, func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"x": 1}, nil
	})
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}

// TestResumeReplayWritesError verifies that a resume whose completed-sibling
// replay cannot be applied (here: a channel that errors on update) surfaces
// the replay error instead of re-dispatching tasks.
func TestResumeReplayWritesError(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddChannel("bad", &errUpdateChannel{failAll: true})
	g.AddNode("entry", noopNode)
	g.AddNode("sibling", func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"bad": 1}, nil
	})
	g.AddNode("ask", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		Interrupt(ctx, "q")
		return nil, nil
	})
	g.AddEdge(types.START, "entry")
	g.AddConditionalEdges("entry", func(runtime.Runtime, map[string]any) ([]any, error) {
		return To("sibling", "ask"), nil
	})
	g.AddEdge("sibling", types.END)
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("InvokeWithOptions() error = %v", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("Interrupts = %+v, want 1 (the sibling's writes are persisted for replay)", res.Interrupts)
	}
	// Nil-input resume: replaying the sibling's "bad" write hits the failing
	// reducer inside resumeFromTuple.
	if _, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t", Resume: "go"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("resume error = %v, want %v", err, errSaverBoom)
	}
}

// --- cache policy errors ---------------------------------------------------

func compileCachePolicyGraph(t *testing.T, fn NodeFunc, cache checkpoint.Cache) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNodeWithPolicies("a", fn, NodePolicies{Cache: &CachePolicy{}})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCache(cache))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

func TestCacheGetErrorFailsTask(t *testing.T) {
	cg := compileCachePolicyGraph(t, noopNode, &cacheErrCache{getErr: errSaverBoom})
	if _, err := cg.Invoke(context.Background(), map[string]any{"x": 1}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("Invoke() error = %v, want %v", err, errSaverBoom)
	}
}

func TestCacheSetErrorFailsRun(t *testing.T) {
	cg := compileCachePolicyGraph(t, func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"x": 1}, nil
	}, &cacheErrCache{setErr: errSaverBoom})
	if _, err := cg.Invoke(context.Background(), map[string]any{"x": 1}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("Invoke() error = %v, want %v", err, errSaverBoom)
	}
}

func TestCacheStoreBadGotoErrors(t *testing.T) {
	cg := compileCachePolicyGraph(t, func(runtime.Runtime, map[string]any) (any, error) {
		return &types.Command{Goto: []any{42}}, nil
	}, &cacheErrCache{})
	if _, err := cg.Invoke(context.Background(), map[string]any{"x": 1}); err == nil ||
		!strings.Contains(err.Error(), "cache writes") {
		t.Fatalf("Invoke() error = %v, want a cache writes error", err)
	}
}

// --- delta overwrite forces snapshot ---------------------------------------

func TestDeltaOverwriteForcesSnapshotOnSave(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	// snapshotFrequency 100: without the Overwrite force-snapshot, the channel
	// would use sentinel storage at this cadence.
	g.AddChannel("items", channels.NewDeltaChannel(intBatchReducer, func() any { return []int{} }, 100))
	g.AddNode("a", func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"items": channels.NewOverwrite([]int{9})}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("InvokeWithOptions() error = %v", err)
	}
	if got := res.Values["items"]; !reflect.DeepEqual(got, []int{9}) {
		t.Fatalf("items = %v, want [9] (overwrite must win over the reducer)", got)
	}
	snap, err := cg.GetState(context.Background(), checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if got := snap.Values["items"]; !reflect.DeepEqual(got, []int{9}) {
		t.Fatalf("GetState items = %v, want [9] (overwrite must force a snapshot blob)", got)
	}
}

// --- runNode direct paths ---------------------------------------------------

func TestRunNodeUnknownNode(t *testing.T) {
	cg := compileLinear(t, noopNode)
	_, _, _, err := cg.runNode(context.Background(), task{node: "ghost"}, nil, nil, 1)
	if err == nil || !strings.Contains(err.Error(), `unknown node "ghost"`) {
		t.Fatalf("runNode() error = %v, want an unknown node error", err)
	}
}

func TestRunNodeNonInterruptPanicRepanics(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("panic", func(runtime.Runtime, map[string]any) (any, error) {
		panic("kaboom")
	})
	g.AddEdge(types.START, "panic")
	g.AddEdge("panic", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	defer func() {
		if r := recover(); r != "kaboom" {
			t.Fatalf("recover() = %v, want the node's non-interrupt panic to propagate", r)
		}
	}()
	_, _, _, _ = cg.runNode(context.Background(), task{node: "panic"}, nil, nil, 1)
}

// --- interrupt consumption accounting ---------------------------------------

func TestInterruptConsumeCountOutsideNode(t *testing.T) {
	if got := InterruptConsumeCount(context.Background()); got != 0 {
		t.Fatalf("InterruptConsumeCount() = %d, want 0 outside a node execution", got)
	}
	// ReplayInterruptConsumption is a no-op without interrupt state.
	ReplayInterruptConsumption(context.Background(), 3)
}

// TestInterruptConsumeCountAndReplay exercises the fn-package-facing
// consumption accounting on a node's interrupt state: ReplayInterruptConsumption
// advances the queue cursor as if values had been consumed (so the next
// Interrupt call skips them), InterruptConsumeCount reports the cursor, and
// generated interrupt IDs stay aligned with a full re-execution.
func TestInterruptConsumeCountAndReplay(t *testing.T) {
	st := &taskInterruptState{resumeQueue: []any{"a", "b", "c"}, nodeName: "n"}
	ctx := context.WithValue(context.Background(), interruptCtxKey{}, st)

	if got := InterruptConsumeCount(ctx); got != 0 {
		t.Fatalf("InterruptConsumeCount() = %d, want 0 before any consumption", got)
	}

	// A replayed execution already consumed "a": advancing by 1 must make the
	// next Interrupt call return "b", not "a".
	ReplayInterruptConsumption(ctx, 1)
	if got := InterruptConsumeCount(ctx); got != 1 {
		t.Fatalf("InterruptConsumeCount() = %d, want 1 after replaying one value", got)
	}
	if v := Interrupt(ctx, "q"); v != "b" {
		t.Fatalf("Interrupt() = %v, want %q (the replayed value must be skipped)", v, "b")
	}

	// Non-positive advances are no-ops.
	ReplayInterruptConsumption(ctx, 0)
	ReplayInterruptConsumption(ctx, -2)
	if got := InterruptConsumeCount(ctx); got != 2 {
		t.Fatalf("InterruptConsumeCount() = %d, want 2 (non-positive replays are no-ops)", got)
	}

	// Skip "c" too; the queue is now exhausted and the next Interrupt call
	// must pause with an ID aligned to a full re-execution (the 4th Interrupt
	// call of the invocation).
	ReplayInterruptConsumption(ctx, 1)
	defer func() {
		r := recover()
		gi, ok := r.(*types.GraphInterrupt)
		if !ok {
			t.Fatalf("recover() = %v, want a *types.GraphInterrupt", r)
		}
		if gi.Interrupt.ID != "n-4" {
			t.Fatalf("interrupt ID = %q, want %q (counter must include replayed consumptions)", gi.Interrupt.ID, "n-4")
		}
		if got := InterruptConsumeCount(ctx); got != 3 {
			t.Fatalf("InterruptConsumeCount() = %d, want 3 (a replayed, b consumed, c replayed)", got)
		}
	}()
	Interrupt(ctx, "final")
}

// TestStepBoundaryUpdateError verifies that a channel failing its empty
// step-boundary update (the untouched-channel notification in applyWrites)
// surfaces as the run's error.
func TestStepBoundaryUpdateError(t *testing.T) {
	g := NewStateGraph()
	g.AddChannel("bad", &errUpdateChannel{}) // errors only on the empty step-boundary update
	g.AddNode("a", func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"x": 2}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	// "bad" is seeded by the input batch; the node's superstep leaves it
	// untouched, so it receives the step-boundary update, which errors.
	if _, err := cg.Invoke(context.Background(), map[string]any{"bad": 1, "x": 1}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("Invoke() error = %v, want %v", err, errSaverBoom)
	}
}

// TestNilInputResumeReplayWritesError covers the same replay failure as
// TestResumeReplayWritesError but through the nil-input boundary-resume
// branch (no explicit Options.Resume).
func TestNilInputResumeReplayWritesError(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddChannel("bad", &errUpdateChannel{failAll: true})
	g.AddNode("entry", noopNode)
	g.AddNode("sibling", func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"bad": 1}, nil
	})
	g.AddNode("ask", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		Interrupt(ctx, "q")
		return nil, nil
	})
	g.AddEdge(types.START, "entry")
	g.AddConditionalEdges("entry", func(runtime.Runtime, map[string]any) ([]any, error) {
		return To("sibling", "ask"), nil
	})
	g.AddEdge("sibling", types.END)
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("InvokeWithOptions() error = %v", err)
	}
	// Nil input, no explicit Resume: the resumeFromTuple call on the
	// boundary-resume branch replays the sibling's "bad" write and fails.
	if _, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("resume error = %v, want %v", err, errSaverBoom)
	}
}

// TestNewTurnInputCheckpointSaveError verifies that a save failure in the
// new-turn (fresh input on an existing checkpoint) input-checkpoint branch
// surfaces as the run's error.
func TestNewTurnInputCheckpointSaveError(t *testing.T) {
	// Puts: run 1 saves input (1) and loop (2); run 2's new-turn input save is
	// the third.
	saver := &failNthPutSaver{Saver: checkpoint.NewMemorySaver(), n: 3}
	cg := compileLinear(t, noopNode, WithCheckpointer(saver))
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"x": 1}, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("first InvokeWithOptions() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"x": 2}, Options{ThreadID: "t"}); !errors.Is(err, errSaverBoom) {
		t.Fatalf("second InvokeWithOptions() error = %v, want %v", err, errSaverBoom)
	}
}
