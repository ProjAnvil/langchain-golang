package fn

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/types"
)

func TestResolvedFutureValue(t *testing.T) {
	f := resolvedFuture(42, nil, nil)
	v, err := f.Get(context.Background())
	if err != nil {
		t.Fatalf("Get error = %v, want nil", err)
	}
	if v != 42 {
		t.Fatalf("Get value = %v, want 42", v)
	}
}

func TestResolvedFutureError(t *testing.T) {
	boom := errors.New("boom")
	f := resolvedFuture(0, boom, nil)
	v, err := f.Get(context.Background())
	if !errors.Is(err, boom) {
		t.Fatalf("Get error = %v, want boom", err)
	}
	if v != 0 {
		t.Fatalf("Get value = %v, want 0", v)
	}
}

func TestFutureGetPanicsGraphInterrupt(t *testing.T) {
	gi := &types.GraphInterrupt{Interrupt: types.Interrupt{Value: "q", ID: "i1"}}
	f := resolvedFuture(0, nil, gi)
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("Get did not panic on a GraphInterrupt future")
		}
		got, ok := r.(*types.GraphInterrupt)
		if !ok {
			t.Fatalf("panic value = %T (%v), want *types.GraphInterrupt", r, r)
		}
		if got != gi {
			t.Fatalf("panic value = %p, want the same pointer %p", got, gi)
		}
	}()
	_, _ = f.Get(context.Background())
}

func TestFutureGetBlocksUntilDone(t *testing.T) {
	f := &Future[int]{done: make(chan struct{})}
	got := make(chan int, 1)
	go func() {
		v, err := f.Get(context.Background())
		if err != nil {
			t.Errorf("Get error = %v, want nil", err)
		}
		got <- v
	}()

	select {
	case <-got:
		t.Fatal("Get returned before the future resolved")
	case <-time.After(50 * time.Millisecond):
	}

	f.val = 7 // happens-before close(done)
	close(f.done)

	select {
	case v := <-got:
		if v != 7 {
			t.Fatalf("Get value = %v, want 7", v)
		}
	case <-time.After(time.Second):
		t.Fatal("Get did not return after close(done)")
	}
}

func TestFutureGetCanceledContext(t *testing.T) {
	f := &Future[int]{done: make(chan struct{})}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := f.Get(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("Get error = %v, want context.Canceled", err)
	}
}

func TestAwaitAllInOrder(t *testing.T) {
	f1 := resolvedFuture(1, nil, nil)
	f2 := resolvedFuture(2, nil, nil)
	f3 := resolvedFuture(3, nil, nil)
	vs, err := AwaitAll(context.Background(), f1, f2, f3)
	if err != nil {
		t.Fatalf("AwaitAll error = %v, want nil", err)
	}
	if len(vs) != 3 || vs[0] != 1 || vs[1] != 2 || vs[2] != 3 {
		t.Fatalf("AwaitAll values = %v, want [1 2 3]", vs)
	}
}

func TestAwaitAllError(t *testing.T) {
	boom := errors.New("boom")
	f1 := resolvedFuture(1, nil, nil)
	f2 := resolvedFuture(0, boom, nil)
	f3 := resolvedFuture(3, nil, nil)
	if _, err := AwaitAll(context.Background(), f1, f2, f3); !errors.Is(err, boom) {
		t.Fatalf("AwaitAll error = %v, want boom", err)
	}
}
