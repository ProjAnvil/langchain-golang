package graph

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

var errFlaky = errors.New("flaky failure")

func alwaysRetry(error) bool { return true }

// fastRetryPolicy returns a policy with near-zero backoff so retry tests run
// fast, jitter disabled so intervals are deterministic.
func fastRetryPolicy(maxAttempts int) *RetryPolicy {
	return &RetryPolicy{
		InitialInterval: time.Millisecond,
		BackoffFactor:   2,
		MaxInterval:     10 * time.Millisecond,
		MaxAttempts:     maxAttempts,
		Jitter:          false,
		RetryOn:         alwaysRetry,
	}
}

func compileRetryGraph(t *testing.T, fn NodeFunc, policies NodePolicies) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNodeWithPolicies("node", fn, policies)
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

func TestRetryPolicyFlakyNodeSucceeds(t *testing.T) {
	var attempts atomic.Int32
	cg := compileRetryGraph(t, func(_ context.Context, _ map[string]any) (any, error) {
		if attempts.Add(1) < 3 {
			return nil, errFlaky
		}
		return map[string]any{"done": true}, nil
	}, NodePolicies{Retry: fastRetryPolicy(3)})

	result, err := cg.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if got := attempts.Load(); got != 3 {
		t.Fatalf("attempts = %d, want 3", got)
	}
	if result.Values["done"] != true {
		t.Fatalf("done = %v, want true", result.Values["done"])
	}
}

func TestRetryPolicyNonRetryableFailsImmediately(t *testing.T) {
	var attempts atomic.Int32
	// RetryOn nil -> DefaultRetryOn, which does not retry a plain error.
	cg := compileRetryGraph(t, func(_ context.Context, _ map[string]any) (any, error) {
		attempts.Add(1)
		return nil, errFlaky
	}, NodePolicies{Retry: &RetryPolicy{InitialInterval: time.Millisecond, MaxAttempts: 5, Jitter: false}})

	_, err := cg.Invoke(context.Background(), nil)
	if !errors.Is(err, errFlaky) {
		t.Fatalf("Invoke() error = %v, want %v", err, errFlaky)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1 (non-retryable error must not retry)", got)
	}
}

func TestRetryPolicyMaxAttemptsExhaustedSurfacesLastError(t *testing.T) {
	var attempts atomic.Int32
	cg := compileRetryGraph(t, func(_ context.Context, _ map[string]any) (any, error) {
		n := attempts.Add(1)
		return nil, fmt.Errorf("failure %d", n)
	}, NodePolicies{Retry: fastRetryPolicy(3)})

	_, err := cg.Invoke(context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), "failure 3") {
		t.Fatalf("Invoke() error = %v, want the last attempt's error (failure 3)", err)
	}
	if got := attempts.Load(); got != 3 {
		t.Fatalf("attempts = %d, want 3", got)
	}
}

func TestRetryPolicyBackoffIncreases(t *testing.T) {
	var mu sync.Mutex
	var attemptTimes []time.Time
	cg := compileRetryGraph(t, func(_ context.Context, _ map[string]any) (any, error) {
		mu.Lock()
		attemptTimes = append(attemptTimes, time.Now())
		mu.Unlock()
		return nil, errFlaky
	}, NodePolicies{Retry: &RetryPolicy{
		InitialInterval: 40 * time.Millisecond,
		BackoffFactor:   2,
		MaxInterval:     time.Minute,
		MaxAttempts:     3,
		Jitter:          false, // jitter off: intervals are deterministic
		RetryOn:         alwaysRetry,
	}})

	start := time.Now()
	_, err := cg.Invoke(context.Background(), nil)
	elapsed := time.Since(start)
	if !errors.Is(err, errFlaky) {
		t.Fatalf("Invoke() error = %v, want %v", err, errFlaky)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(attemptTimes) != 3 {
		t.Fatalf("attempts = %d, want 3", len(attemptTimes))
	}
	gap1 := attemptTimes[1].Sub(attemptTimes[0])
	gap2 := attemptTimes[2].Sub(attemptTimes[1])
	// Loose bounds (no clock injection): gap1 ~= 40ms, gap2 ~= 80ms.
	if gap1 < 30*time.Millisecond || gap1 > 2*time.Second {
		t.Fatalf("first backoff = %v, want ~= 40ms", gap1)
	}
	if gap2 < 65*time.Millisecond || gap2 > 4*time.Second {
		t.Fatalf("second backoff = %v, want ~= 80ms", gap2)
	}
	if gap2 <= gap1 {
		t.Fatalf("backoff did not increase: gap1 = %v, gap2 = %v", gap1, gap2)
	}
	if elapsed < 100*time.Millisecond {
		t.Fatalf("total elapsed = %v, want >= ~120ms of backoff", elapsed)
	}
}

func TestRetryPolicyInterruptIsTerminal(t *testing.T) {
	var attempts atomic.Int32
	cg := compileRetryGraph(t, func(ctx context.Context, _ map[string]any) (any, error) {
		attempts.Add(1)
		Interrupt(ctx, "pause")
		return nil, nil // unreachable
	}, NodePolicies{Retry: fastRetryPolicy(3)})

	result, err := cg.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(result.Interrupts) != 1 {
		t.Fatalf("Interrupts = %+v, want exactly 1", result.Interrupts)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1 (interrupted tasks are never re-executed)", got)
	}
}

func TestRetryPolicyContextCancelDuringBackoff(t *testing.T) {
	var attempts atomic.Int32
	cg := compileRetryGraph(t, func(_ context.Context, _ map[string]any) (any, error) {
		attempts.Add(1)
		return nil, errFlaky
	}, NodePolicies{Retry: &RetryPolicy{
		InitialInterval: 30 * time.Second, // long backoff: cancel must interrupt it
		MaxAttempts:     3,
		Jitter:          false,
		RetryOn:         alwaysRetry,
	}})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	time.AfterFunc(50*time.Millisecond, cancel)

	start := time.Now()
	_, err := cg.Invoke(ctx, nil)
	elapsed := time.Since(start)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Invoke() error = %v, want the parent ctx error (context.Canceled), not the node's %v", err, errFlaky)
	}
	if elapsed > 10*time.Second {
		t.Fatalf("elapsed = %v, want the backoff aborted promptly on cancel", elapsed)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1 (cancel during the first backoff)", got)
	}
}

// retryRecordingSink records RawEvents for balanced start/end assertions.
type retryRecordingSink struct {
	mu     sync.Mutex
	events []RawEvent
}

func (s *retryRecordingSink) EmitRawEvent(ev RawEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, ev)
}

func (s *retryRecordingSink) count(kind RawEventKind, node string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, ev := range s.events {
		if ev.Kind == kind && ev.Node == node {
			n++
		}
	}
	return n
}

func TestRetryPolicyEventsBalancedAcrossAttempts(t *testing.T) {
	var attempts atomic.Int32
	cg := compileRetryGraph(t, func(_ context.Context, _ map[string]any) (any, error) {
		if attempts.Add(1) < 3 {
			return nil, errFlaky
		}
		return nil, nil
	}, NodePolicies{Retry: fastRetryPolicy(3)})

	sink := &retryRecordingSink{}
	if _, err := cg.InvokeStream(context.Background(), nil, Options{}, sink); err != nil {
		t.Fatalf("InvokeStream() error = %v", err)
	}
	if got := attempts.Load(); got != 3 {
		t.Fatalf("attempts = %d, want 3", got)
	}
	if got := sink.count(RawNodeStart, "node"); got != 1 {
		t.Fatalf("RawNodeStart count = %d, want exactly 1 across all attempts", got)
	}
	if got := sink.count(RawNodeEnd, "node"); got != 1 {
		t.Fatalf("RawNodeEnd count = %d, want exactly 1 across all attempts", got)
	}
}

func TestCompileValidatesRetryPolicy(t *testing.T) {
	cases := map[string]*RetryPolicy{
		"negative InitialInterval": {InitialInterval: -time.Second},
		"negative MaxInterval":     {MaxInterval: -time.Second},
		"MaxAttempts below 1":      {MaxAttempts: -1},
	}
	for name, policy := range cases {
		t.Run(name, func(t *testing.T) {
			g := NewStateGraph()
			g.AddNodeWithPolicies("node", func(_ context.Context, _ map[string]any) (any, error) {
				return nil, nil
			}, NodePolicies{Retry: policy})
			g.AddEdge(types.START, "node")
			g.AddEdge("node", types.END)
			if _, err := g.Compile(); err == nil {
				t.Fatalf("Compile() error = nil, want a policy validation error")
			}
		})
	}
}

// httpStatusErr implements `interface{ HTTPStatus() int }` (see DefaultRetryOn).
type httpStatusErr struct{ status int }

func (e httpStatusErr) Error() string { return fmt.Sprintf("http status %d", e.status) }
func (e httpStatusErr) HTTPStatus() int {
	return e.status
}

func TestDefaultRetryOn(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"plain error", errFlaky, false},
		{"net.Error", &net.DNSError{Err: "timeout", Name: "example.com", IsTimeout: true}, true},
		{"wrapped net.Error", fmt.Errorf("call: %w", &net.DNSError{Err: "timeout", Name: "example.com", IsTimeout: true}), true},
		{"context.DeadlineExceeded", context.DeadlineExceeded, true},
		{"context.Canceled", context.Canceled, false},
		{"HTTP 500", httpStatusErr{500}, true},
		{"HTTP 503", httpStatusErr{503}, true},
		{"HTTP 404", httpStatusErr{404}, false},
		{"InvalidUpdateError", &channels.InvalidUpdateError{Channel: "LastValue", Reason: "too many writes"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := DefaultRetryOn(tc.err); got != tc.want {
				t.Fatalf("DefaultRetryOn(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestAddNodeDelegatesToAddNodeWithPolicies(t *testing.T) {
	// AddNode remains the zero-policies path: a node that fails with a
	// retryable-matching error must NOT be retried (no policy installed).
	var attempts atomic.Int32
	g := NewStateGraph()
	g.AddNode("node", func(_ context.Context, _ map[string]any) (any, error) {
		attempts.Add(1)
		return nil, context.DeadlineExceeded // DefaultRetryOn-retryable, but no policy
	})
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.Invoke(context.Background(), nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Invoke() error = %v, want context.DeadlineExceeded", err)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1 (AddNode installs no retry policy)", got)
	}
}
