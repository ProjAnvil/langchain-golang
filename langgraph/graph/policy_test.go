package graph

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
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
		NoJitter:        true,
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
	cg := compileRetryGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
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
	// RetryOn nil -> DefaultRetryOn, which retries plain errors but never a
	// NonRetryable-wrapped one (the exported opt-out, see NonRetryable).
	cg := compileRetryGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
		attempts.Add(1)
		return nil, NonRetryable(errFlaky)
	}, NodePolicies{Retry: &RetryPolicy{InitialInterval: time.Millisecond, MaxAttempts: 5, NoJitter: true}})

	_, err := cg.Invoke(context.Background(), nil)
	if !errors.Is(err, errFlaky) {
		t.Fatalf("Invoke() error = %v, want %v (NonRetryable must not mask the wrapped error)", err, errFlaky)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1 (non-retryable error must not retry)", got)
	}
}

func TestRetryPolicyMaxAttemptsExhaustedSurfacesLastError(t *testing.T) {
	var attempts atomic.Int32
	cg := compileRetryGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
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
	cg := compileRetryGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
		mu.Lock()
		attemptTimes = append(attemptTimes, time.Now())
		mu.Unlock()
		return nil, errFlaky
	}, NodePolicies{Retry: &RetryPolicy{
		InitialInterval: 40 * time.Millisecond,
		BackoffFactor:   2,
		MaxInterval:     time.Minute,
		MaxAttempts:     3,
		NoJitter:        true, // jitter off: intervals are deterministic
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
	cg := compileRetryGraph(t, func(ctx runtime.Runtime, _ map[string]any) (any, error) {
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
	cg := compileRetryGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
		attempts.Add(1)
		return nil, errFlaky
	}, NodePolicies{Retry: &RetryPolicy{
		InitialInterval: 30 * time.Second, // long backoff: cancel must interrupt it
		MaxAttempts:     3,
		NoJitter:        true,
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
	cg := compileRetryGraph(t, func(_ runtime.Runtime, _ map[string]any) (any, error) {
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
		"negative BackoffFactor":   {BackoffFactor: -1},
		"NaN BackoffFactor":        {BackoffFactor: math.NaN()},
		"MaxAttempts below 1":      {MaxAttempts: -1},
	}
	for name, policy := range cases {
		t.Run(name, func(t *testing.T) {
			g := NewStateGraph()
			g.AddNodeWithPolicies("node", func(_ runtime.Runtime, _ map[string]any) (any, error) {
				return nil, nil
			}, NodePolicies{Retry: policy})
			g.AddEdge(types.START, "node")
			g.AddEdge("node", types.END)
			_, err := g.Compile()
			if err == nil {
				t.Fatalf("Compile() error = nil, want a policy validation error")
			}
			if !strings.Contains(err.Error(), `"node"`) {
				t.Fatalf("Compile() error = %v, want it to name the node", err)
			}
		})
	}
}

func TestRetryPolicyBackoffJitterEnabled(t *testing.T) {
	// Jitter on (the zero-value default): each delay is the clamped base plus
	// a uniform random [0, 1s), so it must never fall below the base.
	p := RetryPolicy{
		InitialInterval: 10 * time.Millisecond,
		BackoffFactor:   2,
		MaxInterval:     40 * time.Millisecond,
	}
	for attempt := 1; attempt <= 20; attempt++ {
		base := time.Duration(float64(p.InitialInterval) * math.Pow(p.BackoffFactor, float64(attempt-1)))
		if base > p.MaxInterval {
			base = p.MaxInterval
		}
		got := p.backoff(attempt)
		if got < base {
			t.Fatalf("backoff(%d) = %v, want >= clamped base %v", attempt, got, base)
		}
		if got > base+time.Second {
			t.Fatalf("backoff(%d) = %v, want <= base %v + 1s of jitter", attempt, got, base)
		}
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
		// The relaxed default retries everything except its explicit
		// exclusions (Python _retry.py parity: default_retry_on returns True
		// for anything outside its programming-error list).
		{"plain error", errFlaky, true},
		{"wrapped plain error", fmt.Errorf("call: %w", errFlaky), true},
		{"net.Error", &net.DNSError{Err: "timeout", Name: "example.com", IsTimeout: true}, true},
		{"wrapped net.Error", fmt.Errorf("call: %w", &net.DNSError{Err: "timeout", Name: "example.com", IsTimeout: true}), true},
		{"context.DeadlineExceeded", context.DeadlineExceeded, true},
		{"wrapped context.DeadlineExceeded", fmt.Errorf("call: %w", context.DeadlineExceeded), true},
		{"HTTP 429 wrapped", fmt.Errorf("provider: %w", httpStatusErr{429}), true},
		{"HTTP 404", httpStatusErr{404}, true},
		{"HTTP 500", httpStatusErr{500}, true},
		{"HTTP 503", httpStatusErr{503}, true},
		// Exclusions.
		{"context.Canceled", context.Canceled, false},
		{"wrapped context.Canceled", fmt.Errorf("call: %w", context.Canceled), false},
		{"NonRetryable", NonRetryable(errFlaky), false},
		{"NonRetryable nested in %w", fmt.Errorf("call: %w", NonRetryable(errFlaky)), false},
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

func TestNonRetryableWrapping(t *testing.T) {
	if NonRetryable(nil) != nil {
		t.Fatal("NonRetryable(nil) != nil, want nil")
	}
	err := NonRetryable(errFlaky)
	if !errors.Is(err, errFlaky) {
		t.Fatalf("errors.Is(NonRetryable(errFlaky), errFlaky) = false, want true (sentinel matching must survive the wrap)")
	}
	if err.Error() != errFlaky.Error() {
		t.Fatalf("Error() = %q, want %q (the wrapper is transparent in message)", err.Error(), errFlaky.Error())
	}
}

func TestAddNodeDelegatesToAddNodeWithPolicies(t *testing.T) {
	// AddNode remains the zero-policies path: a node that fails with a
	// retryable-matching error must NOT be retried (no policy installed).
	var attempts atomic.Int32
	g := NewStateGraph()
	g.AddNode("node", func(_ runtime.Runtime, _ map[string]any) (any, error) {
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

func TestRetryPolicyResolvedAppliesDefaults(t *testing.T) {
	resolved := RetryPolicy{}.Resolved()
	if resolved.InitialInterval != defaultInitialInterval ||
		resolved.BackoffFactor != defaultBackoffFactor ||
		resolved.MaxInterval != defaultMaxInterval ||
		resolved.MaxAttempts != defaultMaxAttempts {
		t.Fatalf("Resolved() = %+v, want all defaults applied", resolved)
	}
	if resolved.RetryOn == nil {
		t.Fatal("Resolved().RetryOn = nil, want DefaultRetryOn")
	}

	// Set fields are preserved.
	custom := RetryPolicy{InitialInterval: time.Second, MaxAttempts: 7, NoJitter: true}
	resolved = custom.Resolved()
	if resolved.InitialInterval != time.Second || resolved.MaxAttempts != 7 || !resolved.NoJitter {
		t.Fatalf("Resolved() = %+v, want set fields preserved", resolved)
	}
}

func TestRetryPolicyBackoffDelay(t *testing.T) {
	p := RetryPolicy{
		InitialInterval: 10 * time.Millisecond,
		BackoffFactor:   2,
		MaxInterval:     25 * time.Millisecond,
		NoJitter:        true,
	}
	cases := []struct {
		attempt int
		want    time.Duration
	}{
		{1, 10 * time.Millisecond},
		{2, 20 * time.Millisecond},
		{3, 25 * time.Millisecond}, // clamped to MaxInterval
		{9, 25 * time.Millisecond},
	}
	for _, tc := range cases {
		if got := p.BackoffDelay(tc.attempt); got != tc.want {
			t.Errorf("BackoffDelay(%d) = %v, want %v", tc.attempt, got, tc.want)
		}
	}
}

func TestTimeoutPolicyRejectsNegativeIdleTimeout(t *testing.T) {
	g := NewStateGraph()
	g.AddNodeWithPolicies("node", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return nil, nil
	}, NodePolicies{Timeout: &TimeoutPolicy{RunTimeout: time.Second, IdleTimeout: -time.Second}})
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	if _, err := g.Compile(); err == nil || !strings.Contains(err.Error(), "IdleTimeout") {
		t.Fatalf("Compile() error = %v, want an IdleTimeout validation error", err)
	}
}

// deadlineOnlyViaIs matched context.DeadlineExceeded through a custom Is
// method; under the relaxed DefaultRetryOn every error outside the exclusion
// set is retryable, so the branch-specific coverage is obsolete and the case
// is covered by the plain-error rows of TestDefaultRetryOn.

// --- graph-level default retry policy (WithDefaultRetryPolicy) ---

// flakyNode returns a NodeFunc failing with errFlaky until its attempts
// counter reaches succeedAt, then returning a done write. Pass a huge
// succeedAt for an always-failing node.
func flakyNode(attempts *atomic.Int32, succeedAt int32) NodeFunc {
	return func(_ runtime.Runtime, _ map[string]any) (any, error) {
		if attempts.Add(1) < succeedAt {
			return nil, errFlaky
		}
		return map[string]any{"done": true}, nil
	}
}

func TestWithDefaultRetryPolicyRetriesNodesWithoutOwnPolicy(t *testing.T) {
	// The node was added via AddNode (no own policy). The graph-level default
	// applies, and its nil RetryOn -> the relaxed DefaultRetryOn, so the PLAIN
	// errFlaky is retried and the run succeeds on attempt 3.
	var attempts atomic.Int32
	g := NewStateGraph()
	g.AddNode("node", flakyNode(&attempts, 3))
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg, err := g.Compile(WithDefaultRetryPolicy(&RetryPolicy{
		InitialInterval: time.Millisecond,
		MaxAttempts:     3,
		NoJitter:        true,
	}))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	result, err := cg.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if got := attempts.Load(); got != 3 {
		t.Fatalf("attempts = %d, want 3 (graph default retried a plain error)", got)
	}
	if result.Values["done"] != true {
		t.Fatalf("done = %v, want true", result.Values["done"])
	}
}

func TestPerNodeRetryPolicyOverridesGraphDefault(t *testing.T) {
	// The node's own policy (MaxAttempts 2) wins over the graph default
	// (MaxAttempts 5): the run fails after exactly 2 attempts.
	var attempts atomic.Int32
	g := NewStateGraph()
	g.AddNodeWithPolicies("node", flakyNode(&attempts, math.MaxInt32), NodePolicies{Retry: fastRetryPolicy(2)})
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg, err := g.Compile(WithDefaultRetryPolicy(&RetryPolicy{
		InitialInterval: time.Millisecond,
		MaxAttempts:     5,
		NoJitter:        true,
		RetryOn:         alwaysRetry,
	}))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	_, err = cg.Invoke(context.Background(), nil)
	if !errors.Is(err, errFlaky) {
		t.Fatalf("Invoke() error = %v, want %v", err, errFlaky)
	}
	if got := attempts.Load(); got != 2 {
		t.Fatalf("attempts = %d, want 2 (per-node policy overrides the graph default)", got)
	}
}

func TestGraphDefaultAppliesToNodeWithOtherPoliciesOnly(t *testing.T) {
	// A node registered via AddNodeWithPolicies with only a Timeout policy has
	// Retry == nil, so the graph-level default still applies.
	var attempts atomic.Int32
	g := NewStateGraph()
	g.AddNodeWithPolicies("node", flakyNode(&attempts, 3), NodePolicies{Timeout: &TimeoutPolicy{RunTimeout: time.Minute}})
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg, err := g.Compile(WithDefaultRetryPolicy(fastRetryPolicy(3)))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	if _, err := cg.Invoke(context.Background(), nil); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if got := attempts.Load(); got != 3 {
		t.Fatalf("attempts = %d, want 3 (Retry-less NodePolicies falls back to the graph default)", got)
	}
}

func TestWithoutDefaultRetryPolicyNoRetry(t *testing.T) {
	// No graph default, no per-node policy: never retried, even for an error
	// DefaultRetryOn would happily retry.
	var attempts atomic.Int32
	g := NewStateGraph()
	g.AddNode("node", flakyNode(&attempts, math.MaxInt32))
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	_, err = cg.Invoke(context.Background(), nil)
	if !errors.Is(err, errFlaky) {
		t.Fatalf("Invoke() error = %v, want %v", err, errFlaky)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1 (no policy anywhere means no retry)", got)
	}
}

func TestWithDefaultRetryPolicyNilDisables(t *testing.T) {
	var attempts atomic.Int32
	g := NewStateGraph()
	g.AddNode("node", flakyNode(&attempts, math.MaxInt32))
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg, err := g.Compile(WithDefaultRetryPolicy(nil))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	if _, err := cg.Invoke(context.Background(), nil); !errors.Is(err, errFlaky) {
		t.Fatalf("Invoke() error = %v, want %v", err, errFlaky)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1 (nil default = no retry)", got)
	}
}

func TestWithDefaultRetryPolicyNonRetryableNotRetried(t *testing.T) {
	// The graph default has a nil RetryOn -> DefaultRetryOn, which retries
	// everything EXCEPT a NonRetryable-wrapped error: exactly one attempt.
	// (A custom RetryOn keeps full control, mirroring Python's custom
	// retry_on tuples ignoring the default's exclusions.)
	var attempts atomic.Int32
	g := NewStateGraph()
	g.AddNode("node", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		attempts.Add(1)
		return nil, NonRetryable(errFlaky)
	})
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg, err := g.Compile(WithDefaultRetryPolicy(&RetryPolicy{
		InitialInterval: time.Millisecond,
		MaxAttempts:     5,
		NoJitter:        true,
	}))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	_, err = cg.Invoke(context.Background(), nil)
	if !errors.Is(err, errFlaky) {
		t.Fatalf("Invoke() error = %v, want %v", err, errFlaky)
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1 (NonRetryable beats the graph default)", got)
	}
}

func TestCompileValidatesDefaultRetryPolicy(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("node", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return nil, nil
	})
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	_, err := g.Compile(WithDefaultRetryPolicy(&RetryPolicy{MaxAttempts: -1}))
	if err == nil || !strings.Contains(err.Error(), "default retry policy") {
		t.Fatalf("Compile() error = %v, want a default retry policy validation error", err)
	}
}
