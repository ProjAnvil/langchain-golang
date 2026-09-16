package graph

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// RetryPolicy configures per-node automatic retry with exponential backoff,
// mirroring Python's `langgraph.types.RetryPolicy` (`pregel/_retry.py`). It is
// installed per node via StateGraph.AddNodeWithPolicies or graph-wide as the
// compile-time default via WithDefaultRetryPolicy (a node's own policy always
// takes precedence); nodes with neither carry no policy and are never
// retried.
//
// The retry loop lives in the executor's task wrapper (see
// CompiledGraph.runTask); its interrupt, resume, event, and cancellation
// semantics are documented there.
type RetryPolicy struct {
	// InitialInterval is the backoff before the first retry. Default 500ms.
	InitialInterval time.Duration
	// BackoffFactor multiplies the backoff interval after each failed
	// attempt. Default 2.0.
	BackoffFactor float64
	// MaxInterval caps the backoff interval (before jitter). Default 128s.
	MaxInterval time.Duration
	// MaxAttempts bounds the total number of node executions, including the
	// first attempt. Default 3.
	MaxAttempts int
	// NoJitter disables the jitter added to each backoff interval. Jitter is
	// a uniform random [0, 1s) applied after the MaxInterval clamp (Python
	// parity: `random.uniform(0, 1)`); it is enabled by default — matching
	// Python — unless NoJitter is true.
	NoJitter bool
	// RetryOn decides whether a failed attempt's error is retryable. Nil
	// means DefaultRetryOn.
	RetryOn func(err error) bool
}

// RetryPolicy defaults, mirroring Python's `RetryPolicy` field defaults.
const (
	defaultInitialInterval = 500 * time.Millisecond
	defaultBackoffFactor   = 2.0
	defaultMaxInterval     = 128 * time.Second
	defaultMaxAttempts     = 3
)

// withDefaults returns a copy of p with every unset (zero) field replaced by
// its default. NoJitter is left untouched: its zero value means jitter on,
// which is the Python-parity default.
func (p RetryPolicy) withDefaults() RetryPolicy {
	if p.InitialInterval == 0 {
		p.InitialInterval = defaultInitialInterval
	}
	if p.BackoffFactor == 0 {
		p.BackoffFactor = defaultBackoffFactor
	}
	if p.MaxInterval == 0 {
		p.MaxInterval = defaultMaxInterval
	}
	if p.MaxAttempts == 0 {
		p.MaxAttempts = defaultMaxAttempts
	}
	if p.RetryOn == nil {
		p.RetryOn = DefaultRetryOn
	}
	return p
}

// validate checks p's effective (defaults-applied) values, returning a
// descriptive error for the configurations Compile rejects.
func (p RetryPolicy) validate() error {
	eff := p.withDefaults()
	if eff.InitialInterval < 0 {
		return errors.New("InitialInterval must be >= 0")
	}
	if eff.MaxInterval < 0 {
		return errors.New("MaxInterval must be >= 0")
	}
	if eff.BackoffFactor < 0 || math.IsNaN(eff.BackoffFactor) {
		return errors.New("BackoffFactor must be >= 0 and not NaN")
	}
	if eff.MaxAttempts < 1 {
		return errors.New("MaxAttempts must be >= 1")
	}
	return nil
}

// backoff returns the delay before re-executing after the given (1-based)
// failed attempt: min(MaxInterval, InitialInterval * BackoffFactor^(attempt-1))
// plus, unless NoJitter is set, a uniform random [0, 1s) — Python parity
// (`pregel/_retry.py`: the clamp applies before jitter is added).
func (p RetryPolicy) backoff(attempt int) time.Duration {
	interval := float64(p.InitialInterval) * math.Pow(p.BackoffFactor, float64(attempt-1))
	delay := time.Duration(math.Min(interval, float64(p.MaxInterval)))
	if !p.NoJitter {
		delay += time.Duration(rand.Float64() * float64(time.Second))
	}
	return delay
}

// Resolved returns a copy of p with every unset field replaced by its
// default (see withDefaults). Exported for the fn package's task retry loop.
func (p RetryPolicy) Resolved() RetryPolicy { return p.withDefaults() }

// BackoffDelay returns the delay before re-executing after the given
// (1-based) failed attempt (see backoff). Exported for the fn package.
func (p RetryPolicy) BackoffDelay(attempt int) time.Duration { return p.backoff(attempt) }

// DefaultRetryOn is the default RetryPolicy.RetryOn. It retries EVERY error
// except a small exclusion set, aligning with Python's `default_retry_on`
// (`langgraph/_internal/_retry.py`), which returns True for any exception
// outside its programming-error list (ValueError/TypeError & co.).
//
// Exclusions (never retried):
//
//   - errors wrapped by NonRetryable — the exported opt-out for callers who
//     know an error is permanent (the Go analogue of Python's exception-class
//     list: Go error values carry no such hierarchy, so the classification is
//     explicit instead of type-based);
//   - context.Canceled — the analogue of Python's CancelledError exclusion:
//     the parent run was aborted, so retrying is pointless (the retry loop
//     itself also aborts on ctx.Done() during backoff; see runTask);
//   - *channels.InvalidUpdateError — a write-path programming error (two
//     writes to a non-merging channel in one superstep); retrying a
//     deterministic bug never helps.
//
// Note the previous narrow predicate (net.Error / DeadlineExceeded / 5xx
// HTTPStatus only) is gone: a provider-wrapped 429 or any other transient
// plain error is now retried by default. Callers wanting the old behavior
// supply their own RetryOn. GraphInterrupt is not an error at all (it is a
// panic, converted to a terminal interrupted outcome before the retry loop —
// see runTask).
func DefaultRetryOn(err error) bool {
	if _, ok := errors.AsType[*channels.InvalidUpdateError](err); ok {
		return false
	}
	if errors.Is(err, context.Canceled) {
		return false
	}
	if _, ok := errors.AsType[*nonRetryableError](err); ok {
		return false
	}
	return true
}

// NonRetryable wraps err into an error that DefaultRetryOn will never retry —
// the exported exclusion mechanism for the retry-everything default: Python's
// default_retry_on refuses programming-error exception classes, while Go
// runtime errors carry no such hierarchy, so an error opts out explicitly by
// wrapping. A custom RetryOn keeps full control (mirroring Python's custom
// retry_on tuples, which ignore the default's exclusions); one that wants to
// honor the marker should delegate to DefaultRetryOn.
//
// The wrapper is transparent: Error() and Unwrap() both delegate to err, so
// errors.Is/As matching against the original sentinel keeps working through
// any number of intermediate fmt.Errorf("%w") wraps. NonRetryable(nil)
// returns nil.
func NonRetryable(err error) error {
	if err == nil {
		return nil
	}
	return &nonRetryableError{err: err}
}

// nonRetryableError is the marker type installed by NonRetryable; matched via
// errors.As so the exclusion survives wrapping.
type nonRetryableError struct{ err error }

func (e *nonRetryableError) Error() string { return e.err.Error() }

func (e *nonRetryableError) Unwrap() error { return e.err }

// NodePolicies bundles the optional per-node execution policies installed via
// StateGraph.AddNodeWithPolicies. A nil field means the corresponding policy
// is disabled for the node.
type NodePolicies struct {
	// Retry enables automatic retry of the node's failures (see RetryPolicy).
	Retry *RetryPolicy
	// Cache enables write caching for the node (see CachePolicy).
	Cache *CachePolicy
	// Timeout caps a single node attempt's wall-clock and idle time (see
	// TimeoutPolicy). Mirrors Python's langgraph.types.TimeoutPolicy.
	Timeout *TimeoutPolicy
}

// TimeoutPolicy configures per-node attempt timeouts, mirroring Python's
// langgraph.types.TimeoutPolicy (types.py:450). Installed per node via
// StateGraph.AddNodeWithPolicies. Both fields are optional; a zero-valued
// policy disables timeout for the node.
//
//   - RunTimeout is a hard wall-clock cap on a single node attempt. It is
//     never refreshed. Implemented as a context deadline layered on the
//     node's context, so a node that respects rt.Done()/rt.Err() aborts with
//     context.DeadlineExceeded when it fires.
//   - IdleTimeout is the maximum time a single attempt may go without
//     observable progress. RefreshOn selects the progress signal:
//     "heartbeat" (default "auto") refreshes on rt.Heartbeat() calls.
//
// Cooperative cancellation: Go cannot forcibly kill a goroutine, so a node
// that blocks without checking rt.Done() will overrun the deadline (the same
// limitation Python documents for sync work under asyncio). The watchdog
// goroutine still fires the cancel so well-behaved nodes abort promptly.
type TimeoutPolicy struct {
	// RunTimeout is the hard wall-clock cap per attempt. 0 = no run cap.
	RunTimeout time.Duration
	// IdleTimeout is the max interval without progress per attempt. 0 = no
	// idle cap.
	IdleTimeout time.Duration
	// RefreshOn selects which signals refresh IdleTimeout: "auto" (default)
	// or "heartbeat". "auto" currently refreshes on explicit heartbeats and
	// stream writes (callback-event auto-refresh is a documented follow-up);
	// "heartbeat" refreshes only on explicit rt.Heartbeat() calls.
	RefreshOn string
}

// validate checks the policy's fields, returning a descriptive error for the
// configurations the executor rejects.
func (p TimeoutPolicy) validate() error {
	if p.RunTimeout < 0 {
		return errors.New("RunTimeout must be >= 0")
	}
	if p.IdleTimeout < 0 {
		return errors.New("IdleTimeout must be >= 0")
	}
	if p.RunTimeout == 0 && p.IdleTimeout == 0 {
		return errors.New("TimeoutPolicy must set RunTimeout or IdleTimeout")
	}
	switch p.RefreshOn {
	case "", "auto", "heartbeat":
	default:
		return errors.New(`RefreshOn must be "auto" or "heartbeat"`)
	}
	return nil
}

// CachePolicy configures per-node write caching, mirroring Python's
// `langgraph.types.CachePolicy`. It is installed per node via
// StateGraph.AddNodeWithPolicies and requires a checkpoint.Cache backend
// installed via WithCache; without a backend the policy is inert and the
// node executes uncached.
//
// On a cache miss the executor runs the node and stores the task's WRITES
// (state updates as channel writes plus routing as ReservedTasks writes, the
// same serializer the resume path uses — see completedTaskWrites), not its
// return value. On a hit the stored writes are injected as the task's
// outcome: the node does not execute, no RawNodeStart/RawNodeEnd pair is
// emitted, the injected updates still surface as `updates` stream chunks,
// and cached Command.Goto routing is replayed. Lookup-phase failures
// (KeyFunc or Get error) fail the task during the lookup pass, so the node
// likewise emits no RawNodeStart/RawNodeEnd pair.
type CachePolicy struct {
	// KeyFunc derives the cache key from the task's input (the Send arg when
	// the task was dispatched via types.Send, else the pre-superstep state
	// snapshot). Nil means DefaultCacheKey. A KeyFunc error fails the task
	// with a wrapped error (Python parity: `key_func` errors propagate as
	// task errors).
	//
	// The returned CacheKey may also pin the cache namespace (joined with "/"
	// by the executor; empty falls back to "writes/<node>") and a per-entry
	// TTL (non-zero overrides CachePolicy.TTL; zero falls back to it).
	KeyFunc func(input map[string]any) (types.CacheKey, error)
	// TTL is how long a cached entry lives; 0 means it never expires.
	TTL time.Duration
}

// DefaultCacheKey is the default CachePolicy.KeyFunc: the sha256 hex digest
// of the canonical JSON encoding of input. encoding/json marshals maps with
// sorted keys, so the digest is deterministic for JSON-representable values;
// non-JSON values (funcs, channels, cyclic structures, NaN) produce an
// error, which the executor surfaces as the task's error. The returned
// CacheKey leaves Namespace empty (the executor picks its default) and TTL 0
// (the policy TTL applies).
func DefaultCacheKey(input map[string]any) (types.CacheKey, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return types.CacheKey{}, err
	}
	sum := sha256.Sum256(data)
	return types.CacheKey{Key: hex.EncodeToString(sum[:])}, nil
}
