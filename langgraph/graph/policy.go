package graph

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"net"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
)

// RetryPolicy configures per-node automatic retry with exponential backoff,
// mirroring Python's `langgraph.types.RetryPolicy` (`pregel/_retry.py`). It is
// installed per node via StateGraph.AddNodeWithPolicies; nodes added with
// AddNode carry no policy and are never retried. A graph-level default retry
// is deliberately NOT provided (Python has `retry_policy=` on compile; per-node
// policies suffice — documented divergence, YAGNI).
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

// DefaultRetryOn is the default RetryPolicy.RetryOn. It retries:
//
//   - net.Error (and anything wrapping one): transient network failures.
//   - context.DeadlineExceeded: a deadline hit by the node's OWN work. Parent
//     cancellation is handled separately by the retry loop itself, which
//     aborts on ctx.Done() and surfaces the parent's ctx error (see
//     CompiledGraph.runTask).
//   - errors implementing `interface{ HTTPStatus() int }` with a 5xx status
//     (the Go stand-in for Python's HTTP-server-error retry; 4xx is not
//     retried).
//
// It never retries *channels.InvalidUpdateError-style programming errors, and
// GraphInterrupt is not an error at all (it is a panic, converted to a
// terminal interrupted outcome before the retry loop — see runTask).
//
// Go has no exception hierarchy to mirror Python's `retry_on` exception
// tuple, so this predicate is intentionally small and explicit: callers with
// domain errors should supply their own RetryOn.
func DefaultRetryOn(err error) bool {
	var invalidUpdate *channels.InvalidUpdateError
	if errors.As(err, &invalidUpdate) {
		return false
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var statusErr interface{ HTTPStatus() int }
	if errors.As(err, &statusErr) {
		return statusErr.HTTPStatus() >= 500
	}
	return false
}

// NodePolicies bundles the optional per-node execution policies installed via
// StateGraph.AddNodeWithPolicies. A nil field means the corresponding policy
// is disabled for the node.
type NodePolicies struct {
	// Retry enables automatic retry of the node's failures (see RetryPolicy).
	Retry *RetryPolicy
	// Cache enables write caching for the node (see CachePolicy).
	Cache *CachePolicy
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
	KeyFunc func(input map[string]any) (string, error)
	// TTL is how long a cached entry lives; 0 means it never expires.
	TTL time.Duration
}

// DefaultCacheKey is the default CachePolicy.KeyFunc: the sha256 hex digest
// of the canonical JSON encoding of input. encoding/json marshals maps with
// sorted keys, so the digest is deterministic for JSON-representable values;
// non-JSON values (funcs, channels, cyclic structures, NaN) produce an
// error, which the executor surfaces as the task's error.
func DefaultCacheKey(input map[string]any) (string, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}
