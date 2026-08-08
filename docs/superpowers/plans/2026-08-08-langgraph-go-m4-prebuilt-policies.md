# LangGraph Go Port M4: Prebuilt & Node Policies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Final milestone: per-node retry and cache policies in `langgraph/graph`, and a `langgraph/prebuilt` package whose `ToolNode` wraps the existing `langchain/tools.ToolNode` into a ready-to-use graph node. No `create_react_agent` (redundant with `langchain/agents.CreateAgent` — spec amendment 2026-08-08).

**Architecture:** Retry and cache hook into the executor's per-task dispatch (`runNode` level): retry wraps task execution with backoff; cache stores a task's WRITES (not its return value) keyed by a hash of the task's input, and a cache hit injects the stored writes as the task's outcome without executing. Both are opt-in per node via a new builder method, keeping `AddNode` untouched.

**Tech Stack:** Go 1.23, module `github.com/projanvil/langchain-golang`, standard library only (root module stays zero-dep).

## Global Constraints

- Repo root for all commands: `langchain_golang/`. Root module ZERO third-party dependencies.
- Go 1.23 floor; clean idiomatic Go (owner requirement).
- Existing exported API additive-only: `AddNode`, `Compile`, `Invoke`, `Stream`, etc. unchanged; `langchain/agents` must pass with **zero edits**.
- Retry must never retry `GraphInterrupt` panics (interrupts bubble) and must not duplicate M3 event emissions (exactly one RawNodeStart/RawNodeEnd pair per task regardless of attempts).
- Cache must not interfere with M2 resume: tasks being replayed from pending writes never consult the cache; cached writes injected into a superstep go through the same `applyWrites` path as executed ones.
- Comment style: extensive doc comments, single backticks. Conventional commits.
- Gate after every task: `go build ./... && go vet ./... && go test ./...` from `langchain_golang/`.

## Locked Design Decisions (binding)

- **P1. RetryPolicy** (new file `langgraph/graph/policy.go`):
```go
type RetryPolicy struct {
    InitialInterval time.Duration // default 500ms
    BackoffFactor   float64       // default 2.0
    MaxInterval     time.Duration // default 128s
    MaxAttempts     int           // default 3, includes the first attempt
    Jitter          bool          // default true: additive random [0,1s) per attempt (Python parity)
    RetryOn         func(err error) bool // nil = DefaultRetryOn
}

func DefaultRetryOn(err error) bool // see below
func (p RetryPolicy) withDefaults() RetryPolicy
```
`DefaultRetryOn`: retries `net.Error`, `context.DeadlineExceeded` (from the node's own work, not parent cancellation — see P3), and errors implementing `interface{ HTTPStatus() int }` with 5xx; never retries `GraphInterrupt` (it's a panic, handled separately) or `*channels.InvalidUpdateError`-style programming errors. Keep it small and explicit — document that Go has no exception hierarchy, so callers with domain errors should supply `RetryOn`.
- **P2. Builder API:** `(*StateGraph).AddNodeWithPolicies(name string, fn NodeFunc, policies NodePolicies) *StateGraph` where `type NodePolicies struct { Retry *RetryPolicy; Cache *CachePolicy }`. `AddNode` = `AddNodeWithPolicies(name, fn, NodePolicies{})`. A graph-level default retry is NOT added (YAGNI; Python has one but per-node suffices — document divergence).
- **P3. Retry execution:** inside the task wrapper in `run` (around `runNode`): attempts loop; on error matching `RetryOn` and attempts remaining → sleep `min(MaxInterval, InitialInterval * BackoffFactor^(attempt-1))` + jitter, then re-execute. Sleeps select on `ctx.Done()` (parent cancellation aborts immediately, no retry of ctx.Canceled). `GraphInterrupt` panics propagate untouched (never retried, never swallowed). Events: RawNodeStart/End emitted once around the whole attempt loop; each retry emits nothing extra in M1 event layer; in Stream debug `task_result` only the final outcome appears.
- **P4. CachePolicy + Cache:**
```go
type CachePolicy struct {
    KeyFunc func(input map[string]any) (string, error) // nil = DefaultCacheKey
    TTL     time.Duration                              // 0 = never expires
}

// DefaultCacheKey: sha256 hex of the canonical JSON of input (encoding/json
// with sorted map keys via json.Marshal on map[string]any is deterministic).
func DefaultCacheKey(input map[string]any) (string, error)

// In package langgraph/checkpoint (checkpoint.go) — the cache lives beside the saver:
type Cache interface {
    Get(ctx context.Context, ns, key string) (writes []Write, ok bool, err error)
    Set(ctx context.Context, ns, key string, writes []Write, ttl time.Duration) error
    Clear(ctx context.Context, ns string) error
}
```
`langgraph/checkpoint/memory.go` gains `InMemoryCache` (absolute-expiry TTL checked on read). Executor: at superstep dispatch, a cache-policy node computes the key from the task input (Send arg if present, else the pre-superstep state snapshot) with namespace `"writes/<node>"`; hit → the stored writes become the task's outcome (node NOT executed; M3 updates chunks still emitted so stream consumers stay consistent); miss → execute, and on success store the outcome's writes. Resume-replayed tasks never consult the cache. `(*CompiledGraph).ClearCache(ctx, ns string) error` delegates to the cache. Cache backend installed via `Compile` option `WithCache(c checkpoint.Cache)`.
- **P5. prebuilt.ToolNode:**
```go
// Package prebuilt provides ready-made graph nodes.
package prebuilt

// ToolNode returns a NodeFunc that runs the tool calls of the last AI message
// in state[messagesKey] through tools.ToolNode and returns
// map[string]any{messagesKey: resultMessages}.
func ToolNode(node *tools.ToolNode, opts ...ToolNodeOption) graph.NodeFunc
type ToolNodeOption func(*toolNodeConfig)
func WithMessagesKey(key string) ToolNodeOption // default "messages"
```
Behavior: delegates execution/error-handling to the existing `tools.ToolNode` (no behavior fork); a tool result that IS a `*types.Command` is passed through as the node's Command return (merged update + goto) — if the underlying ToolNode cannot surface Commands today, extend it minimally in `langchain/tools` (additive). Missing/empty tool calls → returns nil (no update). Document divergences: no Send-per-tool-call dispatch (Go executes calls within one node), no reflection arg injection (Go's ToolCallRequest.State is explicit).

## Reference Semantics (Python, verified during M4 research)

- Retry loop: `pregel/_retry.py:641-682` — sleep `min(max_interval, initial * factor**(attempts-1))` + `random.uniform(0,1)` additive jitter; give up at `attempts >= max_attempts`; `GraphBubbleUp`/cancellation never retried; failed-attempt writes cleared (Go: outcomes buffered pre-commit, so nothing to clear — note in code).
- Cache: `_algo.py:668-687` key = hash of key_func(input); `_loop.py:1549-1625` — cached writes loaded into the task and execution skipped; hits emitted as updates with cached=True; TTL absolute expiry on read (`cache/memory/__init__.py:22-53`).
- ToolNode: `prebuilt/tool_node.py` — messages_key default "messages", tool calls from the LAST AI message, parallel execution, handled errors → error ToolMessage.

---

### Task 1: RetryPolicy

**Files:**
- Create: `langgraph/graph/policy.go` (RetryPolicy + DefaultRetryOn + NodePolicies + CachePolicy/Cache scaffolding types IF shared — otherwise Task 2)
- Modify: `langgraph/graph/graph.go` (`AddNodeWithPolicies`, per-node policy storage, retry loop in the task wrapper)
- Test: `langgraph/graph/policy_test.go`

**Interfaces:**
- Produces: P1/P2/P3 exactly. `AddNode` delegates to `AddNodeWithPolicies`. Policies validated at `Compile` (negative intervals, MaxAttempts < 1 → compile error).

- [ ] **Step 1: Write failing tests** — flaky node failing N-1 times then succeeding completes (attempt count asserted); non-retryable error fails immediately (1 attempt); MaxAttempts exhausted → the last error surfaces; backoff intervals increase per policy (inject a sleep-observer or assert total elapsed bounds loosely — prefer a clock-injection-free assertion on attempt timing with generous bounds); jitter off = deterministic intervals; `GraphInterrupt` during a retryable-error-prone node is NOT retried (interrupt surfaces on first panic); ctx cancel during backoff aborts; events balanced (one start/end pair across attempts).
- [ ] **Step 2: Run, verify failure. Step 3: Implement. Step 4: Gate PASS.**
- [ ] **Step 5: Commit** `feat(langgraph/graph): per-node RetryPolicy with backoff and jitter`.

---

### Task 2: CachePolicy + Cache + InMemoryCache

**Files:**
- Modify: `langgraph/checkpoint/checkpoint.go` (`Cache` interface, P4)
- Modify: `langgraph/checkpoint/memory.go` (`InMemoryCache`)
- Modify: `langgraph/graph/policy.go` (CachePolicy, DefaultCacheKey)
- Modify: `langgraph/graph/graph.go` (`WithCache` compile option, dispatch-time cache lookup/injection)
- Test: `langgraph/graph/cache_test.go`, extend `langgraph/checkpoint/checkpoint_test.go` (or new `memory_cache_test.go`)

**Interfaces:**
- Produces: P4 exactly. Cache-hit tasks: skip execution, inject stored writes, still emit M3 `updates` chunks (documented as Python's `cached=True` analog — the Go chunk does not carry a cached flag; document divergence) and exactly one RawNodeStart/End pair is NOT emitted for cache hits (node never ran — document; Python emits no node events for cached tasks either since execution is skipped).

- [ ] **Step 1: Write failing tests** — two identical runs (same input, cache installed): second run's cacheable node does NOT execute (side-effect counter) but its updates land identically; different input → miss → executes; TTL expiry → re-executes; `ClearCache` forces re-execution; Send-arg inputs key on the arg (two Sends with different args don't collide); resume-replayed tasks bypass the cache; cache-hit chunks appear in a `Stream(updates)` consumer; no cache installed → zero overhead/behavior change.
- [ ] **Step 2: Run, verify failure. Step 3: Implement. Step 4: Gate PASS.**
- [ ] **Step 5: Commit** `feat(langgraph): per-node CachePolicy with InMemoryCache`.

---

### Task 3: `langgraph/prebuilt` — ToolNode adapter

**Files:**
- Create: `langgraph/prebuilt/tool_node.go`, `doc.go`
- Test: `langgraph/prebuilt/tool_node_test.go`
- Modify (only if needed for Command passthrough): `langchain/tools/tool_node.go` — additive

**Interfaces:**
- Consumes: `langchain/tools.ToolNode` (read it first: `NewToolNode`/`InvokeToolCalls`/`AppendToolResults` shapes), `graph.NodeFunc`, `types.Command`.
- Produces: P5 exactly.

- [ ] **Step 1: Write failing tests** — a StateGraph with a model stub node writing an AI message with two tool calls → `prebuilt.ToolNode` node executes both (parallel) and appends result ToolMessages under the default key; `WithMessagesKey("chat_history")` variant; tool error → error ToolMessage (default handling); tool returning `*types.Command` → the graph routes/applies it (Command passthrough); no tool calls → nil update; composition with checkpointer + interrupt-before on the tools node.
- [ ] **Step 2: Run, verify failure. Step 3: Implement. Step 4: Gate PASS (incl. `langchain/agents` — it must not be affected by any tools/ addition).**
- [ ] **Step 5: Commit** `feat(langgraph/prebuilt): ToolNode graph-node adapter over langchain/tools`.

---

### Task 4: Docs + spec mark

**Files:**
- Modify: `README.md`, `docs/usage/langgraph.md` (retry/cache/prebuilt sections, one short example each)
- Modify: `docs/superpowers/specs/2026-08-07-langgraph-go-port-design.md` (mark M4 done; the create_react_agent decision is already recorded)

- [ ] **Step 1: Document** RetryPolicy/CachePolicy (with the documented divergences: no graph-level default retry, no cached flag on chunks) and prebuilt.ToolNode (divergences: no Send-per-call dispatch, no arg injection); note the `create_react_agent` ≡ `CreateAgent` equivalence in `docs/usage/langgraph.md`.
- [ ] **Step 2: Mark M4 complete** in the spec (actual date) — this completes all milestones.
- [ ] **Step 3: Gate PASS; commit** `docs: document M4 policies and prebuilt ToolNode`.

---

## Self-Review Notes

- Spec coverage: retry → Task 1; cache → Task 2; prebuilt ToolNode → Task 3; docs → Task 4; create_react_agent explicitly not built per the 2026-08-08 spec amendment. CLI/SDK/deployment remain out of scope.
- Type consistency: `NodePolicies` (P2) introduced in Task 1 carries `Cache *CachePolicy` used in Task 2; `Cache` lives in `checkpoint` beside `Saver` (P4); `WithCache` is a CompileOption like `WithCheckpointer`.
- Risks: (a) retry interacting with the panic-based interrupt machinery — P3 names the rule (never catch GraphInterrupt) and Task 1 tests it; (b) cache-key determinism — P4 pins canonical-JSON+sha256; (c) cache hits bypassing node execution change event semantics — documented in Task 2.
