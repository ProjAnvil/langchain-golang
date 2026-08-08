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
- **P3. Retry execution:** inside the task wrapper in `run` (around `runNode`): attempts loop; on error matching `RetryOn` and attempts remaining → sleep `min(MaxInterval, InitialInterval * BackoffFactor^(attempt-1))` + jitter, then re-execute. Sleeps select on `ctx.Done()` (parent cancellation aborts immediately and surfaces the PARENT's ctx error, not the node's error). Interrupt rule (precise): `runNode`'s deferred recover converts `GraphInterrupt` panics into `interrupted != nil` BEFORE the retry loop sees anything — the attempt loop treats `interrupted != nil` as terminal and never re-executes. Retry-after-resume re-feeds resume values from index 0 (fresh `taskInterruptState` per attempt — Python parity, retry re-invokes from the start; document). Failed-attempt node-internal emissions (InvokeStream sink events, messages/custom chunks) duplicate across attempts — Python parity, document. Events: RawNodeStart/End emitted once around the whole attempt loop; in Stream debug `task_result` only the final outcome appears.
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
`langgraph/checkpoint/memory.go` gains `InMemoryCache` (absolute-expiry TTL checked on read). Executor: at superstep dispatch, a cache-policy node computes the key from the task input (Send arg if present, else the pre-superstep state snapshot) with namespace `"writes/<node>"`. **Outcome mapping (exact):** on miss, store `completedTaskWrites(update, cmd)` (the existing resume.go serializer — state updates as channel writes + goto/Sends as `ReservedTasks` writes); on hit, rebuild `outcome{update, cmd}` from the stored writes via the same channel classification `planResume` uses (ReservedTasks → routing, everything else → update map). `Command.Goto` IS cached and replayed (Python parity: `match_cached_writes` replays task.writes wholesale); conditional routers are unaffected (they run post-commit on fresh merged state). **Cache-bypass rule:** tasks being replayed from pending writes never consult the cache (automatic — they never enter dispatch), AND tasks resuming with a pending interrupt skip the cache lookup too (otherwise a cache entry from a DIFFERENT run would skip the node and silently drop the resume value; Python prevents this via `not t.writes` seeding, Go skips explicitly). **Key errors:** a `KeyFunc`/marshal error fails the task with a wrapped error (Python parity — `key_func` errors propagate as task errors). Key determinism caveat: `json.Marshal` sorts map keys, so keys are deterministic for JSON-representable values; non-JSON values (funcs, channels, cyclic, NaN) produce the task error above. **Hit semantics:** node NOT executed; no RawNodeStart/End pair (the cache Get must happen BEFORE the start event in the dispatch path); M3 `updates` chunks ARE still emitted from the injected writes (no cached flag on the chunk — documented divergence); debug `task` events still fire for cache-hit tasks (Python emits them for prepared tasks — documented). `(*CompiledGraph).ClearCache(ctx, ns string) error` delegates to the cache. Cache backend installed via `Compile` option `WithCache(c checkpoint.Cache)`. Packaging divergence (document in Task 4): Python puts the cache in `langgraph.cache.*`; Go places `Cache` in `checkpoint` beside `Saver` (dependency-safe: checkpoint imports nothing from graph).
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
Behavior: delegates execution/error-handling to the existing `tools.ToolNode` (no behavior fork). **Command convention (concrete):** `core/tools.Tool.Invoke` returns `tools.Result{Content, Artifact, Metadata}` — a tool signals a graph Command by placing a `*types.Command` in `Result.Artifact`. To surface it without forking behavior, `langchain/tools` gains ONE additive method `InvokeToolCallsFull(ctx, calls, state) ([]ToolCallOutcome, error)` with `type ToolCallOutcome struct { Message messages.Message; Command *types.Command }`, implemented in terms of the existing internal execution path (error handling, wrappers, store plumbing untouched; dependency direction `langchain/tools` → `langgraph/types` is acyclic — types imports only `fmt`). The adapter: passes the full state as `ToolCallRequest.State`; merges outcome Commands' `Update` maps and `Goto` lists into the node result (messages update always present; single merged `*types.Command` when any tool returned one — multiple Commands merge update-wise, goto lists concatenate; conflicting semantics documented). Missing/wrong-typed `state[messagesKey]` → descriptive error. Document divergences: no Send-per-tool-call dispatch (Go executes calls within one node), no reflection arg injection (Go's ToolCallRequest.State is explicit). The adapter doc must state that `messagesKey` needs an append reducer registered on the graph (`AddReducer(key, channels.MessagesReducer)`), otherwise the default LastValue replaces history.

## Reference Semantics (Python, verified during M4 research)

- Retry loop: `pregel/_retry.py:641-682` — sleep `min(max_interval, initial * factor**(attempts-1))` + `random.uniform(0,1)` additive jitter; give up at `attempts >= max_attempts`; `GraphBubbleUp`/cancellation never retried; failed-attempt writes cleared (Go: outcomes buffered pre-commit, so nothing to clear — note in code).
- Cache: `_algo.py:668-687` key = hash of key_func(input); `_loop.py:1549-1625` — cached writes loaded into the task and execution skipped; hits emitted as updates with cached=True; TTL absolute expiry on read (`cache/memory/__init__.py:22-53`).
- ToolNode: `prebuilt/tool_node.py` — messages_key default "messages", tool calls from the LAST AI message, parallel execution, handled errors → error ToolMessage.

---

### Task 1: RetryPolicy

**Files:**
- Create: `langgraph/graph/policy.go` (RetryPolicy + DefaultRetryOn + NodePolicies + CachePolicy struct — the struct must land here in Task 1 because NodePolicies references it; the `Cache` interface itself is Task 2)
- Modify: `langgraph/graph/graph.go` (`AddNodeWithPolicies`, per-node policy storage, retry loop in the task wrapper)
- Test: `langgraph/graph/policy_test.go`

**Interfaces:**
- Produces: P1/P2/P3 exactly. `AddNode` delegates to `AddNodeWithPolicies`. Policies validated at `Compile` (negative intervals, MaxAttempts < 1 → compile error).

- [ ] **Step 1: Write failing tests** — flaky node failing N-1 times then succeeding completes (attempt count asserted); non-retryable error fails immediately (1 attempt); MaxAttempts exhausted → the last error surfaces; backoff intervals increase per policy (loose elapsed-time bounds, no clock injection); jitter off = deterministic intervals; a node that calls `graph.Interrupt` is NOT re-executed after the panic (interrupted is terminal — one attempt); ctx cancel during backoff aborts AND surfaces the parent ctx error (not the node's error); events balanced (one start/end pair across attempts).
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

- [ ] **Step 1: Write failing tests** — two identical runs (same input, cache installed): second run's cacheable node does NOT execute (side-effect counter) but its updates land identically; different input → miss → executes; TTL expiry → re-executes; `ClearCache` forces re-execution; Send-arg inputs key on the arg (two Sends with different args don't collide); resume-replayed tasks bypass the cache; **a task resuming with a pending interrupt bypasses the cache even when a different run populated an entry for the same input (the resume value is delivered, the node re-executes)**; key-computation error (non-JSON state value, e.g. a func in state) fails the task with a wrapped error; Command.Goto outcomes are cached and replayed (a Command-returning node hit from cache routes identically); cache-hit chunks appear in a `Stream(updates)` consumer; no RawNodeStart/End for cache hits; no cache installed → zero overhead/behavior change.
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

- [ ] **Step 1: Write failing tests** — a StateGraph (with `AddReducer("messages", channels.MessagesReducer)`) whose model stub writes an AI message with two tool calls → `prebuilt.ToolNode` node executes both (parallel) and appends result ToolMessages under the default key; `WithMessagesKey("chat_history")` variant; tool error → error ToolMessage (default handling); a tool placing `*types.Command` in `Result.Artifact` → the graph applies the merged update and routes via Goto (through the new `InvokeToolCallsFull`); Command + handled-tool-error in the same batch interplay; missing `state[messagesKey]` and wrong-typed value → descriptive error; no tool calls → nil update; composition with checkpointer + interrupt-before on the tools node.
- [ ] **Step 2: Run, verify failure. Step 3: Implement. Step 4: Gate PASS (incl. `langchain/agents` — it must not be affected by any tools/ addition).**
- [ ] **Step 5: Commit** `feat(langgraph/prebuilt): ToolNode graph-node adapter over langchain/tools`.

---

### Task 4: Docs + spec mark

**Files:**
- Modify: `README.md`, `docs/usage/langgraph.md` (retry/cache/prebuilt sections, one short example each)
- Modify: `docs/superpowers/specs/2026-08-07-langgraph-go-port-design.md` (mark M4 done; the create_react_agent decision is already recorded)

- [ ] **Step 1: Document** RetryPolicy/CachePolicy (with the documented divergences: no graph-level default retry; no cached flag on chunks; debug task events fire for cache hits; cache lives in `checkpoint` not a `cache` package — Python packaging divergence) and prebuilt.ToolNode (divergences: no Send-per-call dispatch, no arg injection; Command-via-`Result.Artifact` convention; messagesKey reducer requirement); note the `create_react_agent` ≡ `CreateAgent` equivalence in `docs/usage/langgraph.md`.
- [ ] **Step 2: Mark M4 complete** in the spec (actual date) — this completes all milestones.
- [ ] **Step 3: Gate PASS; commit** `docs: document M4 policies and prebuilt ToolNode`.

---

## Self-Review Notes

- Spec coverage: retry → Task 1; cache → Task 2; prebuilt ToolNode → Task 3; docs → Task 4; create_react_agent explicitly not built per the 2026-08-08 spec amendment. CLI/SDK/deployment remain out of scope.
- Type consistency: `NodePolicies` (P2) introduced in Task 1 carries `Cache *CachePolicy` used in Task 2; `Cache` lives in `checkpoint` beside `Saver` (P4); `WithCache` is a CompileOption like `WithCheckpointer`.
- Risks: (a) retry interacting with the panic-based interrupt machinery — P3 names the rule (interrupted is terminal, never re-executed) and Task 1 tests it; (b) cache-key determinism — P4 pins canonical-JSON+sha256 with an explicit error policy; (c) cache hits bypassing node execution change event semantics — P4 pins the rules (no start/end pair; updates chunks still emitted; debug task events fire) and Task 2 tests them.
- Review history: an adversarial plan review returned FIX-FIRST with 5 issues; all resolved — P4 gained the exact outcome mapping (completedTaskWrites/planResume classification), the interrupted-resume cache-bypass rule, and the key-error policy; P5's Command passthrough was made concrete (`Result.Artifact` convention + additive `InvokeToolCallsFull`); cache-hit event ordering (Get before RawNodeStart) pinned; P3 wording aligned with the actual recover mechanics (plus parent-ctx-error surfacing and resume-retry notes); Task 1/2/3 test lists extended accordingly (interrupted-resume bypass, key error, Command+error interplay, reducer requirement).
