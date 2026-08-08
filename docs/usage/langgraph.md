# Graph runtime (`langgraph/`) — stream modes, serde, SQLite/Postgres checkpoints, retry/cache policies, ToolNode, join edges, functional API

**Languages:** English | [简体中文](langgraph.zh-CN.md)

The public top-level `langgraph/` packages hold the ported graph runtime:
`langgraph/graph` (StateGraph builder + Pregel executor),
`langgraph/channels`, `langgraph/checkpoint` (+ `checkpoint/serde`, with the
durable savers in the nested `checkpoint/sqlite` and `checkpoint/postgres`
modules), `langgraph/types`, `langgraph/prebuilt`, and `langgraph/fn` (the
functional API). `agents.CreateAgent` is built on it; this guide covers the
full M1–M7 surface: the `Stream` API, checkpoint serialization, the SQLite
and Postgres checkpoint savers, per-node retry/cache policies,
`prebuilt.ToolNode`, multi-parent join edges (`AddJoinEdge`), the functional
API (`NewEntrypoint` / `NewTask`), and the M5 saver-interface breaking
changes.

## Stream API (`CompiledGraph.Stream`)

`Stream` runs the graph like `InvokeWithOptions` and yields chunks as they are
emitted, as a Go 1.23 range-over-func iterator (`iter.Seq2`). The `Modes`
option mirrors Python's `stream_mode` and supports multi-mode streams:

```go
import graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"

for chunk, err := range cg.Stream(ctx, input, graphpkg.StreamOptions{
	Modes: []graphpkg.StreamMode{graphpkg.StreamValues, graphpkg.StreamUpdates},
}) {
	if err != nil {
		return err // run failure is yielded as the final pair
	}
	switch chunk.Mode {
	case graphpkg.StreamUpdates:
		// map[string]any{nodeName: map[string]any{channel: value}} per task
	case graphpkg.StreamValues:
		// map[string]any — full graph state after the input batch and after
		// each superstep that changed a channel
	}
	_ = chunk.Namespace // "" for the root graph; subgraph node path otherwise
}
```

- **Modes** — `StreamValues`, `StreamUpdates`, `StreamDebug` (task
  dispatch/completion + checkpoint events), `StreamMessages` (per-token LLM
  chunks; node code opts in by pulling the installed callbacks manager with
  `callbacks.ManagerFromContext`), `StreamCustom` (node-emitted payloads via
  `graphpkg.StreamWriterFromContext`).
- **`StreamChunk`** always carries `Namespace`, `Mode`, and `Payload`
  explicitly — Go does not reshape output by mode count the way Python does
  (bare payload vs tuples).
- **`StreamOptions` embeds `Options`**, so `ThreadID` / `CheckpointID` /
  `Resume` keep their `Invoke` semantics; `Subgraphs: true` includes subgraph
  chunks (with their `Namespace`) instead of dropping them.
- Breaking out of the `range` early cancels the run and waits for its
  goroutine to exit — no leaks.

`Stream` coexists with `InvokeStream` / `NodeEventSink` (the event-ified path
behind `Agent.StreamEvents`); neither replaces the other. One documented
timing divergence: `updates` chunks are emitted post-superstep in
deterministic task order, so they bunch after node-time `messages`/`custom`
chunks instead of interleaving as in Python.

## Checkpoint serde (`langgraph/checkpoint/serde`)

`checkpoint.Serializer` is the persistence encoding contract;
`serde.NewJSONSerializer()` is the in-tree implementation — the portable
subset of Python's `JsonPlusSerializer`:

- JSON-native values (`nil`, `string`, `float64`, `bool`, `map[string]any`,
  `[]any`) encode as plain JSON (tag `"json"`).
- Concrete Go types that plain JSON would degrade round-trip losslessly
  through a **closed type registry** envelope `{"__type__": name, "data":
  payload}` (tag `"json+envelope"`). The registry covers `messages.Message`,
  `[]messages.Message`, `types.Send`, `types.Interrupt`, `time.Time`,
  `[]byte`, `int64`, `int`, and `[]string`. Encoding an unregistered concrete
  type is an error — there is no silent lossy fallback.

> **Divergence note:** Go uses JSON + a closed registry instead of Python's
> msgpack + import-by-name. Checkpoints written by this serializer are **not
> binary-compatible with Python checkpoints** — no Python interop.

## SQLite checkpoint saver (`langgraph/checkpoint/sqlite/`)

A durable `checkpoint.Saver` backed by SQLite, mirroring Python's
`langgraph-checkpoint-sqlite` schema (WAL mode; `checkpoints` / `writes`
tables with `type`+`value` column pairs).

> **Dependency notice:** this is the port's **first third-party dependency** —
> the pure-Go (no cgo) driver `modernc.org/sqlite`, pinned to v1.38.2 for the
> Go 1.23 floor. To keep the root module zero-dependency, the saver lives in
> its own nested Go module (`langgraph/checkpoint/sqlite/go.mod`, with a
> `replace` directive back to the root), mirroring Python's separate
> `langgraph-checkpoint-sqlite` package.

```go
import (
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	sqlite "github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite"
)

saver, err := sqlite.New("checkpoints.db", serde.NewJSONSerializer())
if err != nil {
	return err
}
defer saver.Close()
// Use like any checkpoint.Saver: agents.WithAgentCheckpointer(saver), etc.
```

Because it is a nested module, the root `go test ./...` does **not** cross
into it. Run its tests with:

```bash
make test-sqlite
```

## Per-node retry (`graph.RetryPolicy`)

`RetryPolicy` installs executor-level automatic retry with exponential
backoff on a single node, mirroring Python's `langgraph.types.RetryPolicy`.
Policies attach at node-registration time via `AddNodeWithPolicies`; nodes
added with plain `AddNode` carry no policy and are never retried:

```go
g := graphpkg.NewStateGraph()
g.AddNodeWithPolicies("flaky", flakyNode, graphpkg.NodePolicies{
	Retry: &graphpkg.RetryPolicy{MaxAttempts: 5},
})
```

- **Defaults** (applied to any zero field): `InitialInterval` 500ms,
  `BackoffFactor` 2.0, `MaxInterval` 128s, `MaxAttempts` 3 (total attempts,
  first included). Backoff is `min(MaxInterval, InitialInterval *
  BackoffFactor^(attempt-1))`.
- **Jitter** — a uniform random `[0, 1s)` added after the `MaxInterval`
  clamp — is **on by default** (Python parity); set `NoJitter: true` to
  disable it.
- **`RetryOn`** decides whether a failed attempt's error is retryable; nil
  means `DefaultRetryOn`, which retries `net.Error`s,
  `context.DeadlineExceeded` (a deadline hit by the node's own work — parent
  cancellation aborts the retry loop itself and surfaces the parent's ctx
  error), and errors implementing `interface{ HTTPStatus() int }` with a 5xx
  status. It never retries `channels.InvalidUpdateError`-style programming
  errors or 4xx; supply your own `RetryOn` for domain errors. An interrupted
  node is terminal and never re-executed by the retry loop.

> **Divergence note:** there is deliberately **no graph-level default
> retry** (Python's `retry_policy=` compile kwarg) — per-node policies
> suffice (YAGNI).

## Per-node cache (`graph.CachePolicy` + `checkpoint.Cache`)

`CachePolicy` caches a node's task writes, mirroring Python's
`langgraph.types.CachePolicy`. It needs both halves: a policy on the node
and a `checkpoint.Cache` backend installed at compile time:

```go
g := graphpkg.NewStateGraph()
g.AddNodeWithPolicies("expensive", expensiveNode, graphpkg.NodePolicies{
	Cache: &graphpkg.CachePolicy{TTL: 10 * time.Minute},
})
cg, err := g.Compile(graphpkg.WithCache(checkpoint.NewInMemoryCache()))
```

- What is cached is the task's **writes** (state updates as channel writes,
  plus routing), not its return value. On a hit the node does not execute:
  the stored writes are injected as the task's outcome, so `updates` stream
  chunks are still emitted and cached `Command.Goto` routing is replayed.
- The key defaults to `DefaultCacheKey` — the sha256 hex digest of the
  canonical JSON encoding of the task input (deterministic; non-JSON values
  are an error). `KeyFunc` overrides it; a `KeyFunc` error fails the task.
- `TTL` is the entry lifetime; 0 means the entry never expires.
- `CompiledGraph.ClearCache(ctx, ns)` removes a whole namespace — the
  executor namespaces each node's entries as `"writes/<node>"`. It is a
  no-op when no backend is installed; a policy without a `WithCache` backend
  is inert (the node executes uncached).

> **Divergence notes:** the cache interface and `InMemoryCache` live in the
> **`checkpoint` package beside `Saver`**, not a separate `cache` package
> (Python ships `langgraph.cache.*`; keeping them in `checkpoint` preserves
> the acyclic dependency direction). Stream chunks carry **no cached flag** —
> a cache-hit `updates` chunk is indistinguishable from a live one. Debug
> task events (`StreamDebug`) **still fire on cache hits** (dispatch and
> completion), but no `RawNodeStart`/`RawNodeEnd` event pair is emitted
> because the node never runs.

## `prebuilt.ToolNode`

`langgraph/prebuilt.ToolNode` adapts a `langchain/tools.ToolNode` into a
graph node: it runs the tool calls of the last AI message in the messages
state key and appends one `ToolMessage` per call. The canonical
model ↔ tools loop:

```go
import (
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/prebuilt"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

g := graph.NewStateGraph()
g.AddReducer("messages", channels.MessagesReducer) // required, see below
g.AddNode("model", modelNode)                      // appends an AI message
g.AddNode("tools", prebuilt.ToolNode(toolNode))    // toolNode: *tools.ToolNode
g.AddEdge(types.START, "model")
g.AddConditionalEdges("model", func(_ context.Context, state map[string]any) ([]any, error) {
	msgs, _ := state["messages"].([]messages.Message)
	if len(tools.PendingToolCalls(msgs)) > 0 {
		return graph.To("tools"), nil
	}
	return graph.To(types.END), nil
})
g.AddEdge("tools", "model")
```

- **Reducer requirement:** the messages key needs an append reducer —
  `AddReducer(key, channels.MessagesReducer)` — otherwise the default
  LastValue channel replaces the message history with each update.
- `WithMessagesKey(key)` changes the state key read/written (default
  `"messages"`, Python's `messages_key`).
- Execution and error handling are delegated to the wrapped
  `tools.ToolNode` unchanged: calls run concurrently within the node, and
  tool errors become error `ToolMessage`s per its `HandleToolErrors`. The
  full graph state is passed as `ToolCallRequest.State`, so tools see the
  read-only context Python's `InjectedState` provides.
- **Command convention:** a tool signals graph control flow by placing a
  `*types.Command` in its `Result.Artifact` (surfaced via
  `ToolNode.InvokeToolCallsFull`). When any tool in the batch returned one,
  the node's result is a single merged `*types.Command`: the messages update
  is always present in its `Update` map, the individual commands' `Update`
  maps merge into it, and their `Goto` lists concatenate; on conflicting
  `Update` keys, `Graph`, or `Resume`, the last command in call order wins.

> **Divergence note:** matching `langchain/tools.ToolNode`'s scope, there is
> **no Send-per-tool-call dispatch** (Go executes the batch concurrently
> within one node) and **no reflection-based argument injection**
> (`InjectedState` / `InjectedStore` / `ToolRuntime`) — `ToolCallRequest.State`
> is explicit.

## Join edges (`AddJoinEdge`)

`AddJoinEdge([]string{"a", "b"}, "c")` mirrors Python's
`add_edge(("a", "b"), "c")` — a *waiting edge* backed by a barrier channel
equivalent to `NamedBarrierValue`
(`libs/langgraph/langgraph/graph/state.py:956-966`,
`libs/langgraph/langgraph/channels/named_barrier_value.py`). The child runs
exactly once after ALL parents have committed, whether they finish in the
same superstep or several apart:

```go
g := graph.NewStateGraph()
g.AddNode("a", nodeA) // nodeA/nodeB/nodeC: graph.NodeFunc
g.AddNode("b", nodeB)
g.AddNode("c", nodeC)
g.AddEdge(types.START, "a")
g.AddEdge(types.START, "b")
g.AddJoinEdge([]string{"a", "b"}, "c") // c runs once, after BOTH a and b commit
g.AddEdge("c", types.END)
```

- Several parents finishing in the same superstep still trigger the child
  only once, in the following superstep.
- Arrivals are checkpointed: if parent `a` has arrived and parent `b` is
  interrupted, resuming `b` later still completes the barrier — the partial
  arrival set survives in the checkpoint.
- In a looping graph the barrier resets (`Consume`) after the child commits,
  so it re-arms and can fire again on the next round.
- The `join:a+b:c` barrier channel is control-plane state: it never appears
  in node inputs, snapshots, or stream output.

`AddJoinEdge` returns `*StateGraph` for chaining. Validation failures
accumulate via `setErr` and surface at `Compile`: at least 2 distinct
parents (duplicates are an error), parents must be registered nodes
(checked at Compile time, consistent with `AddEdge`), and neither a parent
nor the child may be `types.START` or `types.END`.

> **Warning (OR semantics, Python parity):** a plain edge, conditional edge,
> `types.Send`, or `Command.Goto` into the join child bypasses the barrier and
> triggers the child directly. Mixing both edge kinds into one child can run
> it multiple times — this is Python's documented behavior, faithfully
> replicated; it is not a bug.

Divergences from Python: Go requires >= 2 distinct parents (Python accepts a
single-element tuple and silently dedups), and Go rejects `types.END` as a
join child while Python allows it (the `state.py:963-964` validation only
rejects `START` as the target and `END` as a source).

> **Not supported:** `defer=True` / `NamedBarrierValueAfterFinish` — the
> edge-driven executor model has no equivalent.

## Postgres checkpoint saver (`langgraph/checkpoint/postgres/`)

A durable `checkpoint.Saver` backed by PostgreSQL — the port of Python's
`langgraph-checkpoint-postgres` (`BasePostgresSaver`). It mirrors Python's
schema — four tables (`checkpoints` / `checkpoint_blobs` /
`checkpoint_writes` / `checkpoint_migrations`) with per-version channel
blobs — and applies pending migrations v0–v9 from `Setup`.

> **Dependency notice:** this is the port's **second third-party
> dependency** — the pure-Go driver `github.com/jackc/pgx/v5`. Like the
> SQLite saver it lives in its own nested Go module
> (`langgraph/checkpoint/postgres/go.mod`, with a `replace` directive back
> to the root), so the root module stays zero-dependency.

```go
import (
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	postgres "github.com/projanvil/langchain-golang/langgraph/checkpoint/postgres"
)

saver, err := postgres.NewFromConnString(ctx,
	"postgres://user:pass@localhost:5432/dbname", serde.NewJSONSerializer())
if err != nil {
	return err
}
defer saver.Close()
// Setup applies the schema and must be called explicitly once before first
// use. It does NOT run inside a transaction — migrations v6–v8 are
// CREATE INDEX CONCURRENTLY, which Postgres forbids in a transaction block.
if err := saver.Setup(ctx); err != nil {
	return err
}
// Use like any checkpoint.Saver: agents.WithAgentCheckpointer(saver), etc.
```

`postgres.New(pool, serde)` takes an existing `*pgxpool.Pool` instead of
opening one. Both savers run the same shared `savertest` contract suite
(`savertest.Run`), so behavior identical across backends is pinned by one
test set.

Because it is a nested module, the root `go test ./...` does **not** cross
into it. Its tests spin up a real in-process database via
`github.com/fergusstrange/embedded-postgres` (the first run downloads ~30MB
of Postgres binaries; `-short` skips these tests):

```bash
make test-postgres
```

> **No cross-language database sharing:** the Go serde is JSON + a closed
> type registry (not Python's msgpack), and the `checkpoint_blobs.version`
> column is `BIGINT` where Python stores TEXT holding decimal strings. A
> database written by one language cannot be read by the other — point each
> language at its own database.

> **Divergence notes:** only the JSON primitives `nil` / `string` / `bool` /
> `float64` inline into the checkpoint JSONB document; `int` / `int64` /
> `map[string]any` / `[]any` and every registry type are versioned into
> `checkpoint_blobs` (Python inlines int/float and sends dict/list to blobs).
> There is no Shallow saver variant and no delta channel-history fast path.
> Smaller divergences — null bytes in metadata strings fail loudly instead
> of being stripped, and `Put` preserves versionless composite channel
> values that Python silently drops — are documented in the package godoc.

## Functional API (`langgraph/fn`)

The functional API — the port of Python's `langgraph.func` (`@entrypoint` /
`@task`) — builds a checkpointed workflow out of plain Go control flow
instead of an explicit graph: loops, branches, and concurrency are ordinary
Go code, while each task's result is checkpointed so an interrupted run
resumes without re-executing finished work. An `Entrypoint` compiles to a
single-node `StateGraph` (three reserved channels `__start__` / `__end__` /
`__previous__`), so interrupt/resume, streaming, and time travel all come
from the existing executor. Reach for the functional API when the control
flow is dynamic and data-dependent; reach for `StateGraph` when the topology
should be explicit, inspectable, and statically validated.

### Entrypoints

`NewEntrypoint` wraps a function as an invokable workflow:

```go
import (
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/fn"
	"github.com/projanvil/langchain-golang/langgraph/graph"
)

entry := fn.NewEntrypoint(fn.EntrypointOpts{
	Checkpointer: checkpoint.NewMemorySaver(),
}, func(ctx context.Context, in string, prev int, hasPrev bool) (int, error) {
	total := len(in)
	if hasPrev {
		total += prev
	}
	return total, nil // also saved as the next invocation's `previous`
})

n, err := entry.Invoke(ctx, "hello", graph.Options{ThreadID: "t-1"}) // n == 5
n, err = entry.Invoke(ctx, "hi", graph.Options{ThreadID: "t-1"})     // n == 7
```

`graph.Options.ThreadID` (together with a checkpointer) ties invocations
into a thread; without a checkpointer every run is stateless.
`EntrypointOpts` also accepts a `checkpoint.Cache` backend (for task cache
policies) and a `graph.RetryPolicy` (retries the entrypoint function as a
whole).

### State across runs: `previous`

`prev` is the save value of the previous completed invocation on the same
thread; `hasPrev` is false — and `prev` the zero value — when there is no
checkpointer, no ThreadID, or no prior completed invocation.

> **Divergence note:** Python passes `previous=None` when nothing is saved;
> Go uses the explicit `hasPrev bool` so a legitimately saved zero value is
> never misread as "nothing saved".

### Decoupling return and save: `Final`

With plain `NewEntrypoint` the returned value doubles as the save value, so
the output type must be assignable to the save type. `NewEntrypointFinal`
returns a `Final[O, S]` to decouple the two (Python's
`entrypoint.final(value=, save=)`):

```go
entry := fn.NewEntrypointFinal(fn.EntrypointOpts{Checkpointer: saver},
	func(ctx context.Context, in string, prev []string, hasPrev bool) (fn.Final[string, []string], error) {
		history := append(slices.Clone(prev), in)
		return fn.Final[string, []string]{
			Value: "ack: " + in, // returned to the caller
			Save:  history,      // threaded into the next run's `prev`
		}, nil
	})
```

### Tasks

`NewTask` wraps a function as a named, checkpoint-replayable unit of work.
`Call` starts it in its own goroutine immediately — there is no Python-style
"next tick" scheduling, because the Go executor is edge-driven and has no
tick concept — and returns a `Future`; `Get` waits for the result.
`AwaitAll` collects a batch of futures:

```go
fetch := fn.NewTask("fetch", func(ctx context.Context, url string) (string, error) {
	return httpGet(ctx, url)
}, fn.TaskOpts{})

entry := fn.NewEntrypoint(fn.EntrypointOpts{Checkpointer: saver},
	func(ctx context.Context, urls []string, prev int, hasPrev bool) ([]string, error) {
		futs := make([]*fn.Future[string], len(urls))
		for i, u := range urls {
			futs[i] = fetch.Call(ctx, u) // concurrent: each Call starts at once
		}
		return fn.AwaitAll(ctx, futs...)
	})
```

`Call` may only be reached from within an entrypoint function, from within
another task, or from a StateGraph node via an `Entrypoint.Invoke` inside
that node (the run dispatcher travels through the context); anywhere else it
panics. The task name must be unique within an entrypoint's call graph — it
identifies the task in deterministic task IDs and in the cache namespace.

> **Divergence note:** there is no bare task-inside-a-StateGraph-node form —
> Python's `@task` called directly in a node relies on Pregel config
> injection with no Go equivalent. The Go shape is invoking an `Entrypoint`
> inside the node (`add.Invoke(ctx, ...)` within the `NodeFunc`).

### Task policies: retry, cache, timeout

`TaskOpts` mirrors the `@task(retry_policy=..., cache_policy=...,
timeout=...)` decorator arguments:

- **Retry** — a `graph.RetryPolicy` with the same semantics as per-node
  retry; nil means never retry.
- **Cache** — a `graph.CachePolicy`, inert unless the enclosing entrypoint
  has a `checkpoint.Cache` backend installed (`EntrypointOpts.Cache`) — the
  same "policy without backend is inert" rule as node caching. Only
  successful results are cached. The key is a hash of the call arguments and
  the namespace is `__fn_writes/<task-name>`; a custom `KeyFunc` receives
  the arguments packed as `map[string]any{"input": in}` (Python's
  `key_func` receives `*args/**kwargs` — documented divergence).
- **Timeout** — caps each attempt. A goroutine cannot be force-killed, so a
  timeout can only cancel the attempt's context and stop waiting for it;
  the abandoned attempt keeps running in the background, so task functions
  should honor their context. (Python likewise does not support timeout for
  sync task functions.)

### Interrupt and resume

An entrypoint — or a task inside it — calls `graph.Interrupt(ctx, value)` to
pause the run for external input. `Invoke` then returns the zero output and
a `*fn.InterruptError` carrying the pending interrupts (recover with
`errors.As`). Resume by invoking again on the same thread with
`graph.Options{ThreadID: ..., Resume: ...}`: the resume value becomes the
return value of the paused `Interrupt` call, and multiple interrupts match
resume values by index order:

```go
entry := fn.NewEntrypoint(fn.EntrypointOpts{Checkpointer: saver},
	func(ctx context.Context, in string, prev int, hasPrev bool) (string, error) {
		approved, _ := graph.Interrupt(ctx, map[string]any{"draft": in}).(bool)
		if !approved {
			return "rejected", nil
		}
		return publish(ctx, in)
	})

_, err := entry.Invoke(ctx, "draft text", graph.Options{ThreadID: "t-1"})
var ierr *fn.InterruptError
if errors.As(err, &ierr) {
	// ierr.Interrupts[0].Value == map[string]any{"draft": "draft text"}
	out, err := entry.Invoke(ctx, "", graph.Options{ThreadID: "t-1", Resume: true})
	// input is ignored on resume; out is whatever publish returned
	_ = out
}
```

`Entrypoint.Stream` runs like `Invoke` and yields chunks, but the mode is
fixed to `updates` (Python's entrypoint default `stream_mode="updates"`).

> **Divergence note:** individual task calls produce no stream chunks —
> tasks execute inside the entrypoint node and are not graph tasks (Python's
> PUSH tasks stream per-task updates).

### Checkpoint replay and determinism

On resume the entrypoint function **re-runs from the beginning**; each
`Call` whose deterministic task ID — a hash of the recovery checkpoint ID,
step, task name, and per-run call index — matches a pending write in the
checkpoint is filled from the persisted result **without re-executing**
(errors are persisted the same way and re-thrown from `Get`). The pattern is
correct exactly when replays are deterministic:

- The task call order must be deterministic across replays of the same
  entrypoint (the per-run call counter restarts from zero) — put
  non-deterministic logic (time, randomness, network) inside tasks, never
  in the entrypoint's control flow. Interrupts must likewise surface in a
  deterministic `Get` order.
- When a run pauses on an interrupt, tasks that started but did not finish
  are canceled; results that already completed land in the checkpoint's
  pending writes before the pause.
- `I` / `O` / `S` and task inputs/outputs must round-trip through the
  checkpoint serde (JSON-native values or the closed type registry); with a
  persistent saver an unregistered type is a descriptive error, never a
  silent downgrade.

> **Not ported:** Python's `@entrypoint(checkpointer=..., store=...)`
> cross-thread `BaseStore` — `EntrypointOpts` has no store field. The full
> 15-item divergence list (replayed errors lose their concrete type, a
> failed run poisons its thread, cache + interrupt-in-task is an unsupported
> combination, ...) lives in the `langgraph/fn` package godoc.

## Breaking changes (M5 saver interface)

M5 evolved the `checkpoint.Saver` contract for functional-API task tracking
and metadata filtering — a sanctioned pre-1.0 break. Custom `Saver`
implementations must be updated:

```go
// before
type ListOptions struct { Before *Config; Limit int }
PutWrites(ctx context.Context, cfg Config, writes []Write, taskID string) error
type Write struct { TaskID string; Channel string; Value any }
// after
type ListOptions struct { Before *Config; Limit int; Filter map[string]any }
PutWrites(ctx context.Context, cfg Config, writes []Write, taskID, taskPath string) error
type Write struct { TaskID string; Channel string; Value any; TaskPath string }
```

Migration notes for custom savers:

- **`PutWrites`** — persist the new `taskPath` argument alongside each write
  (it identifies the task's position within the run, e.g. `a@0/b@0`), or
  ignore it and store `""`.
- **`ListOptions.Filter`** — map-containment semantics over checkpoint
  metadata: a checkpoint matches when its metadata contains every filter
  key/value (`checkpoint.MetadataMatchesFilter`). The Postgres saver
  evaluates it server-side with `@>`; the memory and SQLite savers do the
  equivalent in-process comparison. Filter keys are closed to `source` /
  `step` / `parents` — the fields of `checkpoint.Metadata`.
- **SQLite databases created before M5** keep working: `sqlite.New` detects
  a `writes` table without the `task_path` column at startup and adds it
  with `ALTER TABLE ... ADD COLUMN` (Python added the same column in its own
  migration v9).

## `create_react_agent` ≡ `agents.CreateAgent`

Python's `langgraph.prebuilt.create_react_agent` is **deliberately not
ported** (design decision, 2026-08-08): it has been deprecated upstream
since langgraph v1.0, and its capability — the model ↔ tools loop above — is
a strict subset of `langchain/agents.CreateAgent`, which builds exactly that
loop on this runtime and adds middleware, structured output, and interrupt
support. Use [`agents.CreateAgent`](agents.md); build the loop by hand with
`prebuilt.ToolNode` only when you need graph-level control `CreateAgent`
does not expose.
