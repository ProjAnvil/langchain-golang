# Graph runtime (`langgraph/`) — stream modes, serde, SQLite checkpoints, retry/cache policies, prebuilt ToolNode

The public top-level `langgraph/` packages hold the ported graph runtime:
`langgraph/graph` (StateGraph builder + Pregel executor),
`langgraph/channels`, `langgraph/checkpoint` (+ `checkpoint/serde`),
`langgraph/types`, `langgraph/prebuilt`. `agents.CreateAgent` is built on it;
this guide covers the M3 additions (the `Stream` API, checkpoint
serialization, the durable SQLite checkpoint saver), the M4 additions
(per-node retry/cache policies and `prebuilt.ToolNode`), and the M6 addition
(multi-parent join edges).

## Join edges (multi-parent barrier)

`AddJoinEdge([]string{"a", "b"}, "c")` mirrors Python's
`add_edge(("a", "b"), "c")`: `c` runs exactly once after ALL parents have
committed, whether they finish in the same superstep or several apart, and
re-arms on each loop round. Arrivals are checkpointed, so an interrupted
parent that resumes later still completes the barrier.

> **Warning (OR semantics, Python parity):** a plain edge, conditional edge,
> `types.Send`, or `Command.Goto` into the join child bypasses the barrier and
> triggers the child directly. Mixing both edge kinds into one child can run
> it multiple times.

Divergences from Python: Go requires >= 2 distinct parents (Python accepts a
single-element tuple and silently dedups) and rejects `types.END` as a join
child. `defer=True` / `NamedBarrierValueAfterFinish` are not supported. The
`join:a+b:c` barrier channel is control-plane state: it never appears in node
inputs, snapshots, or stream output.

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

## `create_react_agent` ≡ `agents.CreateAgent`

Python's `langgraph.prebuilt.create_react_agent` is **deliberately not
ported** (design decision, 2026-08-08): it has been deprecated upstream
since langgraph v1.0, and its capability — the model ↔ tools loop above — is
a strict subset of `langchain/agents.CreateAgent`, which builds exactly that
loop on this runtime and adds middleware, structured output, and interrupt
support. Use [`agents.CreateAgent`](agents.md); build the loop by hand with
`prebuilt.ToolNode` only when you need graph-level control `CreateAgent`
does not expose.
