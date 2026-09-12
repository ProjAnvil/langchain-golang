# Streaming

**Languages:** English | [简体中文](streaming.zh-CN.md)

`Agent.StreamEvents` returns a pull-based stream of `StreamEvent` values that
let you observe the run as it happens: per-token model deltas, tool dispatch
lifecycle, and node boundaries. It is the agent-level counterpart of Python's
`astream_events`; the Runnable-level v2 event stream over any
`core/runnables` Runnable is [`runnables.StreamEvents`](#runnable-level-events-runnablestreamevents),
described below, and LangSmith tracing hooks into the same callback layer.

## Event types

| Constant | When emitted | Key fields populated |
|----------|--------------|----------------------|
| `StreamNodeStart` / `StreamNodeEnd` | around every node (`before_agent`, `model`, `tools`, `after_agent`) | `Node` |
| `StreamModelDelta` | per model chunk | `Node`, `Delta`, `Text` |
| `StreamModelEnd` | once per model call, with the assembled AI message | `Node`, `Message` |
| `StreamToolStart` | before each tool dispatch | `Node`, `ToolName`, `ToolArgs` |
| `StreamToolEnd` | after each tool dispatch | `Node`, `ToolName`, `ToolResult` |
| `StreamEnd` | terminal, emitted exactly once last | `State`, `Message` (or `Err`) |

All constants live in the `agents` package, e.g. `agents.StreamModelDelta`.

## Minimal example: print text deltas and tool calls

```go
stream, err := agent.StreamEvents(ctx, []messages.Message{
	messages.User("Summarize the latest commits."),
})
if err != nil {
	panic(err)
}
for {
	ev, ok, err := stream.Next(ctx)
	if err != nil {
		panic(err)
	}
	if !ok {
		break
	}
	switch ev.Type {
	case agents.StreamModelDelta:
		fmt.Print(ev.Text)
	case agents.StreamToolStart:
		fmt.Printf("\n[tool %s args=%v]\n", ev.ToolName, ev.ToolArgs)
	case agents.StreamToolEnd:
		fmt.Printf("\n[tool %s done result=%v]\n", ev.ToolName, ev.ToolResult)
	case agents.StreamEnd:
		if ev.Err != nil {
			log.Printf("run ended: %v", ev.Err)
		}
	}
}
```

`ev.Text` is a convenience string holding the text delta for `StreamModelDelta`
(empty for non-text deltas such as reasoning or tool-call deltas). If you need
the raw content-block protocol event (e.g. reasoning deltas), read `ev.Delta`.

## Order guarantees

- `node_start` / `node_end` pairs always balance per node invocation, even on
  the error or interrupt paths.
- Within a `model` node: zero or more `model_delta` events, then exactly one
  `model_end` with the fully assembled AI message.
- Within a `tools` node: one `tool_start` / `tool_end` pair per dispatched
  tool.
- Exactly one terminal `StreamEnd` as the last event before the stream closes.

When the graph fans out (multiple tasks active in one superstep), their events
interleave on the stream — disambiguate via the `Node` field.

## Runnable-level events (`runnables.StreamEvents`)

`core/runnables.StreamEvents` is the Go counterpart of Python's
`astream_events(v2)` over **any** `Runnable` — chains built with `Pipe` /
`Parallel` / `Retry`, `NewFunc` lambdas, chat models — not just agents. It
wraps one `Runnable.Stream` invocation in a callback-collecting manager and
projects the flat callback stream onto the v2 `runnables.StreamEvent` shape:

```go
import (
	"context"
	"strings"

	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
)

double := runnables.NewFunc(
	func(_ context.Context, in string, _ ...runnables.Option) (string, error) {
		return in + in, nil
	}, schema.String(""), schema.String(""))
upper := runnables.NewFunc(
	func(_ context.Context, in string, _ ...runnables.Option) (string, error) {
		return strings.ToUpper(in), nil
	}, schema.String(""), schema.String(""))
chain := runnables.Pipe(double, upper)

for ev, err := range runnables.StreamEvents(ctx, chain, "go",
	runnables.StreamEventOptions{}) {
	if err != nil {
		return err // run failure is yielded as the final pair
	}
	fmt.Println(ev.Event, ev.Name, ev.RunID, ev.ParentIDs)
}
```

- **Event shape** — `Event` (`on_chain_start` / `on_chain_stream` /
  `on_chain_end` / `on_chain_error`, `on_chat_model_*`, `on_llm_*`,
  `on_tool_*`, `on_retriever_*`), `RunID`, `Name`, `Tags`, `Metadata`,
  root-first `ParentIDs`, and a `Data` payload (`Input` on start, `Chunk` on
  stream, `Output` on end, `Error` on error). The root run's `ParentIDs` is
  nil; a step inside a `Pipe` reports the chain's run ID as its only parent.
- **Filtering** — `StreamEventOptions` mirrors Python's `include_*` /
  `exclude_*` filters: `IncludeNames` / `ExcludeNames`,
  `IncludeTypes` / `ExcludeTypes` (run types `chain` / `chat_model` / `llm`
  / `tool` / `retriever`), and `IncludeTags` / `ExcludeTags`. Each include
  list that is set must admit the event (include-OR); each exclude list must
  not match. Filtering happens at the projection output only, so a
  filtered-out run never breaks the `parent_ids` of its surviving
  descendants.
- **Chat-model aggregation** — one aggregator per model run assembles
  `on_chat_model_end`'s output from the streamed chunks and prefers the v3
  content-block protocol over legacy message chunks when a provider emits
  both: only content-block deltas surface as `on_chat_model_stream` (the
  chunk payload is the protocol event itself — a documented divergence from
  v2's `AIMessageChunk`); message-start/finish and block boundaries fold
  into the surrounding start/end events.
- **Lifecycle** — the iterator is single-use; a run failure is yielded as
  the final `(zero, err)` pair after all events. Breaking out of the
  `range` cancels the run and joins the producer goroutine, so nothing
  leaks. Concurrent child runs interleave freely — pair their events by
  `RunID`, not by global nesting order.

`Agent.StreamEvents` (the seven domain event kinds above) and
`runnables.StreamEvents` (the Runnable-tree v2 events) are two projections
of different layers — pick one surface per run; they do not interlock.

## LangSmith tracing

`core/tracers` ships a LangSmith tracer (`tracers.NewLangChainTracer`) that
rebuilds a run tree from the same callback stream and ships it to the
LangSmith batch API on a background goroutine — batches of up to 64
operations, a 500ms flush cadence, one retry, then `OnError`. Tracing never
fails the business call: handler errors are swallowed and POST failures are
reported, not propagated.

Enable it with the environment (optional vars follow the Python client's
`LANGSMITH_`-before-`LANGCHAIN_` precedence):

```bash
export LANGSMITH_TRACING_V2=true  # or LANGCHAIN_TRACING_V2 / LANGSMITH_TRACING / LANGCHAIN_TRACING
export LANGSMITH_API_KEY=ls__...  # or LANGCHAIN_API_KEY — required
export LANGSMITH_PROJECT=my-app   # optional; LANGSMITH_ENDPOINT overrides the API base
```

Tracing is **off by default** — nothing is sent unless a tracing flag AND an
API key are both set. Construct the tracer explicitly (it returns nil when
the environment does not enable it, and every method is nil-safe, so use it
unconditionally) and own its lifetime:

```go
tracer := tracers.NewLangChainTracer()
defer tracer.Close() // flushes the remainder; idempotent
```

`runnables.StreamEvents` auto-attaches this tracer when the environment
enables tracing, so a streamed run is traced without wiring anything into
the options. Leave the env unset when you attach your own tracer — the
duplicate would conflict server-side. `NewLangChainTracerWithOptions`
constructs one with explicit `LangSmithOptions` (endpoint, project, batch
size, flush cadence, `OnError`).

## Streaming vs non-streaming

`agent.Invoke` runs the loop to completion and returns the final message
history. `agent.StreamEvents` runs the same loop but emits events as it goes.
State semantics are identical between the two paths; streaming is additive
observability, not a different execution model.

## Note on caching

When `WithAgentCache` is configured, the cache is consulted only on the
non-streaming `Invoke` path. `StreamEvents` always bypasses the cache so that
`model_delta` / `model_end` events fire on every run — a cache hit would
otherwise short-circuit the model call and emit nothing.

## Provider notes

- **`partners/openai` usage chunk.** The OpenAI chat model's `Stream` opts
  into usage accounting (`stream_options.include_usage`): the stream ends
  with a usage-only chunk — an empty AI message carrying `UsageMetadata`,
  including cached-token / reasoning-token details where the provider reports
  them. It emits no `model_delta` through `StreamEvents` (the text is
  empty); consume `model.Stream` directly to read it.
- **Structured output while streaming.** The streaming path applies the same
  per-call bind as `Invoke`: middleware `WithResponseFormat` /
  `WithToolChoice` / `WithModelSettings` overrides and a `ProviderStrategy`'s
  model kwargs (via `ModelSettingsBinder`, when the model implements it)
  reach the `Stream` call, not just the non-streaming one; models without
  that capability keep the post-hoc JSON parse of the assembled message.
