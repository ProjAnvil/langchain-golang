# Composing runnables (LCEL)

LangChain's **LCEL** (LangChain Expression Language) lets you chain components
with the `|` operator in Python:

```python
chain = prompt | model | parser
```

Go does **not** support operator overloading, so `|` is unavailable. The Go
equivalent lives in [`core/runnables`](../../core/runnables) and uses free
functions instead:

```go
chain := runnables.Pipe3(prompt, model, parser)
```

The semantics are identical; only the spelling differs. Every composition
returns a value that itself satisfies `Runnable[I, O]`, so chains nest and feed
into the rest of the framework (agents, fallbacks, retries, ...) without
adapters.

## The Runnable contract

Every composable value implements `runnables.Runnable[I, O]`:

```go
type Runnable[I, O any] interface {
	Invoke(ctx context.Context, input I, opts ...Option) (O, error)
	Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error)
	Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error)
	InputSchema() schema.Schema
	OutputSchema() schema.Schema
}
```

`core/language.ChatModel`, `core/tools.Tool` adapters, `prompts.*`, and the
combinators below all satisfy this interface. Because it is generic, the type
chain (`I → A → B → O`) is checked at **compile time** — stronger than Python's
runtime Pydantic checks.

## Pipe — the `|` equivalent

`Pipe(a, b)` composes two runnables so the first's output feeds the second:

```go
import "github.com/projanvil/langchain-golang/core/runnables"

upper := runnables.NewFunc(
	func(_ context.Context, s string, _ ...runnables.Option) (string, error) {
		return strings.ToUpper(s), nil
	},
	schema.String(""), schema.String(""),
)
exclaim := runnables.NewFunc(
	func(_ context.Context, s string, _ ...runnables.Option) (string, error) {
		return s + "!", nil
	},
	schema.String(""), schema.String(""),
)

chain := runnables.Pipe(upper, exclaim)
out, _ := chain.Invoke(context.Background(), "hello")
// out == "HELLO!"
```

`Pipe` returns a `SeqN[I, O]` that satisfies `Runnable[I, O]`.

## Longer chains: Pipe3 .. Pipe6

To keep chains readable past two steps, use `Pipe3`, `Pipe4`, `Pipe5`,
`Pipe6`. Each extra step adds one type parameter, all checked at compile time:

```go
// Python:  prompt | model | parser
// Go:
chain := runnables.Pipe3(prompt, model, parser)

// Four stages (e.g. retrieve → grade → generate → parse):
chain := runnables.Pipe4(retriever, grader, generator, parser)
```

### Beyond six: nest Pipe

A `SeqN` is itself a `Runnable`, so for longer chains you nest. The result is
type-safe and flattens transparently at run time:

```go
chain := runnables.Pipe(
	runnables.Pipe3(a, b, c),
	runnables.Pipe3(d, e, f),
)
```

## Parallel — `RunnableParallel`

Run several runnables on the same input and collect keyed outputs into a map:

```go
pipe := runnables.NewParallel(map[string]runnables.Runnable[string, any]{
	"length":   lengthRunnable,
	"upper":    upperRunnable,
	"reversed": reverseRunnable,
})
out, _ := pipe.Invoke(context.Background(), "hello")
// out == map[string]any{"length": 5, "upper": "HELLO", "reversed": "olleh"}
```

## Branch — `RunnableBranch`

Pick a branch at runtime by condition:

```go
branch, _ := runnables.NewBranch(
	[]runnables.BranchCase[map[string]any, []messages.Message]{
		{Condition: isSmalltalk, Runnable: chitchatRunnable},
		{Condition: needsRetrieval, Runnable: ragRunnable},
	},
	defaultRunnable, // used when no condition matches
)
```

## Fallbacks and retry

Wrap a chain so a failure falls through to alternatives, mirroring Python's
`.with_fallbacks(...)` and `.with_retry(...)`:

```go
primary := runnables.Pipe(prompt, expensiveModel)
resilient, _ := runnables.NewWithFallbacks(primary, cheapModel, cachedModel)

// Retry transient failures up to 3 times:
retrying, _ := runnables.NewRetry(resilient, 3)
```

## Passthrough and Assign — building RAG-style chains

`NewPassthrough` forwards its input unchanged. `NewAssign` adds computed keys
to a map input — the standard pattern for feeding context into a prompt:

```go
// Python:
//   chain = RunnablePassthrough.assign(context=retriever) | prompt | model
//
// Go:
rag := runnables.Pipe(
	runnables.NewAssign(map[string]runnables.Runnable[map[string]any, any]{
		"context": retrieverRunnable, // input map → retrieved docs string
	}),
	runnables.Pipe3(ragPrompt, model, parser),
)
```

## Wiring a chain into an agent

`SeqN[I, O]` is a `Runnable`, so a composed chain plugs directly into the
existing combinator universe — and into anything that consumes a `Runnable`:

```go
// A Pipe result used as a fallback target inside an agent's middleware:
primary := runnables.Pipe(retrieve, summarize)
resilient, _ := runnables.NewWithFallbacks(primary, fallbackSummarizer)
```

## Python ↔ Go cheat sheet

| Python LCEL | Go |
|-------------|-----|
| `a \| b` | `runnables.Pipe(a, b)` |
| `a \| b \| c` | `runnables.Pipe3(a, b, c)` |
| `a \| b \| c \| d \| e \| f` | `runnables.Pipe6(a, b, c, d, e, f)` |
| longer chains | nest `Pipe(Pipe3(...), Pipe3(...))` |
| `RunnableParallel({...})` | `runnables.NewParallel(map[string]Runnable[I, any]{...})` |
| `RunnableBranch([...])` | `runnables.NewBranch(cases, default)` |
| `.with_fallbacks([...])` | `runnables.NewWithFallbacks(primary, ...fallbacks)` |
| `.with_retry()` | `runnables.NewRetry(r, attempts)` |
| `RunnablePassthrough` | `runnables.NewPassthrough[I](schema)` |
| `RunnablePassthrough.assign(...)` | `runnables.NewAssign(map[string]Runnable[map[string]any, any]{...})` |
| `RunnableLambda(fn)` | `runnables.NewFunc(fn, inSchema, outSchema)` |

## What is intentionally absent

Mirroring the project's scoped-port stance, these LCEL-adjacent features are
**not** implemented:

- **The `|` operator itself** — Go has no operator overloading; use `Pipe*`.
- **`astream_log` / `astream_events` over a chain** — streaming over a
  composed chain uses `Runnable.Stream` (pull-based); agent-level event
  streaming goes through `Agent.StreamEvents` (see [streaming](streaming.md)).
- **Pydantic-backed schema validation** — schemas are `schema.Schema`
  (`map[string]any`); they document shape but do not validate at runtime.
- **PNG graph rendering** — JSON / ASCII / Mermaid graph export is supported
  (`runnables.GetGraph`); PNG is not.
