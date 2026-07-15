# Agents — `CreateAgent`

`agents.CreateAgent` is the Go equivalent of Python's
`langchain.agents.create_agent`. It builds a model↔tools loop on top of an
internal graph runtime, with composable middleware hooks around each model call
and each tool call.

## Signature

```go
func CreateAgent(
	model language.ChatModel,
	toolList []coretools.Tool,
	opts ...AgentOption,
) (*Agent, error)
```

- `model` may be `nil` when `WithAgentModel` supplies a `"provider:model"` string.
- `toolList` may be `nil` for a pure-conversation agent.
- The returned `*Agent` exposes `Invoke`, `InvokeWithState`, and `StreamEvents`.

The Go port reaches full parameter parity with Python's `create_agent` for the
in-scope parameter set (16 of 17). The one parameter not ported is
`transformers` (per-callable output transformations such as streaming PII
redaction); the equivalent streaming redaction is delivered via the
`WrapModelStreamHook` middleware instead.

## System prompt

Plain string:

```go
agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentSystemPrompt("You are a helpful assistant."),
)
```

Templated — rendered on every model call via `core/prompts`, with variables
merged from build-time defaults and any per-`Invoke` overrides:

```go
tmpl, _ := prompts.NewPromptTemplate("You are a {{.role}}.")
agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentSystemPromptTemplate(tmpl, map[string]any{"role": "analyst"}),
)
```

## Tools

Define tools with `core/tools`. The simplest is `NewSimple` for a typed
function:

```go
echo, _ := coretools.NewSimple("echo", "echo its input",
	func(ctx context.Context, input string) (coretools.Result, error) {
		return coretools.Result{Content: "echo:" + input}, nil
	})
```

`FromFunc` reflects any Go function into a tool — the `@tool` equivalent. The
function's arguments struct defines the JSON schema:

```go
search, _ := coretools.FromFunc("search", "search the web",
	func(args struct {
		Query string `json:"query"`
	}) (string, error) {
		return runSearch(args.Query), nil
	})
```

Pass them to `CreateAgent`; the model decides when to call them:

```go
agent, _ := agents.CreateAgent(model, []coretools.Tool{echo, search})
```

## Middleware

Middleware wrap the model call and each tool call. Compose them with
`WithAgentMiddleware`:

```go
import "github.com/projanvil/langchain-golang/langchain/agents/middleware"

agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentMiddleware(
		middleware.NewModelFallbackMiddleware(model, fallbackModel), // switch model on error
		middleware.NewModelRetryMiddleware(),                        // retry transient failures
	),
)
```

Fifteen middleware modules ship in-tree:

| Middleware | Purpose |
|------------|---------|
| `NewModelFallbackMiddleware` | Fall back to alternate models on error |
| `NewModelRetryMiddleware` | Retry model calls with backoff |
| `NewSummarizationMiddleware` | Compact long conversation histories |
| `NewModelCallLimitMiddleware` | Cap model calls per run / per thread |
| `NewToolCallLimitMiddleware` | Cap calls to a specific tool |
| `NewToolRetryMiddleware` | Retry failed tool calls |
| `NewHumanInTheLoopMiddleware` | Pause for human approval |
| `NewPIIMiddleware` / `NewPIIStreamTransformer` | Redact PII (batch and streaming) |
| `NewContextEditingMiddleware` | Mutate the model-call context |
| `NewFilesystemFileSearchMiddleware` | Ripgrep-backed file search tool |
| `NewProviderToolSearchMiddleware` | Provider-side tool search |
| `NewShellToolMiddleware` | Persistent shell session tool |
| `NewTodoListMiddleware` | Task-tracking tool |
| `NewLLMToolEmulatorMiddleware` | Emulate tool calls via the LLM |
| `NewLLMToolSelectorMiddleware` | LLM-selected tool subset |

Hook execution order (outermost-first):

```
BeforeAgent → BeforeModel → WrapModelCall → WrapToolCall → AfterModel → AfterAgent
```

Every hook receives a `context.Context`, so any of them can call
`graphpkg.Interrupt` to pause the run for external input (see *Interrupt /
resume* below). A hook can also short-circuit routing by setting
`update["jump_to"]` to `"model"`, `"tools"`, or `"end"`.

## Structured output

Constrain the final response to a schema via `WithAgentResponseFormat`. Three
strategies are available:

```go
import "github.com/projanvil/langchain-golang/core/schema"

sentimentSchema := schema.Object(map[string]schema.Schema{
	"sentiment": schema.Schema{"type": "string", "enum": []any{"pos", "neg", "neu"}},
	"score":     schema.Integer("sentiment score 0-100"),
}, "sentiment", "score")

toolStrategy := agents.NewToolStrategy(sentimentSchema)
//   → schema bound as a callable tool; a matching tool call ends the run.

providerStrategy := agents.NewProviderStrategy(sentimentSchema)
//   → ask the provider for native structured output (best-effort).

autoStrategy := agents.NewAutoStrategy(sentimentSchema)
//   → resolved at build time from the model's capabilities (ToolStrategy when
//     the model supports tool calling, else ProviderStrategy).

agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentResponseFormat(autoStrategy),
)

state, _ := agent.InvokeWithState(ctx, msgs)
result := state["structured_response"] // parsed per the schema
```

> `ProviderStrategy`'s provider-native model-kwargs binding is best-effort: the
> model is separately configured (or prompted) to emit matching JSON, then the
> final text response is parsed against the schema.

## Interrupt / resume (human-in-the-loop)

Pause the run at named nodes with `WithAgentInterruptBefore` /
`WithAgentInterruptAfter`, then resume via `Agent.Graph.InvokeWithOptions` with
the same `ThreadID`. This requires a checkpointer.

```go
agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentCheckpointer(checkpointer),
	agents.WithAgentInterruptBefore(agents.ToolsNodeName), // pause before tools run
)

// First run pauses; resume the same thread:
result, _ := agent.Graph.InvokeWithOptions(ctx,
	map[string]any{"messages": msgs},
	graphpkg.Options{ThreadID: "thread-1"}, // nil Resume resumes a boundary interrupt
)
```

> **API note:** the checkpointer type (`checkpoint.Saver`, `checkpoint.MemorySaver`)
> currently lives in the internal package
> `langchain/internal/agentruntime/checkpoint`. Code outside this module cannot
> import it directly. To wire a custom saver today, implement the `Get` / `Put`
> / `Delete` methods the interface requires on your own type; see
> `langchain/agents/create_agent_test.go` (`TestCreateAgent_InterruptThroughModelHook`)
> for the round-trip shape. A public checkpointer API is on the v1 roadmap.

The related `agentruntime.Interrupt(ctx, value)` primitive — pause *inside* a
node and feed a value back on resume — is available to middleware/node authors
within the module via the internal `graph` package.

## State and context schema

- **`WithAgentStateFields`** — register custom graph-state fields with their
  own reducers (mirrors Python's `state_schema`). A field whose name collides
  with a default key (`messages` / `jump_to` / `structured_response`) overrides
  that key's reducer.
- **`WithAgentContextSchema`** + **`WithContextValues`** / **`ContextValue`** —
  declare and read per-run, read-only context carried through Go's
  `context.Context` (mirrors Python's `context_schema`).

See the `agents` package godoc for the full `WithAgent*` option set
(recursion limit, name, debug, store, cache, ...).

## Streaming

For real-time output, use `agent.StreamEvents` — see the
[streaming guide](streaming.md).

## Subagents (agent-as-tool)

A "subagent" is a named agent invoked from inside another agent's tool. There
is no special API: mirror Python `langchain.agents` and write a tool whose body
calls the inner agent's `InvokeWithState`, returning the final AI message text.

```go
// A named inner agent — the name is what makes it distinguishable.
weather, err := agents.CreateAgent(model, nil, agents.WithAgentName("weather_agent"))

// Hand-rolled subagent tool (the Go equivalent of Python's
//   @tool
//   def call_weather(city): return weather.invoke(...)["messages"][-1].text).
callWeather, err := coretools.NewFunc(
    "call_weather", "Call the weather agent.",
    schema.Object(map[string]schema.Schema{"city": schema.String("city")}, "city"),
    func(ctx context.Context, input map[string]any) (coretools.Result, error) {
        city, _ := input["city"].(string)
        state, err := weather.InvokeWithState(ctx, []messages.Message{messages.Human("weather in " + city)})
        if err != nil {
            return coretools.Result{}, err
        }
        msgs, _ := state["messages"].([]messages.Message)
        for i := len(msgs) - 1; i >= 0; i-- {
            if msgs[i].Role == messages.RoleAI {
                return coretools.Result{Content: messages.Text(msgs[i])}, nil
            }
        }
        return coretools.Result{}, fmt.Errorf("weather agent produced no output")
    },
)

// Supervisor delegates via the tool.
supervisor, err := agents.CreateAgent(model, []coretools.Tool{callWeather}, agents.WithAgentName("supervisor"))
```

Inside the nested run, `agents.NameFromContext(ctx)` returns the inner agent's
name (`"weather_agent"`), not the supervisor's, because `InvokeWithState`
rebinds the run-name context tag. Build the inner agent with `WithAgentName`
so it is distinguishable to middleware, logging, and tracing.

Errors from the inner agent propagate through the tool and surface as an error
`ToolMessage` (via `ToolNode`'s default `HandleToolErrors`), so the supervisor
run still completes and the model can react. Nesting works recursively: each
`InvokeWithState` is an independent graph run with its own recursion limit.

**Streaming limitation.** When the supervisor runs via `StreamEvents`, the
nested agent runs non-streaming — only its final result surfaces as the tool
result; the nested run does not emit `model_delta` events into the parent
stream. Scoped surfacing of a subagent's live events under a separate handle
(`run.subagents`) is not provided; it is part of the deferred stream-transformer
work (Design Decision 4 in the v1-final-parity spec).

## What is intentionally absent

Mirroring the scoped-port stance (the `agentruntime` boundary is private):

- **`transformers` / `run.subagents`** — not exposed; streaming PII redaction is
  delivered via the `WrapModelStreamHook` middleware delta layer instead.
- **`Command` / `Send` returned directly from tools** — out of scope.
- **Public checkpointer / graph-runtime API** — the runtime lives under
  `langchain/internal/agentruntime` and is not exported.
