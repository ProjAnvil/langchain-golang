# Agents — `CreateAgent`

**Languages:** English | [简体中文](agents.zh-CN.md)

`agents.CreateAgent` is the Go equivalent of Python's
`langchain.agents.create_agent`. It builds a model↔tools loop on top of the
public `langgraph/` graph runtime, with composable middleware hooks around each
model call and each tool call.

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

A tool can also write graph state (or jump): place a `*types.Command` (which
implements `messages.ToolOutput`) in the tool's `Result.Artifact`. The
built-in tools node consumes these commands — their updates merge into the
node's state update (`messages` concatenates, other keys are last-write-wins)
and their `Goto` destinations concatenate onto the routing; a `Command.Resume`
or a non-empty `Command.Graph` fails the run. This is how
`TodoListMiddleware`'s `write_todos` tool lands its todos in graph state.

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
| `NewInterruptHumanInTheLoopMiddleware` / `NewHumanInTheLoopMiddleware` | Pause for human approval (interrupt-based / synchronous `Decide` callback) |
| `NewPIIMiddleware` / `NewPIIStreamTransformer` | Redact PII (batch and streaming) |
| `NewContextEditingMiddleware` | Mutate the model-call context |
| `NewFilesystemFileSearchMiddleware` | Ripgrep-backed file search tool |
| `NewProviderToolSearchMiddleware` | Provider-side tool search |
| `NewShellToolMiddleware` | Persistent shell session tool |
| `NewTodoListMiddleware` | Task-tracking tool |
| `NewLLMToolEmulator` | Emulate tool calls via the LLM |
| `NewLLMToolSelectorMiddleware` | LLM-selected tool subset |

Hook execution order (outermost-first):

```
BeforeAgent → BeforeModel → WrapModelCall → WrapToolCall → AfterModel → AfterAgent
```

Every hook receives a `context.Context`, so any of them can call
`graphpkg.Interrupt` to pause the run for external input (see
*Human-in-the-loop* below — `NewInterruptHumanInTheLoopMiddleware` is the
packaged example). A hook can also short-circuit routing by setting
`update["jump_to"]` to `"model"`, `"tools"`, or `"end"`: for `before_agent`
the jump decides whether the run enters the model↔tools loop at all
(`"model"`/`"tools"`) or exits straight through `after_agent` to the end
(`"end"`); for an update-returning `after_agent` hook, `"model"` re-enters the
loop while `"end"` (or no jump) completes the run.

Middleware-contributed tools and state fields are collected automatically:
middleware implementing `middleware.ToolProvider` (`ProvidedTools`) or
`middleware.StateSchemaContributor` (`StateSchema`) register their tools and
state keys with the agent — no manual append to the tool list and no mirrored
`WithAgentStateFields` entry. Two middleware sharing a name (an explicit
`Name()` via `MiddlewareNamer`, or the same Go type) are rejected at build
time, mirroring Python's duplicate-middleware check.

A `wrap_model_call` hook may return `middleware.ExtendedModelResponse`
(implement `WrapModelCallResultHook`): the embedded `Command`'s `Update` is
applied on top of the model node's own state update (middleware keys win
conflicts), while a `Command` carrying `Goto` / `Resume` / `Graph` fails the
run — routing stays with the `jump_to` convention above.

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

Middleware can retune the bind from inside `wrap_model_call` via
`request.Override(...)`. `middleware.WithResponseFormat` — including
ToolStrategy↔ProviderStrategy switches (a ToolStrategy override may only
narrow to structured tools declared in the agent's original response format)
— `middleware.WithToolChoice`, and `middleware.WithModelSettings` are applied
per model call (each time the overriding middleware runs), streaming included;
an `AutoStrategy` re-resolves against the model of the current call, so a
`DynamicModel` swap re-checks it.

## Human-in-the-loop (interrupt mode)

`middleware.NewInterruptHumanInTheLoopMiddleware` pauses the run whenever the
model produces a tool call whose name is registered for review — mirroring
Python's interrupt-based `HumanInTheLoopMiddleware`. The pause is a real
langgraph interrupt: the run stops with the pending call(s) surfaced as
`Result.Interrupts`, and a later resume carries the human's answer back in.

Two preconditions make the pause resumable at all: a checkpointer
(`WithAgentCheckpointer`) and a per-run `graphpkg.Options.ThreadID`. Without
them the interrupt still fires but there is no thread to resume from.

```go
agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(
		map[string]middleware.InterruptConfig{
			"transfer_funds": {
				AllowedDecisions: []middleware.DecisionType{
					middleware.DecisionApprove, middleware.DecisionEdit,
					middleware.DecisionReject, middleware.DecisionRespond,
				},
				// Optional: When, Description / DescriptionFunc, ArgsSchema
				// refine which calls are reviewed and what the reviewer sees.
			},
		},
	)),
	agents.WithAgentCheckpointer(checkpoint.NewMemorySaver()),
)

// First run: pauses after the model asks for a reviewed tool. A paused run
// is NOT an error here — it returns the committed state plus the interrupts.
values, interrupts, err := agent.InvokeWithStateOptions(ctx, msgs,
	graphpkg.Options{ThreadID: "thread-1"})

request, _ := middleware.HITLRequestFromInterrupt(interrupts[0])
// request.ActionRequests — the pending calls (name / args / description)
// request.ReviewConfigs — per-tool review policy (allowed decisions, schema)

// Answer the pause and continue the same thread:
values, _, err = agent.Resume(ctx, graphpkg.Options{
	ThreadID: "thread-1",
	Resume: middleware.HITLResponse{Decisions: []middleware.Decision{{
		Type: middleware.DecisionApprove,
	}}},
})
```

The review runs in a dedicated `"hitl"` graph node wired between the model
node and the tools node (Python runs every `after_model` middleware as its
own node for the same reason): the model's AI message is already committed
when the pause happens, so a resume re-runs only the review node — the model
is never re-invoked for the paused call, and the decisions act on the
committed, stable AIMessage. The revised message replaces the committed one
in place by message ID (an ID-less AI message gets a deterministic ID minted
for this), so a reject/respond never leaves a duplicate copy behind. One
agent supports at most one interrupt-mode HITL middleware (the `"hitl"` node
is shared); it composes with a Decide-mode HITL (below), which runs inline.

`HITLResponse` carries one `Decision` per reviewed tool call, in order.
Exactly one decision per reviewed call is required, and each `Type` must be
in that tool's `AllowedDecisions` — otherwise the resume errors. The four
branches:

- **approve** — execute the call as-is.
- **edit** — replace name/args via `Decision.EditedAction` (`*middleware.ToolCall`);
  the call's ID is kept, so the conversation stays stitched.
- **reject** — do not execute; `Decision.Message` becomes an error
  `ToolMessage` answering the call, which the model sees and can react to.
- **respond** — do not execute; `Decision.Message` becomes the tool's success
  answer on behalf of the human.

Resuming without a value (`Options.Resume` left nil) re-runs the review node
and re-raises the same interrupt with the same ID — a re-pause, useful for
polling UIs. The resume value may also arrive in its JSON wire form
(`map[string]any{"decisions": [...]}` with `"type"` / `"edited_action"` /
`"message"` keys) — what a checkpoint round-trip through a JSON saver or a
non-Go producer leaves behind; `middleware.DecodeHITLResponse` accepts both.

When several interrupts are pending, `Options.Resume` becomes a
`map[string]any` keyed by interrupt NS (first) or ID (second). The NS is also
how nesting stays addressable:

```go
// A supervisor embedding a worker agent as a subgraph. The worker carries
// no checkpointer of its own: it shares the parent run's.
worker, _ := agents.CreateAgent(workerModel, workerTools,
	agents.WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(...)))

parent := graphpkg.NewStateGraph()
parent.AddSubgraph("worker", worker.Graph)
parent.AddEdge(types.START, "worker")
parent.AddEdge("worker", types.END)
supervisor, _ := parent.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))

res, _ := supervisor.InvokeWithOptions(ctx,
	map[string]any{"messages": msgs},
	graphpkg.Options{ThreadID: "sup-1"})
// The worker's review pauses the PARENT; res.Interrupts[0].NS carries the
// nested prefix "worker:<task>/hitl:<task>".

supervisor.InvokeWithOptions(ctx, nil, graphpkg.Options{
	ThreadID: "sup-1",
	Resume: map[string]any{res.Interrupts[0].NS: middleware.HITLResponse{
		Decisions: []middleware.Decision{{Type: middleware.DecisionApprove}},
	}},
})
```

`Options.Graph` scopes a resume to a namespace: set it to the subgraph task's
namespace (the interrupt NS minus its trailing `/hitl:<task>` segment) and a
scalar `Resume` feeds only that child; a `Graph` matching no pending
interrupt's NS is a descriptive error.

### HITL limits and boundaries

- **Structured output is never reviewed.** Under `ToolStrategy`, a
  structured-output tool call ends the run inside the model node — before the
  review ordering — so it cannot pause for HITL (Python has the same
  ordering).
- **Streaming surfaces a pause as an error.** `StreamEvents` (and the plain
  `Invoke`/`InvokeWithState`) treat an interrupted run as a terminal error.
  Resume such threads with `InvokeWithStateOptions`/`Resume` (non-streaming);
  pause-aware streaming is future work.
- **`jump_to: "tools"` bypasses the review.** A hook that jumps straight to
  the tools node addresses it directly; the remapping that inserts the review
  only applies to the model node's normal routing (mirroring Python's
  `jump_to`).
- **A nested agent inside a tool cannot pause the parent.** A tool body that
  calls another agent's `InvokeWithState` runs an independent nested run;
  its interrupts do not bubble up. For a pausable supervisor/worker
  composition, embed the worker with `StateGraph.AddSubgraph` and run the
  parent with a checkpointer, as above.

### Synchronous Decide mode (compatibility)

`middleware.NewHumanInTheLoopMiddleware(interruptOn, decide)` is the older Go
form: instead of pausing, it calls the `decide` callback inline in the model
node (`AfterModel`) with the same `HITLRequest`, and applies the returned
`[]Decision` through the identical four-branch logic. It needs no
checkpointer, but the human must answer inside the call — there is no pause
to resume. The two modes are mutually exclusive per middleware instance (a
nil `Decide` is what selects interrupt mode) and may coexist as separate
middleware on one agent; the inline `AfterModel` hooks always run before the
dedicated hitl node.

## Interrupt boundaries and checkpoint history

Orthogonal to HITL, pause the run at named nodes with
`WithAgentInterruptBefore` / `WithAgentInterruptAfter`, then resume via
`Agent.Resume` (or `Agent.Graph.InvokeWithOptions`) with the same `ThreadID`.
This requires a checkpointer. Boundary interrupts resume with a nil
`Options.Resume`; they compose with the hitl node — a
`WithAgentInterruptBefore(agents.ToolsNodeName)` boundary fires after an
approved HITL decision, before the tools dispatch (and `agents.HITLNodeName`
is addressable the same way).

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

> **API note:** the checkpointer types (`checkpoint.Saver`,
> `checkpoint.NewMemorySaver`) live in the public package
> `github.com/projanvil/langchain-golang/langgraph/checkpoint`. To wire a
> custom saver, implement the versioned `Saver` interface — `GetTuple` /
> `List` (with `ListOptions.Filter` metadata filtering) / `Put` / `PutWrites`
> (with a `taskPath` argument; `checkpoint.Write` carries `TaskPath`) /
> `DeleteThread`, keyed by `checkpoint.Config` (thread ID + checkpoint
> namespace + checkpoint ID); see
> `langchain/agents/create_agent_test.go` (`TestCreateAgent_InterruptBeforeNode`)
> for the round-trip shape. This replaced the M1 `Get` / `Put` / `Delete`
> interface and was extended again by M5 — both sanctioned pre-1.0 breaks; see
> the [graph runtime guide](langgraph.md)'s *Breaking changes* section for the
> before/after signatures and migration notes.

### Checkpoint history and time travel

With a checkpointer installed, every super-step writes an immutable,
ID-addressable checkpoint, so a thread accumulates a full history. The
compiled graph (`Agent.Graph`) exposes it directly:

- `GetState(ctx, checkpoint.Config{ThreadID: ...})` — a `StateSnapshot` of
  the latest (or pinned) checkpoint: channel values, next nodes, config,
  pending interrupts.
- `GetStateHistory(ctx, cfg, opts)` — the thread's snapshots, newest first.
- `UpdateState(ctx, cfg, values, asNode)` — apply a write batch attributed to
  a node and save it as a new checkpoint (human-in-the-loop state edits).
- Time travel: pass `graphpkg.Options{ThreadID: ..., CheckpointID: ...}` to
  `InvokeWithOptions` to pin a historical checkpoint — the run forks from it
  instead of the thread's latest.

The related `graph.Interrupt(ctx, value)` primitive — pause *inside* a
node and feed a value back on resume — is available to middleware/node authors
via the public `github.com/projanvil/langchain-golang/langgraph/graph` package.

> **Durable checkpoints:** `checkpoint.NewMemorySaver` is in-memory. For a
> durable backend, the nested module `langgraph/checkpoint/sqlite` provides a
> SQLite saver — `sqlite.New(path, serde.NewJSONSerializer())` — usable
> anywhere a `checkpoint.Saver` is accepted (including
> `WithAgentCheckpointer`). The nested module `langgraph/checkpoint/postgres`
> provides a Postgres saver —
> `postgres.NewFromConnString(ctx, dsn, serde.NewJSONSerializer())` plus one
> explicit `Setup(ctx)` call — usable the same way. See the
> [graph runtime guide](langgraph.md).

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
[streaming guide](streaming.md). For the lower-level graph surface, the
compiled graph (`Agent.Graph`) also exposes `Stream` with Python-parity
stream modes (`values` / `updates` / `debug` / `messages` / `custom`) — see
the [graph runtime guide](langgraph.md).

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
so it is distinguishable to middleware, logging, and tracing. The name is also
stamped onto every AI message the agent's model produces, so message
histories collected from multiple subagents stay attributable.

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

Mirroring the scoped-port stance (only a subset of `langgraph` is ported):

- **`transformers` / `run.subagents`** — not exposed; streaming PII redaction is
  delivered via the `WrapModelStreamHook` middleware delta layer instead.
- **`Send` returned from tools** — not supported (there is no
  Send-per-tool-call dispatch; parallel tool calls run concurrently inside a
  single node instead).
