# Streaming

`Agent.StreamEvents` returns a pull-based stream of `StreamEvent` values that
let you observe the run as it happens: per-token model deltas, tool dispatch
lifecycle, and node boundaries. It is the Go equivalent of Python's
`astream_events`.

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
