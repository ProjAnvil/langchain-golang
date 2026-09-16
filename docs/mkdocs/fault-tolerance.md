# Fault tolerance & trace privacy

**Languages:** English | [简体中文](zh-CN/fault-tolerance.zh-CN.md)

LangGraph's per-node resilience and privacy policies, installed through
`AddNodeWithPolicies`. Three independent knobs compose freely: `RetryPolicy`
(transient failures), `ErrorHandlerPolicy` (recovery after retries are
exhausted), and `TracePolicy` (payload scrubbing before any tracer sees
events). Runnable examples:
[`examples/fault-tolerance`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/fault-tolerance)
and
[`examples/trace-policy`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/trace-policy).

## Installation

```bash
go get github.com/projanvil/langchain-golang
```

## RetryPolicy

```go
g.AddNodeWithPolicies("flaky_charge", node, graphpkg.NodePolicies{
    Retry: &graphpkg.RetryPolicy{
        MaxAttempts:     3,
        InitialInterval: 10 * time.Millisecond,
        BackoffFactor:   2,
        RetryOn:         graphpkg.DefaultRetryOn, // wrap errors in NonRetryable to abort
    },
})
```

Retries apply before anything else; when a policy is absent the graph-level
default (retry all errors, Python parity) applies.

## Node error handlers

An `ErrorHandlerPolicy` recovers a node that keeps failing after retries:
the handler receives the state snapshot plus a typed `*NodeError` (node name,
attempt, wrapped error) and returns either a plain state update or a
`*types.Command`:

```go
g.AddNodeWithPolicies("capture_receipt", node, graphpkg.NodePolicies{
    ErrorHandler: &graphpkg.ErrorHandlerPolicy{
        Handler: func(state map[string]any, nerr *graphpkg.NodeError) (any, error) {
            return &types.Command{
                Update: map[string]any{"receipt": "queued for manual review"},
                Goto:   graphpkg.To("manual_review"), // route around the failure
            }, nil
        },
    },
})
```

Durability: the task error is persisted as a checkpoint ERROR write *before*
the handler runs, so a crash mid-handler resumes into a handler re-run
instead of re-executing the node (langgraph 1.2.0 `error_handler=` parity).

## TracePolicy

`NodePolicies.Trace` transforms traced payloads on the emit side — before
LangSmith, OTel bridges, or any callback observes them. Scrub inputs, drop
outputs entirely (`OmitPayload`), or both:

```go
g.AddNodeWithPolicies("guarded_node", node, graphpkg.NodePolicies{
    Trace: &graphpkg.TracePolicy{
        ProcessInputs:  redactEmails,            // func(any) any
        ProcessOutputs: graphpkg.OmitPayload,    // event kept, payload dropped
    },
})
```

A processor panic fails **closed** (the run aborts rather than leaking raw
payloads — deliberate divergence, see DIVERGENCES.md). Agents get the same
control per middleware via the `TracePolicyProvider` hook (langchain 1.3.15).

## Environment

No configuration is required; policies are pure code. Tracers that observe
the scrubbed events follow the usual env (`LANGSMITH_TRACING`,
`LANGSMITH_API_KEY`, ...).

## Switching points

- **Fail vs recover**: omit `ErrorHandler` to let the error fail the run;
  add one to salvage it with an update or a `Command` reroute.
- **Scrubbers**: any `func(any) any` — regex redaction, JSON walks, or
  `OmitPayload` when the whole payload is sensitive.
- **Middleware tracing**: in `CreateAgent` graphs, middleware-installed
  `TracePolicyProvider`s compose with node-level policies.
