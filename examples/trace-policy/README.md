# trace-policy

Per-node traced-payload scrubbing via `NodePolicies.Trace`: a custom input redactor plus `graph.OmitPayload` on outputs, observed through a recording callback handler. Raw payloads leak from the unguarded node; the guarded node's are transformed before any tracer sees them.

## Run

```sh
go run ./examples/trace-policy
```

No environment variables required (pure langgraph + callbacks).
