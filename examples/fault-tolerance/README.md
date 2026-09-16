# fault-tolerance

Per-node resilience with `AddNodeWithPolicies`: a `RetryPolicy` carries a flaky node through transient failures, and an `ErrorHandlerPolicy` recovers a permanently failing node — once with a plain update, once by routing around it with a `Command`.

## Run

```sh
go run ./examples/fault-tolerance
```

No environment variables required (pure langgraph, no model).
