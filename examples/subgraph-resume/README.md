# subgraph-resume

An `Interrupt` inside a child graph pauses the parent graph's run; resuming the parent thread with `Options.Resume` feeds the answer down into the subgraph and completes both levels.

## Run

```sh
go run ./examples/subgraph-resume
```

No environment variables required (pure langgraph, no model).
