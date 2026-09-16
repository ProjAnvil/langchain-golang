# hitl-interrupt

Human-in-the-loop pause/resume: `WithAgentCheckpointer(MemorySaver)` + `WithAgentInterruptBefore("tools")` pauses an agent run before tool execution; a second invoke on the same thread resumes it.

## Run

```sh
go run ./examples/hitl-interrupt
```

No environment variables required (uses an in-process fake model).
