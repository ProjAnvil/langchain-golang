# middleware-suite

Three agent middleware composed on one `CreateAgent` agent: `TodoListMiddleware` (write_todos tool + todos state), `SummarizationMiddleware` (old turns collapsed into a summary), and `PIIMiddleware` (email redaction on input and output). The scripted model records every prompt, so each middleware's effect is printed from the model's point of view.

## Run

```sh
go run ./examples/middleware-suite
```

No environment variables required (uses an in-process fake model).
