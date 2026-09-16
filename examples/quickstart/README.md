# quickstart

The smallest `agents.CreateAgent` walkthrough: an offline scripted model drives a full model -> tool -> model loop via `Invoke`, then the same loop is observed through `Agent.StreamEvents`.

## Run

```sh
go run ./examples/quickstart
```

No environment variables required (uses an in-process fake model).
