# streaming

`Agent.StreamEvents` in action: token-level `model_delta` events, node lifecycle events, tool dispatch events, and the terminal `end` event, all from an offline chunked fake model.

## Run

```sh
go run ./examples/streaming
```

No environment variables required (uses an in-process fake model).
