# Graph runtime (`langgraph/`) — stream modes, serde, SQLite checkpoints

The public top-level `langgraph/` packages hold the ported graph runtime:
`langgraph/graph` (StateGraph builder + Pregel executor),
`langgraph/channels`, `langgraph/checkpoint` (+ `checkpoint/serde`),
`langgraph/types`. `agents.CreateAgent` is built on it; this guide covers the
M3 additions: the `Stream` API, checkpoint serialization, and the durable
SQLite checkpoint saver.

## Stream API (`CompiledGraph.Stream`)

`Stream` runs the graph like `InvokeWithOptions` and yields chunks as they are
emitted, as a Go 1.23 range-over-func iterator (`iter.Seq2`). The `Modes`
option mirrors Python's `stream_mode` and supports multi-mode streams:

```go
import graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"

for chunk, err := range cg.Stream(ctx, input, graphpkg.StreamOptions{
	Modes: []graphpkg.StreamMode{graphpkg.StreamValues, graphpkg.StreamUpdates},
}) {
	if err != nil {
		return err // run failure is yielded as the final pair
	}
	switch chunk.Mode {
	case graphpkg.StreamUpdates:
		// map[string]any{nodeName: map[string]any{channel: value}} per task
	case graphpkg.StreamValues:
		// map[string]any — full graph state after the input batch and after
		// each superstep that changed a channel
	}
	_ = chunk.Namespace // "" for the root graph; subgraph node path otherwise
}
```

- **Modes** — `StreamValues`, `StreamUpdates`, `StreamDebug` (task
  dispatch/completion + checkpoint events), `StreamMessages` (per-token LLM
  chunks; node code opts in by pulling the installed callbacks manager with
  `callbacks.ManagerFromContext`), `StreamCustom` (node-emitted payloads via
  `graphpkg.StreamWriterFromContext`).
- **`StreamChunk`** always carries `Namespace`, `Mode`, and `Payload`
  explicitly — Go does not reshape output by mode count the way Python does
  (bare payload vs tuples).
- **`StreamOptions` embeds `Options`**, so `ThreadID` / `CheckpointID` /
  `Resume` keep their `Invoke` semantics; `Subgraphs: true` includes subgraph
  chunks (with their `Namespace`) instead of dropping them.
- Breaking out of the `range` early cancels the run and waits for its
  goroutine to exit — no leaks.

`Stream` coexists with `InvokeStream` / `NodeEventSink` (the event-ified path
behind `Agent.StreamEvents`); neither replaces the other. One documented
timing divergence: `updates` chunks are emitted post-superstep in
deterministic task order, so they bunch after node-time `messages`/`custom`
chunks instead of interleaving as in Python.

## Checkpoint serde (`langgraph/checkpoint/serde`)

`checkpoint.Serializer` is the persistence encoding contract;
`serde.NewJSONSerializer()` is the in-tree implementation — the portable
subset of Python's `JsonPlusSerializer`:

- JSON-native values (`nil`, `string`, `float64`, `bool`, `map[string]any`,
  `[]any`) encode as plain JSON (tag `"json"`).
- Concrete Go types that plain JSON would degrade round-trip losslessly
  through a **closed type registry** envelope `{"__type__": name, "data":
  payload}` (tag `"json+envelope"`). The registry covers `messages.Message`,
  `[]messages.Message`, `types.Send`, `types.Interrupt`, `time.Time`,
  `[]byte`, `int64`, `int`, and `[]string`. Encoding an unregistered concrete
  type is an error — there is no silent lossy fallback.

> **Divergence note:** Go uses JSON + a closed registry instead of Python's
> msgpack + import-by-name. Checkpoints written by this serializer are **not
> binary-compatible with Python checkpoints** — no Python interop.

## SQLite checkpoint saver (`langgraph/checkpoint/sqlite/`)

A durable `checkpoint.Saver` backed by SQLite, mirroring Python's
`langgraph-checkpoint-sqlite` schema (WAL mode; `checkpoints` / `writes`
tables with `type`+`value` column pairs).

> **Dependency notice:** this is the port's **first third-party dependency** —
> the pure-Go (no cgo) driver `modernc.org/sqlite`, pinned to v1.38.2 for the
> Go 1.23 floor. To keep the root module zero-dependency, the saver lives in
> its own nested Go module (`langgraph/checkpoint/sqlite/go.mod`, with a
> `replace` directive back to the root), mirroring Python's separate
> `langgraph-checkpoint-sqlite` package.

```go
import (
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	sqlite "github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite"
)

saver, err := sqlite.New("checkpoints.db", serde.NewJSONSerializer())
if err != nil {
	return err
}
defer saver.Close()
// Use like any checkpoint.Saver: agents.WithAgentCheckpointer(saver), etc.
```

Because it is a nested module, the root `go test ./...` does **not** cross
into it. Run its tests with:

```bash
make test-sqlite
```
