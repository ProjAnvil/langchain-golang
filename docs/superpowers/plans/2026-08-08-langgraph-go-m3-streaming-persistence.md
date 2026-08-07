# LangGraph Go Port M3: Streaming & Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the M3 layer to the public `langgraph/` module: a general `Stream` API with Python-parity stream modes (`values`/`updates`/`debug`/`messages`/`custom`), a checkpoint `Serializer` with a JSON registry implementation, a SQLite checkpoint backend (first third-party dependency, isolated in a nested module), and the M2 leftover debt (Metadata.Step alignment + missing tests).

**Architecture:** `Stream` is a Go-1.23 iterator (`iter.Seq2`) over a new emission layer inside the existing run loop — the loop gains a stream protocol hook emitting `StreamChunk{Namespace, Mode, Payload}` at the exact points M2's structure already exposes (input batch, per-task completion, superstep commit, pause). Serde is a small typed-envelope registry over `encoding/json`. SQLite mirrors Python's two-table schema in a nested Go module so the root module stays dependency-free.

**Tech Stack:** Go 1.23, module `github.com/projanvil/langchain-golang`; nested module `langgraph/checkpoint/sqlite` adds `modernc.org/sqlite` (pure Go, no cgo) — the only third-party dependency in the project.

## Global Constraints

- Repo root for all commands: `langchain_golang/`. The nested sqlite module (Task 4) has its own `go.mod`; its commands run in `langgraph/checkpoint/sqlite/`.
- Go 1.23 floor; root module stays at ZERO third-party dependencies.
- Clean, idiomatic Go (owner requirement): small interfaces, iterators over channels/callbacks where idiomatic.
- `graph.CompiledGraph` existing API frozen (`Invoke`/`InvokeStream`/`Options`/`Result`/event-sink helpers); `langchain/agents` must pass with **zero edits**. `InvokeStream`/`NodeEventSink` are NOT replaced by `Stream` — they coexist (documented; consolidation deferred).
- No Python checkpoint interop: Go serde is JSON + a closed type registry (NOT msgpack, NOT import-by-name). Documented divergence.
- Comment style: extensive doc comments, single backticks. Conventional commits.
- Gate after every task: `go build ./... && go vet ./... && go test ./...` from `langchain_golang/` (plus the nested module's own `go test ./...` once Task 4 exists).
- Do not reformat unrelated files (pre-existing gofmt drift in `langchain/` is known).

## Locked Design Decisions (binding)

- **S1. Stream API shape.**
```go
type StreamMode string
const (
    StreamValues   StreamMode = "values"
    StreamUpdates  StreamMode = "updates"
    StreamDebug    StreamMode = "debug"
    StreamMessages StreamMode = "messages"
    StreamCustom   StreamMode = "custom"
)

type StreamChunk struct {
    Namespace string     // "" = root graph; subgraph node path joined by "/" (derived from the subgraph node path threaded through invokeSubgraph — NOT from checkpoint config, so it works without a checkpointer; when checkpointing is active it coincides with the checkpoint NS)
    Mode      StreamMode
    Payload   any        // mode-dependent, see Task 2/3
}

type StreamOptions struct {
    Options              // embedded: ThreadID/CheckpointID/Resume semantics unchanged
    Modes     []StreamMode // required, non-empty; order irrelevant (emission is chronological)
    Subgraphs bool          // include subgraph chunks (with their Namespace); false = drop them
}

// Stream runs the graph like InvokeWithOptions and yields chunks as they are
// emitted. The iterator is single-use; the run executes on a goroutine and
// chunks are delivered in emission order. Early break cancels the run.
func (g *CompiledGraph) Stream(ctx context.Context, input map[string]any, opts StreamOptions) iter.Seq2[StreamChunk, error]
```
- **S2. Payload shapes (Python parity, adapted):** `values` → `map[string]any` full state snapshot after each superstep (and after the input batch); on pause, the values payload includes the `"__interrupt__"` key. `updates` → per-task `map[string]map[string]any` (`{nodeName: {channel: value}}`) emitted as each task's writes land; interrupt chunk is `map[string]any{"__interrupt__": []types.Interrupt{...}}`. `debug` → `map[string]any{"step": int, "timestamp": string, "type": "task"|"task_result"|"checkpoint", "payload": ...}` with the payload fields listed in Task 2. `messages` → `MessageChunk{Message messages.Message, Metadata map[string]any}` (Task 3). `custom` → whatever the node's StreamWriter emits.
- **S3. Single vs multi mode shaping.** Python reshapes output (bare payload vs tuples) by mode count; Go does NOT — `StreamChunk` always carries `Mode` and `Namespace`. Documented simplification (the type system makes Python's shape-shifting unnecessary).
- **S4. Serde.**
```go
// In package checkpoint (checkpoint.go):
type Serializer interface {
    DumpsTyped(v any) (typ string, data []byte, err error)
    LoadsTyped(typ string, data []byte) (any, error)
}
```
JSON implementation in new package `langgraph/checkpoint/serde` (`NewJSONSerializer()`): primitives/`map[string]any`/`[]any` encode as plain JSON (tag `"json"`); concrete types round-trip via a closed registry envelope `{"__type__": "<name>", "data": <payload>}` (tag `"json+envelope"`); registry covers `messages.Message`, `[]messages.Message`, `types.Send`, `types.Interrupt`, `time.Time`, `[]byte`, `int64`, `int`, `[]string` (int64 AND int preserved explicitly since JSON decodes numbers as float64; `[]string` because `AppendSliceReducer` channel values are typed slices that JSON would decode as `[]any` and then fail the reducer's type check on the next fold). Unknown type on encode → error (no silent lossy fallback); unknown envelope name on decode → error. Documented contract: checkpointed channel values must be JSON-canonical types or registry members — custom structs belong in the registry (extend it) or the saver rejects them.
- **S5. SQLite nested module.** `langgraph/checkpoint/sqlite/` is its own Go module (`module github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite`) with `require github.com/projanvil/langchain-golang v0.0.0` + `replace github.com/projanvil/langchain-golang => ../../../` and `require modernc.org/sqlite`. Schema mirrors Python exactly (Task 4).
- **S6. Metadata.Step alignment (M2 debt).** `UpdateState` checkpoints save with `Step: base.Step+1` (Python `main.py:1734`); new-turn input checkpoints save with `Step = <restored base checkpoint's step>` (Python `_loop.py:1402,1700`: input checkpoint gets `self.step - 1` where `self.step = base.step + 1`); only a thread's FIRST input checkpoint is -1. The one existing test that changes is `snapshot_test.go`'s update-checkpoint `Step == 0` assertion (~line 256) → becomes 1; loop-step assertions elsewhere are unaffected.

## Reference Semantics (from Python source, cited in the M3 research)

- `values`: emitted once per superstep after apply_writes when an output channel was touched (`pregel/_loop.py:691-707`); also after the input batch; on pause, `__interrupt__` merges into values (`_loop.py:1336-1371`).
- `updates`: emitted PER TASK as its writes land (`main.py:2927` → `_loop.py:507-508,1416-1466`); `{task.name: {chan: value}}`; interrupt chunk `{ "__interrupt__": (Interrupt, ...) }`; cached/replayed writes re-emitted on resume (`_loop.py:676-679`).
- `debug`: wrapper `{"step", "timestamp", "type", "payload"}`; `task` payload `{id, name, input, triggers}`; `task_result` payload `{id, name, error, result, interrupts}`; `checkpoint` payload `{config, parent_config, values, metadata, next, tasks}` (`pregel/debug.py:41-206`).
- `messages`: `(message_chunk, metadata)` per LLM token via a callback handler; metadata carries node name/step/ns (`pregel/_messages.py`).
- SQLite schema: `checkpoints(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint BLOB, metadata BLOB, PK(thread_id, checkpoint_ns, checkpoint_id))`; `writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value BLOB, PK(thread_id, checkpoint_ns, checkpoint_id, task_id, idx))`; WAL mode; writes to reserved channels use negative idx with INSERT OR REPLACE, regular writes INSERT OR IGNORE (`libs/checkpoint-sqlite/.../sqlite/__init__.py:139-501`).

---

### Task 1: `langgraph/checkpoint/serde` — Serializer + JSON registry

**Files:**
- Modify: `langgraph/checkpoint/checkpoint.go` — add the `Serializer` interface (S4) only
- Create: `langgraph/checkpoint/serde/json.go`, `langgraph/checkpoint/serde/doc.go`
- Test: `langgraph/checkpoint/serde/json_test.go`

**Interfaces:**
- Produces: `checkpoint.Serializer` (S4); `serde.NewJSONSerializer() checkpoint.Serializer`. Registry envelope names (closed set): `"messages.Message"`, `"[]messages.Message"`, `"types.Send"`, `"types.Interrupt"`, `"time.Time"`, `"[]byte"`, `"int64"`. Everything else must be JSON-round-trippable (`string`, `float64`, `bool`, `nil`, `map[string]any`, `[]any`) else encode errors.

- [ ] **Step 1: Write failing tests** — round-trip every registry type + primitives/maps/slices; `int64` survives as int64 AND `int` survives as int (not float64); `[]string` survives as `[]string` (not `[]any`); `[]byte` survives (not base64 string); `types.Send{Node, Arg}` and `types.Interrupt{Value, ID}` with nested map values round-trip; unregistered concrete struct → encode error; unknown envelope name → decode error; envelope payload containing nested registered values (e.g. `types.Send.Arg` holding `[]messages.Message`) round-trips recursively.
- [ ] **Step 2: Run, verify failure. Step 3: Implement. Step 4: gate PASS.**
- [ ] **Step 5: Commit** `feat(langgraph/checkpoint/serde): typed JSON serializer with closed type registry`.

---

### Task 2: `Stream` core — values/updates/debug modes

**Files:**
- Create: `langgraph/graph/stream.go` — StreamMode/StreamChunk/StreamOptions/Stream + emission hooks
- Test: `langgraph/graph/stream_test.go`
- Modify: `langgraph/graph/graph.go` — call the emission hooks (input batch, per-task completion, superstep commit, pause paths, resume replay); no signature changes
- Modify: `langgraph/graph/subgraph.go` — propagate the emission layer + emission namespace (node path) through `invokeSubgraph` (currently hard-codes a nil sink)
- Modify: `langgraph/graph/resume.go` — `resumeFromTuple` exposes the replayed writes so resume re-emits them as `updates` chunks

**Interfaces:**
- Consumes: M2 executor internals (runState, PlannedTask, applyWrites points, saveCheckpoint points).
- Produces: S1/S2/S3 API. Debug payloads: `task` = `{"id","name","input","triggers"}` (`triggers` is approximate in the edge-driven model — the predecessor node name where known, else the node name; documented); `task_result` = `{"id","name","error","result","interrupts"}`; `checkpoint` = `{"config","parent_config","values","metadata","next"}` (Go's Checkpoint has no per-task `tasks` detail — document the omission).
- Emission timing (from Reference Semantics): `values` after the input batch + after each superstep commit WHERE at least one channel version was bumped (`applyWrites` must return whether anything changed — mirrors Python's `updated_channels ∩ output_keys` gate) + merged `__interrupt__` on pause; `updates` per task as outcomes are collected (post-`wg.Wait`, in deterministic task order — documented divergence from Python's as-they-finish timing, needed because Go applies writes only after all tasks complete; consequence: in multi-mode streams, `updates` chunks bunch after node-time `messages`/`custom` chunks instead of interleaving — qualify S1's "chronological" accordingly); replayed writes on resume re-emitted as `updates`; `debug` task events at dispatch, task_result at completion, checkpoint after each save.

- [ ] **Step 1: Write failing tests** — linear 3-superstep graph: `values` sequence (input + per superstep, NONE for a no-change superstep); `updates` per-task chunks with exact `{node: {key: value}}` payloads; `debug` task/task_result/checkpoint sequence with step numbers (single-turn graphs only — step numbering on update/new-turn paths changes in Task 5); multi-mode combination yields all modes (updates-bunching divergence asserted, not just documented); pause run yields the `__interrupt__` chunks in both `values` and `updates`; resume re-emits replayed `updates`; `Subgraphs: false` drops subgraph chunks, `true` includes them with the correct node-path `Namespace` — test this BOTH with and without a checkpointer; early break cancels (no goroutine leak — assert via a done-channel probe).
- [ ] **Step 2: Run, verify failure. Step 3: Implement.**
- [ ] **Step 4: Gate PASS** (all M1/M2 tests unmodified). **Step 5: Commit** `feat(langgraph/graph): Stream API with values/updates/debug modes`.

---

### Task 3: `messages` and `custom` modes

**Files:**
- Modify: `core/callbacks/callbacks.go` — add `ContextWithManager(ctx, Manager)` / `ManagerFromContext(ctx) (Manager, bool)` (context carrier; additive)
- Create: `langgraph/graph/stream_messages.go` — the messages bridge handler + metadata stamping
- Create: `langgraph/graph/stream_custom.go` — StreamWriter (or fold into stream.go)
- Test: `langgraph/graph/stream_messages_test.go`

**Interfaces:**
- Produces:
  - `messages` payload: `MessageChunk{Message messages.Message, Metadata map[string]any}`; Metadata = `{"langgraph_node": name, "langgraph_step": step, "langgraph_checkpoint_ns": ns}`.
  - Mechanism: when `StreamMessages` is active, the executor installs a `callbacks.Manager` into each node's context (`ContextWithManager`) whose handler maps `EventChatModelStream`/`EventLLMStream` (Chunk = `messages.Message`) to `messages` chunks, deduping the final `EventChatModelEnd` message by message ID (Python parity). Node code opts in by building model configs from `callbacks.ManagerFromContext(ctx)` — documented in the mode's doc comment.
  - `custom`: `type StreamWriter func(payload any)`; `StreamWriterFromContext(ctx) StreamWriter` (nil when custom mode inactive); payloads flow straight to the chunk stream with the emitting node's namespace.

- [ ] **Step 1: Write failing tests** — a fake node that pulls `ManagerFromContext` and emits `EventChatModelStream` chunks + `EventChatModelEnd` → iterator yields per-token `messages` chunks with correct metadata, end-message deduped; node calling `StreamWriterFromContext(ctx)("progress: 50%")` → `custom` chunk with that payload; both inert when their modes are not requested (nil/zero overhead).
- [ ] **Step 2: Run, verify failure. Step 3: Implement. Step 4: Gate PASS.**
- [ ] **Step 5: Commit** `feat(langgraph/graph): messages and custom stream modes`.

---

### Task 4: SQLite checkpoint saver (nested module)

**Files:**
- Create: `langgraph/checkpoint/sqlite/go.mod`, `go.sum`
- Create: `langgraph/checkpoint/sqlite/sqlite.go` (saver), `doc.go`
- Test: `langgraph/checkpoint/sqlite/sqlite_test.go`
- Modify: root `README.md` (one line: nested module + how to test it)

**Interfaces:**
- Consumes: `checkpoint.Saver` (M2), `checkpoint.Serializer` + `serde.NewJSONSerializer()` (Task 1).
- Produces: `sqlite.New(path string, serde checkpoint.Serializer) (*Saver, error)` implementing `checkpoint.Saver`; `:memory:` supported. Schema exactly per Reference Semantics (two tables, WAL, type+value columns, metadata as plain JSON). Write-batch insert rule mirrors Python's BATCH-level decision: if ALL channels in a `PutWrites` batch are reserved (`__error__`→-1, `__tasks__`→-2, `__interrupt__`→-3 — an approximation of Python's WRITES_IDX_MAP `{ERROR:-1, SCHEDULED:-2, INTERRUPT:-3, RESUME:-4}`; Go has no SCHEDULED/RESUME writes and `ReservedError` is currently unused — code-comment this as forward-compat), use INSERT OR REPLACE with the negative idx; otherwise INSERT OR IGNORE with positional idx (first-write-wins). Checkpoint blob = serde-typed serialization of a JSON-safe projection of `Checkpoint` (ChannelValues per-value through the serde; versions maps plain); document the exact projection in code.

- [ ] **Step 1: Scaffold the nested module** (`go mod init`, add requires/replace, `go mod tidy` — requires network for modernc.org/sqlite; if the module fetch fails, STOP with BLOCKED).
- [ ] **Step 2: Write failing tests** — mirror the Python sqlite test categories relevant to a checkpoint saver: put/get-latest/get-by-id round-trip (channel values with messages + interrupts survive serde); **reducer-channel round-trip: `AddReducer("log", channels.AppendSliceReducer)` with `[]string` writes → checkpoint → restore via the SQLite saver → continue the run and fold more writes (catches typed-slice serde corruption)**; `int` state values survive as `int`; List newest-first + Before + Limit; PutWrites visibility incl. the batch-level replace-vs-ignore rule (a mixed state-key+sends batch is IGNORE-on-duplicate; an all-reserved batch is REPLACE); DeleteThread; D3 parent links; concurrent access (WAL); `:memory:`.
- [ ] **Step 3: Implement `sqlite.go`. Step 4: `go build ./... && go vet ./... && go test ./...` in the nested module PASS; root module gate still PASS (root does not import the nested module). Also add a `test-sqlite` target to the root `Makefile` that cds into the nested module and runs its tests (the root `test` target stays unchanged).**
- [ ] **Step 5: Commit** `feat(langgraph/checkpoint/sqlite): SQLite checkpoint saver (nested module, modernc.org/sqlite)`.

---

### Task 5: Metadata.Step alignment + M2 backlog tests

**Files:**
- Modify: `langgraph/graph/snapshot.go`, `graph.go` (S6 step numbering)
- Test: `langgraph/graph/graph_test.go`, `snapshot_test.go`, `subgraph_test.go`, `resume_test.go` (extend)

**Interfaces:**
- S6: `UpdateState` → `Step: base.Step+1`; new-turn input checkpoint continues the step counter (first-ever input stays -1). Existing tests asserting old numbering are updated.
- Backlog tests (from M2 final review): goto-only completed sibling replay (empty update map shape); `Options.CheckpointID` + non-nil input (D2 new-turn from pinned state); interrupt → `UpdateState` → resume HITL flow; subgraph node executed twice under one pinned run (pin-once-per-run documented behavior); subgraph with interrupt inside → descriptive error.

- [ ] **Step 1: Write/adjust tests (failing first where behavior changes). Step 2: Implement S6. Step 3: Gate PASS.**
- [ ] **Step 4: Commit** `feat(langgraph/graph): align Metadata.Step with Python; close M2 test backlog`.

---

### Task 6: Docs + spec mark

**Files:**
- Modify: `README.md`, `docs/usage/agents.md` (if it references streaming/checkpointing), new `docs/usage/langgraph.md` IF the README bullet grows past ~5 lines (judge by the README's existing structure)
- Modify: `docs/superpowers/specs/2026-08-07-langgraph-go-port-design.md` (mark M3 done)

- [ ] **Step 1: Document** the Stream API (one short example), serde (registry + divergence note), the SQLite nested module (dependency notice: first third-party dep, how to run its tests), M3 capabilities in the README langgraph bullet.
- [ ] **Step 2: Mark M3 complete** in the spec (actual date).
- [ ] **Step 3: Gate PASS; commit** `docs: document M3 streaming and persistence`.

---

## Self-Review Notes

- Spec coverage: stream modes → Tasks 2–3; serde → Task 1; SQLite → Task 4; M2 debt → Task 5; docs → Task 6. All M3 spec bullets (as amended 2026-08-08) covered; nothing beyond scope.
- Type consistency: `Serializer` lives in `checkpoint` (Task 1) consumed by Task 4; `StreamChunk`/`StreamOptions` (Task 2) consumed by Task 3; nested module (Task 4) depends on root via replace directive.
- Risks: (a) modernc.org/sqlite fetch requires network at Task 4 Step 1 — the step says BLOCKED on failure; (b) messages mode depends on node opt-in via `ManagerFromContext` — partial parity by design, documented; (c) the nested module is invisible to the root `go test ./...` — Task 4's gate runs it separately, adds a `test-sqlite` Makefile target, and README documents it.
- Review history: an adversarial plan review returned FIX-FIRST with 6 issues; all resolved — serde registry gained `int`/`[]string` + the JSON-canonical contract (with a Task 4 typed-slice round-trip test); subgraph stream namespace derives from the node path (works without a checkpointer); Task 2's file list gained `subgraph.go`/`resume.go` (replayed-writes exposure) and its `values` emission is gated on actual channel change; the SQLite insert rule is batch-level per Python; S6's exact step rule is written down (new-turn input = base step; the one changing test is `snapshot_test.go`'s update-step assertion).
