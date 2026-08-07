# LangGraph Go Port M2: Versioned Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen the public `langgraph/` module from M1's single-checkpoint runtime to a versioned, time-travel-capable core: channel objects, versioned checkpoints with history, pending-writes-based interrupt/resume fidelity, state inspection APIs (`GetState`/`GetStateHistory`/`UpdateState`), and subgraphs — while `langchain/agents` keeps passing with zero changes.

**Architecture:** Retain the edge-driven superstep executor (a 2026-08-08 spec amendment records why this is observably equivalent to Python's PULL triggers for StateGraph-built graphs) and layer Python's version bookkeeping on top: channel objects with batch `Update`, a single global version bump per superstep, `versions_seen` tracking, and per-task pending writes persisted against checkpoints so resume never re-runs completed sibling tasks. `checkpoint.Saver` is replaced by a versioned history interface (pre-1.0 breaking change, sanctioned by the spec).

**Tech Stack:** Go 1.23, module `github.com/projanvil/langchain-golang`, standard library only.

## Global Constraints

- Repo root for all commands: `langchain_golang/`.
- Go 1.23 floor; no new third-party dependencies.
- Clean, idiomatic Go (project owner requirement): small interfaces, no Python-isms leaked into API shapes, no gratuitous reflection.
- State model stays `map[string]any` at the API surface; channel objects are the internal/container layer.
- `graph.CompiledGraph` external API stability: `Invoke`, `InvokeStream`, `Options`, `Result`, `WithCheckpointer`, `WithRecursionLimit`, `WithInterruptBefore/After`, `Interrupt(ctx, value)`, event-sink helpers — signatures unchanged; `langchain/agents` must pass with **zero edits**.
- `checkpoint.Saver` breaking change is sanctioned (pre-1.0). The `agentruntime` shim re-aliases the new API.
- Comment style: extensive doc comments, single backticks. Conventional commits.
- After every task: `go build ./... && go vet ./... && go test ./...` from `langchain_golang/` must pass before committing.
- Known pre-existing issue, NOT to be "fixed" as a side effect: `gofmt -l` reports drift in some `langchain/` files (older-gofmt formatting vs local go1.26.4). Do not reformat unrelated files.

## Reference Semantics (distilled from Python source; cited file:line in `langgraph` repo)

- Checkpoint fields: `v`, `id` (monotonic, `uuid6(clock_seq=step)`), `ts`, `channel_values`, `channel_versions`, `versions_seen` (`libs/checkpoint/langgraph/checkpoint/base/__init__.py:92-123`).
- `apply_writes` (`libs/langgraph/langgraph/pregel/_algo.py:232-345`): per superstep — record `versions_seen[node][chan]` pre-write; compute ONE global `next_version = max(all versions)+1`; batch-update each written channel (`update(vals)`), bump version on change; call `update([])` on every untouched available channel (step-boundary notification); bump on change.
- Channel batch semantics (`libs/langgraph/langgraph/channels/`): `LastValue` errors on >1 write/step (`last_value.py:56-67`); `Topic` flattens list values, `accumulate=False` clears at each step (`topic.py:77-85`); `BinaryOperatorAggregate` folds left-to-right, first value seeds when empty (`binop.py:123-144`); `EphemeralValue` clears on empty update (`ephemeral_value.py:55-61`).
- Interrupt/resume (`pregel/_runner.py:585-613`, `_loop.py:661-664`): on interrupt, completed tasks' writes are persisted via `put_writes`; checkpoint keeps pre-superstep versions; resume replays persisted writes and re-executes only interrupted tasks, feeding resume values in call order.
- `update_state` (`pregel/main.py:2515-2526`): apply values as if written by `as_node`, re-resolve that node's successors, save a new checkpoint with `source="update"`.
- Subgraph `Command(graph=PARENT)` (`graph/state.py:1747-1748`, `pregel/_retry.py:618-631`): bubbles to the parent graph, which applies update+goto itself.

---

### Task 1: Channel objects in `langgraph/channels`

**Files:**
- Create: `langgraph/channels/channel.go` (interface + errors)
- Create: `langgraph/channels/lastvalue.go`, `topic.go`, `binop.go`, `ephemeral.go`
- Test: `langgraph/channels/channel_test.go` (new), existing `channels_test.go` untouched

**Interfaces:**
- Consumes: existing `Reducer` func type and `LastValueReducer`/`AppendSliceReducer`/`MessagesReducer` (unchanged, reused by `BinaryOperator`).
- Produces (consumed by Tasks 3–6):

```go
// Package channels — added to the existing package doc:
// Channel is a stateful container for one graph-state key, the Go analog of
// Python's BaseChannel. The executor feeds it one superstep's writes at a
// time (Update), reads it via Get, and snapshots it via Checkpoint.
type Channel interface {
    // Update applies a whole superstep's writes for this key (possibly
    // empty — an empty slice is the step-boundary notification some
    // channels use to expire themselves). Reports whether the value changed.
    Update(values []any) (changed bool, err error)
    // Get returns the current value or ErrEmptyChannel if never updated.
    Get() (any, error)
    // IsAvailable reports whether Get would succeed.
    IsAvailable() bool
    // Checkpoint returns the serializable snapshot; ok=false means "omit
    // from the checkpoint" (channel empty).
    Checkpoint() (value any, ok bool)
    // FromCheckpoint returns a fresh channel restored from a Checkpoint
    // value (nil when the channel was omitted). It never mutates the receiver.
    FromCheckpoint(value any) Channel
}

var ErrEmptyChannel = errors.New("channels: channel is empty")

type InvalidUpdateError struct{ Channel, Reason string } // Error() string; construct with field names
```

- `NewLastValue() Channel` — >1 value in one `Update` → `*InvalidUpdateError` (Python `last_value.py:56-67`); `Update([v])` sets; empty update → unchanged.
- `NewTopic(accumulate bool) Channel` — each value may be `[]any` (flattened one level); `accumulate=false` clears prior values at the start of every `Update` call; `Get` returns `[]any`, empty → `ErrEmptyChannel`.
- `NewBinaryOperator(op Reducer) Channel` — folds values left-to-right; when empty, the first value becomes the seed without applying `op` (Python `binop.py:126-128`).
- `NewEphemeral(guard bool) Channel` — keeps only the previous step's value: empty `Update` with a stored value clears it (changed=true); `guard=true` errors on >1 value, `guard=false` takes the last.

- [ ] **Step 1: Write failing tests** in `langgraph/channels/channel_test.go` covering, per channel: empty→Get error; single write; multi-write (error for LastValue/guarded Ephemeral; fold for BinaryOperator using `AppendSliceReducer` and `MessagesReducer`; flatten for Topic); empty-update step notification (Ephemeral clears, non-accumulating Topic clears, others unchanged); `Checkpoint`/`FromCheckpoint` round-trip incl. omitted-when-empty. Table-driven, one test func per channel (`TestLastValue`, `TestTopic`, `TestBinaryOperator`, `TestEphemeral`).
- [ ] **Step 2: Run tests, verify compile failure** (`channel.go` missing).
- [ ] **Step 3: Implement the five files.** Each implementation is a small struct (10–40 lines); no reflection beyond the existing reducer helpers.
- [ ] **Step 4: `go build ./... && go vet ./... && go test ./langgraph/... ./langchain/...` — PASS.**
- [ ] **Step 5: Commit** `feat(langgraph/channels): add stateful Channel objects (LastValue/Topic/BinaryOperator/Ephemeral)`.

---

### Task 2: Versioned `langgraph/checkpoint`

**Files:**
- Modify (rewrite): `langgraph/checkpoint/checkpoint.go` — new types + new `Saver` interface
- Create: `langgraph/checkpoint/id.go` — monotonic ID generator
- Create: `langgraph/checkpoint/memory.go` — new `MemorySaver`
- Modify (rewrite): `langgraph/checkpoint/checkpoint_test.go`
- Modify: `langchain/internal/agentruntime/checkpoint/checkpoint.go` — re-alias new names
- Modify: `langgraph/graph/graph.go` — MINIMAL edits only to keep compiling against the new Saver (see Step 4); full executor adoption is Task 3

**Interfaces:**
- Produces (consumed by Tasks 3–6):

```go
type Config struct {
    ThreadID     string
    CheckpointNS string // "" = root graph; subgraph runs use "node" or "a/b"
    CheckpointID string // empty = latest
}

type Checkpoint struct {
    V               int               // Go format version, always 1
    ID              string            // NewID(step); lexicographically sortable = chronological
    TS              time.Time
    ChannelValues   map[string]any
    ChannelVersions map[string]int64
    VersionsSeen    map[string]map[string]int64
    Next            []PlannedTask     // tasks scheduled for the superstep after this checkpoint
}

type PlannedTask struct {
    ID   string         // TaskID(cpID, step, node, arg)
    Node string
    Arg  map[string]any // non-nil only for Send-driven invocations
}

type Metadata struct {
    Source  string            // "input" | "loop" | "update" | "fork"
    Step    int               // -1 for the "input" checkpoint, 0 for the first "loop"
    Parents map[string]string // checkpoint_ns -> parent checkpoint id
}

type Write struct {
    TaskID  string
    Channel string // state key, or one of the Reserved* constants
    Value   any
}

const (
    ReservedInterrupt = "__interrupt__" // Value: types.Interrupt
    ReservedTasks     = "__tasks__"     // Value: types.Send
    ReservedError     = "__error__"     // Value: string
)

type Tuple struct {
    Config        Config
    Checkpoint    Checkpoint
    Metadata      Metadata
    ParentConfig  *Config
    PendingWrites []Write
}

type ListOptions struct {
    Before *Config // only checkpoints strictly before this one
    Limit  int     // 0 = no limit
}

type Saver interface {
    GetTuple(ctx context.Context, cfg Config) (*Tuple, error) // (nil, nil) when absent
    List(ctx context.Context, cfg Config, opts ListOptions) ([]Tuple, error) // newest first
    Put(ctx context.Context, cfg Config, cp Checkpoint, md Metadata, newVersions map[string]int64) (Config, error)
    PutWrites(ctx context.Context, cfg Config, writes []Write, taskID string) error
    DeleteThread(ctx context.Context, threadID string) error
}
```

- `id.go`: `func NewID(clockSeq int) string` — dependency-free monotonic ID: `fmt.Sprintf("%013d-%06x-%016x", time.Now().UnixMilli(), clockSeq&0xffffff, randUint64())` where `randUint64` reads 8 bytes from `crypto/rand`. Fixed-width fields make lexicographic order == chronological order. Document divergence from Python's uuid6 (no interop requirement).
- `memory.go`: `MemorySaver` (zero value ready) storing `map[threadID]map[ns]map[id]stored`, where `stored{cp Checkpoint; md Metadata; writes []Write}`; `List` sorts IDs descending (IDs are sortable), applies `Before`/`Limit`; `GetTuple` computes `ParentConfig` (the second-newest checkpoint in the same thread+ns) and attaches stored pending writes; all reads return deep copies of the maps (shallow for values) so callers can't alias store state.

- [ ] **Step 1: Write failing tests** (`checkpoint_test.go` rewrite): put/get-latest/get-by-id round-trip; multi-checkpoint list newest-first with `Before`+`Limit`; `PutWrites` visible on `GetTuple.PendingWrites`; `DeleteThread`; `NewID` monotonicity (IDs sort ascending for ascending clockSeq); copy-on-read (mutating a returned map doesn't corrupt the store).
- [ ] **Step 2: Run tests, verify they fail.**
- [ ] **Step 3: Implement `checkpoint.go`, `id.go`, `memory.go`.**
- [ ] **Step 4: Keep the module green.** The old `Saver` is gone, so minimal-touch the two dependents:
  - `agentruntime/checkpoint/checkpoint.go`: re-alias the new names (`Checkpoint`, `Saver`, `MemorySaver`, `NewMemorySaver`, plus new `Config`/`Tuple`/`Write`/`Metadata`/`ListOptions`/`PlannedTask`).
  - `langgraph/graph/graph.go`: adapt the existing single-checkpoint call sites to the new interface with the SMALLEST possible change (thread through `Config{ThreadID: opts.ThreadID}`, map old `Get`→`GetTuple`-latest, `Put`→`Put` with `Metadata{Source:"loop"}`, `Delete`→`DeleteThread`; keep old `Checkpoint` field usage working by reading `cp.Checkpoint.ChannelValues`). Do NOT restructure the executor — Task 3 owns that. Interim helper functions are fine; mark them with `// TODO(M2 Task 3): replaced by the versioned executor`.
  - Also adapt `WithCheckpointer` doc comments minimally.
- [ ] **Step 5: `go build ./... && go vet ./... && go test ./...` — PASS** (agents green unmodified).
- [ ] **Step 6: Commit** `feat(langgraph/checkpoint): versioned checkpoints with history (breaking Saver redesign)`.

---

### Task 3: Versioned executor core (`langgraph/graph`)

**Files:**
- Modify: `langgraph/graph/graph.go` (restructure internals; public API unchanged)
- Create: `langgraph/graph/state.go` — `runState` (channel map + versions + versions_seen + step) and `applyWrites`
- Create: `langgraph/graph/taskid.go` — deterministic task IDs
- Test: `langgraph/graph/graph_test.go` (extend — existing tests must pass unmodified)
- Modify: `langgraph/graph/graph.go` builder: add `AddChannel`

**Interfaces:**
- Consumes: Task 1 `channels.Channel`; Task 2 `checkpoint` v2.
- Produces (consumed by Tasks 4–6):
  - `(*StateGraph).AddChannel(key string, prototype channels.Channel) *StateGraph` — registers an explicit channel; `AddReducer(key, r)` now registers `channels.NewBinaryOperator(r)`; unregistered keys default to `channels.NewLastValue()`. Reducer-registered keys keep M1 behavior (fold).
  - Internal `runState`: `channels map[string]channels.Channel`, `versions map[string]int64`, `seen map[string]map[string]int64`, `step int`; methods `snapshot() map[string]any` (available channels), `restore(cp checkpoint.Checkpoint)`, `applyWrites(batch []taskWrites) (updated []string, err error)`.
  - `TaskID(cpID string, step int, node string, arg map[string]any) string` — fnv-1a 64-bit over `cpID`, step, node, and `json.Marshal(arg)` (fall back to `fmt.Sprintf("%#v")` on marshal error), hex-encoded.

**Executor semantics (must match exactly):**
1. On fresh start: input map → one batch of writes → `applyWrites` → if checkpointer+ThreadID: save checkpoint `Metadata{Source:"input", Step:-1}`.
2. Per superstep: run active tasks concurrently (unchanged concurrency + event-sink behavior) → collect outcomes in deterministic order (active-task slice order, NOT map iteration) → `applyWrites`: for each executed task record `seen[task.Node]` for the channels it read (its input keys = all state keys, since Go nodes receive the full state); compute `nextVersion = max(versions)+1` once; per channel, `Update(batch)` in task order, bump to `nextVersion` on change; `Update(nil)` on untouched available channels → save checkpoint `Source:"loop"`, `Step: step+1`, `Next: <resolved next tasks as []PlannedTask>`.
3. `Result.Values` = `runState.snapshot()` (unchanged external shape).
4. Interrupt path stays M1-shaped for now (Task 4 upgrades it); boundary interrupts keep working.

- [ ] **Step 1: Write failing tests** (new test funcs in `graph_test.go`): version bookkeeping across a 3-superstep linear graph (assert versions via a test hook — a checkpoint `Tuple` inspection through `MemorySaver`); per-superstep single global version bump; `LastValue` double-write-in-one-superstep now errors (Send fan-out to two nodes writing the same unregistered key); reducer key fold order is deterministic across runs; `Ephemeral`/non-accumulating `Topic` keys expire between supersteps (register via `AddChannel`).
- [ ] **Step 2: Run tests, verify failure.**
- [ ] **Step 3: Implement `state.go`, `taskid.go`, restructure `graph.go` internals; add `AddChannel`.** Keep every existing public signature; existing `graph_test.go`/`integration_test.go` pass unmodified.
- [ ] **Step 4: Full gate PASS; commit** `feat(langgraph/graph): channel-backed versioned executor core`.

---

### Task 4: Pending writes and interrupt/resume fidelity

**Files:**
- Modify: `langgraph/graph/graph.go` (interrupt + resume paths)
- Create: `langgraph/graph/resume.go` (pending-write replay helpers)
- Test: `langgraph/graph/graph_test.go` (extend)

**Interfaces:**
- Consumes: Tasks 2–3 (`Saver.PutWrites`, `Tuple.PendingWrites`, `PlannedTask`, `TaskID`).
- Produces: upgraded resume semantics used by Task 5 snapshots.

**Semantics (match Python):**
1. On in-node interrupt: persist each COMPLETED sibling task's writes via `PutWrites` (its state-update writes as `Write{taskID, key, value}` entries + goto targets/Sends as `Write{taskID, ReservedTasks, send}`); persist `Write{taskID, ReservedInterrupt, interrupt}` for the interrupted task; save checkpoint with PRE-superstep channel state and `Next = <this superstep's full active task set as []PlannedTask>`.
2. On resume: load tuple; for each planned task in `Next`: tasks whose `PendingWrites` contain only completed-work writes are NOT re-run — their state writes are applied via `applyWrites` and their `ReservedTasks` sends repopulate the next-task queue; the interrupted task (has a `ReservedInterrupt` write) re-executes with its resume queue (existing `taskInterruptState` mechanism).
3. Boundary interrupts (`interrupt_before`/`after`): checkpoint `Next` now stores the FULL planned task set (fixes the M1 single-`Next` limitation — delete that limitation note from `WithInterruptBefore`'s doc comment).
4. `Options.Resume` semantics (nil resume + existing checkpoint = resume; map-by-interrupt-ID; single value → first pending interrupt) unchanged.

- [ ] **Step 1: Write failing tests**: (a) Send fan-out where sibling A completes and sibling B interrupts → resume → A does NOT re-run (probe via a side-effect counter), A's updates present exactly once; (b) Sends issued by a completed task in the interrupting superstep survive the resume; (c) `interrupt_before` on a multi-successor superstep reschedules ALL siblings on resume; (d) resume-by-interrupt-ID map still works; (e) all M1 interrupt tests pass unmodified.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement `resume.go` + the `graph.go` interrupt/resume paths.**
- [ ] **Step 4: Full gate PASS; commit** `feat(langgraph/graph): pending-writes resume fidelity (siblings not re-run, sends survive interrupts)`.

---

### Task 5: State inspection APIs and time travel

**Files:**
- Create: `langgraph/graph/snapshot.go`
- Modify: `langgraph/graph/graph.go` (`Options` gains `CheckpointID`)
- Test: `langgraph/graph/snapshot_test.go`

**Interfaces:**
- Produces:

```go
type StateSnapshot struct {
    Values       map[string]any
    Next         []string            // node names planned for the next superstep
    Config       checkpoint.Config   // selects exactly this snapshot
    Metadata     checkpoint.Metadata
    CreatedAt    time.Time
    ParentConfig *checkpoint.Config
    Interrupts   []types.Interrupt   // pending interrupts, if paused
}

func (g *CompiledGraph) GetState(ctx context.Context, cfg checkpoint.Config) (StateSnapshot, error)
func (g *CompiledGraph) GetStateHistory(ctx context.Context, cfg checkpoint.Config, opts checkpoint.ListOptions) ([]StateSnapshot, error)
func (g *CompiledGraph) UpdateState(ctx context.Context, cfg checkpoint.Config, values map[string]any, asNode string) (checkpoint.Config, error)
```

- `GetState`: latest (or `CheckpointID`-pinned) tuple → snapshot; `Next` from stored `PlannedTask`s; `Interrupts` from `ReservedInterrupt` pending writes.
- `GetStateHistory`: `Saver.List` → one snapshot per checkpoint.
- `UpdateState`: apply `values` as one write batch attributed to `asNode` (unknown node → error; empty `asNode` → error unless the graph has exactly one node, matching Python's ambiguity rule in spirit), re-resolve `asNode`'s static/conditional successors into the new `Next`, save checkpoint `Source:"update"`, return the new Config. Requires a checkpointer.
- `Options.CheckpointID string`: with `ThreadID`, starts/resumes from that historical checkpoint instead of latest (time travel). Fresh input is ignored when resuming from a checkpoint, as today.

- [ ] **Step 1: Write failing tests**: run a graph with checkpointer for 3 supersteps → `GetState` shape (Values/Next/Metadata.Step/CreatedAt/ParentConfig chain); `GetStateHistory` length and newest-first order; `UpdateState` creates a new `Source:"update"` checkpoint and changes subsequent `GetState().Values`; time travel — `InvokeWithOptions` with `CheckpointID` of superstep 1 continues execution from there (new checkpoints fork off it); `UpdateState` with unknown/ambiguous `asNode` errors.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement `snapshot.go` + `Options.CheckpointID`.**
- [ ] **Step 4: Full gate PASS; commit** `feat(langgraph/graph): GetState/GetStateHistory/UpdateState and checkpoint-pinning time travel`.

---

### Task 6: Subgraphs

**Files:**
- Modify: `langgraph/graph/graph.go` (`AddSubgraph`, `normalizeNodeResult` PARENT handling)
- Create: `langgraph/graph/subgraph.go`
- Test: `langgraph/graph/subgraph_test.go`

**Interfaces:**
- Produces:
  - `(*StateGraph).AddSubgraph(name string, child *CompiledGraph) *StateGraph` — registers a node that runs `child` with the parent state map as input and merges the child's final values back as its update. Child checkpointing (when the parent has a checkpointer+ThreadID) uses `CheckpointNS = <parentNS>/<name>`; document divergence from Python's `ns+task_id` namespacing.
  - `types.Command{Graph: types.ParentGraph}` returned by a node INSIDE a child graph bubbles to the parent: the parent's executor applies its `Update`/`Goto` at parent level. `normalizeNodeResult` accepts `Graph == ParentGraph` only from nodes registered via `AddSubgraph` (via an internal wrapper error type `ParentCommandError{Command *types.Command}` that the subgraph wrapper converts into a parent-level `*types.Command` with `Graph` cleared); any other non-empty `Graph` remains an error.

- [ ] **Step 1: Write failing tests**: subgraph as a node in a parent graph (shared keys flow in/out); two-level nesting; child node returns `Command{Graph: ParentGraph, Update: ..., Goto: ...}` → parent applies update and routes; child checkpoints land under the namespaced NS (inspect via `MemorySaver.List` with `CheckpointNS`); `Command.Graph = "bogus"` still errors.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement `subgraph.go` + graph.go hooks.**
- [ ] **Step 4: Full gate PASS; commit** `feat(langgraph/graph): subgraphs as nodes with parent-targeted Commands`.

---

### Task 7: Shim, docs, final gate

**Files:**
- Modify: `langchain/internal/agentruntime/checkpoint/checkpoint.go` (alias surface check — new names aliased in Task 2; verify completeness)
- Modify: `README.md` (langgraph description: versioned checkpoints, time travel, subgraphs), `docs/usage/agents.md` (checkpointer section — the `Saver` interface changed; update any interface listing)
- Modify: `docs/superpowers/specs/2026-08-07-langgraph-go-port-design.md` (mark M2 done)

- [ ] **Step 1: Sweep the shim** — `go doc` the four shim packages vs the `langgraph/*` packages; every exported name reachable before must still resolve. Note: `checkpoint.Saver`'s METHODS changed in Task 2 (sanctioned break); the agents code only names the type, so it compiles.
- [ ] **Step 2: Update README + usage docs** — rewrite the `langgraph/` bullet to mention versioned checkpoints/history/time-travel/subgraphs; fix `docs/usage/agents.md` checkpointer prose if it shows the old `Get/Put/Delete` interface.
- [ ] **Step 3: Mark M2 complete** in the spec heading: `### M2 全保真核心（已完成 2026-08-…）` (use the actual date).
- [ ] **Step 4: Full gate PASS; commit** `docs: document M2 langgraph capabilities`.

---

## Self-Review Notes

- Spec coverage: M2's amended bullets map to Tasks 1 (channels), 2 (versioned checkpoint + breaking saver), 3–4 (version bookkeeping + resume fidelity on the retained edge-driven loop), 5 (history/time travel), 6 (subgraphs + PARENT), 7 (shim/docs). Barrier joins and the PULL rewrite are excluded per the 2026-08-08 spec amendment.
- Type consistency: `PlannedTask`/`Write`/`Reserved*` defined once in Task 2 and consumed by Tasks 3–6; `AddChannel` (Task 3) precedes subgraph/state API usage; `Options.CheckpointID` (Task 5) does not collide with `Options.Resume`.
- Risk: Task 3 touches the executor's core while keeping the public API stable — the mitigating control is that all M1 tests must pass unmodified at every step.
