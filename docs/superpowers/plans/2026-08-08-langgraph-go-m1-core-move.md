# LangGraph Go Port M1: Core Move Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the internal `langchain/internal/agentruntime` graph runtime to a public top-level `langgraph/` package tree in `langchain_golang`, leaving `agentruntime` as a thin alias shim so `langchain/agents.CreateAgent` and all existing tests keep passing unchanged.

**Architecture:** Copy-then-shim. Tasks 1–4 create the new public packages by copying files out of `langchain/internal/agentruntime` (the old tree stays intact, so the module builds and tests green after every task). Task 5 replaces the old tree's contents with type-alias/var shims delegating to the new packages and deletes the duplicated old test files. Task 6 updates docs and runs the final full-module verification.

**Tech Stack:** Go 1.23, module `github.com/projanvil/langchain-golang`, standard library only (no new third-party dependencies).

## Global Constraints

- Repo root for all commands: `langchain_golang/` (the Go module root).
- Go version floor: `go 1.23` (per `go.mod`); do not use stdlib APIs newer than Go 1.23.
- No new third-party dependencies.
- State model is `map[string]any` + per-key reducers — do not introduce generics-based state.
- Comment style: match the existing extensive doc comments; use single backticks for inline code in comments (never Sphinx-style double backticks).
- Commit style: conventional commits as in repo history, e.g. `feat(langgraph): ...`, `refactor(agentruntime): ...`.
- The public API surface of `langchain/agents` must not change; its tests must pass without modification.
- After every task: `go build ./... && go test ./...` from `langchain_golang/` must pass before committing.

### Source files being moved (current locations)

| Current file | New location |
|---|---|
| `langchain/internal/agentruntime/types.go` | `langgraph/types/types.go` |
| `langchain/internal/agentruntime/channels/channels.go` (+ `_test.go`) | `langgraph/channels/channels.go` (+ `_test.go`) |
| `langchain/internal/agentruntime/checkpoint/checkpoint.go` (+ `_test.go`) | `langgraph/checkpoint/checkpoint.go` (+ `_test.go`) |
| `langchain/internal/agentruntime/graph/graph.go`, `events.go`, `graph_test.go`, `integration_test.go` | `langgraph/graph/` (same filenames) |

Import path rewrites applied during the copy:

- `github.com/projanvil/langchain-golang/langchain/internal/agentruntime` → `github.com/projanvil/langchain-golang/langgraph/types` (and identifier prefix `agentruntime.` → `types.`)
- `.../internal/agentruntime/channels` → `github.com/projanvil/langchain-golang/langgraph/channels`
- `.../internal/agentruntime/checkpoint` → `github.com/projanvil/langchain-golang/langgraph/checkpoint`
- `.../internal/agentruntime/graph` → `github.com/projanvil/langchain-golang/langgraph/graph`

---

### Task 1: Create `langgraph/types` package

**Files:**
- Create: `langgraph/types/types.go` (copied from `langchain/internal/agentruntime/types.go`)

**Interfaces:**
- Produces: package `types` with constants `START`, `END`, `ParentGraph` (all `string`), types `Send` (fields `Node string`, `Arg map[string]any`), `Command` (fields `Graph string`, `Update map[string]any`, `Resume any`, `Goto []any`), `Interrupt` (fields `Value any`, `ID string`), `GraphInterrupt` (field `Interrupt Interrupt`, method `Error() string`). Consumed by Tasks 3, 4, 5.

- [ ] **Step 1: Copy the file**

```bash
cd langchain_golang
mkdir -p langgraph/types
cp langchain/internal/agentruntime/types.go langgraph/types/types.go
```

- [ ] **Step 2: Rewrite the header of the new file**

In `langgraph/types/types.go`, replace everything from line 1 through `import "fmt"` (the old file-comment block and the `package agentruntime` clause) with:

```go
// Package types holds the shared control-flow primitives of the Go port of
// Python's `langgraph` (https://github.com/langchain-ai/langgraph): the
// START/END sentinels and the Send/Command/Interrupt values exchanged
// between graph nodes and the Pregel-style executor in langgraph/graph.
//
// This package corresponds to Python's `langgraph.types` plus
// `langgraph.constants` and `langgraph.errors` (GraphInterrupt).
package types

import "fmt"
```

The remainder of the file (the constants and type declarations) is unchanged.

- [ ] **Step 3: Build and test**

Run from `langchain_golang/`:

```bash
go build ./... && go test ./langgraph/... ./langchain/...
```

Expected: build succeeds; all tests pass (the new package has no tests yet; the old tree is untouched).

- [ ] **Step 4: Commit**

```bash
git add langgraph/types/types.go
git commit -m "feat(langgraph): add public types package (Send/Command/Interrupt)"
```

---

### Task 2: Create `langgraph/channels` package

**Files:**
- Create: `langgraph/channels/channels.go` (copied from `langchain/internal/agentruntime/channels/channels.go`)
- Create: `langgraph/channels/channels_test.go` (copied from `langchain/internal/agentruntime/channels/channels_test.go`)

**Interfaces:**
- Produces: package `channels` with `type Reducer func(existing any, update any) (any, error)` and funcs `LastValueReducer`, `AppendSliceReducer`, `MessagesReducer` (same signatures as today). Consumed by Tasks 4, 5.

- [ ] **Step 1: Copy the files**

```bash
cd langchain_golang
mkdir -p langgraph/channels
cp langchain/internal/agentruntime/channels/channels.go langchain/internal/agentruntime/channels/channels_test.go langgraph/channels/
```

- [ ] **Step 2: No code edits needed**

The channels files import only `fmt`, `reflect`, and `github.com/projanvil/langchain-golang/core/messages` — no agentruntime references. Verify with:

```bash
grep -n "agentruntime" langgraph/channels/*.go
```

Expected: no matches.

- [ ] **Step 3: Build and test**

```bash
go build ./... && go test ./langgraph/... ./langchain/...
```

Expected: PASS, including the copied `langgraph/channels` tests.

- [ ] **Step 4: Commit**

```bash
git add langgraph/channels/
git commit -m "feat(langgraph): add public channels package (reducers)"
```

---

### Task 3: Create `langgraph/checkpoint` package

**Files:**
- Create: `langgraph/checkpoint/checkpoint.go` (copied from `langchain/internal/agentruntime/checkpoint/checkpoint.go`)
- Create: `langgraph/checkpoint/checkpoint_test.go` (copied from `langchain/internal/agentruntime/checkpoint/checkpoint_test.go`)

**Interfaces:**
- Consumes: `types.Interrupt` from Task 1.
- Produces: package `checkpoint` with `type Checkpoint struct { Values map[string]any; Next string; PendingInterrupts []types.Interrupt }`, `type Saver interface` (`Get(threadID string) (Checkpoint, bool)`, `Put(threadID string, cp Checkpoint)`, `Delete(threadID string)`), `type MemorySaver`, `func NewMemorySaver() *MemorySaver`. Consumed by Tasks 4, 5.

- [ ] **Step 1: Copy the files**

```bash
cd langchain_golang
mkdir -p langgraph/checkpoint
cp langchain/internal/agentruntime/checkpoint/checkpoint.go langchain/internal/agentruntime/checkpoint/checkpoint_test.go langgraph/checkpoint/
```

- [ ] **Step 2: Rewrite imports and identifiers in both new files**

In `langgraph/checkpoint/checkpoint.go`, replace the import

```go
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
```

with

```go
	"github.com/projanvil/langchain-golang/langgraph/types"
```

and replace every `agentruntime.Interrupt` with `types.Interrupt` (one occurrence: the `PendingInterrupts` field).

In `langgraph/checkpoint/checkpoint_test.go`, apply the same import replacement and replace every `agentruntime.` identifier prefix with `types.` (the test constructs `agentruntime.Interrupt` values). Verify with:

```bash
grep -n "agentruntime" langgraph/checkpoint/*.go
```

Expected: no matches.

- [ ] **Step 3: Build and test**

```bash
go build ./... && go test ./langgraph/... ./langchain/...
```

Expected: PASS, including the copied `langgraph/checkpoint` tests.

- [ ] **Step 4: Commit**

```bash
git add langgraph/checkpoint/
git commit -m "feat(langgraph): add public checkpoint package (Saver, MemorySaver)"
```

---

### Task 4: Create `langgraph/graph` package

**Files:**
- Create: `langgraph/graph/graph.go` (copied from `langchain/internal/agentruntime/graph/graph.go`)
- Create: `langgraph/graph/events.go` (copied from `langchain/internal/agentruntime/graph/events.go`)
- Create: `langgraph/graph/graph_test.go` (copied from `langchain/internal/agentruntime/graph/graph_test.go`)
- Create: `langgraph/graph/integration_test.go` (copied from `langchain/internal/agentruntime/graph/integration_test.go`)

**Interfaces:**
- Consumes: `types.*` (Task 1), `channels.Reducer` etc. (Task 2), `checkpoint.Saver`/`checkpoint.Checkpoint` (Task 3).
- Produces: package `graph` with the full public surface consumed by Task 5's shim and later by `langchain/agents`: `NodeFunc`, `ConditionalEdge`, `To`, `StateGraph` + `NewStateGraph` (methods `AddNode`, `AddReducer`, `AddEdge`, `AddConditionalEdges`, `SetEntryPoint`, `Compile`), `CompileOption` + `WithCheckpointer`/`WithRecursionLimit`/`WithInterruptBefore`/`WithInterruptAfter`, `CompiledGraph` (methods `Invoke`, `InvokeWithOptions`, `InvokeStream`), `Options`, `Result`, `Interrupt(ctx, value)`, `RawEventKind` + `RawNodeStart`/`RawNodeEnd`, `RawEvent`, `NodeEventSink`, `ContextWithEventSink`, `EventSinkFromContext`.

- [ ] **Step 1: Copy the files**

```bash
cd langchain_golang
mkdir -p langgraph/graph
cp langchain/internal/agentruntime/graph/graph.go langchain/internal/agentruntime/graph/events.go langchain/internal/agentruntime/graph/graph_test.go langchain/internal/agentruntime/graph/integration_test.go langgraph/graph/
```

- [ ] **Step 2: Rewrite imports in `langgraph/graph/graph.go`**

Replace the import block

```go
import (
	"context"
	"fmt"
	"sync"

	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/channels"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/checkpoint"
)
```

with

```go
import (
	"context"
	"fmt"
	"sync"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)
```

Then replace every `agentruntime.` identifier prefix with `types.` throughout `graph.go` (occurrences include `agentruntime.START`, `agentruntime.END`, `agentruntime.Command`, `agentruntime.Send`, `agentruntime.Interrupt`, `agentruntime.GraphInterrupt`, `agentruntime.ParentGraph` in doc comments).

Also update the package doc comment at the top of `graph.go`: the sentence "sufficient to run the fixed \"model node <-> tools node\" shape `langchain`'s v1 agent factory needs" stays, but add one sentence after the first paragraph:

```go
// Package graph implements a deliberately scoped Go port of Python's
// `langgraph.graph.StateGraph` builder plus a synchronous, in-process
// Pregel-style executor (see `langgraph.pregel`), sufficient to run the fixed
// "model node <-> tools node" shape `langchain`'s v1 agent factory needs.
// It is the public home of this runtime; `langchain/internal/agentruntime`
// is now a thin alias layer over the `langgraph` packages.
```

- [ ] **Step 3: Rewrite the stale path reference in `langgraph/graph/events.go`**

In `langgraph/graph/events.go`, replace the doc-comment sentence

```go
// of langchain/agents (the graph lives under internal/agentruntime and must not
// reach back up to the public agents package): instead of a callback taking an
```

with

```go
// of langchain/agents (the graph lives in the public langgraph module and must
// not reach back up to the agents package): instead of a callback taking an
```

`events.go` has no agentruntime imports; verify with `grep -n "agentruntime" langgraph/graph/events.go` (expected: no matches).

- [ ] **Step 4: Rewrite imports in `langgraph/graph/graph_test.go`**

Replace the import block entries

```go
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/channels"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/checkpoint"
```

with

```go
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
```

and replace every `agentruntime.` prefix with `types.` throughout the file. This file is `package graph` (internal test); its references to `NewStateGraph`, `Interrupt`, etc. stay unqualified.

- [ ] **Step 5: Rewrite imports in `langgraph/graph/integration_test.go`**

Replace the import block entries

```go
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/channels"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/graph"
```

with

```go
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/types"
```

and replace every `agentruntime.` prefix with `types.` throughout the file (`graph.` and `channels.` prefixes stay valid). The `langchain/tools` import is unchanged — `langchain/tools` does not import `langgraph`, so there is no import cycle.

- [ ] **Step 6: Build and test**

```bash
go build ./... && go test ./langgraph/... ./langchain/...
```

Expected: PASS, including the copied `langgraph/graph` unit tests and `TestAgentLoopWithRealToolNode`.

- [ ] **Step 7: Commit**

```bash
git add langgraph/graph/
git commit -m "feat(langgraph): add public graph package (StateGraph, Pregel-style executor)"
```

---

### Task 5: Convert `agentruntime` to an alias shim

**Files:**
- Modify (rewrite): `langchain/internal/agentruntime/types.go` — becomes the whole `agentruntime` package (aliases)
- Delete: `langchain/internal/agentruntime/doc.go`
- Modify (rewrite): `langchain/internal/agentruntime/channels/channels.go` — becomes aliases
- Delete: `langchain/internal/agentruntime/channels/channels_test.go`
- Modify (rewrite): `langchain/internal/agentruntime/checkpoint/checkpoint.go` — becomes aliases
- Delete: `langchain/internal/agentruntime/checkpoint/checkpoint_test.go`
- Modify (rewrite): `langchain/internal/agentruntime/graph/graph.go` — becomes aliases
- Delete: `langchain/internal/agentruntime/graph/events.go`, `graph_test.go`, `integration_test.go`

**Interfaces:**
- Consumes: everything produced by Tasks 1–4.
- Produces: unchanged public API for all four old packages — existing consumers (`langchain/agents/create_agent.go`, `state_schema.go`, `stream.go`, and their tests) must compile and pass without a single edit.

- [ ] **Step 1: Rewrite `langchain/internal/agentruntime/types.go` and delete `doc.go`**

Replace the entire content of `langchain/internal/agentruntime/types.go` with:

```go
// Package agentruntime is a thin alias layer over the public
// `github.com/projanvil/langchain-golang/langgraph` packages, kept so
// existing consumers (langchain/agents) compile unchanged after the graph
// runtime was promoted out of `internal`. New code should import the
// langgraph packages directly.
package agentruntime

import "github.com/projanvil/langchain-golang/langgraph/types"

const (
	START       = types.START
	END         = types.END
	ParentGraph = types.ParentGraph
)

type Send = types.Send
type Command = types.Command
type Interrupt = types.Interrupt
type GraphInterrupt = types.GraphInterrupt
```

```bash
rm langchain/internal/agentruntime/doc.go
```

- [ ] **Step 2: Rewrite `langchain/internal/agentruntime/channels/channels.go`, delete its test**

Replace the entire content of `langchain/internal/agentruntime/channels/channels.go` with:

```go
// Package channels is a thin alias layer over
// `github.com/projanvil/langchain-golang/langgraph/channels`; see that
// package for documentation. New code should import langgraph/channels
// directly.
package channels

import "github.com/projanvil/langchain-golang/langgraph/channels"

type Reducer = channels.Reducer

var (
	LastValueReducer   = channels.LastValueReducer
	AppendSliceReducer = channels.AppendSliceReducer
	MessagesReducer    = channels.MessagesReducer
)
```

```bash
rm langchain/internal/agentruntime/channels/channels_test.go
```

- [ ] **Step 3: Rewrite `langchain/internal/agentruntime/checkpoint/checkpoint.go`, delete its test**

Replace the entire content of `langchain/internal/agentruntime/checkpoint/checkpoint.go` with:

```go
// Package checkpoint is a thin alias layer over
// `github.com/projanvil/langchain-golang/langgraph/checkpoint`; see that
// package for documentation. New code should import langgraph/checkpoint
// directly.
package checkpoint

import "github.com/projanvil/langchain-golang/langgraph/checkpoint"

type Checkpoint = checkpoint.Checkpoint
type Saver = checkpoint.Saver
type MemorySaver = checkpoint.MemorySaver

var NewMemorySaver = checkpoint.NewMemorySaver
```

```bash
rm langchain/internal/agentruntime/checkpoint/checkpoint_test.go
```

- [ ] **Step 4: Rewrite `langchain/internal/agentruntime/graph/graph.go`, delete the other graph files**

Replace the entire content of `langchain/internal/agentruntime/graph/graph.go` with:

```go
// Package graph is a thin alias layer over
// `github.com/projanvil/langchain-golang/langgraph/graph`; see that package
// for documentation. New code should import langgraph/graph directly.
package graph

import lcgraph "github.com/projanvil/langchain-golang/langgraph/graph"

type NodeFunc = lcgraph.NodeFunc
type ConditionalEdge = lcgraph.ConditionalEdge
type StateGraph = lcgraph.StateGraph
type CompileOption = lcgraph.CompileOption
type CompiledGraph = lcgraph.CompiledGraph
type Options = lcgraph.Options
type Result = lcgraph.Result
type RawEventKind = lcgraph.RawEventKind
type RawEvent = lcgraph.RawEvent
type NodeEventSink = lcgraph.NodeEventSink

const (
	RawNodeStart = lcgraph.RawNodeStart
	RawNodeEnd   = lcgraph.RawNodeEnd
)

var (
	To                   = lcgraph.To
	NewStateGraph        = lcgraph.NewStateGraph
	WithCheckpointer     = lcgraph.WithCheckpointer
	WithRecursionLimit   = lcgraph.WithRecursionLimit
	WithInterruptBefore  = lcgraph.WithInterruptBefore
	WithInterruptAfter   = lcgraph.WithInterruptAfter
	Interrupt            = lcgraph.Interrupt
	ContextWithEventSink = lcgraph.ContextWithEventSink
	EventSinkFromContext = lcgraph.EventSinkFromContext
)
```

```bash
rm langchain/internal/agentruntime/graph/events.go langchain/internal/agentruntime/graph/graph_test.go langchain/internal/agentruntime/graph/integration_test.go
```

- [ ] **Step 5: Build, vet, and run the full test suite**

```bash
go build ./... && go vet ./... && go test ./...
```

Expected: PASS across the whole module with zero changes to `langchain/agents` — this proves the alias shim is transparent. Pay special attention that `langchain/agents` tests (create_agent, state_schema, stream) all pass.

- [ ] **Step 6: Commit**

```bash
git add langchain/internal/agentruntime/
git commit -m "refactor(agentruntime): replace implementation with aliases over public langgraph packages"
```

---

### Task 6: Documentation and final verification

**Files:**
- Modify: `langchain_golang/README.md` (project layout / package listing section)
- Modify: `langchain_golang/docs/superpowers/specs/2026-08-07-langgraph-go-port-design.md` (mark M1 done)

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: up-to-date docs; no code changes.

- [ ] **Step 1: Update README**

Find the section of `README.md` that lists the repository's top-level Go packages (it currently mentions `langgraph` in prose). Add `langgraph/` to the layout listing with a one-line description, e.g.:

```markdown
- `langgraph/` — Go port of Python's LangGraph: StateGraph builder, Pregel-style executor, channels/reducers, and checkpointing (M1 scope; see docs/superpowers/specs/2026-08-07-langgraph-go-port-design.md).
```

Match the exact formatting of the surrounding entries. If the README has no such listing, add the bullet to the most relevant existing section instead of creating a new section.

- [ ] **Step 2: Mark M1 complete in the spec**

In `docs/superpowers/specs/2026-08-07-langgraph-go-port-design.md`, change the M1 heading from `### M1 核心平移` to `### M1 核心平移（已完成 2026-08-08）`.

- [ ] **Step 3: Final full verification**

```bash
go build ./... && go vet ./... && go test ./...
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/specs/2026-08-07-langgraph-go-port-design.md
git commit -m "docs: document public langgraph module and mark M1 complete"
```

---

## Self-Review Notes

- Spec coverage: M1 scope items from the spec (public package skeleton, move of builder/executor/Command/Send/Interrupt/memory checkpoint/event sink, agentruntime as delegating shim, create_agent green) map to Tasks 1–5; docs map to Task 6. Out-of-scope items (subgraphs, stream modes, time travel, persistent backends, functional API) are correctly untouched.
- Type consistency: shim aliases in Task 5 reference exactly the identifiers produced by Tasks 1–4 (`types.*`, `channels.*`, `checkpoint.*`, `graph.*` public surfaces were enumerated from the current source files).
- Risk: if a consumer uses an identifier missed by the Task 5 shim, Step 5's full-module build fails loudly — fix by adding the missing alias, not by editing consumers.
