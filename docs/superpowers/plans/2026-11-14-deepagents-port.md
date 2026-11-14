# deepagents Go Port (MVP Tier 1-4) Implementation Plan — v2 (post-audit)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port deepagents v0.7.38 MVP (backends + FilesystemMiddleware + SubAgentMiddleware + CreateDeepAgent assembler) as a top-level `deepagents/` package in this repo, reusing existing agents/langgraph facilities.

**Architecture:** Assembler over `langchain/agents.CreateAgent` — every capability is middleware on the existing hook surface (ToolProvider, StateSchemaContributor, WrapToolCallHook, HITL interrupt); the only new abstractions are the Backend (virtual FS) and the middleware assembly. Upstream is the authority for behavior; this plan pins Go signatures.

**Tech Stack:** Go 1.26, existing module `github.com/projanvil/langchain-golang`. No new external deps.

**Spec:** `/Users/yuhaochen/Documents/codebase/projanvil/langchain_port/langchain_migration_plan/deepagents-port-spec.md` (v3 — audited twice; plan-audit findings E1-E13 merged). Upstream source (read-only reference): `/Users/yuhaochen/Documents/codebase/projanvil/langchain_port/deepagents/libs/deepagents/deepagents/` (cited as UP:...).

## Global Constraints

- Import direction (hard, plan-audit E1/C1): `deepagents` root → `deepagents/middleware` → `deepagents/backends` → (core, langgraph). No back-edge. No root `deepagents/state.go`.
- Upstream pin: deepagents v0.7.38. Where behavior is unspecified here, match upstream semantics; divergences D1-D14 (spec §8) recorded in Task 9.
- No new go.mod dependencies. No changes outside `deepagents/`, `examples/deepagents/`, Makefile COVERAGE_PKGS, DIVERGENCES.md, CHANGELOG.md, docs.
- Tool errors render as tool result text, never panic/error the run (spec R3).
- Files state: `files` is a PUBLIC key (plan-audit E2/C2). Value `map[string]FileMeta`; update payloads `map[string]*FileMeta`, nil value = delete key; `MapMergeReducer` (backends pkg).
- Private state keys (hardcoded, middleware/state.go): `[]string{middleware.EditedToolCallsStateKey}` ONLY (plan-audit E2/I5). `files` and `todos` are public.
- A tool's `*types.Command` in `tools.Result.Artifact` must NEVER contain a `messages` key — the ToolNode derives the ToolMessage from Result.Content (audit I1; convention in `langchain/agents/middleware/todo.go:68-75`).
- Command = `langgraph/types.Command`. Tool arg names/result texts: upstream literals ONLY (no invented strings — read UP source at implementation).
- Coverage ≥95% per package; `deepagents/...` added to Makefile COVERAGE_PKGS (Task 9).
- Go doc comments cite upstream file:line (repo convention). Commit style: `feat(deepagents): ...` / `test(deepagents): ...`.

## Review Focus

1. **Same-batch `files` collisions** (C2/E4): write+write in one model batch → `_parallel_file_mutation_error` error result (ported); `task`+write in one batch → last-write-wins per key fold at `applyToolNodeCommands` (create_agent.go:3403-3428, D13). Tests: Task 3 reducer table + Task 8 fold-level test.
2. **Backend error → tool text, run survives** — missing file/bad path returns a ToolMessage with the upstream error literal, not a failed invoke. Test: Task 5 error-path table.
3. **No duplicate ToolMessage from task Command** (I1) — task Command Update contains state keys only. Test: Task 8 asserts Command keys ≠ messages.
4. **Path normalization** (E8): relative paths normalized to `/`-rooted (UP utils.py:846-911 `validate_path`), `..` resolved, escape-past-root rejected. Test: Task 1 normalize table.
5. **Grep 15s / glob 10s timeout → upstream timeout text** (E6), not a hang. Test: Task 5 timeout with 50ms injected deadline.

---

### Task 1: backends matchers and path utils

**Files:**
- Create: `deepagents/backends/match.go`, `deepagents/backends/utils.go`
- Test: `deepagents/backends/match_test.go`, `deepagents/backends/utils_test.go`

**Interfaces:**
- Produces:
  - `func MatchGlob(pattern, name string) bool` — `*`, `?`, `**` (spans `/`), `[...]`; no brace expansion (D7). Full-path match.
  - `func GrepLines(content, pattern string, maxCount int) []GrepLine`; `type GrepLine struct { LineNumber int; Text string }` — literal substring per line (NOT regex); `maxCount<=0` → unlimited.
  - `func NormalizePath(path string) (string, error)` — E8: mirror UP:backends/utils.py:846-911 `validate_path`: clean relative → `/`-rooted, resolve `.`/`..` segments, reject paths escaping root with the upstream `invalid_path` error literal (constants UP:backends/protocol.py:39-44).
  - `func NormalizeReadBounds(offset, limit int) (start, count int)` — offset<0→0; limit<=0→0 (empty window).

- [ ] **Step 1: Failing table tests**: MatchGlob (`a/*.go` vs `a/b.go` true / vs `a/b/c.go` false, `a/**/*.go` vs `a/b/c.go` true, `?`, `[abc]`, `a.{x,y}` literal no-brace); GrepLines (literal: pattern `a.c` does NOT match `abc`, matches `xa.cy`); NormalizePath (`x/y`→`/x/y`, `/a/../b`→`/b`, `a/../../x`→error, `//a`→`/a`); NormalizeReadBounds.
- [ ] **Step 2: Run** `go test ./deepagents/backends/ -v` — FAIL (undefined).
- [ ] **Step 3: Implement.** MatchGlob: split on `/`, recursive `**` segment matching; single-segment wildcards via `path.Match` per segment.
- [ ] **Step 4: Run — PASS** + `go vet ./deepagents/...`.
- [ ] **Step 5: Commit** `feat(deepagents): glob/grep matchers and path utils`

### Task 2: Backend interface + FilesystemBackend

**Files:**
- Create: `deepagents/backends/backend.go`, `deepagents/backends/filesystem.go`, `deepagents/backends/doc.go`
- Test: `deepagents/backends/filesystem_test.go`

**Interfaces:**
- Consumes: Task 1.
- Produces (spec §4):
  - `type FileMeta struct { Content string; Encoding string; CreatedAt, ModifiedAt time.Time }` (`"utf-8"`/`"base64"`)
  - `type PermissionMode string` (`allow`/`deny`/`interrupt`) + `type PermissionRule struct { Operations []string; Paths []string; Mode PermissionMode }` (E12 — in backends package)
  - Result types, UP field names: `LsResult{Entries []LsEntry}`, `LsEntry{Name string; Type string; ModifiedAt time.Time}`, `ReadResult{Content string; StartLine, EndLine, TotalLines int; NextOffset int}`, `WriteResult{Path string}`, `EditResult{Replaced int}`, `DeleteResult{Success bool}`, `GlobResult{Files []string}`, `GrepResult{Matches []GrepFileMatches}`, `GrepFileMatches{Path string; Lines []GrepLine}`
  - `type Backend interface` — methods per spec §4, all `(ctx context.Context, ...) (XResult, error)`: `Ls(ctx, path)`, `Read(ctx, path, offset, limit)`, `Write(ctx, path, content)`, `Edit(ctx, path, oldString, newString string, replaceAll bool)`, `Glob(ctx, path, pattern)`, `Grep(ctx, pattern, path, glob, maxCount)`.
  - `type Deleter interface { Delete(ctx context.Context, path string) (DeleteResult, error) }`
  - `func NewFilesystemBackend(rootDir string) (*FilesystemBackend, error)` — root must exist; implements Backend+Deleter. `/x/y`→`filepath.Join(root,"x","y")`; Write mkdirs parents; Edit absent-string error text `"String to replace not found in file"`-family per UP:backends/filesystem.py (read exact literals at implementation).
- [ ] **Step 1: Failing tests** (`t.TempDir()`): write→read pagination (StartLine=1, NextOffset), nested ls sorted, edit absent error, edit multi-without-replaceAll error, Delete via Deleter, glob `**`, grep literal+maxCount, NormalizePath escape error, missing file error (`file_not_found` semantics). Encoding "utf-8".
- [ ] **Step 2: Run — FAIL.** **Step 3: Implement.** **Step 4: Run — PASS** + vet. **Step 5: Commit** `feat(deepagents): Backend interface and FilesystemBackend`

### Task 3: files state plumbing + StateBackend (all in backends pkg — E1)

**Files:**
- Create: `deepagents/backends/state.go`
- Test: `deepagents/backends/state_test.go`

**Interfaces:**
- Consumes: Task 2 FileMeta/Backend.
- Produces (all exported from `backends`):
  - `const FilesStateKey = "files"`
  - `func FilesStateField() middleware.StateField` → `{Name: "files", Reducer: MapMergeReducer}` (middleware = `langchain/agents/middleware` — import is legal: backends → middleware? NO — import direction is middleware → backends. RESOLUTION: `FilesStateField` returns a locally-defined struct? No — middleware.StateField is the collected type. Ruling: backends may NOT import langchain/agents/middleware for this (direction). Instead `MapMergeReducer` is `channels.Reducer`-typed (langgraph/channels — legal for backends), and `FilesStateField` is DELETED; the FilesystemMiddleware (middleware pkg, Task 4) constructs `middleware.StateField{Name: FilesStateKey, Reducer: backends.MapMergeReducer}` itself.
  - `func MapMergeReducer(existing, update any) (any, error)` — typed as `channels.Reducer`; existing/update `map[string]*FileMeta` (nil=delete) or nil; result merged `map[string]FileMeta` copy (C2/UP:middleware/filesystem.py:1324-1377); non-map → typed error.
  - `type StateView struct{...}` with `Files() map[string]FileMeta; Set(path string, meta FileMeta); Delete(path string); Changed() map[string]*FileMeta`
  - `func WithStateView(ctx context.Context, v *StateView) context.Context` / `func StateViewFromContext(ctx context.Context) (*StateView, bool)`
  - `func NewStateBackend() *StateBackend` — Backend+Deleter over ctx StateView; method without view → error `"no state view in context"`. Grep/Glob in-memory over sorted view paths (UP:backends/state.py semantics).
- [ ] **Step 1: Failing tests**: MapMergeReducer table (merge disjoint; nil-value delete; nil existing; delete+write disjoint keys both applied); StateBackend ctx roundtrip (write→read→ls→grep→delete); no-view error.
- [ ] **Step 2: Run — FAIL.** **Step 3: Implement.** **Step 4: Run — PASS** + vet. **Step 5: Commit** `feat(deepagents): files state reducer and StateBackend`

### Task 4: FilesystemMiddleware (ctx injection + permission evaluator)

**Files:**
- Create: `deepagents/middleware/filesystem.go`, `deepagents/middleware/state.go`, `deepagents/middleware/doc.go`
- Test: `deepagents/middleware/filesystem_test.go`, `deepagents/middleware/state_test.go`

**Interfaces:**
- Consumes: Tasks 2-3; `agents.WrapToolCallHook`: `WrapToolCall(ctx context.Context, request middleware.ToolCallRequest, handler middleware.ToolHandler) (messages.Message, error)` (create_agent.go:213-214).
- Produces:
  - `middleware/state.go` (E1/E2): `var ExcludedStateKeys = []string{...}` — copy UP:middleware/subagents.py `_EXCLUDED_STATE_KEYS` exactly (read at implementation); `var PrivateStateKeys = []string{middleware.EditedToolCallsStateKey}` (langchain/agents/middleware/human_in_the_loop.go:73 — verify constant name at implementation).
  - `type FilesystemMiddleware struct { Backend backends.Backend; Permissions []backends.PermissionRule; Tools []coretools.Tool }` (`coretools` = `core/tools`); `func NewFilesystemMiddleware(backend backends.Backend, permissions ...backends.PermissionRule) (*FilesystemMiddleware, error)` (tools filled Task 5).
  - `ProvidedTools() []coretools.Tool`; `StateSchema() []middleware.StateField` — `{Name: backends.FilesStateKey, Reducer: backends.MapMergeReducer}` when backend is `*backends.StateBackend`.
  - `WrapToolCall`: StateBackend → snapshot `request.State[backends.FilesStateKey]` into fresh StateView, `ctx = backends.WithStateView(ctx, view)`, call handler (tools return their own Artifact Command in Task 5; middleware only injects). Export helper `func filesCommand(v *backends.StateView) *types.Command` → `Update: {"files": v.Changed()}` (never a `messages` key).
  - Permission evaluator `func (m *FilesystemMiddleware) Check(tool string, args map[string]any) (backends.PermissionMode, bool)` (mode, matched) — first-match-wins, default allow; op classes read={ls,read_file,glob,grep} write={write_file,edit_file,delete}; path arg per tool mirrors UP `_FS_TOOL_PATH_ARGS` (all seven tools carry a path arg — E7; read UP source for the exact mapping incl. grep/glob `path`).
- [ ] **Step 1: Failing tests**: evaluator table (ops×paths×first-match×default-allow, read-op deny, write-op deny, from UP:middleware/filesystem.py:376-429 cases); WrapToolCall view injection (handler asserts StateViewFromContext); filesCommand shape (Update keys == {"files"}); state.go key lists content.
- [ ] **Step 2: Run — FAIL.** **Step 3: Implement.** **Step 4: Run — PASS** + vet. **Step 5: Commit** `feat(deepagents): FilesystemMiddleware state injection and permission evaluation`

### Task 5: the seven file tools

**Files:**
- Create: `deepagents/middleware/filetools.go`
- Modify: `deepagents/middleware/filesystem.go` (NewFilesystemMiddleware constructs tools)
- Test: `deepagents/middleware/filetools_test.go`

**Interfaces:**
- Consumes: Tasks 1-4; `coretools.NewFunc(name, description, schema, fn)`; `coretools.Result{..., Artifact any}`.
- Produces: tools `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep` (`delete` only for Deleter backends). Schemas — E3 (read UP:middleware/filesystem.py tool defs; arg names pinned here): ls{path}, read_file{file_path, offset=0, limit=100}, write_file{file_path, content}, edit_file{file_path, old_string, new_string, replace_all=false}, delete{path}, glob{path, pattern}, grep{pattern, path, glob, max_count, output_mode ∈ content|files_with_matches|count, default files_with_matches} (E6). All result strings = UP literals (E5, read at implementation): write→`Updated file {path}`; read header `@@ lines {a}-{b} of {t} | next offset {n} @@`; deny→`Error: permission denied for {operation} on {path}`; edit miss→`Error: String not found in file: '{old_string}'`; ls empty→`No files found`.
  - Deny gate before backend call. Interrupt mode → behave as allow until Task 6 wires HITL (placeholder; Task 6 updates these tests — E7/M6).
  - Timeouts (E6): grep 15s, glob 10s ctx deadlines; expiry → Result{Content: UP timeout text}.
  - Writes return Artifact `filesCommand(view)` when StateBackend view changed.
- [ ] **Step 1: Failing table tests per tool** (StateBackend + FilesystemBackend tmpdir): happy-path text literals, error-path text (missing file → `(Result, nil)` with error text — Review Focus #2), timeout (50ms deadline, fake blocking backend — Review Focus #5), deny gate, Artifact on writes only, output_mode variants.
- [ ] **Step 2: Run — FAIL.** **Step 3: Implement filetools.go; wire into NewFilesystemMiddleware.** **Step 4: Run — PASS** + vet. **Step 5: Commit** `feat(deepagents): seven filesystem tools`

### Task 6: permissions → HITL interrupt wiring

**Files:**
- Create: `deepagents/middleware/permissions.go`
- Test: `deepagents/middleware/permissions_test.go`

**Interfaces:**
- Consumes: existing `middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig, ...)` (langchain/agents/middleware/human_in_the_loop.go:90); Tasks 4-5.
- Produces:
  - `func InterruptConfigsFromPermissions(perms []backends.PermissionRule) map[string]middleware.InterruptConfig` — interrupt rules → per-tool entries for ALL seven tools in the matched op class (E7; UP:middleware/_fs_interrupt.py `_build_interrupt_on_from_permissions`); first interrupt rule per tool wins; when-predicate scoping not ported (D11).
  - Interrupt-mode tool behavior: the installed HITL middleware wraps the tool call and short-circuits until approved (no external "decision record" — plan-audit M2); Task 5's placeholder tests updated: interrupt-mode call pauses the run via interrupt, resume-approved proceeds, resume-denied → deny text.
- [ ] **Step 1: Failing tests**: mapping table (write-op rule → write_file/edit_file/delete entries; read-op rule → ls/read_file/glob/grep entries); e2e via CreateAgent with scripted fake model (repo fake ChatModel convention): write_file interrupt → `InvokeWithStateOptions` returns Interrupts non-empty; Resume approved → file written; Resume denied → `Error: permission denied...` ToolMessage.
- [ ] **Step 2: Run — FAIL.** **Step 3: Implement; update Task 5 placeholder tests.** **Step 4: Run — PASS** + vet. **Step 5: Commit** `feat(deepagents): permission-derived HITL interrupt wiring`

### Task 7: SubAgent specs + compilation

**Files:**
- Create: `deepagents/middleware/subagent.go`
- Test: `deepagents/middleware/subagent_test.go`

**Interfaces:**
- Consumes: `agents.CreateAgent(model language.ChatModel, tools []coretools.Tool, opts ...agents.AgentOption)`; `agents.WithAgentMiddleware(mw ...any)`; existing ModelRefusalMiddleware (constructor name: read langchain/agents/middleware/model_refusal.go at implementation); Task 2-3 backends.
- Produces:
  - `type SubAgentSpec struct { Name, Description, Prompt string; Model language.ChatModel; Tools []coretools.Tool; Middleware []any; Backend backends.Backend }`
  - `type CompiledSubAgent struct { Name, Description string; Agent *agents.Agent }` (agents = `langchain/agents`)
  - `func CompileSubAgent(spec SubAgentSpec) (*CompiledSubAgent, error)` — CreateAgent with spec.Prompt as system prompt, ModelRefusalMiddleware installed (audit I2), own FilesystemMiddleware when Backend != nil.
  - `func GeneralPurposeSubAgent(model language.ChatModel) SubAgentSpec` — copy description/prompt from UP:middleware/subagents.py:440-445 and :564-570 (E11).
  - `func ValidateSubAgents(specs []any) error` — SubAgentSpec/*CompiledSubAgent mix; unique non-empty names (UP `_validate_unique_subagent_names`).
  - `func BuildTaskInput(parentState map[string]any, description string) map[string]any` — `messages: [HumanMessage(description)]` + every parent key NOT in ExcludedStateKeys ∪ PrivateStateKeys (E2: `files` PASSES THROUGH — public).
- [ ] **Step 1: Failing tests**: ValidateSubAgents (dup/empty/mix); CompileSubAgent success + refusal middleware present (integration-visible in Task 8); BuildTaskInput: `messages` excluded (rebuilt), `hitl_edited_tool_calls` excluded (PrivateStateKeys), `files` KEPT, custom keys kept.
- [ ] **Step 2: Run — FAIL.** **Step 3: Implement.** **Step 4: Run — PASS** + vet. **Step 5: Commit** `feat(deepagents): SubAgent specs and compilation`

### Task 8: task tool + SubAgentMiddleware

**Files:**
- Create: `deepagents/middleware/subagents.go`
- Test: `deepagents/middleware/subagents_test.go`

**Interfaces:**
- Consumes: Task 7; `agents.Agent.InvokeWithStateOptions(ctx, msgs, graphpkg.Options)` (create_agent.go:1036); `types.Command{Update map[string]any}`.
- Produces:
  - `type SubAgentMiddleware struct { SubAgents []*CompiledSubAgent; Tools []coretools.Tool }` + `func NewSubAgentMiddleware(model language.ChatModel, subagents []any) (*SubAgentMiddleware, error)` — compiles specs; auto-adds GeneralPurposeSubAgent when no `general-purpose` name.
  - task tool (`coretools.NewFunc`, name `task`): schema `{subagent_type: string, description: string}` (E3), unknown keys rejected in fn. Flow (spec §6): select by name (unknown → UP KeyError-style error text, read UP at implementation — E11/M5); input via BuildTaskInput; `InvokeWithStateOptions` with `graphpkg.Options{ThreadID: "task:" + toolCallID}` (E10) and NO checkpointer; content = last non-empty AI text (refusal notice when final AI message has refusal content); **Artifact Command Update = subagent result state minus ExcludedStateKeys ∪ PrivateStateKeys ∪ unchanged-value keys (UP `_is_unchanged`); MUST NOT contain `messages`** (Review Focus #3). `files` passes through (E2): a subagent sharing StateBackend writes files back to the parent.
  - `ProvidedTools()` = [task].
- [ ] **Step 1: Failing tests** (stub subagent = tiny CreateAgent, fake scripted model, one Artifact-Command tool writing a custom state key + one writing a file via StateBackend view): (a) task returns subagent final text as ToolMessage; (b) parent state gains subagent custom key; (c) Command Update keys exclude messages/excluded/private; unchanged-value filtered; (d) **subagent file writeback lands in parent `files`** (E2); (e) unknown agent error text; (f) ThreadID derivation `task:<id>`; (g) fold test (Review Focus #1): same-batch task + write_file Commands on `files` fold last-write-wins without error; same-batch write_file+edit_file both touching files → `_parallel_file_mutation_error`-style error result (ported literal).
- [ ] **Step 2: Run — FAIL.** **Step 3: Implement.** **Step 4: Run — PASS** + vet. **Step 5: Commit** `feat(deepagents): task tool and SubAgentMiddleware`

### Task 9: CreateDeepAgent assembler + example + release prep

**Files:**
- Create: `deepagents/agent.go`, `deepagents/doc.go`, `examples/deepagents/main.go`
- Modify: `Makefile` (COVERAGE_PKGS += `deepagents/...`), `DIVERGENCES.md` (D1-D14 per spec §8), `CHANGELOG.md`, `docs/README.md`
- Test: `deepagents/agent_test.go`

**Interfaces:**
- Consumes: Tasks 4-8; `agents.WithAgentSystemPrompt/WithAgentMiddleware/WithAgentCheckpointer/WithAgentStore/WithAgentRecursionLimit`; `middleware.NewSummarizationMiddleware(...)` requiring `middleware.SummarizerFunc` (summarization.go:94 — E9).
- Produces:
  - `type DeepAgentOptions struct { SystemPrompt string; Tools []coretools.Tool; Middleware []any; SubAgents []any; Backend backends.Backend; Permissions []backends.PermissionRule; InterruptOn map[string]middleware.InterruptConfig; Checkpointer checkpoint.Saver; Store store.Store; Summarizer middleware.SummarizerFunc }` + `DeepAgentOption` constructors `WithSystemPrompt/WithTools/WithMiddleware/WithSubAgents/WithBackend/WithPermissions/WithInterruptOn/WithCheckpointer/WithStore/WithSummarizer` (E9).
  - `func CreateDeepAgent(model language.ChatModel, opts ...DeepAgentOption) (*agents.Agent, error)` — stack: FilesystemMiddleware(backend=opts.Backend or NewStateBackend(), perms) → SubAgentMiddleware(model, opts.SubAgents) → SummarizationMiddleware (Summarizer = opts.Summarizer, nil → default constructed from model via a single-summarize-call func) → ModelRefusalMiddleware → HITL when `InterruptConfigsFromPermissions(perms)` ∪ opts.InterruptOn non-empty (user entries win per tool) → opts.Middleware → CreateAgent(model, tools, WithAgentSystemPrompt(opts.SystemPrompt), WithAgentMiddleware(stack...), **WithAgentRecursionLimit(9999)** (E13), checkpointer/store passthrough). No default prompt (I5), no execute tool (D5).
- [ ] **Step 1: Failing e2e tests** (fake scripted model): (a) bound tools include the 7 file tools + task; (b) write_file→read_file roundtrip via InvokeWithState with StateBackend; (c) permission interrupt pause/resume; (d) delegation: model calls task → subagent fake model calls write_file on its own FilesystemMiddleware over the SHARED StateBackend → parent state `files` gains the entry (E2 end-to-end).
- [ ] **Step 2: Run — FAIL.** **Step 3: Implement agent.go + example** (StateBackend + one declarative subagent; model via openaicompat guarded by env var, skip note when unset). **Step 4: Full gates**: `go build ./... && go test ./... && golangci-lint run && make coverage` (≥95% incl. deepagents). **Step 5: Commit** `feat(deepagents): CreateDeepAgent assembler, example, docs`

---

## Self-review notes

- Spec coverage: §4→Tasks 1-3, §5→Tasks 4-6, §6→Tasks 7-8, §7→Task 9, §8→Task 9 (D1-D14), §9 gates→Task 9.
- E1 ruling recorded in Task 3 Interfaces: `FilesStateField` was dropped; the middleware package constructs the StateField from `backends.FilesStateKey` + `backends.MapMergeReducer`, keeping the backends→(no-middleware) import direction.
- All plan-audit edits E1-E13 landed: E1 layout (Tasks 3-4 files), E2 files-public + EditedToolCallsStateKey (Tasks 7-9), E3 arg names (Tasks 5, 8), E4 collision story (Task 8 tests g), E5 literals (Task 5), E6 timeouts+output_mode (Tasks 5), E7 all-seven interrupt (Tasks 4, 6), E8 normalize (Task 1), E9 Summarizer (Task 9), E10 ThreadID (Task 8), E11 citations (Tasks 7, 8), E12 PermissionRule home (Task 2), E13 recursion+D14 (Task 9, spec §8).
