# ToolErrorMiddleware（v0.12.3）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Python `ToolErrorMiddleware`（langchain 1.4.7，libs/langchain_v1/langchain/agents/middleware/tool_error.py）to Go as a first-class agent middleware, including the ToolNode error-path layering it needs to actually observe execution errors. Release v0.12.3 (patch).

**Architecture:** One task, two files of consequence. Python mechanism (verified by runtime probe on langgraph 1.2.19 + prebuilt 1.1.1 + langchain 1.4.7, 2026-11-02): default `handle_tool_errors` converts ONLY `ToolInvocationError` (arg binding/validation) inside `_execute_tool_sync` (inside the middleware wrap chain); all other execution errors are RE-RAISED, escaping through the wrap chain where `ToolErrorMiddleware.on_error` sees them; `on_error` returning content → error ToolMessage(status="error"); returning None → the error propagates (default handler re-raises non-invocation errors → run fails); GraphBubbleUp never reaches on_error. Go today converts EVERY error via catch-all `DefaultHandleToolErrors` inside `execute()` (inside the wrap chain), so a wrap-chain middleware never sees raw errors.

**Tech Stack:** Go 1.26+; no new dependencies.

**Spec:** none (single-task patch; Python file is the spec, mechanism runtime-verified by controller).

## Global Constraints

- English commit messages, English code comments, English release changelog (user directive this round).
- Python semantics from the local checkouts only: langchain checkout `libs/langchain_v1/langchain/agents/middleware/tool_error.py`, langgraph checkout `libs/prebuilt/langgraph/prebuilt/tool_node.py` (`_run_one`/`_execute_tool_sync`/`_default_handle_tool_errors`/`_infer_handled_errors` region). Upstream tests: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_error.py` (5 tests — mirror them).
- Gates: `go build ./... && go vet ./... && go test -race ./...`; `golangci-lint run ./...` zero issues; `make fmt-check`; no coverage-set packages are touched (langchain/agents is not in COVERAGE_PKGS — verify, and if any touched package is in the set, `make coverage` ≥95%).
- TDD mandatory; conventional commit citing the upstream PR (self-verify via git log in the Python checkout); DIVERGENCES updated in the same commit; no public API breakage (additive only).
- A working Python probe venv exists at `langchain/libs/langchain_v1/.venv` (langsmith upgraded to 0.14.17, pytest/blockbuster/langchain-tests/pytest-mock installed, langgraph bumped to 1.2.19 + prebuilt 1.1.1 + checkpoint 4.3.1) — use it to run upstream tests or ad-hoc probes; do not "clean it up".

## Controller rulings (binding, adjudicated 2026-11-02)

1. **Ruling — default stays catch-all.** Go `DefaultHandleToolErrors` keeps its catch-all convert-everything semantics this round (zero breaking; Python 1.1.1+'s "only ToolInvocationError converts, rest propagate" default change is recorded as a DIVERGENCES entry instead). `ToolErrorMiddleware` is the explicit opt-in path to Python's new posture.
2. **Ruling — layering move.** `handleToolErrors` application moves from `ToolNode.execute()` (inside the wrap chain) to `runOne()` (outside the wrap chain), mirroring Python `_run_one`. With no wrap chain configured the observable behavior must be byte-identical to today (conversion still happens, just one frame later); with a wrap chain, middleware now observe raw errors (Python shape). Existing wrap middleware (HITL, tool_retry, etc.) and all their tests must not regress.
3. **Ruling — None-means-propagate sentinel.** When `OnError` declines (returns not-handled), the middleware returns the error wrapped in a new exported sentinel type (e.g. `tools.UnhandledToolError`) that the runOne-level conversion check unwraps and re-raises WITHOUT conversion — so declined errors fail the run like Python's None path, bypassing the catch-all default.
4. **Ruling — binding errors.** Go cannot uniformly distinguish argument-binding errors from execution errors (no shared binding layer; both surface as errors from `Tool.Invoke`). The middleware therefore observes both (Python's on_error sees only execution errors). Record as a DIVERGENCES note (language-forced) unless the implementer finds Go's tools package does expose a uniform binding-error marker — verify before recording.

## Task 1: ToolErrorMiddleware + ToolNode layering

**Files:**
- Create: `langchain/agents/middleware/tool_error.go` (+ test file next to the other middleware tests)
- Modify: `langchain/tools/tool_node.go` (move handleToolErrors application to runOne level; add the UnhandledToolError unwrap), test files for ToolNode error paths
- Modify: `langchain/agents/middleware/__init__`-equivalent export surface (where middleware are re-exported, if the package has one)
- Modify: `DIVERGENCES.md` (two entries: default handle_tool_errors semantics divergence; binding-vs-execution observability note)

**Behavior spec (mirror upstream tests `test_tool_error.py`):**
- [ ] Step 1 RED tests: (a) through create_agent, a failing tool's error reaches OnError; returned content becomes exactly one ToolMessage with status error (ResponseMetadata{"status":"error"}), name=tool name, content=the returned content (string or content blocks); raw error message not leaked when OnError's content omits it. (b) OnError declining → the ORIGINAL error propagates out of agent Invoke (not converted). (c) tool filter: non-matching tools' errors bypass the middleware entirely. (d) constructor validation: missing OnError → error. (e) control-flow errors (interrupt) never reach OnError and propagate. (f) no-middleware default behavior unchanged (error converted, run succeeds) — regression guard for the layering move. (g) existing wrap middleware (HITL/tool_retry) tests green.
- [ ] Step 2 implement + gates + commit `feat(agents): port ToolErrorMiddleware (#<self-verified PR>)` with the layering move and DIVERGENCES entries in the same commit.
