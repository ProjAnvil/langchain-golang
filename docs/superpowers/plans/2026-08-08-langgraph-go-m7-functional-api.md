# LangGraph Go Port M7: Functional API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the functional API (Python `langgraph.func` 的 `@entrypoint` / `@task` 对标物） as a new package `langgraph/fn`: generic `Task[I,O]` / `Future[O]` / `Entrypoint[I,O,S]` with checkpoint-backed task-result replay, task retry/cache/timeout policies, `previous` state injection, and `Final[O,S]` value/save decoupling — all built as a thin wrapper over the existing single-node `graph.StateGraph` + `checkpoint.Saver` machinery, no new execution model. 前置修复一个执行器层 blocker：单节点内多次顺序 `Interrupt` 的 resume 值错位（F7，RESUME-write 机制）。

**Architecture:** `Entrypoint` compiles the user function into a one-node `graph.StateGraph` with three reserved channels (`__start__`=Ephemeral, `__end__`/`__previous__`=LastValue, Python 对齐 `func/__init__.py:576-609`)。A per-run `dispatcher`（由 fn 包装层持有、经 `context.WithValue` 注入，模式同 `graph.Interrupt` 的 ctx 注入，`graph/graph.go:1127-1128`）drives `Task.Call`: goroutine 立即执行、确定性 task ID（fnv-1a over cpID/ns/step/name/parentPath/callIdx，扩展 `graph/taskid.go`）、per-run call counter 重放、retry/cache 复用 `graph.RetryPolicy`/`graph.CachePolicy`+`checkpoint.Cache`。结果持久化闭环在 fn 层：运行开始 `GetTuple` 载入 pending writes 供重放回填；运行返回后按钉死时序（cancel → seal → 快照 → `PutWrites`）把 `__return__`/`__error__` 结果追加到最新 checkpoint（cpID/step 在追加时重盖章，保证跨暂停链一致）。resume 值对齐由 graph 包的 RESUME-write 机制保证（暂停时持久化已消费前缀 `__resume__` writes，resume 时重建完整有序队列）。

**Tech Stack:** Go 1.23，module `github.com/projanvil/langchain-golang`，根 module 零第三方依赖（fn 包只 import 标准库 + 本仓库 `langgraph/graph` / `langgraph/checkpoint` / `langgraph/channels` / `langgraph/types`）。

## Global Constraints

- 所有命令的工作目录：`langchain_golang/`（仓库根 module）。嵌套 sqlite module（Task 2 触碰）的命令在 `langgraph/checkpoint/sqlite/` 下跑，或经根 Makefile 的 `make test-sqlite`。
- 根 module 保持**零第三方依赖**（新依赖只允许出现在嵌套 module；M7 不新增任何依赖）。
- Go 1.23 下限（`iter.Seq2` 等 1.23 特性可用）。
- 底线全绿：`go build ./... && go vet ./... && go test ./...`；嵌套 module 回归 `make test-sqlite`（Task 2 改 sqlite.go 的保留 channel 映射，必须跑）。`langchain/agents` 零改动、行为零变化。
- **单元测试对标 Python**：从 `langgraph/libs/langgraph/tests/test_pregel.py`（函数式用例集中于 1269-1397、4985-5072、5486-5947、6048-6090、6269-6880 行段）语义级移植，不是象征性覆盖；每个用例注明 Python 侧出处 `文件:行号`。
- 提交信息风格仿 `git log` 的 conventional 格式：`feat(langgraph): ...` / `docs(langgraph): ...`（参考 `git log --oneline` 既有条目）。
- 注释风格：充分 doc comment、单反引号（与 M1-M4 既有代码一致）。
- 每个任务结束跑门禁：`go build ./... && go vet ./... && go test ./...`（预期：build/vet 无输出，test 全部 `ok`，无 `FAIL`）。
- 不重排无关文件；改动面仅限本计划列出的文件。

## Locked Design Decisions (binding)

### F1. fn 包公开 API（完整代码，含 doc comment；实现逐字落地）

```go
// Package fn implements the functional API of the Go langgraph port,
// mirroring Python's `langgraph.func` (`@entrypoint` / `@task`): an
// Entrypoint wraps a plain function as a checkpointed workflow, and Task
// wraps a function as an asynchronously executed, checkpoint-replayable unit
// whose result survives interrupt/resume cycles.
//
// The package is a thin wrapper over the existing graph machinery: an
// Entrypoint compiles to a single-node graph.StateGraph with three reserved
// channels (__start__/__end__/__previous__, Python parity
// `func/__init__.py:576-609`), so interrupt/resume/stream/time-travel all
// come from the existing executor. See the package documentation for the
// documented divergences (no store, timeout semantics, replayed error
// typing, StateGraph-node usage pattern).
package fn

import (
    "context"
    "iter"
    "time"

    "github.com/projanvil/langchain-golang/langgraph/checkpoint"
    "github.com/projanvil/langchain-golang/langgraph/graph"
    "github.com/projanvil/langchain-golang/langgraph/types"
)

// TaskOpts bundles the optional per-task execution policies, mirroring the
// `@task(retry_policy=..., cache_policy=..., timeout=...)` decorator
// arguments.
type TaskOpts struct {
    // Retry enables automatic retry of the task's failures (see
    // graph.RetryPolicy). Nil means never retry.
    Retry *graph.RetryPolicy
    // Cache enables result caching for the task (see graph.CachePolicy). It
    // is inert unless the enclosing Entrypoint has a checkpoint.Cache
    // backend installed (EntrypointOpts.Cache) — the same "policy without
    // backend is inert" rule as graph node caching. Only SUCCESSFUL results
    // are cached (graph parity: errored/interrupted tasks store nothing).
    Cache *graph.CachePolicy
    // Timeout caps each task attempt. Go cannot kill a goroutine, so a
    // timeout can only cancel the attempt's context and stop waiting for it
    // (the abandoned goroutine still runs to completion in the background;
    // task functions should honor their context). A timed-out attempt fails
    // with context.DeadlineExceeded, which graph.DefaultRetryOn treats as
    // retryable. Zero means no timeout.
    Timeout time.Duration
}

// Task is a named, callable unit of work, created by NewTask. The name
// replaces Python's `module.qualname` introspection: it identifies the task
// in the cache namespace (`__fn_writes/<name>`) and in deterministic task
// IDs, so it must be unique within an entrypoint's call graph.
type Task[I, O any] struct { /* unexported */ }

// NewTask wraps f as a Task. f runs in its own goroutine per Call; it must
// be safe for concurrent use if the caller holds several Futures of the same
// task at once.
func NewTask[I, O any](name string, f func(context.Context, I) (O, error), opts TaskOpts) *Task[I, O]

// Call starts the task and returns a Future for its result. Call may only be
// reached from within an Entrypoint function, from within another task, or
// from a StateGraph node via an Entrypoint.Invoke inside that node (the run
// dispatcher travels through the context); anywhere else Call panics —
// the Go analogue of Python's runtime error for tasks called outside an
// entrypoint/StateGraph.
//
// When the enclosing run resumed from a checkpoint whose pending writes
// contain this call's deterministic task ID, the Future is filled from the
// persisted result and f does NOT re-execute (checkpoint replay).
func (t *Task[I, O]) Call(ctx context.Context, in I) *Future[O]

// ClearCache removes every cached result of this task from cache,
// mirroring Python's `_TaskFunction.clear_cache`. It is a no-op when the
// task has no Cache policy.
func (t *Task[I, O]) ClearCache(ctx context.Context, cache checkpoint.Cache) error

// Future is a pending task result with exactly two outcome states: a value
// or an error (a task that interrupted carries a *types.GraphInterrupt
// instead — see Get).
type Future[O any] struct { /* unexported: done chan, val, err, gi */ }

// Get blocks until the task completes and returns its result. A task that
// called graph.Interrupt re-panics that *types.GraphInterrupt from Get, so
// an interrupt raised inside a task pauses the enclosing run exactly as if
// the entrypoint function itself had interrupted (Python parity: interrupts
// from call tasks are the parent's responsibility, `_algo.py:844-846`).
// Callers must not recover this panic. Get also honors ctx cancellation.
func (f *Future[O]) Get(ctx context.Context) (O, error)

// AwaitAll waits for every future and returns their values in argument
// order; the first error (in argument order among failed futures) aborts
// with that error. A *types.GraphInterrupt carried by any future propagates
// as a panic (same rule as Future.Get).
func AwaitAll[T any](ctx context.Context, futs ...*Future[T]) ([]T, error)

// EntrypointOpts bundles the optional Entrypoint backends and policies,
// mirroring `@entrypoint(checkpointer=..., cache=..., retry_policy=...)`.
// Python's `store=` (cross-thread BaseStore) is NOT ported — documented
// divergence.
type EntrypointOpts struct {
    // Checkpointer enables cross-invocation state (previous) and
    // interrupt/resume with task-result replay. Nil disables persistence.
    Checkpointer checkpoint.Saver
    // Cache is the backend for task-level Cache policies (TaskOpts.Cache)
    // and is simply installed on the internal graph via graph.WithCache.
    Cache checkpoint.Cache
    // Retry retries the entrypoint function as a whole (it is the internal
    // graph node's retry policy, installed via AddNodeWithPolicies).
    Retry *graph.RetryPolicy
}

// Entrypoint is a function compiled to a checkpointed workflow, mirroring
// Python's `entrypoint`-decorated callable. I is the input type, O the
// return type, S the save type threaded through `previous`.
type Entrypoint[I, O, S any] struct { /* unexported: opts, *graph.CompiledGraph */ }

// NewEntrypoint compiles f into an Entrypoint. prev is the save value of the
// previous invocation on the same thread (graph.Options.ThreadID); hasPrev
// is false — and prev the zero value — when there is no checkpointer, no
// ThreadID, or no prior completed invocation (Python passes None; Go uses an
// explicit bool so a zero S is never misread). For the plain form the
// returned O value is also written as the save value, so O must be
// assignable to S (use NewEntrypointFinal to decouple them); a mismatch
// surfaces as a descriptive error on the NEXT invocation's previous
// assertion, not silently.
//
// Construction panics if f is nil or the internal graph fails to compile
// (programmer errors; the fixed graph shape cannot otherwise fail).
func NewEntrypoint[I, O, S any](opts EntrypointOpts,
    f func(ctx context.Context, in I, prev S, hasPrev bool) (O, error)) *Entrypoint[I, O, S]

// Final decouples the value returned to the caller from the value saved for
// the next invocation's `previous`, mirroring `entrypoint.final(value=,
// save=)` (`func/__init__.py:475-514`).
type Final[O, S any] struct {
    // Value is returned to the caller (written to the __end__ channel).
    Value O
    // Save is persisted for the next invocation's previous (written to the
    // __previous__ channel).
    Save S
}

// NewEntrypointFinal is NewEntrypoint for functions returning Final[O, S].
func NewEntrypointFinal[I, O, S any](opts EntrypointOpts,
    f func(ctx context.Context, in I, prev S, hasPrev bool) (Final[O, S], error)) *Entrypoint[I, O, S]

// Invoke runs the entrypoint. opts.ThreadID (+ a Checkpointer) enables
// previous-injection and resumability; opts.Resume feeds a value to a paused
// run's pending graph.Interrupt (input is ignored on resume, mirroring
// graph.Options.Resume semantics); opts.CheckpointID pins a historical
// checkpoint (time travel).
//
// When the run pauses on interrupts, Invoke returns the zero O and an
// *InterruptError carrying them (recover via errors.As). Any other run
// failure is returned as a plain error.
func (e *Entrypoint[I, O, S]) Invoke(ctx context.Context, in I, opts graph.Options) (O, error)

// Stream runs the entrypoint like Invoke and yields stream chunks. The mode
// is fixed to graph.StreamUpdates (Python's entrypoint default
// stream_mode="updates", `func/__init__.py:532`): each chunk's payload is
// map[string]any{"entrypoint": <return value>} for the completion chunk and
// map[string]any{"__interrupt__": []types.Interrupt{...}} on pause —
// reserved channel keys (__start__/__end__/__previous__) are filtered out,
// and individual task calls do NOT produce chunks (tasks run inside the
// node; they are not graph tasks — documented divergence from Python, whose
// PUSH tasks stream per-task updates). Early break cancels the run.
func (e *Entrypoint[I, O, S]) Stream(ctx context.Context, in I, opts graph.Options) iter.Seq2[graph.StreamChunk, error]

// InterruptError is returned by Invoke when the run paused on one or more
// interrupts, mirroring the "__interrupt__" key of Python's invoke result.
type InterruptError struct {
    // Interrupts are the pausing interrupts, in the run's collection order.
    Interrupts []types.Interrupt
}

func (e *InterruptError) Error() string // "fn: entrypoint interrupted (N pending)"
```

### F2. 确定性 task ID（扩展 `graph/taskid.go`）

```go
// FnTaskID computes the deterministic identity of a functional-API task
// invocation: an fnv-1a 64-bit hash over the base checkpoint's ID, the
// checkpoint namespace, the run's step, the task name, the parent call path,
// and the per-path call index, hex-encoded. It is the Go analogue of
// Python's PUSH/Call task ID (`pregel/_algo.py:834-842`:
// task_id_func(checkpoint_id, checkpoint_ns, step, name, PUSH, parent_path,
// call_idx)). Same-task calls from a loop hash differently by callIdx, so
// checkpoint replay keys on call identity rather than task name.
func FnTaskID(cpID, ns string, step int, name, parentPath string, callIdx int) string {
    h := fnv.New64a()
    fmt.Fprintf(h, "%s\x00%s\x00%d\x00%s\x00%s\x00%d\x00", cpID, ns, step, name, parentPath, callIdx)
    return fmt.Sprintf("%016x", h.Sum64())
}
```

- `parentPath`：entrypoint 直接调用为 `""`；在 task 内调用为父 task 的调用路径（根级 task 的路径为 `name@callIdx`，如 `"a@0"`；嵌套一级为 `"a@0/b@0"`，以此类推——空 parentPath 不加前导 `/`）。**不含 cpID**，因此重盖章（F4）无需递归重写。
- `callIdx`：dispatcher 内 per-parentPath 计数器（`map[string]int`），Call 时取号。每次 entrypoint 重跑从零重放；文档约束重跑时调用顺序必须确定（同 Python determinism 一节）。
- `cpID`/`step` 的取值见 F4——**追加时重盖章**使两侧恒一致，无需运行前可知。

### F3. dispatcher（per-run，fn 包装层持有，ctx 注入）

```go
// dispatcher drives Task.Call execution and replay for one entrypoint run.
// The fn wrapper layer (Invoke/Stream) owns it — it is NOT on the node
// stack, so an interrupt panic unwinding the node never loses buffered
// results. It reaches Task.Call through the context, the same injection
// pattern as graph.Interrupt (graph/graph.go:1127-1128).
type dispatcher struct {
    mu      sync.Mutex
    counts  map[string]int              // parentPath -> next call index (per-run replay counter)
    replay  map[string]checkpoint.Write // deterministic task ID -> persisted result write; nil on a fresh run
    cpID    string                      // replay base: the checkpoint the run resumed from
    ns      string                      // checkpoint namespace (always "" — fn runs are root-namespace)
    step    int                         // replay base: that checkpoint's Metadata.Step
    cache   checkpoint.Cache            // EntrypointOpts.Cache; nil disables task caching
    results []taskResult                // every outcome completed this run (execution, replay, cache hit)
    sealed  bool                        // set at run end (after cancel); record() drops everything once sealed
}

type taskResult struct {
    name       string
    parentPath string
    callIdx    int
    value      any    // set when isErr is false (falsy values included — the channel carries the state, not truthiness)
    errMsg     string // set when isErr is true
    isErr      bool
}

type dispatcherKey struct{} // context key for *dispatcher
type callPathKey struct{}   // context key for the caller's call path (string)
```

- `Task.Call` 在 `dispatcherFromContext(ctx) == nil` 时 panic（对齐 Python 运行期错误）。
- 嵌套：task goroutine 的 ctx 由 Call ctx 派生并写入新 callPath；entrypoint 内 Invoke 另一个 Entrypoint 时内层 dispatcher 经 `context.WithValue` 自然遮蔽外层。
- run 结束（正常/出错/interrupt panic 传播）时 fn 层按钉死时序收尾（F4）：先 `cancel()` run-scoped ctx，再 `seal()`——已启动未完成的 task 被取消（文档化分歧：Python 丢弃未完成 PUSH 任务），seal 前已完成并 record 的结果供持久化，seal 后完成的（不响应 ctx 的在途 task）一律丢弃，关闭 record 与 persist 之间的竞争与丢失窗口。

### F4. 结果持久化闭环（spec 已定稿，此处钉死 cpID/step 语义与收尾时序）

- **运行开始**：`Invoke`/`Stream` 在 checkpointer 非 nil 且 `opts.ThreadID != ""` 时 `GetTuple(Config{ThreadID, CheckpointID: opts.CheckpointID})`（空 CheckpointID = latest）。重放表载入条件（gate）：
  1. `opts.Resume != nil` **且 `len(tup.Checkpoint.Next) > 0`**（interrupt 恢复：tuple 是暂停 checkpoint，`Source=="loop"`、`Next` 非空——Next 为空的 checkpoint 不是暂停点，不重放）；或
  2. tuple 的 `Metadata.Source == "input"` 且 pending writes 中含 `__return__`/`__error__`（上一轮在首个超步提交前失败——entrypoint 错误重试，对齐 Python `invoke(None, config)` 重跑重抛；见 `_runner.py:748-754`）。
  
  命中 gate 时：`replay[taskID] = w`（`w.Channel ∈ {__return__, __error__}`，**last write wins**——链式暂停会向不同 checkpoint 各追加一次，同一 checkpoint 重复追加的值相同）；`cpID = tup.Checkpoint.ID`，`step = tup.Metadata.Step`。新鲜轮次（tuple `Source=="loop"` 或无 fn writes）**不重放**——防止新轮次误命中上轮结果（Python 靠新 input checkpoint 的 checkpoint_id 换哈希天然隔离；Go 的 gate 达成同效，文档化）。
  
  **gate 2 的推论（文档化分歧，doc.go 与 Task 6.3 测试钉死）**：entrypoint 出错后，同 ThreadID 的**任何**后续 Invoke——包括换全新输入——都会再次重放重抛该错误；用户须换新 ThreadID 开新轮次才能逃离错误态（Python 用新输入开新轮次即可逃离，Go 的 gate 2 不看输入，不能）。
- **运行返回后**（正常完成、返回错误、interrupt 暂停、用户函数 panic 的 defer 路径——四条都跑），钉死时序：**① 显式 `cancel()` run-scoped ctx（`defer cancel()` 仍保留作 panic 路径兜底）→ ② `d.seal()` → ③ `d.snapshotResults()`（持 `d.mu` 拷贝）→ ④ 逐条 PutWrites**。checkpointer 非 nil、ThreadID 非空、快照非空时，`GetTuple`（latest）定位运行产生的最新 checkpoint L（暂停→暂停 checkpoint；完成→最终 loop checkpoint；出错→运行前的 input checkpoint），对每条 result 重盖章并追加：

```go
id := graph.FnTaskID(L.Checkpoint.ID, L.Config.CheckpointNS, L.Metadata.Step, r.name, r.parentPath, r.callIdx)
w := checkpoint.Write{Channel: checkpoint.ReservedReturn, Value: r.value}
if r.isErr {
    w = checkpoint.Write{Channel: checkpoint.ReservedError, Value: r.errMsg}
}
// 每条 result 一次 PutWrites（Saver.PutWrites 对整个 batch 盖同一个 taskID，
// checkpoint/checkpoint.go:186-189 + memory.go:121-125）
err := saver.PutWrites(ctx, L.Config, []checkpoint.Write{w}, id)
```

  重盖章正确性：恢复运行时重放查询用 `FnTaskID(tup.ID, ns, tup.Step, ...)` 计算，与上一次追加时写入 L=tup 的 ID 恒等；链式暂停（pause→resume→pause）时整张结果表（含重放命中条目）重新缓冲、向新暂停 checkpoint 重盖章追加，逐环一致。pending writes 本就惰性读取，事后追加语义正确（spec）。
- **错误重抛**：重放命中 `__error__` 时 `Get` 返回 `errors.New(errMsg)`——**丢失原错误类型**（Go 无法序列化 error 值；文档化分歧，Python pickle 异常对象）。task 执行失败（重试耗尽）即记录 `__error__` 结果；entrypoint 未捕获则经 node error 使 graph run 返回错误，fn 层仍把结果表追加到最新 checkpoint（=input checkpoint），下次同线程 Invoke 经 gate 2 重放重抛。
- **falsy 结果**：结果两态由 channel（`__return__` vs `__error__`）区分，`false`/`0`/`""` 结果正常往返（对标 `test_pregel.py:5486` test_falsy_return_from_task）。
- **M5 注意**：M5 计划给 `Saver.PutWrites` 增加 `taskPath string` 参数、给 `checkpoint.Write` 增加 `TaskPath` 字段（根接口扩展）。若 M5 已落地，本计划的 PutWrites 调用点传入 `r.parentPath`；未落地则按当前四参签名。`loadReplay` 只按 `Channel` 过滤 pending writes，Write 新增字段不影响它。实现时以 `checkpoint/checkpoint.go` 实际签名为准并同步调整。

### F5. graph 包前置导出（Task 1，均为纯追加、零行为变化）

```go
// graph/policy.go 追加：
// Resolved returns a copy of p with every unset field replaced by its
// default (see withDefaults). Exported for the fn package's task retry loop.
func (p RetryPolicy) Resolved() RetryPolicy { return p.withDefaults() }

// BackoffDelay returns the delay before re-executing after the given
// (1-based) failed attempt (see backoff). Exported for the fn package.
func (p RetryPolicy) BackoffDelay(attempt int) time.Duration { return p.backoff(attempt) }

// checkpoint/checkpoint.go 常量块追加：
// ReservedReturn persists a functional-API task's return value (fn package),
// mirroring Python's `__return__` (`_internal/_constants.py:22`).
ReservedReturn = "__return__"
```

### F6. Entrypoint 内部编译形态

- 单节点图：节点名常量 `"entrypoint"`；`AddChannel("__start__", channels.NewEphemeral(true))`、`AddChannel("__end__", channels.NewLastValue())`、`AddChannel("__previous__", channels.NewLastValue())`；`AddNodeWithPolicies("entrypoint", nodeFn, graph.NodePolicies{Retry: opts.Retry})`；`SetEntryPoint("entrypoint")`；`Compile` 时按需追加 `graph.WithCheckpointer` / `graph.WithCache`。
- 节点函数返回 `map[string]any{"__end__": value, "__previous__": save}`（两 channel 各一条 write，LastValue 单写约束满足）。
- 保留 key 不进用户可见面：Invoke 只取 `res.Values["__end__"]`；Stream 过滤/改写 updates payload（F1 doc comment）。
- `previous` 注入：新鲜轮次从运行前 GetTuple 的 `ChannelValues["__previous__"]` 读入并写入 input batch（`input["__previous__"] = rawPrev`，仅在键存在时）；resume 轮次 graph 忽略 input，`__previous__` 随 checkpoint 状态恢复（暂停 checkpoint 含上轮值）——两条路径节点函数都统一从 `state["__previous__"]` 读取并做 `.(S)` 断言，断言失败报描述性错误（serde 契约违反，不静默降级）。
- 输入：input batch `{"__start__": in}`；节点从 `state["__start__"]` 取并 `.(I)` 断言（失败同样报描述性错误）。`__start__` 用 Ephemeral 对齐 Python（`func/__init__.py:594`）；其"末超步值滞留 checkpoint"的既有分歧（`channels/ephemeral.go:17-24`）恰好保证 resume 重跑能读到原输入。

### F7. 顺序 interrupt 的 resume 值对齐（RESUME-write 机制，graph 包修复，Task 2）

**Bug（审阅实证）**：`resumeValuesFor`（`graph/resume.go:246-262`）对 scalar resume 只构造 `[]any{resume}`（长度=当前 checkpoint 的未消费 interrupt 数）；单节点内多次顺序 `Interrupt` 时，第二次及以后的 resume 把新值错配给队列头部（实证：entrypoint/节点循环 2 次 interrupt，第三次 Invoke 把 answer2 错配给 q0 → 再暂停死循环）。根因：Go 每次暂停新建 checkpoint，新 checkpoint 的 pending writes 只含本轮新触发的 interrupt，**已消费的 resume 前缀丢失**。Python 靠 interrupt 消费/触发时把已消费列表经 `(RESUME, ...)` write 持久化、恢复时重放完整前缀（`types.py:905-925`：`conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])`；`RESUME = "__resume__"`，`_internal/_constants.py:11`）。

**修复设计**（graph 包，同时覆盖普通 StateGraph 节点与 fn entrypoint 节点）：

1. `checkpoint` 包新增常量：

```go
// ReservedResume persists the ordered prefix of resume values a re-run task
// has already consumed (one write per value, in consumption order), so a
// later resume rebuilds the full ordered queue instead of misaligning the
// new value onto an already-answered interrupt (Python parity: `types.py`
// interrupt() sends (RESUME, scratchpad.resume); `RESUME = "__resume__"`,
// `_internal/_constants.py:11`).
ReservedResume = "__resume__"
```

2. `runNode`/`runTask`（`graph/graph.go:1088-1144`）增返 `consumed []any`：`runNode` 在 recover 到 `*types.GraphInterrupt` 时拷贝 `ist.resumeQueue[:ist.idx]`（触发 panic 时 `idx == len(resumeQueue)`，即本轮消费的全部有序前缀）；`runTask` 透传（重试/错误路径丢弃——出错不产暂停 checkpoint，前缀无落点）。`outcome` 结构（`graph/graph.go:857` 构造处）新增 `consumed` 字段。
3. 暂停路径（`graph/graph.go:916-923`）：中断任务除原有的 ReservedInterrupt writes 外，把 consumed 前缀逐值写 ReservedResume（同一 `PutWrites` 批次，interrupt writes 在前、resume writes 在后；slice 顺序即消费顺序，saver 按序存储）。新函数（resume.go）：

```go
// persistInterruptAndResume records a paused in-node task's pending
// interrupts plus the ordered prefix of resume values it has already
// consumed (one ReservedResume write per value, in consumption order), so
// the next resume rebuilds the full ordered queue (see resumeValuesFor).
// Boundary interrupts keep using persistInterrupts — no node ran, so there
// is no consumed prefix.
func persistInterruptAndResume(ctx context.Context, saver checkpoint.Saver, cfg checkpoint.Config, taskID string, interrupts []types.Interrupt, consumed []any) error {
    writes := make([]checkpoint.Write, 0, len(interrupts)+len(consumed))
    for _, intr := range interrupts {
        writes = append(writes, checkpoint.Write{Channel: checkpoint.ReservedInterrupt, Value: intr})
    }
    for _, v := range consumed {
        writes = append(writes, checkpoint.Write{Channel: checkpoint.ReservedResume, Value: v})
    }
    return saver.PutWrites(ctx, cfg, writes, taskID)
}
```

4. `resumeValuesFor` 改签名并重建完整队列；`planResume`（`graph/resume.go:176-230`）在 byTask 分类循环里按序收集该任务的 ReservedResume writes 传入：

```go
// resumeValuesFor computes the resume queue for a re-run task: the persisted
// prefix of already-consumed resume values (ReservedResume writes, in
// consumption order) followed by the values matched from THIS resume call,
// mirroring Python's accumulated (RESUME, ...) scratchpad list
// (`types.py:905-925`). Matching rules for the new value are unchanged: a
// map[string]any addresses pending interrupts by ID (unmatched ones re-fire
// on re-run), a nil resume appends nothing (the pending interrupt re-fires,
// the run re-pauses), any other scalar feeds the first pending interrupt.
// Values for interrupts already answered in earlier cycles are carried by
// prefix, so a map entry naming an already-answered interrupt ID is ignored.
func resumeValuesFor(pending []types.Interrupt, prefix []any, resume any) []any {
    queue := append([]any{}, prefix...)
    if len(pending) == 0 || resume == nil {
        return queue
    }
    if byID, ok := resume.(map[string]any); ok {
        for _, p := range pending {
            if v, ok := byID[p.ID]; ok {
                queue = append(queue, v)
            }
        }
        return queue
    }
    return append(queue, resume)
}
```

5. **自包含性**：每个暂停 checkpoint 携带该任务当前的完整消费前缀（= 上一轮 checkpoint 重建的前缀 + 本轮新消费值），链式暂停逐环推进；既有单 interrupt 流程 prefix 为空，行为零变化。interrupt ID 稳定性：`taskInterruptState.counter` 随消费递增，重跑后第 k+1 次 Interrupt 调用重触发同 ID（`"<node>-<counter>"`，`graph/graph.go:1178-1184`），跨轮一致。
6. **sqlite 嵌套 module**：`reservedWriteIdx`（`langgraph/checkpoint/sqlite/sqlite.go:55-65`）补 `checkpoint.ReservedResume: -4`（对齐 Python `WRITES_IDX_MAP` 的 `RESUME: -4`；该 map 注释中"no SCHEDULED/RESUME writes"一句同步更新为"SCHEDULED 无对应物"）。M5 的 postgres 映射已含 RESUME=-4（spec M5 节），兼容。
7. 注释清扫：resume.go 文件头注释（13-25 行）与 `planResume` doc comment 中"re-executes only interrupted tasks, feeding them their resume queues"的表述补充前缀重建语义；`graph/graph.go` 暂停路径注释（904-910 行）补 ReservedResume writes 一条。

## Reference Semantics (Python 移植基线，均已逐行核实)

- Entrypoint 编译形态：三保留 channel + 单 PregelNode + ChannelWrite 拆 value/save（`langgraph/func/__init__.py:576-609`）；`entrypoint.final`（同文件 475-514）；previous 注入语义与 `test_entrypoint_stateful`（`tests/test_pregel.py:6329-6367`）。
- task/future：`@task` 不立即执行、返回 future（`func/__init__.py:86-94` → `pregel/_call.py:276-298`）；恢复重放：已运行 task 从 pending writes 回填 future、错误重抛（`pregel/_runner.py:734-786`，尤其 748-754 行 `RETURN`/`ERROR` 两分支）；`call_counter` 区分循环调用（`_runner.py:720-733` scratchpad.call_counter）。
- 确定性 task ID：`prepare_push_task_functional`（`pregel/_algo.py:800-842`），哈希输入 = checkpoint_id, checkpoint_ns, step, name, PUSH, parent_path, call_idx。
- cache 命名空间：`(CACHE_NS_WRITES, identifier(func))`，`CACHE_NS_WRITES="__pregel_ns_writes"`（`_internal/_constants.py:29`；`_algo.py:858-870`）；Go 对应物 `__fn_writes/<name>`（spec 钉定）。
- task 结果持久化到**当前 checkpoint**（不新建超步 checkpoint）；resume 值按 interrupt 调用序匹配；非确定性逻辑必须放进 task（`func/__init__.py` task/entrypoint docstring）。
- RESUME-write 机制：interrupt 消费/触发时把已消费列表经 `(RESUME, ...)` write 持久化（`types.py:905-925`），恢复时 scratchpad.resume 重放完整前缀；`RESUME = "__resume__"`（`_internal/_constants.py:11`），`WRITES_IDX_MAP` 中 `RESUME: -4`。

---

### Task 1: 前置导出 — FnTaskID + RetryPolicy 导出包装 + ReservedReturn

**Files:**
- Modify: `langgraph/graph/taskid.go` — 追加 `FnTaskID`（F2 完整代码）
- Modify: `langgraph/graph/policy.go` — 追加 `Resolved`/`BackoffDelay`（F5）
- Modify: `langgraph/checkpoint/checkpoint.go` — 常量块追加 `ReservedReturn`（F5）
- Test: `langgraph/graph/taskid_test.go`（新建）

**Interfaces:**
- Produces: `graph.FnTaskID(cpID, ns string, step int, name, parentPath string, callIdx int) string`（Task 4/6 消费）；`graph.RetryPolicy.Resolved() RetryPolicy`、`graph.RetryPolicy.BackoffDelay(attempt int) time.Duration`（Task 4 消费）；`checkpoint.ReservedReturn = "__return__"`（Task 3/4/6 消费）。

- [ ] **Step 1: Write failing tests** — `taskid_test.go`：①同一组输入两次调用 `FnTaskID` 结果相同（确定性）；②逐字段摄动（cpID、step、name、parentPath、callIdx 各变一次）结果两两不同；③返回 16 位小写 hex（`len == 16`，字符 ∈ [0-9a-f]）；④与 `graph.TaskID` 同输入风格一致（fnv-1a、`\x00` 分隔）——用 `fnv.New64a()` 手工复算 `FnTaskID("cp", "", 0, "mapper", "", 0)` 比对相等。
- [ ] **Step 2: Run `go test ./langgraph/graph/ -run TestFnTaskID -v`，verify failure（编译错误：FnTaskID 未定义）。**
- [ ] **Step 3: Implement** — 三个文件的追加（F2/F5 代码逐字）。
- [ ] **Step 4: Gate PASS**（`go build ./... && go vet ./... && go test ./...`；既有测试零改动全绿——纯追加不触碰任何既有符号）。
- [ ] **Step 5: Commit** — `git add langgraph/graph/taskid.go langgraph/graph/taskid_test.go langgraph/graph/policy.go langgraph/checkpoint/checkpoint.go && git commit -m "feat(langgraph): export FnTaskID, retry helpers, and __return__ channel for the fn package"`.

---

### Task 2: graph 包 — RESUME-write 机制（顺序 interrupt 的 resume 值对齐修复，F7）

**Files:**
- Modify: `langgraph/checkpoint/checkpoint.go` — 常量块追加 `ReservedResume`（F7 代码逐字）
- Modify: `langgraph/graph/resume.go` — 新增 `persistInterruptAndResume`；`resumeValuesFor` 改签名+语义；`planResume` 收集前缀；文件头与 doc comment 清扫（F7.7）
- Modify: `langgraph/graph/graph.go` — `runNode`/`runTask` 增返 `consumed`、`outcome` 加字段、暂停路径（903-937 行段）改调 `persistInterruptAndResume`、暂停路径注释补一条；边界 interrupt 两处 `persistInterrupts`（752、985 行）不动
- Modify: `langgraph/checkpoint/sqlite/sqlite.go` — `reservedWriteIdx` 补 `checkpoint.ReservedResume: -4` + 注释更新（嵌套 module）
- Test: `langgraph/graph/resume_test.go`（扩展）

**Interfaces:**
- Consumes: 现有 resume 机制（`interruptsFromWrites`/`planResume`/`resumeValuesFor`/`persistInterrupts`，`graph/resume.go:29-276`；暂停路径 `graph/graph.go:903-937`；`taskInterruptState`，`graph/graph.go:1146-1153`）。
- Produces（后续 fn 任务的依赖面）：
  - `checkpoint.ReservedResume = "__resume__"`（常量；fn 任务不直接消费，但 Task 6.4/7.5 的链式暂停用例行为依赖本机制）。
  - `func persistInterruptAndResume(ctx context.Context, saver checkpoint.Saver, cfg checkpoint.Config, taskID string, interrupts []types.Interrupt, consumed []any) error`。
  - `func resumeValuesFor(pending []types.Interrupt, prefix []any, resume any) []any`（签名变化；包内唯一调用点 `planResume` 同步改，无包外调用者）。
  - `runNode`/`runTask`/`outcome` 的包内签名变化（`graph` 包外无调用者；`langchain/agents` 只用公开 API，零影响）。

- [ ] **Step 1: Write failing tests**（`resume_test.go` 扩展）：
  1. **节点版顺序 interrupt**（移植 `test_pregel.py:6048-6090` test_node_before_multiple_interrupt_cycles_graph_api）：`prepare` 节点（count+10）→ `multi_interrupt` 节点（`first := graph.Interrupt(ctx, "First question?")`、`second := graph.Interrupt(ctx, "Second question?")`，返回 `{"data": first+","+second}`）。三次 Invoke（ThreadID "1"；第二次 `Resume: "first_answer"`、第三次 `Resume: "second_answer"`）：首次暂停值 "First question?"；第二次暂停值 "Second question?"（**不是**重触发第一个，也不是错配）；第三次返回 `{"count":10, "data":"first_answer,second_answer"}`。修复前此测试在第三次 Invoke 错配（answer 喂给队列头部）或重暂停——即验证失败形态。
  2. **暂停 checkpoint 的 writes 内容**：第二次暂停后，最新 tuple 的 PendingWrites 中该任务含一条 `ReservedInterrupt`（Value 为 "Second question?" 的 types.Interrupt）+ 一条 `ReservedResume`（Value == "first_answer"），且 interrupt write 在前、resume write 在后。
  3. **三轮链式前缀累积**：单节点循环 3 次 interrupt（值为 q0/q1/q2），四次 Invoke（Resume "a"/"b"/"c"）→ 依次暂停 q0/q1/q2 后完成，返回 `[a,b,c]` 按序拼接；第三个暂停 checkpoint 的 ReservedResume writes 为 ["a","b"]（按序两条）。
  4. **nil-resume 重暂停语义不变**：用例 1 首次暂停后以 nil Resume + 空 input Invoke（`invoke(None)` 语义）→ 重暂停同 interrupt（"First question?"），既有行为不回归。
- [ ] **Step 2: Run `go test ./langgraph/graph/ -run TestResume -v`，verify failure**（用例 1 第三次 Invoke 错配/重暂停；用例 2/3 找不到 ReservedResume writes）。
- [ ] **Step 3: Implement** — F7 逐字（四个文件）。既有测试回归说明：若有既有测试断言暂停 checkpoint pending writes 的**精确集合**，修复后多出的 ReservedResume writes 属本修复的预期行为变化——把 RESUME writes 补进该测试预期并在 commit message 注明（除此外不允许改既有测试）。
- [ ] **Step 4: Gate PASS**（`go build ./... && go vet ./... && go test ./...` + `go test -race ./langgraph/graph/` + `make test-sqlite`——sqlite.go 改动必须跑嵌套 module 测试）。
- [ ] **Step 5: Commit** — `git add langgraph/checkpoint/checkpoint.go langgraph/graph/resume.go langgraph/graph/graph.go langgraph/graph/resume_test.go langgraph/checkpoint/sqlite/sqlite.go && git commit -m "feat(langgraph/graph): persist consumed resume prefix to align sequential interrupt resumes"`.

---

### Task 3: fn 包骨架 — Future + dispatcher + ctx 注入 + AwaitAll

**Files:**
- Create: `langgraph/fn/future.go` — `Future[O]`、`AwaitAll`（F1 签名）
- Create: `langgraph/fn/dispatcher.go` — `dispatcher`/`taskResult`/ctx keys/`newDispatcher`/`dispatcherFromContext`/`contextWithDispatcher`（F3）、`record`/`seal`/`snapshotResults`、`replayedCall`（Task 4 实现）所需的 replay 查询辅助
- Create: `langgraph/fn/doc.go` — 包注释（本轮只写包级概述；文档化分歧清单在 Task 8 补全）
- Test: `langgraph/fn/future_test.go`、`langgraph/fn/dispatcher_test.go`（均为 `package fn` 白盒测试）

**Interfaces:**
- Consumes: `checkpoint.Write`、`checkpoint.ReservedReturn`/`checkpoint.ReservedError`（Task 1）、`types.GraphInterrupt`（`langgraph/types/types.go:76`）。
- Produces:
  - `func newDispatcher(cache checkpoint.Cache) *dispatcher`
  - `func contextWithDispatcher(ctx context.Context, d *dispatcher) context.Context`
  - `func dispatcherFromContext(ctx context.Context) *dispatcher`（nil = 非法上下文）
  - `func (d *dispatcher) nextCallIdx(parentPath string) int`（加锁取号）
  - `func (d *dispatcher) record(r taskResult)`（持 `d.mu` 追加；**`d.sealed` 为 true 时丢弃**——run 收尾后迟到完成的 task 不进结果表，F4 时序②③）
  - `func (d *dispatcher) seal()`（持锁置 `sealed`）
  - `func (d *dispatcher) snapshotResults() []taskResult`（持锁拷贝整张表）
  - `func (d *dispatcher) loadReplay(tup *checkpoint.Tuple, opts graph.Options)`（F4 gate + 建表；Task 6 才接真实 tuple，本任务实现函数体并单测。gate 1 要求 `opts.Resume != nil && len(tup.Checkpoint.Next) > 0`；gate 2 要求 `tup.Metadata.Source == "input"` 且 pending writes 含 `__return__`/`__error__`）
  - `Future[O]`：内部 `done chan struct{}`（关闭即完成）、`val O`、`err error`、`gi *types.GraphInterrupt`；`Get` 语义见 F1（`gi` 非 nil 时 panic(gi)）。
  - 解析辅助（包级泛型函数，Go 不允许非泛型类型上有泛型方法）：`func resolvedFuture[O any](val O, err error, gi *types.GraphInterrupt) *Future[O]`。
  - `AwaitAll`：按参数顺序收集；任一 `Get` panic 的 GraphInterrupt 直接向上传播（不 recover）。

- [ ] **Step 1: Write failing tests** —
  - `future_test.go`：`resolvedFuture(42, nil, nil).Get(ctx)` 返回 `(42, nil)`；`resolvedFuture(0, errors.New("boom"), nil).Get` 返回 `(""/0, boom)`；带 `gi` 的 future `Get` panic 且 recover 到的值是同一 `*types.GraphInterrupt`；`Get` 在未完成的 future 上阻塞、另一 goroutine close 后返回（用 `time.After` 断言先阻塞后返回）；`Get` 带已取消 ctx 在未完成 future 上返回 `context.Canceled`。`AwaitAll`：三个已解析 future 按序返回 `[1,2,3]`；中间一个带错误时返回该错误且顺序正确。
  - `dispatcher_test.go`：①`nextCallIdx("")` 连续返回 0,1,2；`nextCallIdx("a@0")` 与 `""` 各自独立计数；②`dispatcherFromContext` 在无注入 ctx 上返回 nil、注入后返回同一指针；③`record` 追加顺序保持，`snapshotResults` 返回拷贝（改快照不影响原表）；`seal()` 后 `record` 被丢弃（快照长度不变）；④`loadReplay`：构造 `checkpoint.Tuple{Metadata: checkpoint.Metadata{Source: "loop"}, PendingWrites: []checkpoint.Write{{TaskID: "a", Channel: checkpoint.ReservedReturn, Value: 1}, {TaskID: "b", Channel: checkpoint.ReservedInterrupt, Value: types.Interrupt{}}}, Checkpoint.Next: []checkpoint.PlannedTask{{ID: "x", Node: "entrypoint"}}}` 且 `opts.Resume = "v"` → replay 表只含 `a`（`__interrupt__` 被过滤），`cpID/step` 取自 tuple；`opts.Resume != nil` 但 `Next` 为空 → **不重放**（gate 1 的 Next 条件）；`opts.Resume == nil` 且 Source="loop" → 不重放（replay 为 nil）；Source="input" + 含 fn writes + Resume==nil → 重放（gate 2）；Source="input" 无 fn writes → 不重放。
- [ ] **Step 2: Run `go test ./langgraph/fn/ -v`，verify failure（包不存在/符号未定义）。**
- [ ] **Step 3: Implement** — `future.go`/`dispatcher.go`/`doc.go`（F3 结构逐字 + 上述辅助函数）。
- [ ] **Step 4: Gate PASS。**
- [ ] **Step 5: Commit** — `git add langgraph/fn/ && git commit -m "feat(langgraph/fn): Future, per-run dispatcher, and AwaitAll skeleton"`.

---

### Task 4: Task — NewTask/Call 执行语义 + retry/cache/timeout + 错误与 interrupt 处理

**Files:**
- Create: `langgraph/fn/task.go` — `TaskOpts`/`Task`/`NewTask`/`Call`/`ClearCache`（F1）+ 包级泛型执行辅助
- Test: `langgraph/fn/task_test.go`（`package fn` 白盒）

**Interfaces:**
- Consumes: Task 3 全部 dispatcher/Future 符号；`graph.FnTaskID`、`graph.RetryPolicy.Resolved/BackoffDelay`、`graph.DefaultRetryOn`、`graph.CachePolicy`/`graph.DefaultCacheKey`（`graph/policy.go:172-195`）；`checkpoint.Cache`（`checkpoint/checkpoint.go:147-156`）；`checkpoint.NewInMemoryCache()`（`checkpoint/memory.go:220`）。
- Produces:
  - `func NewTask[I, O any](name string, f func(context.Context, I) (O, error), opts TaskOpts) *Task[I, O]`（name 空 → panic "fn: task name must be non-empty"；f nil → panic）
  - `func (t *Task[I, O]) Call(ctx context.Context, in I) *Future[O]`
  - `func (t *Task[I, O]) ClearCache(ctx context.Context, cache checkpoint.Cache) error`
  - 缓存命名空间：`func fnCacheNS(name string) string { return "__fn_writes/" + name }`（spec 钉定；Python `CACHE_NS_WRITES + module.qualname` 的 Go 对应）
  - 缓存键：`pol.KeyFunc(map[string]any{"input": in})`，nil KeyFunc → `graph.DefaultCacheKey`（文档化：Python key_func 收 `*args/**kwargs`，Go 包成单键 map）
  - 包级泛型辅助（Go 方法不能带类型参数）：`func startTask[I, O](d *dispatcher, ctx context.Context, t *Task[I, O], parentPath string, callIdx int, in I) *Future[O]`、`func callSafely[I, O](ctx context.Context, t *Task[I, O], in I) (O, *types.GraphInterrupt, error)`、`func runAttempt[I, O](ctx context.Context, t *Task[I, O], in I) (O, *types.GraphInterrupt, error)`、`func replayedCall[O any](d *dispatcher, name, parentPath string, callIdx int, w checkpoint.Write) *Future[O]`

**Call 流程（实现逐字按此）：**

```go
func (t *Task[I, O]) Call(ctx context.Context, in I) *Future[O] {
    d := dispatcherFromContext(ctx)
    if d == nil {
        panic("fn: Task.Call must be called from within an Entrypoint function, another task, or an Entrypoint invoked inside a StateGraph node")
    }
    parentPath, _ := ctx.Value(callPathKey{}).(string)
    callIdx := d.nextCallIdx(parentPath)

    // 1. Checkpoint replay: a persisted result fills the future without
    //    re-executing (pregel/_runner.py:745-756).
    if d.replay != nil {
        id := graph.FnTaskID(d.cpID, d.ns, d.step, t.name, parentPath, callIdx)
        if w, ok := d.replay[id]; ok {
            return replayedCall[O](d, t.name, parentPath, callIdx, w) // 见下
        }
    }
    // 2. Cache lookup (independent second mechanism; only with a backend).
    if t.opts.Cache != nil && d.cache != nil {
        if fut, ok := cachedCall(ctx, d, t, parentPath, callIdx, in); ok {
            return fut
        }
    }
    // 3. Fresh execution in its own goroutine.
    return startTask(d, ctx, t, parentPath, callIdx, in)
}
```

- `replayedCall[O]`：`w.Channel == checkpoint.ReservedReturn` → `w.Value.(O)`（断言失败 → 返回带描述性错误 `fmt.Errorf("fn: replayed result of task %q has type %T, want the declared output type", ...)` 的 future，**不静默**）；`__error__` → `resolvedFuture` 带 `errors.New(w.Value.(string))`；两种情况都 `d.record(...)`（重放结果必须重新缓冲——链式暂停要向新 checkpoint 重追加，F4；seal 后 record 自然丢弃，无竞态）。
- `startTask` goroutine 核心：

```go
taskPath := t.name + "@" + strconv.Itoa(callIdx) // 根级调用: "a@0"（空 parentPath 不加前导 "/"）
if parentPath != "" {
    taskPath = parentPath + "/" + taskPath // 嵌套: "a@0/b@0"
}
taskCtx := context.WithValue(ctx, callPathKey{}, taskPath)
go func() {
    defer close(fut.done)
    var retry *graph.RetryPolicy
    if t.opts.Retry != nil {
        r := t.opts.Retry.Resolved()
        retry = &r
    }
    for attempt := 1; ; attempt++ {
        val, gi, err := runAttempt(taskCtx, t, in)
        if gi != nil { // interrupt 透传：不记录、不重试，Get 时 re-panic
            fut.gi = gi
            return
        }
        if err == nil {
            fut.val = val
            d.record(taskResult{name: t.name, parentPath: parentPath, callIdx: callIdx, value: val})
            cacheStore(ctx, d, t, in, val) // 仅成功且 Cache+backend 非 nil 时 Set
            return
        }
        if taskCtx.Err() != nil { // run 取消（interrupt 暂停/父取消）：放弃，不记录
            fut.err = err
            return
        }
        if retry == nil || attempt >= retry.MaxAttempts || !retry.RetryOn(err) {
            fut.err = err
            d.record(taskResult{name: t.name, parentPath: parentPath, callIdx: callIdx, isErr: true, errMsg: err.Error()})
            return
        }
        timer := time.NewTimer(retry.BackoffDelay(attempt))
        select {
        case <-taskCtx.Done():
            timer.Stop()
            fut.err = taskCtx.Err()
            return
        case <-timer.C:
        }
    }
}()
```

- `runAttempt`：`Timeout <= 0` 直接 `callSafely`；否则 per-attempt `context.WithTimeout` + 子 goroutine（结果 channel 缓冲 1，被放弃的 goroutine 不阻塞不泄漏）：

```go
func runAttempt[I, O](ctx context.Context, t *Task[I, O], in I) (O, *types.GraphInterrupt, error) {
    var zero O
    if t.opts.Timeout <= 0 {
        return callSafely(ctx, t, in)
    }
    attCtx, cancel := context.WithTimeout(ctx, t.opts.Timeout)
    defer cancel()
    type outcome struct {
        val O
        gi  *types.GraphInterrupt
        err error
    }
    ch := make(chan outcome, 1)
    go func() {
        v, g, e := callSafely(attCtx, t, in)
        ch <- outcome{v, g, e}
    }()
    select {
    case o := <-ch:
        return o.val, o.gi, o.err
    case <-attCtx.Done():
        if ctx.Err() != nil { // 父/run 取消，不是 task 自身超时
            return zero, nil, ctx.Err()
        }
        return zero, nil, context.DeadlineExceeded // 放弃等待（goroutine 不可强杀，文档化）
    }
}
```

- `callSafely`：`defer recover()`——`*types.GraphInterrupt` → 返回 gi；其他 panic → `fmt.Errorf("fn: task %q panicked: %v", t.name, r)`（普通错误，参与 RetryOn 判定）；正常返回透传。
- `cachedCall`：`key, err := keyFunc(map[string]any{"input": in})`——KeyFunc 出错 → 返回带包装错误的 future（`fmt.Errorf("fn: task %q cache key: %w")`，对齐 graph 节点缓存"key_func 错误使任务失败"）并 record `__error__`；`d.cache.Get(ctx, fnCacheNS(t.name), key)` 命中且 `writes[0].Channel == ReservedReturn` → `.(O)` 断言解析 future + `d.record`；未命中/过期 → `(nil, false)`。
- `cacheStore`：仅 `t.opts.Cache != nil && d.cache != nil` 时 `d.cache.Set(ctx, fnCacheNS(t.name), key, []checkpoint.Write{{Channel: checkpoint.ReservedReturn, Value: val}}, t.opts.Cache.TTL)`；Set 失败 → task 以包装错误失败（graph 节点缓存同规则）。
- `ClearCache`：`t.opts.Cache == nil` → nil；否则 `cache.Clear(ctx, fnCacheNS(t.name))`。

- [ ] **Step 1: Write failing tests**（全部白盒构造 `newDispatcher(nil)` + `contextWithDispatcher`）：
  1. **非法上下文 panic**：裸 `context.Background()` 上 `Call` panic，recover 到字符串含 "Entrypoint"。
  2. **立即执行**：`Call` 返回时任务已在跑（任务内 close 一个 started channel，主 goroutine 不等 Get 即可 `<-started`）；`Get` 返回 `(in*2, nil)`。
  3. **并发 futures**：同一 task 循环 Call 5 次（输入 0..4），`AwaitAll` 按序返回（输入乱序 sleep 证明并发：总耗时 < 串行和）。
  4. **call counter 确定性**：循环 3 次 Call 同 task → dispatcher.counts[""]== 3 且三次 callIdx 为 0,1,2（白盒断言）。
  5. **嵌套 task**：task A 内 Call task B（B 的 parentPath == `"a@0"`，经白盒断言 B 收到的 callPath——根级无前导 `/`）→ 结果正确且 B 的 callIdx 独立计数。
  6. **retry**：f 前两次返回 `&net.DNSError{IsTimeout: true}`（DefaultRetryOn 命中）第三次成功 → `Get` 成功且调用计数==3；`RetryPolicy{MaxAttempts: 2, InitialInterval: time.Millisecond, NoJitter: true}` 下始终失败 → `Get` 返回最后错误且计数==2。
  7. **retry 不重试**：`RetryOn: func(error) bool { return false }` → 计数==1。
  8. **cache**：dispatcher 带 `checkpoint.NewInMemoryCache()`，`Cache: &graph.CachePolicy{}` 的 task 以同输入 Call 两次（第二个用**新 dispatcher**模拟下一轮），计数==1；`ClearCache` 后再 Call 计数==2。
  9. **cache key 包装**：自定义 `KeyFunc` 断言收到的 map 恰为 `map[string]any{"input": in}`。
  10. **timeout**：`Timeout: 50*time.Millisecond` + f sleep 500ms（不响应 ctx）→ `Get` 在 ~50ms 返回 `context.DeadlineExceeded`；被放弃 goroutine 最终完成不阻塞（测试正常退出）。
  11. **panic 转错误**：f panic("boom") → `Get` 返回错误含 `task "x" panicked: boom`。
  12. **GraphInterrupt 透传**：f 调 `graph.Interrupt(ctx, "q")`（ctx 来自白盒注入的 `interruptCtxKey`……注意：`graph.Interrupt` 的 ctx key 是 graph 包未导出的——测试改为直接让 f `panic(&types.GraphInterrupt{Interrupt: types.Interrupt{Value: "q", ID: "n-1"}})`）→ `Get` re-panic 同一对象；dispatcher.results 无该条记录。
  13. **run 取消不记录**：f 阻塞在 `<-ctx.Done()`，主 goroutine cancel run ctx 后 `Get` 返回 `context.Canceled`，results 为空。
  14. **seal 丢弃**：f sleep 100ms（不响应 ctx）正常返回成功；主 goroutine 在 10ms 时 `seal()`——任务最终完成但 `snapshotResults()` 为空（迟到结果被丢弃，无 race——配合 `-race` 验证）。
- [ ] **Step 2: Run `go test ./langgraph/fn/ -v`，verify failure（NewTask 未定义）。**
- [ ] **Step 3: Implement** — `task.go`（上述代码逐字落地）。
- [ ] **Step 4: Gate PASS**（含 `-race` 跑一次 `go test -race ./langgraph/fn/`——dispatcher 计数器/结果缓冲跨 goroutine，必须无 data race）。
- [ ] **Step 5: Commit** — `git add langgraph/fn/task.go langgraph/fn/task_test.go && git commit -m "feat(langgraph/fn): Task with deterministic IDs, retry, cache, and timeout"`.

---

### Task 5: Entrypoint — 编译单节点图 + Invoke/Stream + previous + Final

**Files:**
- Create: `langgraph/fn/entrypoint.go` — `EntrypointOpts`/`Entrypoint`/`Final`/`NewEntrypoint`/`NewEntrypointFinal`/`Invoke`/`Stream`/`InterruptError`（F1）+ F6 编译形态 + `newDispatcher` 装配（本任务只做新鲜轮次：Invoke 不调 `loadReplay`、无 `persistResults`——重放与持久化由 Task 6 新增函数并接线，本任务的 Invoke 返回序列只有 `run → err → InterruptError → __end__`）
- Test: `langgraph/fn/entrypoint_test.go`（`package fn`——可用黑盒 `package fn_test` 风格仅经公开 API 断言；选 `package fn` 统一）

**Interfaces:**
- Consumes: Task 1-4 全部；`graph.NewStateGraph`/`AddChannel`/`AddNodeWithPolicies`/`SetEntryPoint`/`Compile`/`WithCheckpointer`/`WithCache`（`graph/graph.go:102-346`）；`channels.NewEphemeral(true)`/`channels.NewLastValue()`；`graph.Options{ThreadID, CheckpointID, Resume}`、`graph.Result{Values, Interrupts}`（`graph/graph.go:381-453`）；`graph.StreamOptions{Options, Modes, Subgraphs}` + `graph.StreamChunk`（`graph/stream.go:51-86`）；`checkpoint.NewMemorySaver()`（`checkpoint/memory.go:31`）。
- Produces: F1 的 Entrypoint 全部公开 API。内部常量：`channelStart = "__start__"`、`channelEnd = "__end__"`、`channelPrevious = "__previous__"`、`entrypointNode = "entrypoint"`。
- 节点函数与 Invoke 流程按 F6 代码逐字（input batch、类型断言、`{"__end__": v, "__previous__": s}` 返回）。
- `Stream` 实现要点：返回的 iterator 首次拉取时做与 Invoke 相同的 tuple/dispatcher 装配，然后委托 `e.graph.Stream(ctx, input, graph.StreamOptions{Options: opts, Modes: []graph.StreamMode{graph.StreamUpdates}})`；逐 chunk 改写：payload 为 `map[string]any` 且含 `entrypointNode` 键时，取内层 map 的 `__end__` 值替换整个内层 map（`{"entrypoint": <value>}`），`__interrupt__` chunk 原样透传。Task 6 在此基础上于内层 iterator 耗尽或提前 break 后补 `persistResults` 调用（本任务无此调用）。

- [ ] **Step 1: Write failing tests**（Python 出处逐条标注）：
  1. **基本 invoke**（`test_pregel.py:6307` test_entrypoint_without_checkpointer）：无 checkpointer，`NewEntrypoint[map[string]any, map[string]any, map[string]any]` 两次 Invoke 同 ThreadID，`hasPrev` 恒 false、prev 恒零值（收集每次调用的 hasPrev 断言 `[false, false]`；Python 中 previous 恒 None）。
  2. **previous 跨 invoke**（`test_pregel.py:6329` test_entrypoint_stateful）：`checkpoint.NewMemorySaver()`，`NewEntrypoint[map[string]any, map[string]any, map[string]any]`，f 返回 `{"previous": prev-or-nil, "current": in}`（hasPrev=false 时 previous 键写 nil）；同 ThreadID 连调三次 `{"a":"1"}/{"a":"2"}/{"a":"3"}`，断言三次返回与 Python 完全同形（第二次 previous = 第一次返回值，第三次嵌套）。
  3. **Final value≠save**（`test_pregel.py:6785` test_entrypoint_with_return_and_save）：`NewEntrypointFinal[string, int, []string]`，f 返回 `Final{Value: len(prev), Save: append(prev, in)}`；三次 Invoke("hello"/"goodbye"/"definitely") 返回 0/1/2，且 f 内观察到的 prev 依次为 `hasPrev=false`、`["hello"]`、`["hello","goodbye"]`。
  4. **Stream**：单任务 entrypoint 返回 "done"，`Stream` 收 chunks：恰好一条 `Mode == graph.StreamUpdates`、payload `map[string]any{"entrypoint": "done"}`；无 `__start__`/`__previous__` 泄漏到任何 chunk payload。
  5. **entrypoint 内 interrupt + resume**（无 task 版，`test_pregel.py:4985` 的骨架）：f 内 `v := graph.Interrupt(ctx, "Provide value")`，返回 `fmt.Sprintf("got %v", v)`；首次 Invoke 返回 `*InterruptError`（`errors.As` 提取，Interrupts[0].Value == "Provide value"）；`Invoke(ctx, in, graph.Options{ThreadID: "1", Resume: "bar"})` 返回 "got bar"。
  6. **entrypoint 函数错误传播**：f 返回错误 → Invoke 返回该错误（非 InterruptError）。
- [ ] **Step 2: Run `go test ./langgraph/fn/ -v`，verify failure。**
- [ ] **Step 3: Implement** — `entrypoint.go`（F1/F6 逐字）。
- [ ] **Step 4: Gate PASS**（`-race` 一并跑）。
- [ ] **Step 5: Commit** — `git add langgraph/fn/entrypoint.go langgraph/fn/entrypoint_test.go && git commit -m "feat(langgraph/fn): Entrypoint over a single-node StateGraph with previous and Final"`.

---

### Task 6: 结果持久化闭环 — 重放载入 + 结果追加 + 重盖章 + 错误重抛

**Files:**
- Modify: `langgraph/fn/entrypoint.go` — 新增 `persistResults`；`Invoke`/`Stream` 接通 `loadReplay(tup, opts)` 与 `persistResults`（Invoke 的返回序列与收尾时序按下方 Interfaces 钉死）
- Modify: `langgraph/fn/dispatcher.go` — 如 Step 1 测试暴露缺口则补（预期不缺：`loadReplay`/`seal`/`snapshotResults` 已在 Task 3 实现并单测）
- Test: `langgraph/fn/persist_test.go`（`package fn`）

**Interfaces:**
- Consumes: 全部前序任务（含 Task 2 的 RESUME-write 机制——用例 4 的链式双暂停 resume 值按序匹配依赖它）；`checkpoint.Saver.GetTuple/PutWrites`（`checkpoint/checkpoint.go:174-191`）；`graph.FnTaskID`（Task 1）。
- Produces:
  - `func (e *Entrypoint[I, O, S]) persistResults(ctx context.Context, opts graph.Options, d *dispatcher) error`——F4 的"运行返回后"流程逐字。收尾时序（钉死）：**run 返回 → 显式 `cancel()` run-scoped ctx（`defer cancel()` 保留作 panic 兜底）→ `d.seal()` → `d.snapshotResults()` → 逐条重盖章 PutWrites**；四条返回路径（正常/错误/InterruptError/用户 panic 的 defer 最佳努力——panic 路径 persist 失败后原始 panic 继续传播，不遮蔽）都走同一时序。
  - Invoke 的返回序列（钉死顺序）：`run → cancel/seal/snapshot/persist（best-effort）→ run err 非 nil 返回 err → Interrupts 非空返回 *InterruptError → 取 __end__ 断言返回`。persistResults 自身出错且 run err 为 nil 时返回包装错误 `fmt.Errorf("fn: persisting task results: %w", err)`。

- [ ] **Step 1: Write failing tests**：
  1. **resume 重放副作用计数不重跑**（`test_pregel.py:1269` test_imp_task 主体）：task `mapper`（计数器+1，返回 `strings.Repeat(strconv.Itoa(in), 2)`）；entrypoint 对 `[0,1]` 并行 Call 两个 future、Get 后 `graph.Interrupt(ctx, "question")`，返回每个结果拼接 answer。首次 Invoke → InterruptError，计数==2；`Resume: "answer"` → 返回 `["00answer","11answer"]` 且**计数仍 ==2**（白盒+行为双重断言：计数器不变证明重放）。
  2. **falsy 结果重放**（`test_pregel.py:5486`）：task 返回 `false`；entrypoint 先 Get 再 interrupt；resume 后 task 不重跑（计数==1）且结果确为 false（不是零值误读——entrypoint 返回 `strconv.FormatBool(v)` 断言 "false"）。
  3. **错误持久化重抛**：task `bad` 计数+1 后返回错误 "boom"；entrypoint 不捕获。首次 Invoke 返回含 "boom" 的错误；检查 `MemorySaver` 最新 tuple 的 PendingWrites 含一条 `Channel == "__error__"`、`Value == "boom"`；同 ThreadID 再次 Invoke（同输入）→ task **不重跑**（计数==1）、Invoke 返回错误含 "boom"（gate 2 重放重抛；重抛语义下命中 `__error__` 即重抛、永不重跑——与 Python `_runner.py:751-754` 一致）；**同 ThreadID 换全新输入第三次 Invoke → 仍重抛 "boom"**（gate 2 不看输入——钉死该行为为文档化分歧，doc.go 清单⑪；Python 可用新输入逃离，Go 不能）；随后换新 ThreadID 跑成功路径（task 改为返回成功的新实例），确认全新线程不受影响。
  4. **链式双暂停重盖章**（`test_pregel.py:5818` test_task_before_interrupt_resume）：entrypoint 内 task `setup` 返回 2，然后循环 2 次第 i 次 `graph.Interrupt(ctx, fmt.Sprintf("q%d", i))`；三次 Invoke（首次 + Resume "answer1" + Resume "answer2"）→ 最终返回 `["answer1","answer2"]`（resume 值按序匹配，依赖 Task 2 的 RESUME-write 修复），setup 计数==1；中间第二次 Invoke 返回的 InterruptError 的 Value == "q1"；重盖章生效断言：`Saver.List` 取出两个暂停 checkpoint B（首次暂停）与 C（第二次暂停），两者的 fn writes 都存在但 taskID 不同，且分别等于用各自 tuple 的 `Checkpoint.ID`/`Metadata.Step` 复算的 `graph.FnTaskID(...)`。
  5. **interrupt 时不丢缓冲**（spec 闭环①：dispatcher 由 fn 层持有）：task A 立即完成、task B sleep 200ms；entrypoint 依次执行 `futA := A.Call(ctx, ...)`、`vA := futA.Get(ctx)`（A 完成并入缓冲）、`_ = B.Call(ctx, ...)`（B 在途，不 Get）、`graph.Interrupt(ctx, "q")`。首次 Invoke → InterruptError；此时 B 被 run cancel/seal 放弃、结果不记录（即使 200ms 后完成也被 seal 丢弃），A 的结果已落 pending writes（断言最新 tuple 含一条 A 的 `__return__` write）。resume（Resume "ok"）后 A **不重跑**（计数==1）、B 重跑完成（计数==2：首轮在途 1 次 + resume 1 次），最终返回正确。
  6. **无 checkpointer 零持久化**：`EntrypointOpts{}` 跑带 task 的 entrypoint + interrupt（无 checkpointer 时 interrupt 不可恢复——断言首次 Invoke 返回 InterruptError 即够，不 resume）。
- [ ] **Step 2: Run `go test ./langgraph/fn/ -v`，verify failure（重放/持久化未接线：重放测试的副作用计数翻倍、PendingWrites 断言找不到 fn writes）。**
- [ ] **Step 3: Implement** — `persistResults` + Invoke/Stream 接线（F4 代码与时序逐字）。
- [ ] **Step 4: Gate PASS**（含 `-race`）。
- [ ] **Step 5: Commit** — `git add langgraph/fn/ && git commit -m "feat(langgraph/fn): persist task results to checkpoints and replay them on resume"`.

---

### Task 7: Python 测试移植套件（端到端）

**Files:**
- Test: `langgraph/fn/functional_test.go`（新建，`package fn`；本任务集中放移植套件，与按机制组织的既有测试文件并存）

**Interfaces:**
- Consumes: Task 1-6 全部公开 API（含 Task 2 的 RESUME-write 修复——用例 5/7 的顺序 interrupt resume 对齐依赖它）；`graph.NewStateGraph`（"task 在 StateGraph 节点内调用"用例的父图）。

**移植清单**（每条标注 Python 出处；checkpointer 一律 `checkpoint.NewMemorySaver()`——Python 参数化多 saver，Go 移植以 memory saver 为准，sqlite 契约由 saver 自身测试覆盖）：

- [ ] **Step 1: Write failing tests（如已绿则作为回归直接通过——允许，因为 Task 4-6 的机制测试已覆盖部分行为；此处按 Python 用例语义补齐断言）**：
  1. `test_imp_task`（`test_pregel.py:1269-1329`）——已在 Task 6.1 覆盖主体；此处补 **AwaitAll 形态**：`futures := []*fn.Future[string]{...}; fn.AwaitAll(ctx, futures...)` 返回有序结果。
  2. `test_imp_nested`（`test_pregel.py:1332-1394`）——嵌套 task（mapper 内 Call submapper）+ interrupt/resume：两轮计数各自精确（mapper==2、submapper==2），resume 后不变。
  3. `test_interrupt_functional`（`test_pregel.py:4985-5016`）——foo task → interrupt → bar task；resume "bar" 后 `{"a":"foobar","b":"bar"}`（用 `map[string]any` 输入输出）。
  4. `test_interrupt_task_functional`（`test_pregel.py:5019-5072`）——**interrupt 在 task 内**：bar task 内 `graph.Interrupt(ctx, ...)`（task 的 f 收到的 ctx 带有 entrypoint 节点的 interrupt 状态——本用例验证该接线成立：Get re-panic → run 暂停 → resume 后 bar 重跑、Interrupt 消费 resume 值）；第二段（同 task 连续两次 interrupt、两次 resume）一并移植（依赖 Task 2 的前缀机制对齐第二段）。
  5. `test_multiple_interrupts_functional`（`test_pregel.py:5710-5742`）——循环 `[1,2,3]`：double task + interrupt 交替；三次 resume "a"/"b"/"c" 后 `{"values":[2,"a",4,"b",6,"c"]}` 且 double 计数==3（**call counter 确定性**：同 task 循环调用各自结果正确；resume 值按序匹配依赖 Task 2 修复）。
  6. `test_multiple_interrupts_functional_cache`（`test_pregel.py:5745-5815`）——`Cache: &graph.CachePolicy{}` 的 double，输入 `[1,1,2,2,3,3]`：首轮+六次 resume 后计数==3；**新 ThreadID 全程重跑计数仍==3**（cache 跨线程命中）；`ClearCache` 后再跑计数==6。
  7. `test_task_before_interrupt_resume` + `test_multiple_tasks_before_interrupt_resume`（`test_pregel.py:5818-5904`）——Task 6.4 已覆盖前者；移植后者（step_a/step_b/ask 三 task，resume "continue" → `{"computed":12,"answer":"continue"}`，a/b 计数各==1）。
  8. `test_named_tasks_functional`（`test_pregel.py:6830-6879`）——Go 形态：`NewTask` 显式命名（`custom_foo`/`other_foo`/…），同名不同函数/同函数不同名串联调用结果链正确（`"foo|bar|baz|custom_baz|qux"` 式断言）；两个不同名 task 包装同一函数时 task ID 不同（白盒复算 `graph.FnTaskID` 不等）。
  9. `test_multiple_subgraphs_mixed_state_graph`（`test_pregel.py:6611-6676`）——**task 在 StateGraph 节点内调用**的 Go 形态：父 `graph.NewStateGraph()` 节点的 NodeFunc 内调用 `add.Invoke(ctx, ...)` / `multiply.Invoke(ctx, ...)`（entrypoint 无 checkpointer，父图带 MemorySaver）。断言中间步骤与 Python 一致：`call_same_subgraph` 节点内两次串联 `add.Invoke`——第一次 `add.Invoke([2,3])` 得 5，第二次 `add.Invoke([5,10])` 得 15——终态 `{"result": 15}`；`call_multiple_subgraphs` 形态 `{"add_result":5,"multiply_result":6}`。（文档化分歧：Python 允许 StateGraph 节点内裸调 @task，依赖 Pregel config 注入；Go 无等价物，节点内使用 task 须经 Entrypoint——doc.go 声明。）
  10. `test_imp_exception`（`test_pregel.py:8288-8318`）——entrypoint **捕获** task 错误继续执行：my_task 两次 + task_with_exception 一次（错误被捕获）→ 返回 "done"；同 ThreadID 第二次 Invoke 正常（新轮次重执行——计数器验证新轮次无误重放）。
  11. `test_entrypoint_output_schema_with_return_and_save`（`test_pregel.py:6755-6783`）——Python 断 jsonschema，Go 无 jsonschema 概念：**不移植**，在测试文件头部注释说明。
- [ ] **Step 2: Run `go test ./langgraph/fn/ -v`——若有失败，回到对应 Task 修实现（不允许改测试迁就实现，除非确认测试断言写错）。**
- [ ] **Step 3: Gate PASS**（`go build ./... && go vet ./... && go test ./...` + `go test -race ./langgraph/fn/` + `make test-sqlite` 回归）。
- [ ] **Step 4: Commit** — `git add langgraph/fn/functional_test.go && git commit -m "test(langgraph/fn): port Python functional API test suite"`.

---

### Task 8: 文档 + spec 标记

**Files:**
- Modify: `langgraph/fn/doc.go` — 完整包文档
- Modify: `docs/superpowers/specs/2026-08-08-langgraph-go-m5-m8-design.md` — M7 节标记完成（实际日期）

- [ ] **Step 1: doc.go 补齐文档化分歧清单**（逐条，最低集合）：①Timeout=ctx 取消+放弃等待（goroutine 不可强杀；Python sync task 同样不支持 timeout）；②interrupt 时已启动未完成 task 被取消（ctx cancel + dispatcher seal），已完成结果落 pending writes（Python 丢弃未完成 PUSH 任务，语义对应）；**正常返回时未 Get 的在途 task 同样被取消、其结果不持久化**；③无 checkpointer 时 hasPrev=false（Python None → 显式 bool）；④不支持 `store`（BaseStore 未移植）；⑤重放重抛的错误丢失具体类型（仅存 message；Python pickle 异常对象）；⑥StateGraph 节点内使用 task 的 Go 形态 = 节点内 Invoke 一个 Entrypoint（Python 的裸 @task-in-node 依赖 Pregel config 注入，无 Go 等价物）；⑦`Stream` 固定 updates 模式，task 级 chunk 不产生（task 在节点内执行，非图任务）；⑧确定性约束：重跑时同 entrypoint 的 task 调用顺序必须确定（同 Python determinism 一节），非确定性逻辑放进 task；⑨serde 契约：I/O/S 与 task 输入输出必须 JSON 可往返或在 serde 封闭注册表内，持久化 saver 下未注册类型报描述性错误而非静默降级；⑩cache key 输入包装为 `map[string]any{"input": in}`（Python key_func 收 *args/**kwargs）；⑪entrypoint 出错后同 ThreadID 的任何后续 Invoke（含换全新输入）都会重放重抛该错误——须换新 ThreadID 开新轮次（Python 用新输入即可逃离错误态，Go 的 gate 2 不看输入，不能）；⑫`CheckpointID` 钉住暂停 checkpoint + 新鲜输入的 time-travel fork 是新轮次：task 全部重跑、不命中该 checkpoint 的 pending writes（重放只在 `Resume` 恢复或 gate 2 错误重试时发生）。
- [ ] **Step 2: spec 的 M7 节标题后加一行**：`状态：已完成（2026-08-08，实施计划 docs/superpowers/plans/2026-08-08-langgraph-go-m7-functional-api.md）`——若 M8 已先行刷新 README，则同步确认 README 的 Supported/Not supported 条目与本实现一致（不一致只改条目归属，不重写 M8 文案）。
- [ ] **Step 3: Gate PASS；commit** — `git add langgraph/fn/doc.go docs/superpowers/specs/2026-08-08-langgraph-go-m5-m8-design.md && git commit -m "docs(langgraph): document fn package divergences; mark M7 done"`.

---

## Self-Review Notes

- **Spec 覆盖**：M7 节每条要求的落点——泛型显式构造 API（F1，Task 4/5）；Future 两态 + Get 阻塞（Task 3）；dispatcher ctx 注入 + 非法上下文 panic（Task 3/4，对齐 `graph/graph.go:1127-1128`）；确定性 task ID（F2/Task 1，扩展 `graph/taskid.go`）；per-run call counter（Task 4）；retry/cache/timeout 复用（Task 4，`graph.RetryPolicy`/`graph.CachePolicy`/`checkpoint.Cache`，命名空间 `__fn_writes/<name>`）；三保留 channel 单节点图 + 控制面过滤（F6/Task 5）；previous 注入 + 无 checkpointer hasPrev=false（Task 5）；Final value≠save（Task 5）；持久化闭环三点（F4/Task 6：dispatcher fn 层持有、运行开始 GetTuple 载重放、运行返回后按钉死时序 GetTuple+PutWrites 追加）；错误处理节（Task.Call panic、serde/类型断言描述性错误、task panic 转 `__error__`、entrypoint panic 传播）；测试移植清单（Task 7，用户指定 11 项全覆盖）。不支持 store、Timeout 语义等分歧（Task 8 doc.go）。
- **与 spec 的两处细化**（不改语义，实现层钉死）：①F4 的 cpID/step **追加时重盖章**——Go 执行器 interrupt 会新建暂停 checkpoint（`graph/graph.go:913`），运行前不可能知道结果落点 ID，重盖章使重放查询（以恢复 checkpoint 自身 ID/Step 计算）与持久化写入恒等，链式暂停逐环一致；②重放 gate 增加 `Source=="input"` 分支以覆盖错误重试（Python 靠 invoke(None) 重载同一 checkpoint 达成，Go 的 input checkpoint 在出错后仍是 latest）——其"新输入也重抛"的推论已升级为文档化分歧⑪并配 Task 6.3 断言。
- **类型一致性**：Task 1 产出 `graph.FnTaskID`/`Resolved`/`BackoffDelay`/`ReservedReturn` 被 Task 3/4/6 按同名同签名消费；Task 2 产出 `ReservedResume` 与包内 `resumeValuesFor` 新签名（graph 包内闭环，无包外调用者），Task 6.4/7.4/7.5 的链式暂停与顺序 interrupt 用例行为依赖它；Task 3 的 `dispatcher`/`loadReplay`/`record`/`seal`/`snapshotResults` 被 Task 4/6 消费（`seal`/`snapshotResults` 支撑 Task 6 的收尾时序②③）；`checkpoint.Cache.Get/Set/Clear`（`checkpoint/checkpoint.go:147-156`）与 `fnCacheNS` 字符串拼接匹配；`Saver.PutWrites` 每条 result 单发（batch 级单 taskID 盖章约束，`memory.go:121-125`）。M5 若已落地 PutWrites 五参签名（+taskPath）与 Write.TaskPath 字段，Task 6 按 F4 注记适配。
- **风险**：①非确定性控制流重放对错值——只能文档约束（同 Python）；②错误重放缺失会静默重跑失败任务——Task 6.3 专测；③task 内 interrupt 依赖 task goroutine 的 ctx 携带 graph 包的 interrupt 状态——Task 7.4 验证，并发 task 同时 interrupt 存在 `taskInterruptState` 计数竞争的理论窗口（Get 顺序消费的模式下不可达，doc.go 注明 interrupt 须按 Get 顺序确定性出现）；④serde 未注册类型——断言失败给描述性错误（不静默）。
- **Review history**：首轮对抗性审阅返回 1 blocker + 2 major + 5 minor + 1 nit，全部修复——B1（链式双暂停 resume 值错位）新增 Task 2 移植 RESUME-write 机制（F7：checkpoint.ReservedResume、`persistInterruptAndResume`、`resumeValuesFor` 前缀重建、sqlite `reservedWriteIdx` 补 -4），Task 6.4/7.5 保持原写法（修复后应通过）；M1（gate 2 毒化线程）升级为文档化分歧⑪ + Task 6.3 新输入重抛断言；M2（persist 竞态/丢失窗口）F4 钉死 cancel→seal→快照→persist 时序 + doc.go 分歧②补句；m1 根级 taskPath 无前导 `/`（`"a@0"`，与 Task 4 Step 1 用例 5 断言对齐）；m2 统一命名 `replayedCall`；m3 `loadReplay` gate 1 补 `len(Next) > 0`；m4 doc.go 分歧⑫（CheckpointID 钉暂停点 fork 全重跑）；m5 F4 注记补 Write.TaskPath 句；nit Task 7.9 补中间步骤说明（2+3=5、5+10=15）。
