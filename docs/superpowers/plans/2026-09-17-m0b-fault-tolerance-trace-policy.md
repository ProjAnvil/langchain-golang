# M0b 容错合规 parity Implementation Plan（发布 v0.8.1）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 node-level error handler（langgraph 1.2.0 容错三件套的第三件，上游完整语义）与 TracePolicy（langgraph 1.2.11 / langchain 1.3.15），以 **v0.8.1** 发布（用户指定，替代原定的 v0.9.0；后续里程碑版本号相应前移）。

**Architecture:** error handler 走「runTask 失败分支 → 仅在有 handler 时包成 `*NodeError` → execute 闭包先持久化 ERROR write（复用已就绪的 `checkpoint.ReservedError` 槽位）再以同 superstep 延续身份运行 handler → resume 分类把『只有 ERROR write 的任务』判为 handler 重跑」；TracePolicy 走「emit 侧 payload 变换」——新增通用 `callbacks.NewPayloadPolicyManager` 包装器，在节点 taskCtx 与 agent middleware 洋葱层安装，保证任何 tracer（langsmith/console）看到的都是变换后载荷，processor panic 时 fail-closed 丢弃载荷。

**Tech Stack:** Go 1.26；无新依赖。测试用既有 fake model / NewMemorySaver / httptest 模式。

**Spec:** `docs/superpowers/specs/2026-09-16-parity-catchup-p0p1-design.md` §6（M0b）；事实基线：2026-09-17 代码勘察报告（行号引用见各任务）。

## Global Constraints

- 零现有导出接口改动：`NodePolicies` 加字段可以（struct 加字段非 breaking）；不改任何既有函数签名
- 无 handler 的节点行为**逐字节不变**（错误传播路径零变化——`*NodeError` 包装仅发生在配置了 handler 的节点）
- 代码注释英文 + Python parity 出处；commit 用 conventional 前缀
- 全程 modern-go 规范（实施 subagent 须遵守：`for i := range n`、`errors.AsType`、`slices`/`maps` helpers、`t.Context()` 等）
- 完成后门禁：`go test ./...` 全绿 + `make test-sqlite test-redis test-postgres` 全绿 + golangci-lint（预编译 v2.13.2，`/tmp/gcl-bin/golangci-lint`）0 issues
- 发布：v0.8.1 + 嵌套 pin v0.8.1 + 嵌套 tag v0.3.4（push 与 gh release 已获用户授权）
- 行号基于 2026-09-17 勘察（main=019455f），实施时以实际代码为准（允许 ±20 行漂移，语义锚点为准）

---

### Task 1: NodeError + ErrorHandlerPolicy + 运行期集成

**Files:**
- Modify: `langgraph/graph/policy.go`（NodePolicies :185-193 加字段；文件尾部加两个新类型）
- Modify: `langgraph/graph/graph.go`（runTask :2022-2050 失败分支；execute 闭包 :1461-1478；task 结构体定义处加字段）
- Create: `langgraph/graph/error_handler_test.go`

**Interfaces:**
- Produces（后续任务与用户依赖的确切签名）:

```go
// langgraph/graph/policy.go

// ErrorHandlerPolicy installs a recovery handler that runs after a node's
// retries are exhausted (or immediately on failure when no RetryPolicy is
// installed), mirroring Python langgraph 1.2.0's add_node(error_handler=).
type ErrorHandlerPolicy struct {
	// Handler receives the current state snapshot and a typed NodeError.
	// It may return a plain map[string]any state update or a *types.Command
	// (normalized exactly like a node result), or an error, which wraps and
	// propagates in place of the original failure.
	Handler func(state map[string]any, err *NodeError) (any, error)
}

// NodeError is the typed failure handed to an ErrorHandlerPolicy.Handler.
type NodeError struct {
	// Node is the failing node's name.
	Node string
	// Attempt is the 1-based attempt count that exhausted the retry budget.
	// It is 0 when reconstructed on resume (the persisted write carries the
	// message only).
	Attempt int
	// Err is the original error.
	Err error
}

func (e *NodeError) Error() string // "node %q attempt %d: %v" — Attempt 为 0 时省略 attempt 段
func (e *NodeError) Unwrap() error
```

`NodePolicies` 加字段（注释说明 install 路径仅 per-node；刻意不提供 graph 级默认以杜绝全局递归）：

```go
	// ErrorHandler runs after retries are exhausted (see ErrorHandlerPolicy).
	// Deliberately per-node only: there is no compile-time default handler,
	// and a handler's own failure never re-enters any handler (Python's
	// recursion guard).
	ErrorHandler *ErrorHandlerPolicy
```

task 结构体（graph.go 内的私有 struct task）加两个字段：

```go
	// runErrorHandler marks a resumed task whose node failed with an
	// persisted ERROR write: the executor runs the node's error handler
	// instead of the node function.
	runErrorHandler bool
	// resumeErrMsg carries the persisted error message for a
	// runErrorHandler task (the reconstructed NodeError's Err).
	resumeErrMsg string
```

- [ ] **Step 1: 写失败测试** `langgraph/graph/error_handler_test.go`（本任务先写运行期 6 个；resume 相关在 Task 2）

```go
package graph

// 覆盖矩阵（spec §6.1）：
// 1. TestErrorHandlerRunsAfterRetriesExhausted：Retry{MaxAttempts:2} + 节点恒败
//    → handler 恰在 attempt==2 后收到 NodeError{Node:"node", Attempt:2}；state 快照可读（预置 state key）
// 2. TestErrorHandlerNoRetryRunsImmediately：无 Retry 策略 + 一次失败 → handler 立即执行，Attempt==1
// 3. TestErrorHandlerPlainUpdateBecomesOutcome：handler 返回 map[string]any{"recovered": true}
//    → run 结果 Values 含 "recovered"（同 superstep 生效，图正常走 END）
// 4. TestErrorHandlerCommandRoutes：二节点图 a→b，a 失败，handler 返回
//    &types.Command{Update: {"a_ok": true}, Goto: []any{"b"}}（跳过静态边直接路由 b）→ 断言 b 执行且 a_ok 在 state
//    （构造 Command 需绕过 ParentGraph 校验：Goto 用节点名，Graph 留空）
// 5. TestErrorHandlerFailurePropagates：handler 返回错误 → Invoke 返回该错误且 errors.Is 能同时匹配
//    原始错误与 handler 错误（要求传播错误用 fmt.Errorf("...: %w", 原始) 包装 handler 错时含两者——
//    具体包装格式：graph: error handler for node %q: %w（handler err），并保留对原始 err 的链：用 errors.Join(handlerErr, nodeErr.Err)）
// 6. TestErrorHandlerTimeoutTriggersHandler：NodePolicies{Timeout:{RunTimeout: 50ms}, ErrorHandler}
//    + 节点 select <-rt.Done() 返回 rt.Err() → handler 收到的 NodeError.Err 满足
//    errors.Is(nerr.Err, context.DeadlineExceeded)
// 用例 1/2/3/5/6 用 compileRetryGraph 同款单节点图 helper（policy_test.go:37）；
// 用例 4 需新 helper compileTwoNodeGraph。失败节点用 var fails atomic.Int32 计数。
```

- [ ] **Step 2:** `go test ./langgraph/graph/ -run TestErrorHandler -v` → 全部 FAIL（编译错或行为缺失）

- [ ] **Step 3: 实现**

(a) `policy.go`：加上述两个类型与 NodePolicies 字段。

(b) `graph.go` runTask 失败分支（:2040 `return nil, nil, nil, nil, rerr`）改为——仅当该节点配置了 ErrorHandler 时把错误包成 `*NodeError{Node: t.node, Attempt: attempt, Err: rerr}` 返回，否则原样返回（零行为变化）：

```go
		if retry == nil || attempt >= retry.MaxAttempts || !retry.RetryOn(rerr) {
			if g.errorHandlerPolicy(t.node) != nil {
				rerr = &NodeError{Node: t.node, Attempt: attempt, Err: rerr}
			}
			return nil, nil, nil, nil, rerr
		}
```

新增 helper（graph.go，与 policies 访问同处）：

```go
func (g *CompiledGraph) errorHandlerPolicy(node string) *ErrorHandlerPolicy {
	if policies, ok := g.policies[node]; ok {
		return policies.ErrorHandler
	}
	return nil
}
```

(c) execute 闭包（:1478 `outcomes[i] = outcome{...}` 之前）：runTask 返回 err 后的恢复路径 + resume 的 handler 重跑路径：

```go
			update, cmd, interrupts, consumed, err := g.runTask(taskCtx, t, state, resumeValues[t.id])
			if t.runErrorHandler {
				// Resumed handler task: run the handler with the persisted
				// message instead of the node function (crash recovery —
				// Python re-executes the handler, not the node).
				ne := &NodeError{Node: t.node, Err: errors.New(t.resumeErrMsg)}
				result, herr := g.errorHandlerPolicy(t.node).Handler(state, ne)
				update, cmd, err = normalizeErrorHandlerResult(result, herr, ne)
			} else if ne, ok := errors.AsType[*NodeError](err); ok {
				// Persist the ERROR write BEFORE running the handler, so a
				// crash mid-handler resumes into a handler re-run (mirrors
				// Python's committed task error write; ReservedError slots
				// already exist in every saver).
				if werr := cpSink.putWrites(ctx, *currentCfg, []checkpoint.Write{
					{Channel: checkpoint.ReservedError, Value: ne.Err.Error()},
				}, t.plannedID(*currentCfg, rs.step+1)); werr != nil {
					err = fmt.Errorf("graph: persisting error write for thread %q: %w", opts.ThreadID, werr)
				} else {
					result, herr := g.errorHandlerPolicy(t.node).Handler(state, ne)
					update, cmd, err = normalizeErrorHandlerResult(result, herr, ne)
				}
			}
```

注意：execute 闭包内 `ctx`/`cpSink`/`currentCfg`/`opts`/`rs` 等变量的实际名以闭包现场为准（勘察显示 cpSink/currentCfg/opts 均在闭包可达域；putWrites 调用形态参照 :1735-1748 的既有用法）。新增归一化 helper：

```go
// normalizeErrorHandlerResult applies node-result normalization to a
// handler's return, wrapping handler failures so both the handler error and
// the original NodeError stay matchable.
func normalizeErrorHandlerResult(result any, herr error, ne *NodeError) (map[string]any, *types.Command, error) {
	if herr != nil {
		return nil, nil, fmt.Errorf("graph: error handler for node %q: %w (original: %w)", ne.Node, herr, ne.Err)
	}
	return normalizeNodeResult(result)
}
```

- [ ] **Step 4:** `go test ./langgraph/graph/ -run TestErrorHandler -v` 全 PASS；`go test ./langgraph/graph/` 全包回归 PASS
- [ ] **Step 5:** `git commit -m "feat(langgraph): node-level error handlers (exhausted-retry recovery, Command routing)"`

### Task 2: Resume 分类——handler 崩溃恢复与无 handler 的历史错误

**Files:**
- Modify: `langgraph/graph/resume.go`（planResume :305-337 的 write 分类 switch 与最终分类 switch）
- Create/扩展: `langgraph/graph/error_handler_test.go`（resume 用例）

**Interfaces:**
- Consumes: Task 1 的 `task.runErrorHandler` / `task.resumeErrMsg` / `errorHandlerPolicy`
- Produces: resume 语义——「任务只有 ERROR write 且节点配了 handler → 重跑 handler」「配了 handler 但图代码已移除 handler → resume 报存储的错误」

- [ ] **Step 1: 写失败测试**（追加到 error_handler_test.go；全部用 NewMemorySaver + ThreadID，进程内两次 Invoke 模拟崩溃恢复）

```
7. TestErrorHandlerResumeRerunsHandlerNotNode：节点恒败 + handler 记录 nerr.Err.Error() 到 state 后【人为让 handler 只在第二次调用时恢复】
   （第一次调用 handler 时返回错误 → 第一次 Invoke 失败；同一 ThreadID 第二次 Invoke）
   → 断言：节点函数总执行次数不变（1 次——resume 没有重跑节点）；第二次 Invoke 成功且 state 含恢复值；
     handler 第二次收到的 NodeError.Err.Error() 与第一次相同（来自持久化 write）
8. TestErrorHandlerCompletedWritesWinOverErrorWrite：handler 成功完成的线程，之后因【另一个节点 interrupt】resume
   → 断言带 ERROR write 的任务被 replay 而非重跑 handler（完成写优先）
   （构造：图 a(失败+handler 恢复) → b(interrupt 节点，WithInterrupts 或 HumanInput 模式任选现有机制)；第一次 Invoke 在 b 前中断；第二次 resume 完成）
9. TestResumeErrorWriteWithoutHandlerFails：构造历史 write 只能来自曾配 handler 的图——用同一 Saver 先跑带 handler 的图留下 ERROR write，
   再用【同名节点但无 handler】的图 Compile 后 resume 同 ThreadID → Invoke 返回错误，错误信息含持久化的错误消息
```

- [ ] **Step 2:** 运行确认 FAIL

- [ ] **Step 3: 实现** planResume（:305-337）：

(a) write 分类 switch 增加：

```go
			case checkpoint.ReservedError:
				if msg, ok := w.Value.(string); ok {
					taskErrMsg = msg
				}
```

（`taskErrMsg` 为该循环外对每个 pt 声明的局部变量，进入循环体前清零。）

(b) 最终分类 switch 在 `case len(update) > 0 || len(sends) > 0:`（replay，保持首位）之后、`default:` 之前插入：

```go
			case taskErrMsg != "":
				if g.errorHandlerPolicy(pt.Node) != nil {
					plan.tasks = append(plan.tasks, task{id: pt.ID, node: pt.Node, arg: pt.Arg,
						runErrorHandler: true, resumeErrMsg: taskErrMsg})
				} else {
					// The write was produced by a graph that had a handler;
					// this build removed it — surface the stored failure
					// instead of silently re-running the node.
					return plan, fmt.Errorf("graph: task %s failed (persisted error): %s", pt.ID, taskErrMsg)
				}
```

（planResume 现签名若不返回 error，改为返回 `(resumePlan, error)` 并在调用点 `resumeFromTuple` 透传；这是包内私有函数签名，允许改。`g` 的可达性：若 planResume 是自由函数，把 `errorHandlerPolicy` 的判定结果作为参数传入或改为方法——以现场代码为准，语义不变。）

- [ ] **Step 4:** `go test ./langgraph/graph/ -run 'TestErrorHandler|TestResume' -v` 全 PASS + 全包回归
- [ ] **Step 5:** `git commit -m "feat(langgraph): resume re-runs persisted error handlers (crash recovery semantics)"`

### Task 3: TracePolicy——emit 侧 payload 变换

**Files:**
- Create: `core/callbacks/payload_policy.go` + `core/callbacks/payload_policy_test.go`
- Create: `langgraph/graph/trace_policy.go`
- Modify: `langgraph/graph/policy.go`（NodePolicies 加 Trace 字段）
- Modify: `langgraph/graph/graph.go`（execute 闭包 taskCtx 构造处 :1463-1465 之后安装 policy manager）
- Create: `langgraph/graph/trace_policy_test.go`

**Interfaces:**
- Produces:

```go
// langgraph/graph/trace_policy.go

// TracePolicy controls what traced payloads a node contributes, mirroring
// langgraph 1.2.11's add_node(trace_policy=) (PR #8523): transforms are
// applied where events are EMITTED, before any tracer (LangSmith, console,
// future OTel) observes them. Processor panics fail CLOSED: the payload is
// dropped (deliberate divergence from upstream's fail-open, since the
// feature's motivation is PII/compliance — see DIVERGENCES.md).
type TracePolicy struct {
	// ProcessInputs transforms start-kind event payloads (inputs).
	ProcessInputs func(value any) any
	// ProcessOutputs transforms end-kind event payloads (outputs).
	ProcessOutputs func(value any) any
}

// OmitPayload drops a payload entirely; assign it to ProcessInputs or
// ProcessOutputs (Python's omit_payload helper equivalent).
func OmitPayload(any) any { return nil }
```

```go
// core/callbacks/payload_policy.go

// NewPayloadPolicyManager wraps parent so every event's Input/Output passes
// through the supplied transforms before handlers observe it. Derived
// managers (Child, WithMetadata) retain the policy.
func NewPayloadPolicyManager(parent Manager, processInputs, processOutputs func(any) any) Manager
```

实现要点：私有 wrapper struct 内嵌 `Manager`；重写 `Emit`（拷贝 event，`Input != nil && processInputs != nil` 时变换，`Output != nil && processOutputs != nil` 时变换；变换函数 panic 时 recover 并将对应字段置 nil——fail-closed）；重写 `Child(parentID)` 与 `WithMetadata(...)` 返回再包装的派生 manager。

`NodePolicies` 加字段：

```go
	// Trace controls traced payloads for events emitted within this node
	// (see TracePolicy).
	Trace *TracePolicy
```

节点侧安装（graph.go execute 闭包，`taskCtx := context.WithValue(em.nodeContext(...), plannedTaskIDKey{}, ...)` 之后）：

```go
			if tp := g.tracePolicy(t.node); tp != nil {
				if parent := callbacks.ManagerFromContext(taskCtx); parent != nil {
					taskCtx = callbacks.ContextWithManager(taskCtx,
						callbacks.NewPayloadPolicyManager(parent, tp.ProcessInputs, tp.ProcessOutputs))
				}
			}
```

（`tracePolicy` helper 仿 `errorHandlerPolicy`；`callbacks` 包已 import 于 graph 包——若未 import 则补。）

- [ ] **Step 1: 写失败测试**

```
core/callbacks/payload_policy_test.go：
- TestPayloadPolicyManagerTransformsInputOutput：手写 recordingHandler 收 Event；构造 Manager + 两个 handler（模拟 langsmith+console 双 tracer），
  经 NewPayloadPolicyManager 包装后 Emit(EventChainStart{Input: "secret"}) / Emit(EventChainEnd{Output: "out"})
  → 两个 handler 都收到变换后的值；原始值不可见于任何 handler
- TestPayloadPolicyManagerPanicFailsClosed：processInputs panic → event 仍送达但 Input 为 nil
- TestPayloadPolicyManagerChildRetainsPolicy：m.Child("x") 派生 manager 的 Emit 仍被变换
- TestPayloadPolicyManagerNilTransformsPassthrough：两个 nil → 事件原样

langgraph/graph/trace_policy_test.go：
- TestNodeTracePolicyAppliesToRunnablesInsideNode：图节点内调用一个经 runnables.WithCallbacks? 触发 chain 事件的 runnable
  （参照 core/runnables 现有测试对 startChainRun 的触发方式；或节点内直接从 ctx 取 callbacks.ManagerFromContext 后 Emit——若前者过重）
  → NodePolicies{Trace: &TracePolicy{ProcessInputs: OmitPayload, ProcessOutputs: OmitPayload}}
  → recordingHandler 收到的节点内事件 Input/Output 均为 nil，Node 名等元数据不变
```

- [ ] **Step 2:** FAIL → **Step 3:** 实现上述三处 → **Step 4:** PASS + 全包回归 → **Step 5:** `git commit -m "feat(langgraph): per-node TracePolicy (emit-side payload transforms, fail-closed)"`

### Task 4: Agent middleware 的 TracePolicyProvider 挂点

**Files:**
- Modify: `langchain/agents/middleware/types.go`（新增接口）
- Modify: `langchain/agents/create_agent.go`（洋葱组装 :1877-1909）
- Create: `langchain/agents/middleware/trace_policy_test.go`

**Interfaces:**
- Consumes: `callbacks.NewPayloadPolicyManager`（Task 3）
- Produces:

```go
// langchain/agents/middleware/types.go

// TracePolicyConfig mirrors langgraph's TracePolicy transforms without
// importing the graph package: a middleware providing it controls traced
// payloads for everything inside its model-call wrapping layer (langchain
// 1.3.15's middleware trace_policy, #38910).
type TracePolicyConfig struct {
	ProcessInputs  func(value any) any
	ProcessOutputs func(value any) any
}

// TracePolicyProvider is the optional hook middlewares implement to scrub
// traced payloads (PII redaction before any tracer observes them).
type TracePolicyProvider interface {
	TracePolicy() TracePolicyConfig
}
```

洋葱层接线：在 WrapModelCallHook / WrapModelCallResultHook 的 handler 包装处（create_agent.go:1877-1909），每遇到实现了 TracePolicyProvider 的 middleware，在调用 hook 前把 policy manager 装进 ctx（policy 作用于该 middleware 包住的整段——内层 middleware 与模型 emit 的事件都被变换；嵌套 middleware 的 policy 逐层复合，如洋葱）：

```go
			handler = func(c context.Context, r middleware.ModelRequest) (middleware.ModelResponse, error) {
				if tp, ok := mws[i].(TracePolicyProvider); ok {
					cfg := tp.TracePolicy()
					if cfg.ProcessInputs != nil || cfg.ProcessOutputs != nil {
						if parent := callbacks.ManagerFromContext(c); parent != nil {
							c = callbacks.ContextWithManager(c,
								callbacks.NewPayloadPolicyManager(parent, cfg.ProcessInputs, cfg.ProcessOutputs))
						}
					}
				}
				return hook.WrapModelCall(c, r, next)
			}
```

（WrapModelCallResultHook 分支同样处理；`callbacks` import 以 create_agent.go 现状为准。）

- [ ] **Step 1: 失败测试**（middleware 包）：

```
- TestMiddlewareTracePolicyScrubsModelPayloads：自定义 middleware 同时实现 WrapModelCallHook（直通 next）与 TracePolicyProvider（ProcessInputs: 脱敏 func 把 map 里 "ssn" 键值替换为 "[REDACTED]"）
  + fake model（经 runnables Config 携带 recording handler，模拟 tracer）
  → CreateAgent + Invoke → recording handler 收到的 chat start 事件 Input 中 ssn 字段已脱敏；无 policy 的对照 middleware 不影响
- TestNestedMiddlewarePoliciesCompose：外层 policy 替换整个 Input 为 "<omitted>"、内层脱敏 → 最终 handler 只看到 "<omitted>"（外层后变换）
```

- [ ] **Step 2:** FAIL → **Step 3:** 实现 → **Step 4:** PASS + `go test ./langchain/agents/...` 回归 → **Step 5:** `git commit -m "feat(agents): middleware TracePolicyProvider hook (per-middleware payload scrubbing)"`

### Task 5: 文档 + 发布 v0.8.1

**Files:**
- Modify: `CHANGELOG.md`（[Unreleased] → [0.8.1] - 2026-09-17）
- Modify: `DIVERGENCES.md`（TracePolicy fail-closed 条目由「planned」改为正式记录，含理由）
- Modify: `docs/superpowers/specs/2026-09-16-parity-catchup-p0p1-design.md` §4 版本表（M0b→v0.8.1；M1→v0.9.0、M2→v0.10.0、M3→v0.11.0）

CHANGELOG 0.8.1 段内容：

```markdown
## [0.8.1] - 2026-09-17

### Added
- langgraph: node-level error handlers (`NodePolicies.ErrorHandler`, mirroring langgraph 1.2.0 `error_handler=`): handlers receive the state snapshot and a typed `NodeError` after retries are exhausted and may return a plain update or a `Command`; the task error is persisted as a checkpoint ERROR write before the handler runs, so a crash mid-handler resumes into a handler re-run instead of the node.
- langgraph: per-node `TracePolicy` (`NodePolicies.Trace`, mirroring langgraph 1.2.11) — emit-side payload transforms applied before any tracer observes chain events; `OmitPayload` helper included. Processor panics fail closed (deliberate divergence, see DIVERGENCES.md).
- agents: middleware `TracePolicyProvider` hook (mirroring langchain 1.3.15) for per-middleware payload scrubbing.
```

- [ ] 发布序列（已授权）：`go test ./... && make test-sqlite test-redis test-postgres` 全绿 + lint 0 → commit → `git tag v0.8.1` → `git push origin main v0.8.1` → 三个嵌套 go.mod pin v0.8.1 + tidy + commit + `git tag langgraph/checkpoint/{sqlite,redis,postgres}/v0.3.4` + push → `gh release create v0.8.1`（模板：Highlights / Features / Docs / Full Changelog）→ 嵌套 v0.3.4 release ×3

---

## Self-Review 记录

- **Spec 覆盖**：§6.1→Task 1+2（含完整测试矩阵 9 用例：耗尽/state 可读/plain update/Command 路由/handler 失败传播/超时触发/resume 重跑 handler/完成写优先/无 handler 报错）；§6.2→Task 3+4（emit 侧、多 tracer 一致、fail-closed、OmitPayload helper、add_node 与 middleware 双挂点、PII 协同）；§6 的降级预案未触发（run loop 改动限定在 execute 闭包与 runTask 失败分支，成比例）；版本 v0.8.1→Task 5。
- **占位符**：无 TBD；两处「以现场代码为准」（闭包变量名、planResume 签名）是勘察行号漂移的显式适应指令，附有语义锚点。
- **类型一致性**：`NodeError`/`ErrorHandlerPolicy`/`TracePolicy`/`TracePolicyConfig`/`OmitPayload`/`NewPayloadPolicyManager`/`task.runErrorHandler`/`task.resumeErrMsg`/`errorHandlerPolicy`/`tracePolicy` 各任务引用一致。
- **风险**：① planResume 签名改动的调用点数量未知（勘察显示 resumeFromTuple 单点）——实施时 grep 全部调用；② ERROR write 与既有 interrupt 兄弟任务写的交互——用例 8 覆盖；③ execute 闭包内 putWrites 的 persist 语义与 :1735-1748 的 superstep 末持久化的叠加——用例 8 的 replay 优先级覆盖。
