# M4 核心语义修复（parity wave 2）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 2026-10-06 差距报告的 P1 六项核心语义修复（标准模型异常、interrupt response_schema、无效工具调用修复、HITL 精化、内部调用标记、delta channel 水合），对应上游 core #39538/#38887-adjacent、v1 #40530/#40463/#39773/#39252、langgraph #8886/#8548/#8538/#9141/#9142/#9165/#9170。

**Architecture:** 错误层级挂在既有 `core/lcerrors`（sentinels + `ProviderError` 载体，`errors.Is` 兼容）；HITL/无效调用修复都在 `create_agent` model node 与 HITL middleware 的既有钩子点上做增量；内部调用标记用 context token 在 `langgraph/graph/stream.go` 投影侧过滤（机制分歧，行为对齐 Python InternalCallTransformer）；delta 修复复用 `reconstructDeltaChannels` 的祖先遍历逻辑下沉为共享函数后接入 run 路径与 UpdateState。

**Tech Stack:** Go 1.26；无新依赖。测试用既有 fake model / NewMemorySaver / httptest 模式。

**Spec:** `docs/superpowers/specs/2026-10-06-parity-wave2-design.md`

## Global Constraints

- 零既有导出签名破坏：struct 加字段/新函数可以，不改既有函数签名（`NewProviderError` 参数不变，`ModelKind` 内部推导）
- 注释英文 + Python parity 出处（PR 号）；commit 用 conventional 前缀
- modern-go 规范：`for i := range n`、`errors.AsType`、`slices`/`maps` helpers、`t.Context()`
- 每 task：先写失败测试→确认失败→最小实现→全绿→commit；不 push、不打 tag
- 门禁：`go test ./...` 全绿 + `make test-sqlite test-redis test-postgres` + golangci-lint 0 issues + 受影响包覆盖率 ≥95%
- 行号基于 2026-10-06 勘察（HEAD=e8906bc），允许 ±20 行漂移，语义锚点为准
- Python 事实基线：`is_retryable` 仅 RateLimit/API(5xx)/Connection/Timeout 为 true；HITL 拒绝文案/编辑通知用 Python 原文逐字对齐

---

### Task 1: ModelError 层级 + 状态码分类（core/lcerrors）

**Files:**
- Modify: `core/lcerrors/errors.go`
- Modify: `core/lcerrors/errors_test.go`（如不存在则创建）

**Interfaces:**
- Produces（后续任务与用户依赖）:

```go
// ModelErrorKind classifies a provider failure, mirroring Python
// langchain-core 1.6 standard model exception types (#39538).
type ModelErrorKind string

const (
	ModelKindNone             ModelErrorKind = ""
	ModelKindAuth             ModelErrorKind = "authentication"    // 401
	ModelKindPermissionDenied ModelErrorKind = "permission_denied" // 403
	ModelKindInvalidRequest   ModelErrorKind = "invalid_request"   // 400/422
	ModelKindNotFound         ModelErrorKind = "not_found"         // 404
	ModelKindRateLimit        ModelErrorKind = "rate_limit"        // 429
	ModelKindServer           ModelErrorKind = "server_error"      // 5xx
	ModelKindConnection       ModelErrorKind = "connection"        // transport
	ModelKindTimeout          ModelErrorKind = "timeout"           // 408 / net timeout
	ModelKindContextOverflow  ModelErrorKind = "context_overflow"  // 400 + body marker
)

// ErrModelAuth/ErrModelPermissionDenied/ErrModelInvalidRequest/ErrModelNotFound/
// ErrModelServer/ErrModelConnection: sentinel errors (var ErrModelAuth = errors.New("lc: model authentication failed") etc.)
// ErrModelRateLimit = ErrRateLimited, ErrModelTimeout = ErrTimeout (alias, keep single sentinel identity)

// ProviderError gains: ModelKind ModelErrorKind
func (e *ProviderError) IsModelRetryable() bool // Python is_retryable parity
func ModelErrorKindForStatus(statusCode int, body string) ModelErrorKind // pure, exported for tests/partners
```

- [ ] **Step 1: 写失败测试**（`core/lcerrors/errors_test.go`）

```go
func TestModelErrorKindForStatus(t *testing.T) {
	cases := []struct {
		status int
		body   string
		want   ModelErrorKind
	}{
		{401, "", ModelKindAuth},
		{403, "", ModelKindPermissionDenied},
		{400, "", ModelKindInvalidRequest},
		{422, "", ModelKindInvalidRequest},
		{404, "", ModelKindNotFound},
		{429, "", ModelKindRateLimit},
		{500, "", ModelKindServer},
		{503, "", ModelKindServer},
		{400, `{"error":{"message":"This model's maximum context length is 4096 tokens..."}}`, ModelKindContextOverflow},
		{400, `{"error":{"code":"context_length_exceeded"}}`, ModelKindContextOverflow},
		{200, "", ModelKindNone},
	}
	for _, tc := range cases {
		if got := ModelErrorKindForStatus(tc.status, tc.body); got != tc.want {
			t.Errorf("status=%d body=%q: got %q want %q", tc.status, tc.body, got, tc.want)
		}
	}
}

func TestProviderErrorModelKindAndRetryable(t *testing.T) {
	pe := NewProviderError("openai", "/responses", 429, "", 0)
	if pe.ModelKind != ModelKindRateLimit || !pe.IsModelRetryable() {
		t.Fatalf("429: kind=%v retryable=%v", pe.ModelKind, pe.IsModelRetryable())
	}
	if !errors.Is(pe, ErrModelRateLimit) || !errors.Is(pe, ErrRateLimited) {
		t.Fatal("429 must satisfy ErrModelRateLimit and ErrRateLimited")
	}
	for _, status := range []int{401, 403, 400, 404} {
		pe := NewProviderError("openai", "/responses", status, "", 0)
		if pe.IsModelRetryable() {
			t.Errorf("status %d must not be retryable", status)
		}
	}
}

func TestProviderErrorSentinels(t *testing.T) {
	// errors.Is on each sentinel must be satisfied by the mapped status, and
	// unwrap must keep working for the legacy sentinels.
	if !errors.Is(NewProviderError("x", "", 401, "", 0), ErrModelAuth) { t.Fatal("401") }
	if !errors.Is(NewProviderError("x", "", 403, "", 0), ErrModelPermissionDenied) { t.Fatal("403") }
	if !errors.Is(NewProviderError("x", "", 404, "", 0), ErrModelNotFound) { t.Fatal("404") }
	if !errors.Is(NewProviderError("x", "", 500, "", 0), ErrModelServer) { t.Fatal("500") }
	if !errors.Is(NewProviderError("x", "", 408, "", 0), ErrModelTimeout) { t.Fatal("408") }
}
```

- [ ] **Step 2: 跑测试确认失败**：`go test ./core/lcerrors/ -run 'ModelError|ProviderError'` → 编译失败（未定义符号）
- [ ] **Step 3: 实现**：`errors.go` 加 `ModelErrorKind` 常量组、sentinels（`ErrModelRateLimit`/`ErrModelTimeout` 直接 `=` 既有 `ErrRateLimited`/`ErrTimeout` 别名）；`ProviderError` 加 `ModelKind` 字段；`NewProviderError` 内部 `ModelKind = ModelErrorKindForStatus(status, body)`，并对 `ModelKind != None` 时把 sentinel 串进 wrap 链（新增 `Sentinel error` 字段或包一层 `fmt.Errorf("%w", ...)` 链，确保 `errors.Is` 命中，同时保留 `Unwrap() error` 原语义——推荐：加 `sentinel error` 私有字段，`Is(target error) bool` 方法匹配 sentinel）；`IsModelRetryable` 按 kind ∈ {RateLimit, Server, Connection, Timeout} 返回 true（ContextOverflow/Auth/Permission/InvalidRequest/NotFound/None → false）；body 嗅探串列表：`context_length_exceeded`、`context window`、`maximum context length`、`prompt is too long`（小写包含匹配）。`classifyStatus` 改为基于 `ModelErrorKindForStatus`（保持既有返回值兼容：429→ErrRateLimited、408→ErrTimeout、其余非 2xx→ErrProvider 不变，但 ProviderError.ModelKind 已细化）
- [ ] **Step 4: 跑包全测**：`go test ./core/lcerrors/ ./core/httpclient/` 全绿（httpclient 既有 errors 测试回归）
- [ ] **Step 5: commit**：`feat(core/lcerrors): standard model error kinds + status classification (langchain-core #39538)`

### Task 2: retry 中间件 default_retry_on 语义（middleware）

**Files:**
- Modify: `langchain/agents/middleware/retry.go`
- Modify: `langchain/agents/middleware/model_retry.go`
- Modify: `langchain/agents/middleware/tool_retry.go`
- Test: `langchain/agents/middleware/model_retry_test.go`（追加）

**Interfaces:**
- Consumes: Task 1 的 `ProviderError.IsModelRetryable()`、`lcerrors.ProviderError`
- Produces:

```go
// retry.go
// DefaultRetryOn mirrors Python langchain 1.4 default_retry_on (#39538):
// classified model errors retry only when retryable; unclassified errors retry.
func DefaultRetryOn(err error) bool
```

- [ ] **Step 1: 失败测试**（model_retry_test.go 追加；fake sleep 模式沿用文件内既有测试）

```go
func TestModelRetryDefaultRetryOnClassifiedErrors(t *testing.T) {
	// A non-retryable classified error (401) must NOT be retried: handler
	// called exactly once, OnFailure=continue path taken.
	calls := 0
	mw, err := NewModelRetryMiddleware() // defaults now use DefaultRetryOn
	if err != nil { t.Fatal(err) }
	authErr := lcerrors.NewProviderError("openai", "/responses", 401, "", 0)
	out, rerr := mw.WrapModelCall(t.Context(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		calls++
		return ModelResponse{}, authErr
	})
	if calls != 1 { t.Fatalf("handler called %d times, want 1", calls) }
	if rerr != nil { t.Fatalf("OnFailure=continue must swallow: %v", rerr) }
	if len(out.Result) != 1 { t.Fatalf("expected failure AIMessage, got %+v", out.Result) }

	// A retryable classified error (429) must be retried MaxRetries times.
	calls = 0
	mw2, _ := NewModelRetryMiddleware(WithModelRetryMaxRetries(2), WithModelRetrySleep(func(time.Duration) {}))
	rerr = nil
	_, _ = mw2.WrapModelCall(t.Context(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		calls++
		return ModelResponse{}, lcerrors.NewProviderError("openai", "/responses", 429, "", 0)
	})
	if calls != 3 { t.Fatalf("handler called %d times, want 3 (1 + 2 retries)", calls) }

	// An unclassified error must keep retrying (Python: non-MLError -> True).
	calls = 0
	_, _ = mw2.WrapModelCall(t.Context(), ModelRequest{}, func(context.Context, ModelRequest) (ModelResponse, error) {
		calls++
		return ModelResponse{}, errors.New("boom")
	})
	if calls != 3 { t.Fatalf("unclassified: handler called %d times, want 3", calls) }
}
```

- [ ] **Step 2: 确认失败**：`go test ./langchain/agents/middleware/ -run TestModelRetryDefaultRetryOn` → FAIL（401 也被重试）
- [ ] **Step 3: 实现**：`retry.go` 加 `DefaultRetryOn`（`var pe *lcerrors.ProviderError; if errors.AsType(&pe) { return pe.IsModelRetryable() }; return true`）；`NewModelRetryMiddleware`/`NewToolRetryMiddleware` 的 `RetryOn` 默认值从 nil（全重试）改为 `DefaultRetryOn`；两个 struct 的字段注释与 `WithModelRetryOn` 文档同步更新；tool_retry 同样处理
- [ ] **Step 4: 全测**：`go test ./langchain/agents/...` 全绿（既有测试若有依赖「全重试默认」的用例，按新语义修正断言并在 commit message 注明）
- [ ] **Step 5: commit**：`feat(agents): default_retry_on semantics for model/tool retry middleware (langchain #39538)`

### Task 3: gemini 错误映射进层级（partners/gemini）

**Files:**
- Modify: `partners/gemini/chatmodel.go`（Invoke/Stream 错误返回处）
- Test: `partners/gemini/errors_test.go`

**Interfaces:**
- Consumes: Task 1 `ModelErrorKindForStatus`、`NewProviderError`
- Produces: gemini 错误满足 `errors.Is(err, lcerrors.ErrModelServer)` 等（HTTP 语义尽可能从 genai APIError 提取 status/body；无法提取时包 `ErrModelConnection` 兜底）

- [ ] **Step 1: 失败测试**：构造 genai 错误（或 httpStatusError 等价物）→ 断言 `errors.Is(err, lcerrors.ErrModelRateLimit)` 等
- [ ] **Step 2: 确认失败 → Step 3: 实现**（genai 错误类型断言提取 code/status → `lcerrors.NewProviderError("gemini", "generateContent", status, msg, 0)` 包装，原错误进 wrap 链）→ **Step 4: `go test ./partners/gemini/`**（既有 100% 覆盖不回落）→ **Step 5: commit**：`feat(gemini): map genai errors onto standard model error kinds`

### Task 4: interrupt response_schema（langgraph）

**Files:**
- Modify: `langgraph/types/types.go`（Interrupt 结构体）
- Modify: `langgraph/graph/graph.go`（Interrupt helper :2480 区域）
- Modify: `langgraph/graph/resume.go`（interruptsFromWrites 序列化）
- Test: `langgraph/graph/interrupt_schema_test.go`（新）

**Interfaces:**
- Produces:

```go
// types.go — Interrupt gains:
	// ResponseSchema is the JSON Schema of the value expected when resuming
	// this interrupt, if the node provided one (Python langgraph 1.2.13
	// interrupt(response_schema=...), #8886). nil when unset.
	ResponseSchema map[string]any `json:"response_schema,omitempty"`

// graph.go
// InterruptWithSchema behaves like Interrupt but records response_schema on
// the surfaced interrupt (#8886).
func InterruptWithSchema(ctx context.Context, value any, responseSchema map[string]any) any
```

- [ ] **Step 1: 失败测试**：节点内 `InterruptWithSchema(ctx, "pick one", map[string]any{"type":"object","properties":{"choice":{"type":"string"}},"required":["choice"]})` → invoke 得到 `Interrupts[0].ResponseSchema != nil`；GetState 快照同样携带；普通 `Interrupt()` 的 ResponseSchema 为 nil
- [ ] **Step 2: 确认失败**（字段不存在编译失败即视为红）
- [ ] **Step 3: 实现**：结构体加字段；`InterruptWithSchema` 复用 Interrupt 的 panic/resume 机制（把 schema 存进 interruptCtx state 或直接在 mint 的 Interrupt 上赋值——按现有实现把 st.counter/ID 路径抽出共享）；`interruptsFromWrites` 反序列化时带出该字段（确认 reserved write 的 JSON 往返）
- [ ] **Step 4: `go test ./langgraph/...`** 全绿 → **Step 5: commit**：`feat(langgraph): interrupt response_schema (#8886)`

### Task 5: create_agent 无效工具调用修复（#40530）

**Files:**
- Modify: `langchain/agents/create_agent.go`（buildModelNode :2009/:2056 区域，响应落 update 前）
- Test: `langchain/agents/invalid_tool_calls_test.go`（新）

**Interfaces:**
- Consumes: `messages.Message.InvalidToolCalls []ToolCall`（messages.go:66，元素含 Name/ID；错误串在 content-block 形态才有，此处用 Python 固定文案）
- Produces: 未应答 invalid call → 追加 `messages.ToolMessage{ToolCallID, Name, Content: "Tool call {name} with id {id} could not be executed - arguments were malformed or truncated.", ResponseMetadata: {"status": "error"}}` 进 `update["messages"]`

- [ ] **Step 1: 失败测试**：fake model 返回 AIMessage{InvalidToolCalls: [{ID:"bad1", Name:"search"}]}（无有效 ToolCalls）→ invoke 后 state messages 里存在 ToolMessage{ToolCallID:"bad1", status:error}；再次 invoke 时历史完整（AI invalid + error ToolMessage）；已应答的（历史里已有 ToolMessage 同 ID）不重复追加；invalid 无 ID 的跳过
- [ ] **Step 2: 确认失败** → **Step 3: 实现**：model node 构造 update 处，对 `resp.Result` 里每条 AIMessage 的 InvalidToolCalls 生成未应答集合的 error ToolMessage（回答集合 = state 既有 + 本次新增 ToolMessage 的 ToolCallID）追加到 newMessages 尾部 → **Step 4: `go test ./langchain/...`**（含既有 fake-model 全链路回归）→ **Step 5: commit**：`fix(agents): repair invalid tool calls with error ToolMessages (#40530)`

### Task 6: HITL 拒绝理由成帧（#39773）

**Files:**
- Modify: `langchain/agents/middleware/human_in_the_loop.go`（processHumanDecision 拒绝分支 :514 区域）
- Test: 既有 `human_in_the_loop_test.go` 追加

- [ ] **Step 1: 失败测试**：Decision{Type: Reject, Message: "Custom response message"} → ToolMessage content == "User rejected the tool call for `test_tool` with reason: Custom response message"；无 Message → 既有默认文案不变
- [ ] **Step 2: 确认失败** → **Step 3: 实现**：有 Message 时 content = fmt.Sprintf("User rejected the tool call for `%s` with reason: %s", name, message)（去掉 id 段，Python #39773 语义）→ **Step 4: 全测** → **Step 5: commit**：`fix(agents/middleware): frame custom HITL rejection reasons (#39773)`

### Task 7: HITL 编辑通知 + return_direct 以执行名为准（#40463）

**Files:**
- Modify: `langchain/agents/middleware/human_in_the_loop.go`（AfterModel 决策处理 + 新增 WrapModelCall 无关的 WrapToolCall 实现 + `EditNotice *string` 字段与 `WithEditNotice` HITL 选项）
- Modify: `langchain/agents/create_agent.go`（return_direct 路由 :1145-1169 若以模型原始调用名解析则改为执行名）
- Test: 既有 HITL 测试文件追加

**Interfaces:**
- Consumes: `ToolCallRequest.State`（types.go:243）、`processHumanDecision` 保留原 call.ID 的行为
- Produces:

```go
// human_in_the_loop.go
const editedToolCallsStateKey = "hitl_edited_tool_calls" // state: map[string]ToolCall (editedID -> action)

// EditNotice, when non-nil, is prepended to the ToolMessage content of a tool
// call a reviewer replaced via an edit decision. nil = default notice text;
// WithEditNotice("") disables. Python #40463.
EditNotice *string

// HITLOption
func WithEditNotice(s string) HITLOption // &s, "" disables
```

- [ ] **Step 1: 失败测试**：(a) edit 决策后工具执行，ToolMessage content 以默认通知开头（Python 原文 "Note: a human reviewer replaced this tool call before it ran. ..."）；(b) `WithEditNotice("")` 关闭；(c) 自定义通知文本生效；(d) edit 决策后 state 含 `hitl_edited_tool_calls[origID]`；(e) return_direct：模型调用 tool A（非 return_direct），编辑为 tool B（return_direct）→ 该回合结束路由按 B 判定
- [ ] **Step 2: 确认失败** → **Step 3: 实现**：AfterModel 处理 edit 时写 state key；middleware 实现 `WrapToolCall(ctx, req, handler)`：`req.State[editedToolCallsStateKey]` 命中 `req.ToolCall.ID` 时 handler 结果为 ToolMessage 则 content = notice + "\n\n" + 原 content（仅 ToolMessage，其他形态原样）；return_direct 路由审计后按执行 ToolMessage 的 name 解析 → **Step 4: 全测 + HITL 既有 cross-process 测试回归** → **Step 5: commit**：`feat(agents/middleware): HITL edit notice + executed-name return_direct routing (#40463)`

### Task 8: 内部调用标记 + messages 投影过滤（#39252，机制分歧）

**Files:**
- Create: `core/callbacks/internal_call.go`
- Modify: `langgraph/graph/stream.go`（StreamMessages 投影 :307 区域 + StreamEvents 的 chat-model 消息事件路径）
- Test: `core/callbacks/internal_call_test.go`、`langgraph/graph/stream_internal_call_test.go`

**Interfaces:**
- Produces:

```go
// core/callbacks/internal_call.go
// WithInternalCall marks a context as carrying a middleware-internal model
// call. Events emitted under this context are excluded from messages-mode
// stream projections and message stream events (Python langchain 1.4
// internal_call_metadata/InternalCallTransformer, #39252; Go uses a
// context token instead of metadata dicts — unspoofable, DIVERGENCES.md).
func WithInternalCall(ctx context.Context) context.Context
func IsInternalCall(ctx context.Context) bool
```

- [ ] **Step 1: 失败测试**：graph 两节点：节点 A 以 `WithInternalCall(ctx)` 触发 fake model（经 callbacks 发 chat_model 事件），节点 B 正常触发 → StreamMessages 只见 B 的 chunk；StreamEvents 的 message 事件只有 B；`IsInternalCall(WithInternalCall(bg))` == true、`IsInternalCall(bg)` == false；values/updates 模式不受影响
- [ ] **Step 2: 确认失败**（A 的事件泄漏进投影）→ **Step 3: 实现**：`internal.go` unexported struct key；stream.go 投影入口处 `if callbacks.IsInternalCall(event ctx/run ctx) { skip }`（事件携带的 ctx 来源按现实现定位——从 emit 侧闭包捕获的 ctx 判断）→ **Step 4: 全测** → **Step 5: commit**：`feat(callbacks,langgraph): internal-call marking filters messages projections (#39252)`

### Task 9: delta run 路径水合 + UpdateState 先重建（#8548/#9170/#9165/#9142）

**Files:**
- Modify: `langgraph/graph/snapshot.go`（`reconstructDeltaChannels` :307 下沉出可复用的 `hydrateDeltaChannels(ctx, saver, tup, rs)`；UpdateState :164 applyWrites 前调用）
- Modify: `langgraph/graph/graph.go`（new-turn :1246 restore 后水合）
- Modify: `langgraph/graph/resume.go`（resume :228 restore 后水合）
- Test: `langgraph/graph/delta_hydration_test.go`（新）

**Interfaces:**
- Produces（包内私有，供三处调用）：

```go
// hydrateDeltaChannels reconstructs sentinel delta channels into rs from the
// ancestor history of tup (shared by GetState-read, UpdateState, new-turn and
// resume paths). Channels absent from history remain absent.
func (g *CompiledGraph) hydrateDeltaChannels(ctx context.Context, tup *checkpoint.Tuple, rs *runState)
```

- [ ] **Step 1: 失败测试（先移植上游场景）**：(a) freq=10 图，turn1 积累 [1,10]，新 turn2 invoke（nil/新输入）→ turn2 内节点读到 [1,10,...] 而非从空开始；turn2 触发快照时 blob 含完整累积（不是缩减值）；(b) resume 同理（interrupt 后 resume 继续累积）；(c) UpdateState 在 delta 通道 sentinel 状态的 checkpoint 上写其它 key → 新 checkpoint 里 delta 通道值 = 原累积（不是缺失）；(d) UpdateState 直接写 delta key → 强制快照 = 祖先累积 + 更新值
- [ ] **Step 2: 确认失败**（现状 (a) 读到空、(c) 通道缺失、(d) 只含更新值）→ **Step 3: 实现**：抽出共享水合函数（基于 reconstructDeltaChannels 逻辑参数化 protos/saver 来源），三处接入；`IsAvailable()` 语义保持（不存在则不落 ChannelValues）→ **Step 4: 全 delta 测试回归**（delta_integration/delta_pertask/save_checkpoint/snapshot_errors/durability 五个文件——`delta_pertask_test.go` 中断言旧行为 `[2,10]` 的用例改为新语义 `[1,10,2,10]`，commit message 注明行为对齐）→ **Step 5: commit**：`fix(langgraph): hydrate delta channels on run/resume/update paths (#8548 #9170 #9165 #9142)`

### Task 10: never-written delta 通道短路（#9141）

**Files:**
- Modify: `langgraph/graph/snapshot.go`（reconstructDeltaChannels walk 循环）
- Test: `langgraph/graph/snapshot_errors_test.go` 追加

- [ ] **Step 1: 失败测试**：注入 counting saver（包装 GetTuple 计数）：含 never-written delta 通道的图 GetState → walk 不为该 key 消耗额外 GetTuple（短路后总调用数 = written keys 所需 + 常数）；行为不变（值仍缺失）
- [ ] **Step 2: 确认失败**（现状全链走完）→ **Step 3: 实现**：walk 前用 `tup.Checkpoint.ChannelVersions`（或 `VersionsSince` 等价物）过滤掉从未有版本的 key（首 checkpoint 也没有即断定 never-written；保守起见仅当祖先链首个 checkpoint 的 ChannelValues/写集均无该 key 时剪枝）→ **Step 4: 全测** → **Step 5: commit**：`perf(langgraph): short-circuit never-written delta channels in history walk (#9141)`

### Task 11: delta 计数器缺口（F5）

**Files:**
- Modify: `langgraph/graph/graph.go`（advanceDeltaCounters :2017 迭代源从 `rs.channels` 改为 `rs.channels ∪ channelProtos 中 delta keys`）
- Modify: `langgraph/graph/checkpoint_sink.go`（flushExit :356 区域去重计数）
- Test: `langgraph/graph/save_checkpoint_test.go` 追加

- [ ] **Step 1: 失败测试**：(a) 注册但从未 materialize 的 delta 通道，resume 后 metadata.CountersSinceDeltaSnapshot 的 superstep 计仍推进（5000 界可触发）；(b) exit 模式 pause→resume 后 superstep 计数 == 实际 superstep 数（非双倍）
- [ ] **Step 2: 确认失败** → **Step 3: 实现**：advance 迭代并集；flushExit 只在最终持久化 checkpoint 时 advance（中间 saveCheckpoint 调用跳过或回滚——按现实现选最小改动：给 saveCheckpoint 加 `advanceCounters bool` 私参路径或把 exit 路径的中间调用改走不计数变体）→ **Step 4: `make test-sqlite test-redis test-postgres` + 全测** → **Step 5: commit**：`fix(langgraph): delta counter advance over registered protos; exit-mode double count`

### Task 12: 子图 delta 读路径（#8538）— 审计优先

**Files:**
- Modify: `langgraph/graph/subgraph.go` / `snapshot.go`（如修）
- Test: `langgraph/graph/subgraph_delta_test.go`（新，先证明问题）

- [ ] **Step 1: 写父图 GetState(childNS) 场景测试**：子图 sentinel delta 通道经父图读取 → 现状不重建。若测试证明子图自身 `child.GetState` 已可重建且父图路径文档化为「用子图实例读」（查现有 subgraph 测试惯例），则改为 DIVERGENCES 记录 + 测试固化该边界，本任务结束（commit `docs(divergences): ...`）
- [ ] **Step 2（仅当修）**：NS 前缀登记表（父图编译时登记子图 compiled 引用）→ GetState 按 NS 解析 protos/saver 水合 → 全测 → commit `fix(langgraph): resolve subgraph saver/protos for delta reads (#8538)`

### Task 13: #9103 防御性测试（不实现机制）

**Files:**
- Test: `langgraph/graph/snapshot_test.go` 追加

- [ ] **Step 1**：interrupt 后 resume 完成 → GetState 的 Interrupts 为空（无已应答残留）；interrupt-mode HITL resume 后同理。现状若已绿则固化，红则按 Go resume 语义修复 → **Step 2: commit**：`test(langgraph): answered interrupts not surfaced by GetState (#9103)`

### Task 14: 收尾——CHANGELOG / DIVERGENCES / 报告更新 + 全量门禁

**Files:**
- Modify: `CHANGELOG.md`（Unreleased：逐项记账，引用上游 PR 号）
- Modify: `DIVERGENCES.md`（E 机制分歧；Task 12 若走文档路线）
- Modify: `../reports/12-latest-release-port-gap.md`（§3.1/§3.2/§3.3/§3.4/§3.5/§3.6 状态更新）

- [ ] **Step 1**: `go build ./... && go vet ./...`
- [ ] **Step 2**: `go test ./...`（root）全绿
- [ ] **Step 3**: `make test-sqlite test-redis test-postgres` 全绿
- [ ] **Step 4**: golangci-lint 0 issues（`/tmp/gcl-bin/golangci-lint run ./...` 或仓库 Makefile 目标）
- [ ] **Step 5**: `make coverage` 受影响包 ≥95%
- [ ] **Step 6**: 文档三处更新 + commit：`docs: changelog + divergences for parity wave 2 M4`
