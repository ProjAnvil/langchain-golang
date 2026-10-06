# langchain-golang parity wave 2（上游 2026-10 release 追齐）设计文档

- 日期：2026-10-06
- 状态：已按用户指令「开始规划并实现差距内容」直接进入实施（自主会话，用户已预授权 plan→implement 流水线；不 push、不打 tag）
- 关联研究：`reports/12-latest-release-port-gap.md`（2026-10-06 差距报告）+ 两轮代码勘察（Go 侧错误处理/HITL/interrupt/delta 现状，2026-10-06）
- 上游基线：core 1.6.6 · langchain v1 1.4.3 · langgraph 1.2.13（本地 checkout tag）
- 实施约束：全程 modern-go 规范；TDD（每任务先红后绿）；门禁 `go test ./...` + `make test-sqlite test-redis test-postgres` + golangci-lint 0 issues + 受影响包 ≥95% 覆盖率

## 1. 背景

2026-10-06 差距报告确认：8 月报告的差距清单在 v0.9.2 已基本关闭；剩余可移植项集中
在「核心运行时语义」（P1）与「partner 功能面」（P2）两波。本设计覆盖 **wave 2 = M4
（核心语义修复）**，M5（partner 批次）另行规划。

## 2. 决策记录

用户已拍板（本轮指令）：按报告 §4 推荐顺序开始规划并实现。

按推荐代答（可推翻）：

1. **langgraph_redis middleware 包（semantic cache / conversation memory /
   semantic router / tool cache / Redis BaseStore）→ backlog**：Redis 官方第三方扩
   展，属新增 partner 子系统，需求触发再立项（与 Bedrock 同等待遇）。
2. **M4 范围 = P1 全部六项**（见 §3），M5 = partner 批次（configuration_update、
   async tools、mid-conversation tools、模型 profile、Azure workload identity、
   reasoning_effort 标准参数、LangSmith gateway、ToolErrorMiddleware、
   init_chat_model 新 provider）。
3. **internal-call 过滤机制走 Go 惯用分歧**：用 context 携带不可伪造 token 替代
   Python 的 metadata-dict + StreamTransformer 基础设施（行为等价，机制不同，记
   DIVERGENCES）。
4. **Python #9103（get_state 不再上报已应答 interrupt）**：Go 的 resume 语义在同
   一 invoke 内重跑任务，架构上不存在「INTERRUPT+RESUME 双 pending write」形态；
   列为 M4 审计项（补一个防御性测试证明无此问题即关闭，不实现机制）。
5. delta channel 的既有行为（run 路径不水合）**未**在 DIVERGENCES.md 声明，按上游
   1.2.11–1.2.13 修复语义对齐（water-tight test-first）。

## 3. M4 范围（本 wave 实施）

### A. 标准模型异常层级（core #39538 + openai #39300）

Python 1.6 在 `langchain_core.exceptions` 定义 `ModelError` 家族。Go 侧已有
`core/lcerrors`（sentinels + `ProviderError` + `classifyStatus`）与
`exceptions.go`（`ContextOverflowError` 已存在）。

**设计**：在 `core/lcerrors` 新增模型错误 sentinels 与分类，复用 `ProviderError`
作载体（加 `ModelKind` 字段），保持 `errors.Is` 兼容：

- sentinels：`ErrModelAuth`(401)、`ErrModelPermission`(403)、
  `ErrModelInvalidRequest`(400/422)、`ErrModelNotFound`(404)、
  `ErrModelRateLimit`(429，映射既有 `ErrRateLimited`)、`ErrModelServer`(5xx)、
  `ErrModelConnection`(transport)、`ErrModelTimeout`(408/timeout，映射既有
  `ErrTimeout`)、`ContextOverflowError`(400 + body 嗅探 context-window 标记)。
- `classifyStatus` 扩展为完整 401/403/404/400/422/429/408/5xx 映射；
  `ProviderError.ModelKind` 记录种类，`ProviderError.IsModelRetryable()` 落
  Python `is_retryable` 语义（仅 RateLimit/API(5xx)/Connection/Timeout 为 true）。
- retry 中间件：新增 `RetryOnModelError`（ModelError 且 IsRetryable）；默认
  `RetryOn` 保持「全重试」，但文档指引用户切换；加便捷选项
  `WithModelRetryOnDefault()` 使用 `default_retry_on` 等价语义（分类错误按
  is_retryable，未分类错误重试）。
- partner：openai/anthropic/openaicompat 经 `httpclient` 自动获得分类；gemini
  包一层 genai 错误 → ProviderError 映射。
- 400-body 嗅探标记（对齐 Python openai 的 ContextWindowExceededError 映射）：
  `context_length_exceeded` / `context window` / `maximum context length` /
  `prompt is too long`。

### B. `interrupt(response_schema=...)`（langgraph #8886）

`types.Interrupt` 加 `ResponseSchema map[string]any` 字段；`graph.Interrupt` 增加
变体 `InterruptWithSchema(ctx, value, schema)`；surface 路径（`GetState` 快照、
run 结果 Interrupts、resume.go 的 `interruptsFromWrites` 序列化）带上该字段。
默认 nil，零影响。

### C. create_agent 无效工具调用修复（v1 #40530）

Go 现状：invalid tool calls 被静默丢弃（路由只看 `ToolCalls`）。Python 修复：为
未应答的 invalid call 合成 error ToolMessage 反馈给模型。

**设计**：model node（`create_agent.go` buildModelNode）在响应落 state 前，扫描
`resp.Result` 中 AIMessage 的 `InvalidToolCalls`（`InvalidToolCallBlock`，含
Name/ID/Error），对没有对应 ToolMessage 应答的条目追加
`ToolMessage{ToolCallID, Name, Content: "Tool call {name} with id {id} could not
be executed - arguments were malformed or truncated.", ResponseMetadata:
{"status":"error"}}`（Python 原文）。路由不变（invalid-only 回合仍结束，但纠错
反馈进入历史）。流式路径的 invalid calls 已在 provider 层汇入
`message.InvalidToolCalls`，同一修复覆盖。

### D. HITL 精化（v1 #40463 + #39773；#39247 Go 已 fail-closed 无需动）

- **拒绝理由成帧**（#39773）：有 Message 时 content =
  ``User rejected the tool call for `{name}` with reason: {reason}``（替换现
  状的裸理由）；无 Message 保持既有默认文案（Go 现文案已对齐 Python id 版）。
- **编辑通知**（#40463）：HITL AfterModel 处理 edit 决策时，把
  `editedID → editedAction` 写入 state key `hitl_edited_tool_calls`（Go state 为
  map[string]any，middleware 已有 state 写通道）；新增 `WrapToolCall` 钩子实现：
  `request.ToolCall.ID ∈ edited set` 时，将 `EditNotice`（新字段，默认 Python
  原文，`WithEditNotice("")` 关闭）前置到结果 ToolMessage content。Go 编辑保
  留原 call ID（`processHumanDecision` 已验证），跨 resume 由 state 持久化。
- **return_direct 以实际执行的工具名为准**（#40463 factory 部分）：审计
  `create_agent.go:1145-1169` 路由，若以模型原始 tool_calls 名解析 return_direct，
  改为以已执行 ToolMessage 名解析（编辑后名称可能不同）。

### E. 内部调用标记 + messages 投影过滤（v1 #39252）

Go 架构下 middleware 内部模型调用不经 graph（SummarizerFunc 无 ctx），现状无泄
漏向量；缺的是**公开机制**供第三方 middleware 标记内部调用。

**设计**（Go 惯用分歧，机制不同行为等价）：
- `langchain/agents`（或 `core/callbacks`）暴露
  `callbacks.WithInternalCall(ctx) context.Context`（unexported struct key，
  不可伪造）与 `callbacks.IsInternalCall(ctx) bool`。
- `langgraph/graph/stream.go` 的 StreamMessages 投影与 StreamEvents 的
  chat-model 消息事件：事件来源 ctx 标记为 internal 时跳过（值事件/更新事件不受
  影响——Python 只过滤 messages 投影与 raw log）。
- 测试：tagged 内部调用不出现在 messages 模式与 StreamEvents，正常调用不受影
  响；不改变 state 提交行为。

### F. DeltaChannel 正确性批次（langgraph 1.2.11–1.2.13 六修复的 Go 语义对齐）

勘察结论（Go 现状）：
- F1 **run 路径零水合**（对应 #8548/#9170）：resume/new-turn/UpdateState 只
  `rs.restore`（sentinel 通道不重建），节点读到空值，快照 blob 可能丢历史累积。
  修复：run 路径 restore 后调用与 `reconstructDeltaChannels` 同源的重建（列出
  sentinel delta keys → 祖先 walk → `ReplayWrites`），只在值确实存在时落
  ChannelValues。
- F2 **UpdateState 先重建再应用**（对应 #9165/#9142）：UpdateState 在 applyWrites
  前重建 sentinel delta 通道，强制快照包含「累积 + 更新」；不碰 delta key 的更新
  保持通道原值进新 checkpoint。
- F3 **never-written 短路**（对应 #9141）：`reconstructDeltaChannels` 对从未写过
  的 key 走全链 O(history)；用 `ChannelVersions` 是否含 key（或首 checkpoint 即无
  blob 且无 writes 的剪枝）短路。
- F4 **子图读路径**（对应 #8538）：`parent.GetState(childNS)` 不重建子图 sentinel
  delta 通道（protos 不在父图）。修复：GetState 命中子图 NS 时按注册的子图
  compiled graph（或 NS 前缀登记表）解析 protos+saver。
- F5 **计数器缺口**：never-materialized 通道不推进 superstep 计（advance 迭代
  `rs.channels`）；exit 模式 `flushExit` 二次 advance 导致多计。修为按
  channelProtos 迭代 + flushExit 不重复计数。
- 全部 test-first：每个修复先移植上游回归场景（红）再修（绿）。若 F4 复杂度失
  衡，允许降级为「文档化限制 + DIVERGENCES 记录」并在计划中显式标记。

## 4. 不做（本 wave）

- M5 partner 批次全部项（另立里程碑）。
- langgraph_redis middleware 包 + Redis BaseStore（backlog）。
- `wrap_tool_call(state_schema=)`（Go 显式 state 读取分歧已声明，不追）。
- Python #9103 机制实现（见决策 4，只补防御性测试）。
- push / release / tag（用户未授权）。

## 5. 验收

1. 新增行为均有 TDD 测试（先红后绿），引用上游 PR 号。
2. 全量门禁绿：`go test ./...`（root）、`make test-sqlite test-redis
   test-postgres`、golangci-lint 0 issues、受影响包覆盖率 ≥95%。
3. `CHANGELOG.md` Unreleased 节记账；机制分歧（E、F4 若降级）记 `DIVERGENCES.md`。
4. `reports/12-latest-release-port-gap.md` §3.1–§3.6 状态在完成后更新为已落地。
