# Parity Wave 4（上游 2026-10-29 追齐：langgraph 1.2.19 / core 1.7.3）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 2026-11-01 差距报告（reports/14）的全部可移植项——P1 七项（delta 重放排序、exit 分支污染、message id、openai 流钉扎、trace_policy tracer-only、tool-call-delta 协议、Mermaid 转义）+ P2 八项 + P3 有效项 + 分歧决策记录，对齐上游 langgraph 1.2.19 / langchain-core 1.7.3。

**Architecture:** langgraph 持久化语义四连修（T1–T4，同文件链严格串行）；partner 侧两个自包含修复（T5 openai、T7 协议跨切）；anthropic 链三步（T8 profile 基建 → T9 mid-conversation → T10 effort）；langgraph 小项与流式收尾（T11–T13）；docs 收口（T14）。每项对齐 Python 参考实现（本地 checkout 于 langchain-core==1.7.3 / langgraph 1.2.19 tag）。

**Tech Stack:** Go 1.26+（本机 go1.27.3）；无新外部依赖。测试全离线（httptest / miniredis / embedded postgres / savertest）。

**Spec:** `docs/superpowers/specs/2026-11-01-parity-wave4-design.md`（决策 D1–D10 为唯一事实源；冲突以 spec 为准）

## Global Constraints

- Python 参考路径：`/Users/yuhaochen/Documents/codebase/projanvil/langchain_port/langchain`（@ `langchain-core==1.7.3`）与 `/Users/yuhaochen/Documents/codebase/projanvil/langchain_port/langgraph`（@ `1.2.19`）。上游语义一律以这两个 checkout 为准，commit message 引用上游 PR 号。
- Go 仓库：`/Users/yuhaochen/Documents/codebase/projanvil/langchain_port/langchain_golang`，分支 `feat/parity-wave4`。
- 每任务门禁：`go build ./... && go vet ./... && go test -race ./...`（root）；触及嵌套模块（langgraph/checkpoint/{sqlite,postgres,redis}）时 `make test-sqlite test-redis test-postgres`；`golangci-lint run ./...` 零 issue；触及 coverage 追踪包时 `make coverage` 通过（95% 红线）。
- TDD 强制：先写失败测试并亲自看它失败，再写实现。测试用真实代码，mock 仅限传输边界。
- Conventional commits（英文）：`fix(langgraph): ... (#8544)` 风格。一个任务多个逻辑提交可以。
- 不新增外部依赖；不破坏既有 Go 公共 API（新增可以）；刻意分歧必须落 DIVERGENCES.md。
- 不做任务清单之外的重构。

## Review Focus

1. 并发 task 写同一 delta channel 时，三个持久 saver 的重放顺序必须等于 live 派发顺序（T1 的 savertest 用例必须真的用 fnv 逆序 ID 构造）。
2. fork + exit-durability 组合：钉旧 checkpoint 的 run 不得污染新分支（T2 用例须先建出新分支再回钉）。
3. 消息 ID 跨持久化/重放稳定：RemoveMessage 按 ID 定位必须成功（T3/T13）。
4. trace_policy 脱敏不得改变流输出（T4：流侧原始、tracer 侧脱敏双向断言）。
5. 流式 tool args 的部分解析健壮性：中途非法 JSON、空 args、legacy `tool_call_chunk` 消费者回退（T5/T7）。

---

### Task 1: langgraph delta 重放排序（#8544 + #9114）

**Files:**
- Modify: `langgraph/`（writes 落盘调用点：`graph/graph.go` 提交块 ~1856-1866、pause writes、exit writes——填充 `Write.TaskPath`）
- Modify: `langgraph/checkpoint/snapshot.go`（`Write` 已有 TaskPath 字段 ~53-54；确认序列化往返）
- Modify: `langgraph/checkpoint/delta_helpers.go`（exit 锚点前缀 `0000000X` → 排序在后前缀）
- Modify: `langgraph/checkpoint/sqlite/`（writes 表加 `task_path` 列 + 迁移 + ORDER BY）、`langgraph/checkpoint/postgres/`（同）、`langgraph/checkpoint/redis/`（zset 排序键加 task_path）
- Test: savertest 新用例 + 各 saver 现有测试

**Interfaces:**
- Produces: `Write.TaskPath` 在所有持久化路径被填充；saver 读取按 `(task_path, task_id, idx)` 排序。后续任务（T2/T3）依赖此排序语义。

- [ ] **Step 1: 失败测试**——savertest：同一 superstep 两个 task（构造 task_id 使 fnv 排序与派发顺序相反）写同一 delta channel；断言重放顺序 == 派发顺序。先在 sqlite/postgres/redis 三处看它失败（memory 应已通过）。
- [ ] **Step 2: 失败测试**——恢复的 exit run 写入按 live 顺序重放（多步 exit 写入 + SQL saver 重放顺序断言）。
- [ ] **Step 3: 实现**——putWrites 各调用点填 TaskPath；三 saver schema 迁移 + ORDER BY；exit 前缀改排序在后（对齐 Python #9114 后方案）；serde 往返带上 task_path。
- [ ] **Step 4: 全门禁**（含三个嵌套模块 + lint）。
- [ ] **Step 5: Commit**（`fix(langgraph): order delta channel replay by task path (#8544)` 等分拆提交）。

### Task 2: langgraph exit 模式分支污染（#9269）

**Files:**
- Modify: `langgraph/graph/checkpoint_sink.go`（sink 构造处检测钉住 checkpoint ≠ 命名空间最新；`flushExit` ~405-429 对写过 channel 走 `channelsToSnapshot` 而非锚定写入）
- Test: `langgraph/graph/` 新增 fork 场景测试（内存 + 至少一个 SQL saver）

**Interfaces:**
- Consumes: T1 的写入/重放语义。
- Produces: 无新公共 API（行为修复）。

- [ ] **Step 1: 失败测试**——run 到 checkpoint A → 继续到 B（新分支）→ 钉 A 跑 exit 模式；断言 B 分支读不到该 run 写入、A 侧经快照读得到。参考 Python `_loop.py:1403-1409`。
- [ ] **Step 2: 实现 + 门禁。**
- [ ] **Step 3: Commit**（`fix(langgraph): snapshot exit writes forked onto a superseded checkpoint (#9269)`）。

### Task 3: messages DeltaChannel 补 ID（#9275）

**Files:**
- Modify: `langgraph/channels/channels.go`（MessagesReducer 身份可探测）、`langgraph/graph/graph.go`（提交块 + `persistDeltaInputWrites`）、`langgraph/graph/resume.go`（`completedTaskWrites` 邻域）、`langgraph/graph/snapshot.go`（`UpdateState` ~178-180）、exit 累计器路径
- Test: 上述各路径新用例

**Interfaces:**
- Produces: 内部 `ensureMessageIDs` 钩子（不导出）；按 reducer 身份门控。

- [ ] **Step 1: 失败测试**——四条路径各写无 ID 消息，重载后按 ID 定位（RemoveMessage）成功；调用方传入对象不被修改（复制替换）。
- [ ] **Step 2: 实现 + 门禁。**
- [ ] **Step 3: Commit**（`fix(langgraph): give every message written to a messages delta channel an id (#9275)`）。

### Task 4: trace_policy 只作用于 tracer（#9486）

**Files:**
- Modify: `langgraph/graph/graph.go` ~1493-1501（policy 包裹范围收窄）、`core/callbacks/payload_policy.go`（如需排除流桥 handler）
- Modify: `DIVERGENCES.md`（trace_policy 条目更新）
- Test: `langgraph/graph/` 新用例

- [ ] **Step 1: 失败测试**——节点带 omit-payload TracePolicy：messages 流看到原始载荷、fake tracer handler 看到脱敏载荷。参考 Python 8d5637b2d（`_read.py`/`main.py`）。
- [ ] **Step 2: 实现 + 门禁。**
- [ ] **Step 3: Commit**（`fix(langgraph): apply trace_policy to tracers only (#9486)`）。

### Task 5: OpenAI 流 item 钉扎（#41121）

**Files:**
- Modify: `partners/openai/stream.go`（`item_output_indices` 钉扎；`upsertToolCall`/`upsertReasoningBlock`/finish 路径）
- Test: `partners/openai/` 流组装测试（事件序列注入，无需网络）

**Interfaces:**
- Produces: 无公共 API 变化（内部钉扎 map）。

- [ ] **Step 1: 失败测试**——done 事件 output_index 漂移的多 tool call 序列，断言块落位正确。参考 Python `base.py:5376-5433`。
- [ ] **Step 2: 实现 + 门禁。**
- [ ] **Step 3: Commit**（`fix(openai): pin Responses stream items to first-seen output_index (#41121)`）。

### Task 6: core 小项批（#41512 + #41540 + #41006 + llm-cache globals）

**Files:**
- Modify: `core/runnables/graph_mermaid.go`（`_escape_mermaid_label` + scalar-only metadata 格式化）
- Create: `core/messages/notification.go`（`AddNotificationMetadata`/`GetNotificationMetadata`，写 `AdditionalKwargs["lc_source"]/["lc_data"]`；MergeRuns 拼接 lc_source）
- Modify: `core/tracers/langsmith.go` + `core/tracers/run.go`（`Address`/`Replicas` 选项 + run payload 转发）
- Modify: `core/caches/cache.go`（`GetLLMCache`/`SetLLMCache` 全局访问器）
- Test: 各文件对应 `_test.go`

- [ ] **Step 1: 失败测试×4**——Mermaid 特殊字符元数据转义 + 列表/映射格式化规则；notification metadata 写读 + MergeRuns 拼接；tracer options → payload 字段映射；llm-cache 全局 set/get。
- [ ] **Step 2: 实现 + 门禁。**（四个独立提交。）
- [ ] **Step 3: Commit**（`fix(core): escape node metadata in Mermaid labels (#41512)` 等）。

### Task 7: tool-call-delta 流协议（#41257/#41246，Wave B）

**Files:**
- Modify: `core/streamevents/events.go`（新类型常量 `tool-call-delta`；`pushDelta` 双标签兼容 + 从 content-block-start 播种 id/name；`Dispatch` 处理 start）
- Modify: `core/language/stream_events.go` 桥、`langchain/agents/stream.go` 桥（start 骨架、delta 去重 id/name、稳定 per-key 索引）
- Modify: `partners/openai/stream.go`（retag delta）、`partners/anthropic/stream.go`（retag delta）、`partners/openai/chat_completions_stream.go`（逐片转发 args）
- Modify: `partners/gemini/stream.go`、`partners/ollama/`（SDK 有增量则升级，否则 DIVERGENCES 记录）
- Test: 各处流组装/golden 序列测试

**Interfaces:**
- Produces: 流上的 `tool-call-delta` delta 类型（`NonStandardContentBlock{Type:"tool-call-delta", Value:{"args": <slice>}}`）；消费端同时接受 legacy `tool_call_chunk`。

- [ ] **Step 1: 失败测试**——streamevents 消费端：start 骨架播种 + delta 累积 + legacy 回退 + 中途非法 JSON 容错。
- [ ] **Step 2: 失败测试**——openai/anthropic 发射端 golden 序列（start 骨架、纯 args 切片、finish 带迟到字段）；chat-completions 逐片。
- [ ] **Step 3: 实现**（spec D6 顺序：类型 → 消费端 → 桥 → 发射端 → chat-completions）。
- [ ] **Step 4: 全门禁 + standardtests 确认不回归。**
- [ ] **Step 5: Commit**（`feat(core): stream tool-call args as tool-call-delta slices (#41257)` 等分拆）。

### Task 8: modelprofiles 数据物化 + mid_conversation 键（#41180 schema 半边）

**Files:**
- Modify: `modelprofiles/model_profile.go`（`MidConversationSystemMessages`/`MidConversationTools` 键）
- Create: `modelprofiles/profiles.json`（refresh CLI 生成后提交）
- Modify: `modelprofiles/cli/`（离线 fixture 测试保持）+ 查询入口（按 provider+model）
- Test: `modelprofiles/` 用例

**Interfaces:**
- Produces: `modelprofiles` 查询函数（provider+model → `*ModelProfile`），T9/T10 依赖。

- [ ] **Step 1: 失败测试**——键解析 + 查询命中（anthropic/openai 已知模型带 mid_conversation 标志）+ 未知模型 miss。
- [ ] **Step 2: 实现 + 物化数据 + 门禁。**
- [ ] **Step 3: Commit**（`feat(modelprofiles): add mid_conversation profile keys and materialized data (#41180)`）。

### Task 9: anthropic mid-conversation system/tools（#41180 行为半边）

**Files:**
- Modify: `partners/anthropic/chatmodel.go` ~370-419（profile 门控放置规则；`tool_addition`/`tool_removal` system-block 格式）
- Test: `partners/anthropic/` 消息映射测试（profile on/off 两态）

**Interfaces:**
- Consumes: T8 查询入口。

- [ ] **Step 1: 失败测试**——profile 允许时中途 SystemMessage 按 Python 1.7.7 放置规则保留（`_previous_turn_all_allows_system`）；未允许时保持提升 + 日志。参考 Python `chat_models.py:715-760`。
- [ ] **Step 2: 实现 + 门禁 + standardtests。**
- [ ] **Step 3: Commit**（`feat(anthropic): gate mid-conversation system messages and tools on model profile (#41180)`）。

### Task 10: anthropic effort/output_config（#41536 超集）

**Files:**
- Modify: `partners/anthropic/chatmodel.go`（`WithEffort`、payload `output_config`、profile `reasoning_effort_levels` 校验含 `xhigh` adaptive 默认、getter）
- Test: `partners/anthropic/` 用例

**Interfaces:**
- Consumes: T8 查询入口。

- [ ] **Step 1: 失败测试**——payload 形状、级别校验（合法/非法/默认解析）。
- [ ] **Step 2: 实现 + 门禁。**
- [ ] **Step 3: Commit**（`feat(anthropic): effort/reasoning_effort with output_config and profile levels (#41536)`）。

### Task 11: langgraph 小项批（#9420 + #9383 + #9449 + redis #220 测试）

**Files:**
- Modify: `langgraph/channels/delta.go`（默认 1000→100）
- Modify: `langgraph/store/store.go`（`validateNamespace` 拒绝 `\x00`）+ `storetest`
- Modify: `langgraph/graph/export.go` + `langgraph/prebuilt/tool_node.go`（`ToolNames()` duck-type → `__tools` 元数据）
- Test: `langgraph/checkpoint/redis/` GetTuple pending-writes parity 用例

- [ ] **Step 1: 失败测试×4**（含 NUL 桶碰撞用例 `["a\x00b"]` ≠ `["a","b"]`）。
- [ ] **Step 2: 实现 + 门禁。**（四个提交。）
- [ ] **Step 3: Commit**（`feat(langgraph)!: default delta snapshots to 100 updates (#9420)` 等）。

### Task 12: #9235 写失败不落 checkpoint + #9302 delta→plain 迁移

**Files:**
- Modify: `langgraph/graph/graph.go` 同步提交路径（先缓冲 writes 再 save；写失败则 checkpoint 不持久化）、`checkpoint_sink.go` 异步 worker 对齐
- Modify: `langgraph/channels/binop.go`（`FromCheckpoint` 解包 delta snapshot blob）、`langgraph/checkpoint/snapshot.go`（游走考虑"有版本无值"哨兵键）
- Test: 上述路径用例（含失败注入 saver）

- [ ] **Step 1: 失败测试×2**——PutWrites 失败 → 无新 checkpoint（同步与异步）；delta channel 字段改 plain reducer 后读回保值。
- [ ] **Step 2: 实现 + 门禁。**
- [ ] **Step 3: Commit**（`fix(langgraph): don't save a checkpoint past a failed delta write (#9235)`、`fix(langgraph): keep a delta channel's value when its field changes to a plain reducer (#9302)`）。

### Task 13: 流式三连（#9463 + #9458 + #9281）

**Files:**
- Modify: `langgraph/graph/stream_messages.go`（补发节点返回消息，id 去重）、`graph.go` `run()`（带新 ThreadID 时丢弃继承的 emitter/bridge）
- Modify: `langgraph/runtime/runtime.go` + `graph/graph.go`（`RunControl` 经 Options/Runtime 接入循环；drain 检查在 interrupt_before 判定之后）
- Test: 对应用例（含 drain 优雅停 + 断点优先）

- [ ] **Step 1: 失败测试×3**。参考 Python `10f1fabb3`（drain）、`_messages.py`（node-output 消息）。
- [ ] **Step 2: 实现 + 门禁。**
- [ ] **Step 3: Commit**（`feat(langgraph): stream node-returned messages (#9463)` 等三个提交）。

### Task 14: docs 收口（DIVERGENCES / CHANGELOG / report 14 追记）

**Files:**
- Modify: `DIVERGENCES.md`（response_schema 上游方向补记、trace_policy 条目更新、gemini/ollama 流分歧若有、ToolErrorMiddleware/chat_loaders defer 补记）
- Modify: `CHANGELOG.md`（[Unreleased] → 版本条目，从 git log 汇总）
- Modify: `../../reports/14-latest-release-port-gap.md`（落地追记横幅）

- [ ] **Step 1: 逐项核对 git log 与 DIVERGENCES 现状，写入。**
- [ ] **Step 2: Commit**（`docs: parity wave 4 changelog and divergences`）。
