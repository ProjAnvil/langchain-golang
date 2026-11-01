# Parity Wave 4 Design（上游 2026-10-29 追齐：langgraph 1.2.19 / core 1.7.3）

**Date**: 2026-11-01 · **Authority**: `reports/14-latest-release-port-gap.md`（四路审计，全部 verdict 带 file:line 证据）
**Python ground truth**: `langchain/` @ `langchain-core==1.7.3` · `langgraph/` @ `1.2.19`（工作区绝对路径下的本地 checkout）
**Go target**: `langchain_golang/` 自 `v0.11.1`（37baf07）起，分支 `feat/parity-wave4`

本 spec 是 wave4 的唯一决策记录；plan 从本 spec 展开，实现冲突以本 spec 为准。范围 = report 14 的 P1 七项 + P2 八项 + P3 有效项 + 分歧决策落地。

## 决策记录

### D1 — Delta 重放排序（#8544/#9114，P1）
Go 的三个持久 saver（sqlite/postgres/redis）按 `(task_id, idx)` 排序重放 pending writes，task_id 是与派发顺序无关的 fnv 哈希；`Write.TaskPath` 字段存在但从未写入。修复对齐 Python：**写入时填充 `TaskPath`，读取按 `(task_path, task_id, idx)` 排序**。
- 表结构：三个 saver 的 writes 表各加 `task_path TEXT NOT NULL DEFAULT ''` 列（ALTER TABLE 迁移；旧行回退到现行排序，无破坏）。跨语言 checkpoint DB 本就不共享（DIVERGENCES 已声明），schema 可自由演进；版本号 bump。
- exit 锚点合成 ID：前缀从 `"0000000X"`（排序在真实 ID 之前）改为排序在后的前缀（`"ffffffff"` 风格，对齐 Python #9114 后的方案），保证恢复的 exit run 写入按 live 顺序重放。v0.11.1 的"锚点 checkpoint ID 预留"语义不受影响（那是 checkpoint ID 排序，不是 write 排序）。
- MemorySaver 保持插入顺序（已正确）。
- 验收：savertest 级用例——同一 superstep 两个 task 以 fnv 顺序与派发顺序相反的 ID 写同一 delta channel，断言重放顺序 == 派发顺序（sqlite/postgres/redis 三处）；恢复的 exit run 写入重放保持 live 顺序。

### D2 — Exit 模式分支污染（#9269，P1）
Go 无 `checkpoint_superseded` 概念：以 `Options.CheckpointID` 钉在非最新 checkpoint 上跑 exit 模式时，写入锚在共享父链上，被新分支的祖先游走拾取 → 持久化跨分支污染。修复对齐 Python `_loop.py:1403-1409`：sink 构造时检测"钉住的 checkpoint ≠ 命名空间最新"；此时对本次 run 写过的 channel **强制快照**（进 `channelsToSnapshot`）而不是锚定写入。验收：fork 场景——A checkpoint 后继续到 B，再钉 A 跑 exit run，断言 B 分支读不到该 run 写入、A 分支经由快照读得到。

### D3 — messages DeltaChannel 补 ID（#9275，P1）
`MessagesReducer`（channels.go:67-151）原样追加无 ID 消息。修复：在持久化路径上加 `ensureMessageIDs` 钩子——task writes 提交块（graph.go:1856-1866 邻域）、`persistDeltaInputWrites`（graph.go:1920-1931）、`UpdateState`（snapshot.go:178-180）、exit 累计器——**按 reducer 身份门控**（仅 MessagesReducer 的 channel）；空/缺 ID 的消息复制后赋 UUID 再落盘，不改动调用方传入的对象（Python："a write that changes is replaced"）。Go 输入面没有字符串/tuple 形态，只移植 ID 半边。验收：写无 ID 消息 → 重新加载后可按 ID 定位（RemoveMessage 生效）；UpdateState / 图输入 / fork Command 路径同验。

### D4 — trace_policy 只作用于 tracer（#9486，P1）
Go 把整个节点回调管理器包进 `NewPayloadPolicyManager`（graph.go:1493-1501），messages 桥（stream.go:304-309）收到的是脱敏载荷——上游已改为仅 tracer 生效。修复：policy 只包裹 tracer 类 handler（或将流桥 handler 排除在包裹集外）。现有 DIVERGENCES 条目只声明了 fail-closed 半边，本条修复后同步更新该条目。验收：节点带 TracePolicy omit 时，messages 模式流看到原始载荷，fake tracer handler 看到脱敏载荷。

### D5 — OpenAI 流 item 钉扎（#41121，P1）
移植 Python `base.py:5376-5433` 的 `item_output_indices` 钉扎：`ItemID` → 首次见到的 `OutputIndex`，upsert/finish 一律用钉扎值。验收：构造 output_index 在 done 事件上漂移的事件序列，断言 tool call/reasoning 块落位正确。

### D6 — tool-call-delta 流协议（#41257/#41246，Wave B）
Go 语义已是增量，协议形状不对。决策：
- 新增 delta 类型常量 `tool-call-delta`（以 `NonStandardContentBlock{Type:"tool-call-delta", Value:{"args": <slice>}}` 承载，不破 sealed interface）。
- openai/anthropic 发射端改发该类型（纯 args 切片）；首次见到某 tool call index 时先发 `content-block-start` 骨架（`ToolCallChunkBlock{ID, Name, Args:""}`）；迟到的 extras/id/name 移到 `content-block-finish`。
- 消费端 `ChatModelStream.pushDelta` 同时接受 `tool-call-delta` 与 legacy `tool_call_chunk`；id/name 优先从 content-block-start 播种。
- 两座桥（core/language/stream_events.go、langchain/agents/stream.go）发 start 骨架、delta 去掉重复 id/name、稳定 per-key 索引（不用 `len(events)` 回退）。
- chat-completions 适配器改为逐片转发 args（不再缓冲到 `[DONE]`）。gemini/ollama：SDK 若暴露增量 args 则升级，否则保留整块 + DIVERGENCES 记录。

### D7 — modelprofiles 数据物化 + mid_conversation 键（#41180，schema 半边）
`ModelProfile` 增加 `mid_conversation_system_messages` / `mid_conversation_tools` 键；用现有 refresh CLI 物化 `profiles.json` 并提交（生成步骤一次性联网，测试保持离线 fixtures）；提供 `modelprofiles` 查询入口（按 provider+model）。此任务是 T9/T10 的前置。

### D8 — anthropic 中途对话 system/tools（#41180，行为半边）
以 profile 门控：`mid_conversation_system_messages` 允许时，中途 SystemMessage 按 Python 1.7.7 `_format_messages` 的放置规则处理（`_previous_turn_all_allows_system`），不再无条件提升到顶层 `system`；`mid_conversation_tools` + `tool_addition`/`tool_removal` system-block 格式同步移植。profile 未声明允许时保持现状提升（并可加对齐 Python 的 hoist warning 日志）。

### D9 — anthropic effort/output_config（#41536，超集）
构造选项 `WithEffort`；payload 发 `output_config`；profile `reasoning_effort_levels` 校验（含 `xhigh` adaptive 默认）。Go 无 identifying-params 概念，本期只做 payload + 校验 + getter 暴露，tracing 侧待 Go 有 identifying-params 机制再接。

### D10 — Wave D 各小项
#9420 delta 快照默认 1000→100（CHANGELOG 记 Changed）；#9235 写失败不落 checkpoint（同步路径先缓冲 writes 再 save，失败则本轮 checkpoint 不持久化；异步 sink 对齐）；#9302 delta→plain reducer 迁移（`FromCheckpoint` 解包 delta snapshot blob + 历史游走考虑"有版本无值"哨兵）；#9383 store namespace label 拒绝 `\x00`（内存 store 用 `\x00` 做内部分隔符，存在真实桶碰撞）；#9463 messages 模式补发节点返回的消息（id 去重，经节点桥）；#9458 带新 ThreadID 的嵌套 invoke 不继承父 emitter；#9281 drain 真正被消费（RunControl 进 Options/Runtime，interrupt_before 优先于 drain）；#9449 `__tools` 节点元数据；#41006 tracer `address`/`replicas`；#41540 notification metadata 助手；llm-cache globals；redis #220 parity 测试。

### 分歧决策（维持 + 记录）
- `Interrupt.ResponseSchema` resume 值校验：**维持不校验**（DIVERGENCES.md:36），但条目补记上游 1.2.15+ 已改为校验 + prebuilt 1.1.1 传播错误。
- 独立 ToolErrorMiddleware：**不移植**（ToolNode `HandleToolErrors` 默认策略已覆盖行为面），DIVERGENCES 补记。
- chat_loaders（上游 classic-only）：**defer**，DIVERGENCES 补记。
- LangSmith Gateway：**defer**（沿用 report 13 决策）。
- RemoteGraph：维持 backlog。

## 门禁
每任务：`go build ./... && go vet ./... && go test -race ./...`（root）+ 触及嵌套模块时 `make test-sqlite test-redis test-postgres` + `golangci-lint run ./...` 零 issue + 触及 coverage 追踪包时 `make coverage` ≥95%。收尾：全模块全绿 + 全仓 lint + final whole-branch review（最强模型）+ CHANGELOG/DIVERGENCES/report 14 追记。
