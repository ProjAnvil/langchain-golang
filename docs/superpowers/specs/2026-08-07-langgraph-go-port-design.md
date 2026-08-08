# LangGraph Go 移植设计

日期：2026-08-07
状态：已批准（用户于 2026-08-07 确认）
上游基线：`langchain-ai/langgraph` main @ `ea5f9cc9f`（1.2.x 之后）

## 目标

将 Python `langgraph` 移植为本仓库（`langchain_golang`，module `github.com/projanvil/langchain-golang`）中的公开 Go 模块，使 Go 用户可以用与 Python 版一致的语义构建有状态、可中断、可恢复的图工作流，并与本仓库已移植的 `core`（messages/runnables/tools）体系集成。

## 已否决的备选方案

- **全保真逐文件重写（方案 B）**：严格镜像 Python 模块结构从头写。工作量最大且会废弃现有 `agentruntime` 的已测代码。不作为起点，但其语义（版本化 checkpoint、完整 channel 类型、Pregel 超步细节）是 M2 的演进目标。
- **依赖第三方 Go 实现（方案 C，如 dshills/langgraph-go）**：alpha 阶段、API 与 Python 版不对齐、无法与本仓库 `core` 集成。否决。
- **泛型强类型 state**：与 Python 动态语义和现有代码均不一致，否决。state 统一为 `map[string]any`，配合每 key reducer/channel。

## 总体方案（方案 A）

以现有 `langchain/internal/agentruntime` 为地基，晋升为仓库根部的公开 `langgraph/` 模块，然后按里程碑增量补全至全保真语义。

### 包布局（镜像 Python monorepo 依赖图）

```text
langgraph/
├── types/       # Command / Send / Interrupt / GraphInterrupt（平移自 agentruntime/types.go）
├── channels/    # LastValue / Topic / BinaryOperatorAggregate 等 channel 抽象
├── checkpoint/  # Saver 接口、版本化 Checkpoint、InMemorySaver
├── graph/       # StateGraph builder + CompiledGraph
├── pregel/      # 超步执行引擎（M2 起从 graph 内的 run 循环中抽出）
└── prebuilt/    # ToolNode 等（M4）
```

依赖方向：`types` ← `channels` ← `checkpoint` ← `pregel` ← `graph` ← `prebuilt`，禁止反向依赖。

### 状态模型

- Graph state 为 `map[string]any`；节点函数不得修改传入的 state map（只读），通过返回值表达更新。
- 每 key 的合并语义由 channel/reducer 决定；未注册的 key 默认 LastValue（last write wins）。
- 现有 `MessagesReducer`（ID 感知合并）、`AppendSliceReducer`、`LastValueReducer` 逻辑复用。

### 与现有 `agentruntime` 的关系

每个里程碑完成后，`langchain/internal/agentruntime` 改为薄封装，委托到公开 `langgraph` 包；`langchain/agents.CreateAgent` 的对外行为与测试必须保持不变（全绿）。

## 里程碑划分

每个里程碑独立可交付、可验证。

### M1 核心平移（已完成 2026-08-08）

- 建立公开 `langgraph/` 包骨架（types/channels/checkpoint/graph）。
- 平移现有能力：StateGraph builder、同步超步执行循环（超步内并行）、`Command`/`Send`/`Interrupt`、interrupt_before/after、单 checkpoint/thread 的 `MemorySaver`、事件 sink（现有 `InvokeStream` 能力）。
- `agentruntime` 改为委托封装；`create_agent` 切换到新包，全部现有测试通过。
- 显式不做：subgraphs、stream modes、时间旅行、持久化后端、函数式 API。

### M2 全保真核心（已完成 2026-08-08）

- 版本化 checkpoint：每 thread 多 checkpoint（checkpoint_id 单调）、state history（`List`/`GetStateHistory`）、时间旅行（从任意历史 checkpoint fork/恢复）。
- checkpoint 结构对齐 Python：channel values + channel versions + versions_seen + pending writes。
- channel 升级为对象抽象（对齐 `langgraph.channels`：`LastValue`、`Topic`、`BinaryOperatorAggregate`、`EphemeralValue`），内部复用 M1 reducer 逻辑。
- ~~超步循环升级为 PULL 订阅模型~~ **设计决策（2026-08-08 修订）**：保留边驱动调度。调研证实对纯 StateGraph 构建的图，Python 的 PULL 触发器（`branch:to:<node>` channel）与边驱动调度在"哪些节点运行、顺序、输入"上可观测等价（`pregel/_algo.py:1260-1277` vs 边解析）；PULL 机器是 Python 实现细节而非语义。M2 改为在边驱动循环上叠加版本簿记（channel_versions/versions_seen/每超步单一全局版本递增），以获得同等的时间旅行与恢复语义，同时保持 Go 实现的简洁。
- `checkpoint.Saver` 接口破坏性升级为版本化模型（GetTuple/List/Put/PutWrites/DeleteThread，key 为 thread_id+checkpoint_ns+checkpoint_id）；`langgraph` 尚处 pre-1.0，允许 break。`graph.CompiledGraph` 的 `Invoke/InvokeStream/Options{ThreadID,Resume}/Result` 外部 API 保持稳定，`langchain/agents` 零改动。
- 多父节点 barrier join（Python `NamedBarrierValue`）不做：当前 Go builder 无多起点边 API，spec 未要求。
- subgraph 支持：`Command.Graph = ParentGraph`、子图作为节点；子图 checkpoint 使用 namespace 前缀（与 Python 的 ns+task_id 方案存在文档化差异）。

### M3 流式与持久化（已完成 2026-08-08）

- Stream modes：`values` / `updates` / `debug` / `messages` / `custom`（对齐 Python `stream_mode`；多模式复用、子图 namespace 前缀）。**设计决策（2026-08-08）**：Go API 形态为 `Stream(ctx, input, StreamOptions) iter.Seq2[StreamChunk, error]`（Go 1.23 range-over-func，惯用且背压友好），区别于 Python 的生成器+元组；`InvokeStream`/`NodeEventSink` 保留不动（agents 依赖），未来可再统一。`messages` 模式通过 `core/callbacks` Handler 桥接 `core/language` 的 `EventChatModelStream` 事件 + 执行器按节点注入元数据（节点名/ns/step）实现，节点需自行接线 callbacks（文档化）。
- Checkpoint serde：`Serializer` 接口 + JSON 实现（对齐 `JsonPlusSerializer` 的可移植子集；Go 用 JSON + 封闭类型注册表信封代替 Python 的 msgpack+import-by-name，不与 Python checkpoint 二进制兼容，明确记录）。
- SQLite checkpoint 后端（对齐 `langgraph-checkpoint-sqlite` 的 schema：checkpoints/writes 两表、type+value 双列）。**设计决策（2026-08-08）**：Go stdlib 无 SQLite 驱动；为保持根 module 零第三方依赖，SQLite saver 放在**独立嵌套 module** `langgraph/checkpoint/sqlite/`（自己的 go.mod，`require` 根 module + 纯 Go 驱动 `modernc.org/sqlite`），镜像 Python 的 `langgraph-checkpoint-sqlite` 分包。这是整个移植引入的第一个第三方依赖。
- M2 遗留技术债清理：`Metadata.Step` 与 Python 对齐（update checkpoint step+1、new-turn input 延续计数）；补齐 M2 终审遗留测试（goto-only sibling replay、CheckpointID+非空 input、interrupt→update_state→resume、子图双重进入、子图内 interrupt 报错路径）。

### M4 prebuilt 与节点策略（已完成 2026-08-08）

- `prebuilt/`：`ToolNode` 图节点适配器（包装现有 `langchain/tools.ToolNode` 为可直接 `AddNode` 的图节点，补 `messages_key` 选项与工具返回 `Command` 的透传）。~~`create_react_agent` 等价物~~ **设计决策（2026-08-08）**：不做。Python 的 `create_react_agent` 自 v1.0 起已 deprecated（`chat_agent_executor.py:274-278`），其能力（model↔tools 循环）是本仓库 `langchain/agents.CreateAgent` 的真子集，重复建设违背 YAGNI；文档中说明等价关系即可。
- 节点 retry / cache 策略：`RetryPolicy`（initial_interval/backoff_factor/max_interval/max_attempts/jitter/retry_on，执行器层级逐节点重试，对齐 `pregel/_retry.py`）；`CachePolicy` + `Cache` 接口 + `InMemoryCache`（缓存的是任务 writes 而非返回值，key 为节点输入的哈希，对齐 `pregel/_algo.py:668-687`/`_loop.py:1549-1625`）。
- 明确不做：CLI、SDK（LangGraph Server 客户端）、部署相关功能。

## 错误处理

- 沿用仓库惯例：显式 `error` 返回，节点内 `Interrupt` 沿用 panic/recover 机制（`GraphInterrupt`），不支持的功能返回明确错误而非静默降级。
- 图构建期错误（重复节点、悬空边等）在 `Compile` 时聚合报告，与现有实现一致。

## 测试策略

- 沿用仓库 table-driven 测试惯例。
- 每个里程碑以 Python 侧对应测试文件为参照（如 `libs/langgraph/tests/test_pregel.py`、`libs/checkpoint/tests/test_memory.py`）挑选可移植用例写 Go 测试。
- 回归底线：`langchain/agents` 与 `agentruntime` 现有测试在每个里程碑后全绿（`go test ./...`）。

## 风险

1. **Pregel 引擎细节**：`pregel/main.py` 数千行，超步/中断/恢复语义复杂；M2 是最重里程碑，需要逐节对照 Python 源码。
2. **Serde 不兼容**：Go 版 checkpoint 无法与 Python 版互通，文档中明确声明。
3. **动态类型摩擦**：`map[string]any` 下的类型断言错误只能在运行期暴露，靠测试覆盖弥补。
