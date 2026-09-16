# 图运行时（`langgraph/`）—— Stream 模式、serde、SQLite/Postgres checkpoint、retry/cache 策略、ToolNode、join 边、函数式 API

**Languages:** [English](../langgraph.md) | 简体中文

公开的顶层 `langgraph/` 包承载了移植版图运行时：
`langgraph/graph`（StateGraph 构建器 + Pregel 执行器）、
`langgraph/channels`、`langgraph/checkpoint`（含 `checkpoint/serde`，持久化
saver 位于嵌套模块 `checkpoint/sqlite` 与 `checkpoint/postgres`）、
`langgraph/types`、`langgraph/prebuilt`，以及 `langgraph/fn`（函数式 API）。
`agents.CreateAgent` 构建于其上；本指南覆盖 M1–M7 全量内容：`Stream`
API、checkpoint 序列化、SQLite 与 Postgres checkpoint saver、节点级
retry/cache 策略、`prebuilt.ToolNode`、多父节点 join 边（`AddJoinEdge`）、
interrupt 与 resume（含子图中断）及 durability 模式、函数式 API
（`NewEntrypoint` / `NewTask`），以及 M5 saver 接口的 breaking 变更。

## Stream API（`CompiledGraph.Stream`）

`Stream` 像 `InvokeWithOptions` 一样运行图，并以 Go 1.23 range-over-func
迭代器（`iter.Seq2`）的形式在 chunk 产生时逐个吐出。`Modes` 选项对齐
Python 的 `stream_mode`，支持多模式流：

```go
import graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"

for chunk, err := range cg.Stream(ctx, input, graphpkg.StreamOptions{
	Modes: []graphpkg.StreamMode{graphpkg.StreamValues, graphpkg.StreamUpdates},
}) {
	if err != nil {
		return err // run failure is yielded as the final pair
	}
	switch chunk.Mode {
	case graphpkg.StreamUpdates:
		// map[string]any{nodeName: map[string]any{channel: value}} per task
	case graphpkg.StreamValues:
		// map[string]any — full graph state after the input batch and after
		// each superstep that changed a channel
	}
	_ = chunk.Namespace // "" for the root graph; subgraph node path otherwise
}
```

- **Modes** —— `StreamValues`、`StreamUpdates`、`StreamDebug`（任务
  派发/完成 + checkpoint 事件）、`StreamMessages`（逐 token 的 LLM
  chunk；节点代码通过 `callbacks.ManagerFromContext` 取出已安装的
  callbacks manager 来 opt in）、`StreamCustom`（节点经
  `graphpkg.StreamWriterFromContext` 发出的自定义负载）。
- **`StreamChunk`** 始终显式携带 `Namespace`、`Mode` 与 `Payload` —— Go
  不像 Python 那样按模式数量改变输出形态（裸负载 vs 元组）。
- **`StreamOptions` 内嵌 `Options`**，因此 `ThreadID` / `CheckpointID` /
  `Resume` 保持其 `Invoke` 语义；`Subgraphs: true` 会包含子图 chunk
  （携带其 `Namespace`）而不是丢弃它们。
- 提前跳出 `range` 会取消本次运行并等待其 goroutine 退出 —— 无泄漏。

`Stream` 与 `InvokeStream` / `NodeEventSink`（`Agent.StreamEvents` 背后的
事件化路径）共存；二者互不替代。一条文档化的时序分歧：`updates` chunk
在超步结束后按确定性的任务顺序发出，因此它们会聚集在节点期的
`messages`/`custom` chunk 之后，而不是像 Python 那样交错出现。

## Checkpoint serde（`langgraph/checkpoint/serde`）

`checkpoint.Serializer` 是持久化编码契约；
`serde.NewJSONSerializer()` 是仓库内实现 —— Python `JsonPlusSerializer`
的可移植子集：

- JSON 原生值（`nil`、`string`、`float64`、`bool`、`map[string]any`、
  `[]any`）编码为普通 JSON（tag `"json"`）。
- 普通 JSON 会有损的具体 Go 类型，通过**封闭类型注册表**的信封
  `{"__type__": name, "data": payload}`（tag `"json+envelope"`）无损往返。
  注册表覆盖 `messages.Message`、`[]messages.Message`、`types.Send`、
  `types.Interrupt`、`time.Time`、`[]byte`、`int64`、`int` 与 `[]string`。
  编码未注册的具体类型会报错 —— 不存在静默的有损回退。

> **分歧说明：** Go 使用 JSON + 封闭注册表，而非 Python 的 msgpack +
> 按名导入。该序列化器写出的 checkpoint **与 Python checkpoint 不二进制
> 兼容** —— 无 Python 互通。

## SQLite checkpoint saver（`langgraph/checkpoint/sqlite/`）

基于 SQLite 的持久化 `checkpoint.Saver`，对齐 Python
`langgraph-checkpoint-sqlite` 的 schema（WAL 模式；`checkpoints` /
`writes` 两表，均为 `type`+`value` 列对）。

> **依赖声明：** 这是本移植的**第一个第三方依赖** —— 纯 Go（无 cgo）
> 驱动 `modernc.org/sqlite`，为 Go 1.23 底线固定在 v1.38.2。为保持根
> module 零依赖，该 saver 位于独立的嵌套 Go module
> （`langgraph/checkpoint/sqlite/go.mod`，带指回根 module 的 `replace`
> 指令），对齐 Python 独立的 `langgraph-checkpoint-sqlite` 包。

```go
import (
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	sqlite "github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite"
)

saver, err := sqlite.New("checkpoints.db", serde.NewJSONSerializer())
if err != nil {
	return err
}
defer saver.Close()
// Use like any checkpoint.Saver: agents.WithAgentCheckpointer(saver), etc.
```

由于它是嵌套 module，根 module 的 `go test ./...` **不会**进入其中。
运行其测试：

```bash
make test-sqlite
```

## 节点级 retry（`graph.RetryPolicy`）

`RetryPolicy` 在单个节点上安装执行器级别的自动重试（指数退避），对齐
Python 的 `langgraph.types.RetryPolicy`。策略在注册节点时经
`AddNodeWithPolicies` 挂载；用普通 `AddNode` 添加的节点不带策略，永不
重试：

```go
g := graphpkg.NewStateGraph()
g.AddNodeWithPolicies("flaky", flakyNode, graphpkg.NodePolicies{
	Retry: &graphpkg.RetryPolicy{MaxAttempts: 5},
})
```

- **默认值**（应用于任何零值字段）：`InitialInterval` 500ms、
  `BackoffFactor` 2.0、`MaxInterval` 128s、`MaxAttempts` 3（总尝试次数，
  含首次）。退避为 `min(MaxInterval, InitialInterval *
  BackoffFactor^(attempt-1))`。
- **Jitter** —— 在 `MaxInterval` 截断之后追加的 `[0, 1s)` 均匀随机
  抖动 —— 默认**开启**（Python 对齐）；设 `NoJitter: true` 关闭。
- **`RetryOn`** 判定失败尝试的错误是否可重试；nil 表示
  `DefaultRetryOn`，它重试**一切**错误，只排除一个很小的集合：被
  `graph.NonRetryable` 包装的错误（为已知永久性错误提供的显式退出通道 ——
  Go 错误不像 Python 默认判定所参考的那样带有异常类层级）、
  `context.Canceled`（父级运行已中止；退避期间重试循环本身也会因
  `ctx.Done()` 中止），以及 `*channels.InvalidUpdateError`（写入路径的编程
  错误）。早先的窄判定（仅网络错误 / 5xx）已移除 —— provider 的 429 或任何
  其他普通瞬时错误现在默认重试；需要更窄的行为请提供自定义 `RetryOn`。被
  中断的节点是终态，重试循环不会重新执行它。

**图级默认 retry。** `Compile(graphpkg.WithDefaultRetryPolicy(
&graphpkg.RetryPolicy{MaxAttempts: 5}))` 安装一个图级默认策略，应用于所有
不带自身策略的节点（对齐 Python 的 `retry_policy=` 编译参数）。节点自身
的策略总是胜出；默认策略不会被子图继承（每个子图单独编译）；传 nil 关闭
它。函数式 API 中，`EntrypointOpts.Retry` 走同一机制，因此它既重试
entrypoint 函数，也重试没有自身 `TaskOpts.Retry` 的每个 task。

## 节点级 cache（`graph.CachePolicy` + `checkpoint.Cache`）

`CachePolicy` 缓存节点任务的 writes，对齐 Python 的
`langgraph.types.CachePolicy`。它需要两半配合：节点上的策略，以及编译
时安装的 `checkpoint.Cache` 后端：

```go
g := graphpkg.NewStateGraph()
g.AddNodeWithPolicies("expensive", expensiveNode, graphpkg.NodePolicies{
	Cache: &graphpkg.CachePolicy{TTL: 10 * time.Minute},
})
cg, err := g.Compile(graphpkg.WithCache(checkpoint.NewInMemoryCache()))
```

- 缓存的是任务的 **writes**（以 channel write 形式存在的状态更新，外加
  路由），而非其返回值。命中时节点不执行：存储的 writes 被注入为任务
  结果，因此 `updates` stream chunk 照常发出，缓存的 `Command.Goto`
  路由也会被重放。
- 键默认为 `DefaultCacheKey` —— 任务输入的规范化 JSON 编码的 sha256
  十六进制摘要（确定性；非 JSON 值报错）。`KeyFunc` 可覆盖它；
  `KeyFunc` 出错会使任务失败。
- `TTL` 是条目生命周期；0 表示条目永不过期。
- `CompiledGraph.ClearCache(ctx, ns)` 清除整个命名空间 —— 执行器把每个
  节点的条目命名空间化为 `"writes/<node>"`。未安装后端时它是 no-op；
  有策略而没有 `WithCache` 后端时策略惰性失效（节点不缓存地执行）。

> **分歧说明：** cache 接口与 `InMemoryCache` 位于 **`checkpoint` 包内、
> 与 `Saver` 并列**，而非独立的 `cache` 包（Python 提供
> `langgraph.cache.*`；放在 `checkpoint` 内保持了无环的依赖方向）。
> Stream chunk **不携带 cached 标志** —— 缓存命中的 `updates` chunk 与
> 实时产生的无法区分。Debug 任务事件（`StreamDebug`）在缓存命中时**仍然
> 触发**（派发与完成），但不会发出 `RawNodeStart`/`RawNodeEnd` 事件对，
> 因为节点从未运行。

## `prebuilt.ToolNode`

`langgraph/prebuilt.ToolNode` 把 `langchain/tools.ToolNode` 适配为图节点：
它执行 messages 状态键中最后一条 AI 消息里的工具调用，并为每个调用追加
一条 `ToolMessage`。典型的 model ↔ tools 循环：

```go
import (
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/prebuilt"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

g := graph.NewStateGraph()
g.AddReducer("messages", channels.MessagesReducer) // required, see below
g.AddNode("model", modelNode)                      // appends an AI message
g.AddNode("tools", prebuilt.ToolNode(toolNode))    // toolNode: *tools.ToolNode
g.AddEdge(types.START, "model")
g.AddConditionalEdges("model", func(_ context.Context, state map[string]any) ([]any, error) {
	msgs, _ := state["messages"].([]messages.Message)
	if len(tools.PendingToolCalls(msgs)) > 0 {
		return graph.To("tools"), nil
	}
	return graph.To(types.END), nil
})
g.AddEdge("tools", "model")
```

- **Reducer 要求：** messages 键需要 append reducer ——
  `AddReducer(key, channels.MessagesReducer)` —— 否则默认的 LastValue
  channel 会在每次更新时替换掉消息历史。
- `WithMessagesKey(key)` 更改读写的状态键（默认 `"messages"`，即 Python
  的 `messages_key`）。
- 执行与错误处理原样委托给被包装的 `tools.ToolNode`：调用在节点内并发
  执行，工具错误按其 `HandleToolErrors` 变为错误 `ToolMessage`。完整图
  状态作为 `ToolCallRequest.State` 传入，因此工具能看到 Python
  `InjectedState` 提供的只读上下文。
- **Command 约定：** 工具通过在 `Result.Artifact` 中放置
  `*types.Command` 来表达图控制流（经 `ToolNode.InvokeToolCallsFull`
  浮出水面）。当批次中任一工具返回了 Command，节点的结果就是单个合并的
  `*types.Command`：messages 更新始终出现在其 `Update` map 中，各命令
  的 `Update` map 合并进其中，`Goto` 列表拼接；`Update` 键、`Graph` 或
  `Resume` 冲突时，按调用顺序靠后的命令胜出。

> **分歧说明：** 与 `langchain/tools.ToolNode` 的范围一致，**不提供按
> 工具调用派发 Send**（Go 在一个节点内并发执行整个批次），也**不提供
> 基于反射的参数注入**（`InjectedState` / `InjectedStore` /
> `ToolRuntime`）—— `ToolCallRequest.State` 是显式传入的。

## Join 边（`AddJoinEdge`）

`AddJoinEdge([]string{"a", "b"}, "c")` 对齐 Python 的
`add_edge(("a", "b"), "c")` —— 一条 *waiting edge*，背后是与
`NamedBarrierValue` 等价的 barrier channel
（`libs/langgraph/langgraph/graph/state.py:956-966`、
`libs/langgraph/langgraph/channels/named_barrier_value.py`）。子节点在所有
父节点都提交后恰好运行一次，无论它们在同一超步内完成还是相隔多个超步：

```go
g := graph.NewStateGraph()
g.AddNode("a", nodeA) // nodeA/nodeB/nodeC: graph.NodeFunc
g.AddNode("b", nodeB)
g.AddNode("c", nodeC)
g.AddEdge(types.START, "a")
g.AddEdge(types.START, "b")
g.AddJoinEdge([]string{"a", "b"}, "c") // c runs once, after BOTH a and b commit
g.AddEdge("c", types.END)
```

- 多个父节点在同一超步内完成，子节点也只触发一次，在下一个超步执行。
- 到达记录会进 checkpoint：若父节点 `a` 已到达而父节点 `b` 被中断，稍后
  resume `b` 仍能补全 barrier —— 部分到达集合在 checkpoint 中存活。
- 在循环图中，子节点提交后 barrier 复位（`Consume`），因此它会在下一轮
  重新武装、可再次触发。
- `join:a+b:c` barrier channel 是控制面状态：从不出现在节点输入、快照或
  stream 输出中。

`AddJoinEdge` 链式返回 `*StateGraph`。校验失败经 `setErr` 累积、在
`Compile` 时报错：至少 2 个去重后的父节点（重复父节点报错）；父节点与
子节点都必须是已注册的节点（在 Compile 时检查，与 `AddEdge` 一致）；
父节点与子节点均不得为 `types.START` 或 `types.END`。

> **警告（OR 语义，Python 对齐）：** 指向 join 子节点的普通边、条件边、
> `types.Send` 或 `Command.Goto` 会绕过 barrier 直接触发子节点。同一节点
> 混用两类边可能使其运行多次 —— 这是 Python 的既定行为，Go 忠实复刻，
> 不是 bug。

与 Python 的分歧：Go 要求 ≥2 个去重父节点（Python 接受单元素元组并静默
去重）；Go 拒绝 `types.END` 作为 join 子节点，而 Python 允许
（`state.py:963-964` 的校验只拒绝 `START` 作终点、`END` 作起点）。

> **不支持：** `defer=True` / `NamedBarrierValueAfterFinish` —— 边驱动
> 执行器模型中没有等价物。

## 图导出（`GetGraph` / `DrawMermaid`）

`StateGraph` 与 `CompiledGraph` 都可导出图的静态结构，对齐 Python 的
`get_graph()` + `draw_mermaid()`
（`libs/langgraph/langgraph/pregel/main.py:845`、
`libs/core/langchain_core/runnables/graph_mermaid.py`）：

```go
cg, _ := g.Compile()
fmt.Println(cg.DrawMermaid()) // == cg.GetGraph().DrawMermaid()
```

`GetGraph()` 返回 `core/runnables.Graph`：全部已注册节点加上
`types.START` / `types.END` 哨兵（绘制为圆角的首/末节点）、静态边与
join 边（每个 join 父节点到子节点各一条实线），条件边绘制成虚线。
Mermaid 输出遵循 Python 格式 —— `id(label)` 节点、
`-- &nbsp;label&nbsp; -->` 带标签边、`-.->` 条件边，以及末尾的
`classDef` 样式。

Python 通过 mock state 干跑（dry-run）整张图来发现条件边目标；本移植改为
**最佳努力探测**（默认开启，可用 `GetGraph(WithRouterProbing(false))`
关闭）：用普通 `AddConditionalEdges` / `SetConditionalEntryPoint` 注册的
router 会被空状态调用一次，返回的节点名、`types.END`、`*types.Send` 目标
绘制成虚线边。报错或 panic 的 router（例如读取了探测空状态中不存在的
state key）在图中会被省略。若需静态可枚举、编译期校验的边，请改用带
path map 的注册方式 —— 即 Python 的 `path_map`：

```go
g.AddConditionalEdgesWithPathMap("model", router, map[string]string{
	"tools": "tools",     // key == target: 边不带标签
	"end":   types.END,   // key 不同: 绘制成 ` -. &nbsp;end&nbsp; .-> `
})
```

### 子图展开（`WithXrayDepth`）

子图节点（`AddSubgraph`）默认绘制为单节点。传入 `WithXrayDepth` —— 即
Python 的 `xray` —— 可原地展开：负数深度展开所有嵌套层级（对应 Python 的
`xray=True`），正数深度展开对应层数。被展开的节点替换为其子图（裁掉
START/END 哨兵）并以 `parent:child` 前缀并入，父图中指向该节点的边重接到
子图的首/末节点，Mermaid 按前缀渲染 `subgraph` 块：

```go
fmt.Println(cg.GetGraph(graph.WithXrayDepth(-1)).DrawMermaid())
```

### Mermaid 选项、interrupt 元数据与 ASCII 图

`DrawMermaid` 透传 `core/runnables` 的 mermaid 选项，对齐 Python
`draw_mermaid` 的关键字参数：

```go
cg.DrawMermaid(
	runnables.WithStyles(false),       // 只输出 `graph TD;` + 边
	runnables.WithCurveStyle("basis"), // frontmatter config.flowchart.curve
	runnables.WithFrontmatterConfig(map[string]any{"theme": "neutral"}),
	runnables.WithNodeStyles(runnables.NodeStyles{Default: "fill:#f9f"}),
)
```

经 `WithInterruptBefore` / `WithInterruptAfter` 注册的节点在导出图中携带
`__interrupt = before` / `after` / `before,after` 元数据，并渲染进节点标签
（仅编译后的图 —— 与 Python 一致，边界是 compile 选项）。

`runnables.Graph.DrawASCII` 把同一张图渲染为 ASCII 盒线图（Python 的
`draw_ascii`）。布局采用 Go 自实现的 longest-path 分层而非 Python 的
grandalf Sugiyama，输出遵循同一惯例但不与 Python 逐字节一致。

与 Python 的分歧：router 目标发现是最佳努力探测（Python 是完整 dry-run
模拟）；ASCII 布局不逐字节对齐；不支持 PNG 渲染
（`runnables.Graph.DrawPNG` 报错）。

## Postgres checkpoint saver（`langgraph/checkpoint/postgres/`）

基于 PostgreSQL 的持久化 `checkpoint.Saver` —— Python
`langgraph-checkpoint-postgres`（`BasePostgresSaver`）的移植。它对齐
Python 的 schema —— 四张表（`checkpoints` / `checkpoint_blobs` /
`checkpoint_writes` / `checkpoint_migrations`），channel blob 按版本存储
—— 并在 `Setup` 中应用待执行的 v0–v9 migrations。

> **依赖声明：** 这是本移植的**第二个第三方依赖** —— 纯 Go 驱动
> `github.com/jackc/pgx/v5`。与 SQLite saver 一样，它位于独立的嵌套 Go
> module（`langgraph/checkpoint/postgres/go.mod`，带指回根 module 的
> `replace` 指令），因此根 module 保持零依赖。

```go
import (
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	postgres "github.com/projanvil/langchain-golang/langgraph/checkpoint/postgres"
)

saver, err := postgres.NewFromConnString(ctx,
	"postgres://user:pass@localhost:5432/dbname", serde.NewJSONSerializer())
if err != nil {
	return err
}
defer saver.Close()
// Setup applies the schema and must be called explicitly once before first
// use. It does NOT run inside a transaction — migrations v6–v8 are
// CREATE INDEX CONCURRENTLY, which Postgres forbids in a transaction block.
if err := saver.Setup(ctx); err != nil {
	return err
}
// Use like any checkpoint.Saver: agents.WithAgentCheckpointer(saver), etc.
```

`postgres.New(pool, serde)` 接受一个已有的 `*pgxpool.Pool`，而不是自行
打开连接池。两个 saver 跑同一套共享的 `savertest` 契约套件
（`savertest.Run`），跨后端一致的行为由同一组测试钉死。

由于它是嵌套 module，根 module 的 `go test ./...` **不会**进入其中。其
测试通过 `github.com/fergusstrange/embedded-postgres` 在进程内启动真实
数据库（首次运行需联网下载约 30MB 的 Postgres 二进制；`-short` 跳过这些
测试）：

```bash
make test-postgres
```

> **跨语言数据库不互通：** Go serde 是 JSON + 封闭类型注册表（而非
> Python 的 msgpack），且 `checkpoint_blobs.version` 列在 Go 中是
> `BIGINT`，Python 则是存十进制字符串的 TEXT。一种语言写入的数据库无法
> 被另一种语言读取 —— 每种语言各用各的数据库。

> **分歧说明：** 只有 JSON 原生值 `nil` / `string` / `bool` / `float64`
> 内联进 checkpoint 的 JSONB 文档；`int` / `int64` / `map[string]any` /
> `[]any` 及一切注册表类型都按版本存进 `checkpoint_blobs`（Python 内联
> int/float，把 dict/list 送进 blobs）。没有 Shallow saver 变体，也没有
> delta channel 历史快速路径。更小的分歧 —— 元数据字符串中的 null 字节
> 直接报错而非静默剔除、`Put` 保留 Python 会静默丢弃的无版本复合 channel
> 值 —— 记录在包 godoc 中。

## Interrupt、resume 与 durability

### 暂停：`graph.Interrupt` 与边界中断

节点用 `graph.Interrupt(ctx, value)` 在运行中暂停自身（即 Python 的
`langgraph.types.interrupt`）；`Compile(WithInterruptBefore("tools"))` /
`WithInterruptAfter("tools")` 在命名节点前后暂停（即 Python 的
`interrupt_before=` / `interrupt_after=`）。两种方式都干净地停下，
`InvokeWithOptions` 返回暂停的 `Result` —— `Interrupts` 非空，`Values`
为暂停时刻的状态。线程上的 `GetState` 展示同样的待处理 interrupt
（`StateSnapshot.Interrupts`）。

恢复方式是带 `Options{ThreadID: ..., Resume: ...}` 再次调用。resume 值
会成为被暂停的 `Interrupt` 调用的返回值（节点从头重新执行，并按调用顺序
消费排队的 resume 值）；边界暂停用 nil `Resume` 恢复 —— 即 Python 的
`invoke(None, config)` —— 因为不存在需要回填值的节点内调用。

### Resume 寻址：NS、ID 与 `Options.Graph`

`types.Interrupt` 同时携带 `ID`（节点作用域，如 `ask-1`）与 `NS` ——
发起中断的任务的完整 checkpoint 命名空间 —— resume 匹配优先查 NS：

- 标量 `Resume` 喂给唯一的待处理 interrupt；待处理 interrupt 多于一个时
  标量是错误 —— 改用 map。
- `map[string]any` 形式的 `Resume` 先按 `NS`、再按 `ID` 寻址各 interrupt
  （`NS` 字段出现之前持久化的 interrupt 反序列化出空 NS，仅按 ID 匹配）。
  两个键都未命中的 interrupt 不会被喂值：其 `Interrupt` 调用再次触发，
  运行带着剩余部分再次暂停 —— 部分寻址的 resume 会再次暂停。
- `Options.Graph` 把匹配收窄到一个命名空间（即 Python 的
  `Command(resume=..., graph=ns)`）：可以是该 NS 本身或嵌套其下的任何
  interrupt。此时只要命中恰好一个待处理 interrupt，标量也可用；命中零个
  待处理 interrupt 的 `Graph` 是列出可用 NS 值的描述性错误；有 `Graph`
  而无 `Resume` 是错误；`types.ParentGraph` 被拒绝。`Graph` 与 `Resume`
  一起传播进子图运行，因此用目标 NS 的任意祖先命名空间寻址嵌套
  interrupt 在任意深度都有效。

### 子图中断暂停父运行

经 `AddSubgraph` 注册的子图若发生中断 —— 节点内 `Interrupt` 或它自己的
编译期边界 —— 不再使父运行报错。节点包装器把子图的暂停位置记录进父图的
暂停 checkpoint（`Metadata.Parents`），并把子图的 interrupt 原样上抛 ——
每个都携带其完整嵌套 NS
（`<parentNS>/<node>:<taskID>/<inner>:<taskID>`）—— 作为子图节点的
interrupt 结果。父图像任何普通暂停一样在 `Result.Interrupts` 中返回它们；
随后的 `Options.Resume` 被转发进子图运行，后者钉在其自身的暂停
checkpoint 上并从那里继续。

- 单个待处理子图 interrupt 在任意嵌套深度下都可用标量 `Resume`；按 NS
  为键的 map 按发起中断的任务分发（例如子图内的 `Send` 扇出按任务各中断
  一次：每个 interrupt 在共享的子命名空间下有各自的 NS）。
- 每个子图任务在各自的 `<parentNS>/<node>:<taskID>` 命名空间下
  checkpoint，因此进入同一子图节点的 `Send` 扇出 —— 以及循环节点的每次
  重新执行 —— 都获得独立命名空间；并发或重复的调用绝不共享（也绝不
  分叉）同一段 checkpoint 历史。
- 跨进程 resume 可用：恢复只依赖已持久化的线程状态，绝不依赖内存中的
  运行状态。在同一个 saver 上重新 `Compile` —— 或换一个进程（配 SQLite
  或 Postgres saver）—— 仅凭 `Options{ThreadID, Resume}` 即可恢复。

### Durability：编译期默认值与按运行覆盖

`Compile(WithDurability(d))` 选择执行器何时刷新 checkpoint 写入，对齐
Python 的 `langgraph.types.Durability`：`DurabilitySync`（下一个超步开始
前同步持久化每个超步 —— Go 默认）、`DurabilityAsync`（后台写入器，返回前
刷新；Python 的运行期默认）与 `DurabilityExit`（运行结束时延迟刷新）。
Python 在运行期解析 durability；Go 在编译期解析 —— 这是已文档化的分歧 ——
并提供等价的按运行覆盖。

单次运行可用 `Options.WithRunDurability` 覆盖编译期模式，对齐 Python
invoke/stream 的 `durability` 参数；覆盖会传播进子图运行：

```go
opts := graph.Options{ThreadID: "t"}.WithRunDurability(graph.DurabilityExit)
cg.InvokeWithOptions(ctx, input, opts)
```

暂停 checkpoint —— interrupt、interrupt_before/after 边界 —— 在所有
durability 模式下都同步刷新（async/exit 运行走专门的暂停路径而非延迟
刷新），因此暂停的运行始终可持久化、可恢复。

## 函数式 API（`langgraph/fn`）

函数式 API —— Python `langgraph.func`（`@entrypoint` / `@task`）的移植
—— 用普通 Go 控制流而非显式图来构建带 checkpoint 的工作流：循环、分支
与并发就是普通的 Go 代码，而每个任务的结果都会进 checkpoint，因此被中断
的运行在恢复时无需重新执行已完成的工作。`Entrypoint` 编译为单节点
`StateGraph`（三个保留 channel `__start__` / `__end__` / `__previous__`），
因此 interrupt/resume、流式输出与时间旅行全部来自现有执行器。控制流是
动态、数据依赖的场合用函数式 API；希望拓扑显式、可检视、可静态校验的
场合用 `StateGraph`。

### Entrypoint

`NewEntrypoint` 把一个函数包装为可调用的工作流：

```go
import (
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/fn"
	"github.com/projanvil/langchain-golang/langgraph/graph"
)

entry := fn.NewEntrypoint(fn.EntrypointOpts{
	Checkpointer: checkpoint.NewMemorySaver(),
}, func(ctx context.Context, in string, prev int, hasPrev bool) (int, error) {
	total := len(in)
	if hasPrev {
		total += prev
	}
	return total, nil // also saved as the next invocation's `previous`
})

n, err := entry.Invoke(ctx, "hello", graph.Options{ThreadID: "t-1"}) // n == 5
n, err = entry.Invoke(ctx, "hi", graph.Options{ThreadID: "t-1"})     // n == 7
```

`graph.Options.ThreadID`（配合 checkpointer）把多次调用串成一条线程；
没有 checkpointer 时每次运行都是无状态的。`EntrypointOpts` 还可接受
`checkpoint.Cache` 后端（供任务 cache 策略使用）、`graph.RetryPolicy`
（作为图级默认安装，因此既重试 entrypoint 函数，也重试没有自身
`TaskOpts.Retry` 的每个 task），以及 `store.Store`（跨线程 BaseStore，
见本节末尾的说明）。

### 跨轮状态：`previous`

`prev` 是同一线程上一次已完成调用的保存值；当没有 checkpointer、没有
ThreadID、或没有先前已完成的调用时，`hasPrev` 为 false —— 此时 `prev`
为零值。

> **分歧说明：** Python 在没有保存值时传 `previous=None`；Go 用显式的
> `hasPrev bool`，使合法保存的零值永远不会被误读为"没有保存值"。

### 解耦返回值与保存值：`Final`

普通 `NewEntrypoint` 中返回值同时充当保存值，因此输出类型必须可赋值给
保存类型。`NewEntrypointFinal` 返回 `Final[O, S]` 以解耦二者（Python 的
`entrypoint.final(value=, save=)`）：

```go
entry := fn.NewEntrypointFinal(fn.EntrypointOpts{Checkpointer: saver},
	func(ctx context.Context, in string, prev []string, hasPrev bool) (fn.Final[string, []string], error) {
		history := append(slices.Clone(prev), in)
		return fn.Final[string, []string]{
			Value: "ack: " + in, // returned to the caller
			Save:  history,      // threaded into the next run's `prev`
		}, nil
	})
```

### Task

`NewTask` 把函数包装为具名的、可经 checkpoint 重放的工作单元。`Call`
立即在它自己的 goroutine 中启动任务 —— 没有 Python 式的"下一 tick"调度，
因为 Go 执行器是边驱动的、没有 tick 概念 —— 并返回 `Future`；`Get`
等待结果。`AwaitAll` 收集一批 future：

```go
fetch := fn.NewTask("fetch", func(ctx context.Context, url string) (string, error) {
	return httpGet(ctx, url)
}, fn.TaskOpts{})

entry := fn.NewEntrypoint(fn.EntrypointOpts{Checkpointer: saver},
	func(ctx context.Context, urls []string, prev int, hasPrev bool) ([]string, error) {
		futs := make([]*fn.Future[string], len(urls))
		for i, u := range urls {
			futs[i] = fetch.Call(ctx, u) // concurrent: each Call starts at once
		}
		return fn.AwaitAll(ctx, futs...)
	})
```

`Call` 只能在 entrypoint 函数内、另一个 task 内、或 StateGraph 节点内经
节点内的 `Entrypoint.Invoke` 触达（运行期 dispatcher 随 context 传递）；
在任何其他地方调用都会 panic。任务名在一个 entrypoint 的调用图内必须
唯一 —— 它标识确定性任务 ID 与 cache 命名空间中的该任务。

> **分歧说明：** 不存在裸的 task-in-StateGraph-node 形态 —— Python 在
> 节点内直接调用的 `@task` 依赖 Pregel config 注入，无 Go 等价物。Go 的
> 形态是在节点内 `Invoke` 一个 `Entrypoint`（在 `NodeFunc` 内
> `add.Invoke(ctx, ...)`）。

### 任务策略：retry、cache、timeout

`TaskOpts` 对齐 `@task(retry_policy=..., cache_policy=..., timeout=...)`
装饰器参数：

- **Retry** —— `graph.RetryPolicy`，语义与节点级 retry 相同；nil 表示
  永不重试。
- **Cache** —— `graph.CachePolicy`，除非所属 entrypoint 安装了
  `checkpoint.Cache` 后端（`EntrypointOpts.Cache`），否则惰性失效 —— 与
  节点缓存"有策略无后端即惰性"的规则相同。只缓存成功的结果。键是调用
  参数的哈希，命名空间为 `__fn_writes/<task-name>`；自定义 `KeyFunc`
  收到的参数打包为 `map[string]any{"input": in}`（Python 的 `key_func`
  收到的是 `*args/**kwargs` —— 文档化分歧）。
- **Timeout** —— 限制每次尝试的时长。goroutine 无法被强杀，因此超时
  只能取消该次尝试的 context 并放弃等待；被放弃的尝试仍在后台继续运行，
  所以任务函数应当响应 context 取消。（Python 对同步任务函数同样不支持
  timeout。）

### Interrupt 与 resume

entrypoint —— 或其内部的 task —— 调用 `graph.Interrupt(ctx, value)` 暂停
运行以等待外部输入。此时 `Invoke` 返回零值输出与携带待处理 interrupt 的
`*fn.InterruptError`（用 `errors.As` 取出）。恢复方式是在同一线程上再次
调用并传入 `graph.Options{ThreadID: ..., Resume: ...}`：resume 值会成为
被暂停的 `Interrupt` 调用的返回值；多个 interrupt 的 resume 值按索引顺序
匹配：

```go
entry := fn.NewEntrypoint(fn.EntrypointOpts{Checkpointer: saver},
	func(ctx context.Context, in string, prev int, hasPrev bool) (string, error) {
		approved, _ := graph.Interrupt(ctx, map[string]any{"draft": in}).(bool)
		if !approved {
			return "rejected", nil
		}
		return publish(ctx, in)
	})

_, err := entry.Invoke(ctx, "draft text", graph.Options{ThreadID: "t-1"})
var ierr *fn.InterruptError
if errors.As(err, &ierr) {
	// ierr.Interrupts[0].Value == map[string]any{"draft": "draft text"}
	out, err := entry.Invoke(ctx, "", graph.Options{ThreadID: "t-1", Resume: true})
	// input is ignored on resume; out is whatever publish returned
	_ = out
}
```

`Entrypoint.Stream` 与 `Invoke` 一样运行并吐出 chunk，但模式固定为
`updates`（Python entrypoint 的默认 `stream_mode="updates"`）。

> **分歧说明：** 单次任务调用不产生 stream chunk —— 任务在 entrypoint
> 节点内执行，不是图任务（Python 的 PUSH 任务会流式输出逐任务更新）。

### Checkpoint 重放与确定性

resume 时 entrypoint 函数**从头重跑**；每个 `Call` 的确定性任务 ID ——
由恢复所用 checkpoint ID、checkpoint namespace、step、任务名、父调用
路径与按路径计数的调用索引组成的哈希 —— 若
命中 checkpoint 中的 pending write，则直接用持久化结果填充 future 而
**不重新执行**（错误同样持久化，并从 `Get` 重抛）。只有当重放是确定
的时候，这个模式才正确：

- 同一 entrypoint 的多次重放中，任务调用顺序必须确定（调用计数器每次
  重放都从零重计，且按父调用路径独立计数）—— 把非确定性逻辑（时间、
  随机、网络）放进 task 内，绝不要
  放进 entrypoint 的控制流。Interrupt 同样必须以确定的 `Get` 顺序浮现。
- 运行因 interrupt 暂停时，已启动但未完成的 task 被取消；已完成的结
  果在暂停前落入 checkpoint 的 pending writes。
- `I` / `O` / `S` 及任务的输入输出必须能经 checkpoint serde 往返
  （JSON 原生值或封闭类型注册表）；使用持久化 saver 时，未注册类型是
  明确的描述性错误，永不静默降级。

> **Store：** Python `@entrypoint(checkpointer=..., store=...)` 的跨线
> 程 `BaseStore` **已支持**：设置 `EntrypointOpts.Store`（如
> `store.NewInMemoryStore()`），它经 `graph.WithStore` 安装到内部图，并在
> entrypoint 函数及其派发的每个 task 内以 `rt.Store` 浮现 —— 同一实例跨调
> 用、跨线程共享（使用前需判 nil）。内存 store 还支持语义检索：
> `store.NewInMemoryStoreWithIndex(store.IndexConfig{Embed: embedder,
> Fields: []string{"text"}})`（对齐 Python 的 `InMemoryStore(index=...)`）
> 在 `Put` 时对 value 的每个索引字段做嵌入，带 `Query` 的 `Search` 按余弦
> 相似度降序排序；用普通 `NewInMemoryStore` 构建的 store 没有索引，会静默
> 忽略 `Query`（即 Python 的 `index=None`）。其余分歧清单（重放的错误丢失具
> 体类型、失败的运行会"毒化"其线程、cache +
> interrupt-in-task 是不支持的组合，等等）见 `langgraph/fn` 包 godoc。

## Breaking 变更（M5 saver 接口）

M5 为函数式 API 的任务追踪与元数据过滤演进化了 `checkpoint.Saver` 契约
—— 这是 spec 允许的 pre-1.0 break。自定义 `Saver` 实现必须更新：

```go
// before
type ListOptions struct { Before *Config; Limit int }
PutWrites(ctx context.Context, cfg Config, writes []Write, taskID string) error
type Write struct { TaskID string; Channel string; Value any }
// after
type ListOptions struct { Before *Config; Limit int; Filter map[string]any }
PutWrites(ctx context.Context, cfg Config, writes []Write, taskID, taskPath string) error
type Write struct { TaskID string; Channel string; Value any; TaskPath string }
```

自定义 saver 的迁移指引：

- **`PutWrites`** —— 把新的 `taskPath` 参数随每条 write 一并持久化（它
  标识任务在一次运行中的位置，如 `a@0/b@0`），或忽略它并存 `""`。
- **`ListOptions.Filter`** —— 对 checkpoint 元数据的 map 包含语义：当
  元数据包含 filter 的每一个键值对时该 checkpoint 匹配
  （`checkpoint.MetadataMatchesFilter`）。Postgres saver 在服务端用 `@>`
  求值；memory 与 SQLite saver 做等价的进程内比较。Filter 键封闭于
  `source` / `step` / `parents` —— 即 `checkpoint.Metadata` 的字段。
- **M5 之前创建的 SQLite 数据库**继续可用：`sqlite.New` 启动时检测
  `writes` 表缺少 `task_path` 列则以 `ALTER TABLE ... ADD COLUMN` 添加
  （Python 在它自己的 migration v9 中添加了同一列）。

## `create_react_agent` ≡ `prebuilt.CreateReactAgent`

Python 的 `langgraph.prebuilt.create_react_agent` 以
`langgraph/prebuilt.CreateReactAgent(model, toolList, opts...)` 提供。它自
langgraph v1.0 起已在上游被 `langchain.agents.create_agent` 取代，因此这里
是对 `langchain/agents.CreateAgent` 的轻量委托：每个选项都是按原义应用的
`agents.AgentOption`，返回的也是同一个包裹编译图（`agent.Graph`）的
`*agents.Agent`。新代码请直接使用 [`agents.CreateAgent`](../agents.md)；只在
需要 `CreateAgent` 未暴露的图级控制时，才用 `prebuilt.ToolNode` 手工搭建
循环。
