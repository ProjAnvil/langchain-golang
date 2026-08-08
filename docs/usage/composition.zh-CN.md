# 组合 runnable（LCEL）

**Languages:** [English](composition.md) | 简体中文

LangChain 的 **LCEL**（LangChain Expression Language）让你在 Python 中用
`|` 运算符串联组件：

```python
chain = prompt | model | parser
```

Go **不**支持运算符重载，因此没有 `|`。Go 等价物位于
[`core/runnables`](../../core/runnables)，改用自由函数：

```go
chain := runnables.Pipe3(prompt, model, parser)
```

语义完全一致，只是写法不同。每种组合返回的值本身都满足
`Runnable[I, O]`，因此链可以嵌套，并无须适配器即可喂给框架的其余部分
（agent、fallback、retry 等等）。

## Runnable 契约

每个可组合的值都实现 `runnables.Runnable[I, O]`：

```go
type Runnable[I, O any] interface {
	Invoke(ctx context.Context, input I, opts ...Option) (O, error)
	Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error)
	Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error)
	InputSchema() schema.Schema
	OutputSchema() schema.Schema
}
```

`core/language.ChatModel`、`core/tools.Tool` 适配器、`prompts.*` 以及下文的
组合子都满足这个接口。因为它是泛型，类型链（`I → A → B → O`）在**编译期**
检查 —— 比 Python 的运行时 Pydantic 校验更强。

## Pipe —— `|` 的等价物

`Pipe(a, b)` 组合两个 runnable，使前者的输出喂给后者：

```go
import "github.com/projanvil/langchain-golang/core/runnables"

upper := runnables.NewFunc(
	func(_ context.Context, s string, _ ...runnables.Option) (string, error) {
		return strings.ToUpper(s), nil
	},
	schema.String(""), schema.String(""),
)
exclaim := runnables.NewFunc(
	func(_ context.Context, s string, _ ...runnables.Option) (string, error) {
		return s + "!", nil
	},
	schema.String(""), schema.String(""),
)

chain := runnables.Pipe(upper, exclaim)
out, _ := chain.Invoke(context.Background(), "hello")
// out == "HELLO!"
```

`Pipe` 返回满足 `Runnable[I, O]` 的 `SeqN[I, O]`。

## 更长的链：Pipe3 .. Pipe6

两步以上时，为保持可读性使用 `Pipe3`、`Pipe4`、`Pipe5`、`Pipe6`。每多一步
多一个类型参数，全部在编译期检查：

```go
// Python:  prompt | model | parser
// Go:
chain := runnables.Pipe3(prompt, model, parser)

// Four stages (e.g. retrieve → grade → generate → parse):
chain := runnables.Pipe4(retriever, grader, generator, parser)
```

### 超过六步：嵌套 Pipe

`SeqN` 本身也是 `Runnable`，更长的链靠嵌套。结果是类型安全的，并在运行时
透明地拍平：

```go
chain := runnables.Pipe(
	runnables.Pipe3(a, b, c),
	runnables.Pipe3(d, e, f),
)
```

## Parallel —— `RunnableParallel`

在同一输入上并行运行多个 runnable，把带键的输出收集进一个 map：

```go
pipe := runnables.NewParallel(map[string]runnables.Runnable[string, any]{
	"length":   lengthRunnable,
	"upper":    upperRunnable,
	"reversed": reverseRunnable,
})
out, _ := pipe.Invoke(context.Background(), "hello")
// out == map[string]any{"length": 5, "upper": "HELLO", "reversed": "olleh"}
```

## Branch —— `RunnableBranch`

按条件在运行时挑选分支：

```go
branch, _ := runnables.NewBranch(
	[]runnables.BranchCase[map[string]any, []messages.Message]{
		{Condition: isSmalltalk, Runnable: chitchatRunnable},
		{Condition: needsRetrieval, Runnable: ragRunnable},
	},
	defaultRunnable, // used when no condition matches
)
```

## Fallback 与 retry

包装一条链，失败时落到备选，对应 Python 的 `.with_fallbacks(...)` 与
`.with_retry(...)`：

```go
primary := runnables.Pipe(prompt, expensiveModel)
resilient, _ := runnables.NewWithFallbacks(primary, cheapModel, cachedModel)

// Retry transient failures up to 3 times:
retrying, _ := runnables.NewRetry(resilient, 3)
```

## Passthrough 与 Assign —— 搭建 RAG 风格的链

`NewPassthrough` 原样转发输入。`NewAssign` 向 map 输入追加计算得到的键 ——
这是把 context 喂进 prompt 的标准模式：

```go
// Python:
//   chain = RunnablePassthrough.assign(context=retriever) | prompt | model
//
// Go:
rag := runnables.Pipe(
	runnables.NewAssign(map[string]runnables.Runnable[map[string]any, any]{
		"context": retrieverRunnable, // input map → retrieved docs string
	}),
	runnables.Pipe3(ragPrompt, model, parser),
)
```

## 把链接进 agent

`SeqN[I, O]` 是 `Runnable`，组合好的链可以直接插入现有的组合子体系 ——
以及任何消费 `Runnable` 的地方：

```go
// A Pipe result used as a fallback target inside an agent's middleware:
primary := runnables.Pipe(retrieve, summarize)
resilient, _ := runnables.NewWithFallbacks(primary, fallbackSummarizer)
```

## Python ↔ Go 速查表

| Python LCEL | Go |
|-------------|-----|
| `a \| b` | `runnables.Pipe(a, b)` |
| `a \| b \| c` | `runnables.Pipe3(a, b, c)` |
| `a \| b \| c \| d \| e \| f` | `runnables.Pipe6(a, b, c, d, e, f)` |
| longer chains | nest `Pipe(Pipe3(...), Pipe3(...))` |
| `RunnableParallel({...})` | `runnables.NewParallel(map[string]Runnable[I, any]{...})` |
| `RunnableBranch([...])` | `runnables.NewBranch(cases, default)` |
| `.with_fallbacks([...])` | `runnables.NewWithFallbacks(primary, ...fallbacks)` |
| `.with_retry()` | `runnables.NewRetry(r, attempts)` |
| `RunnablePassthrough` | `runnables.NewPassthrough[I](schema)` |
| `RunnablePassthrough.assign(...)` | `runnables.NewAssign(map[string]Runnable[map[string]any, any]{...})` |
| `RunnableLambda(fn)` | `runnables.NewFunc(fn, inSchema, outSchema)` |

## 有意不做的部分

遵循本项目的有界移植立场，以下 LCEL 周边特性**不**实现：

- **`|` 运算符本身** —— Go 没有运算符重载；用 `Pipe*`。
- **链上的 `astream_log` / `astream_events`** —— 组合链上的流式使用
  `Runnable.Stream`（拉取式）；agent 级的事件流走 `Agent.StreamEvents`
  （见 [streaming](streaming.md)）。
- **Pydantic 支撑的 schema 校验** —— schema 是 `schema.Schema`
  （`map[string]any`）；它们描述形状，但不在运行时校验。
- **PNG 图渲染** —— 支持 JSON / ASCII / Mermaid 图导出
  （`runnables.GetGraph`）；不支持 PNG。
