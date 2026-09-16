# 容错与追踪隐私

**Languages:** [English](../fault-tolerance.md) | 简体中文

LangGraph 的节点级韧性与隐私策略，经 `AddNodeWithPolicies` 安装。三个独立
开关可自由组合：`RetryPolicy`（瞬态故障）、`ErrorHandlerPolicy`（重试耗尽
后的恢复）、`TracePolicy`（任何 tracer 观察到事件之前的载荷脱敏）。可运行
示例：
[`examples/fault-tolerance`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/fault-tolerance)
与
[`examples/trace-policy`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/trace-policy)。

## 安装

```bash
go get github.com/projanvil/langchain-golang
```

## RetryPolicy

```go
g.AddNodeWithPolicies("flaky_charge", node, graphpkg.NodePolicies{
    Retry: &graphpkg.RetryPolicy{
        MaxAttempts:     3,
        InitialInterval: 10 * time.Millisecond,
        BackoffFactor:   2,
        RetryOn:         graphpkg.DefaultRetryOn, // 用 NonRetryable 包裹错误可立即中止
    },
})
```

重试先于一切；未配置策略时应用图级默认（全部错误都重试，Python parity）。

## 节点错误处理器

`ErrorHandlerPolicy` 恢复重试后仍失败的节点：handler 收到状态快照与类型化
的 `*NodeError`（节点名、尝试次数、包装的错误），返回普通状态更新或
`*types.Command`：

```go
g.AddNodeWithPolicies("capture_receipt", node, graphpkg.NodePolicies{
    ErrorHandler: &graphpkg.ErrorHandlerPolicy{
        Handler: func(state map[string]any, nerr *graphpkg.NodeError) (any, error) {
            return &types.Command{
                Update: map[string]any{"receipt": "queued for manual review"},
                Goto:   graphpkg.To("manual_review"), // 绕开故障节点改道
            }, nil
        },
    },
})
```

持久化语义：任务错误先作为 checkpoint 的 ERROR write 落盘，*然后*才运行
handler——handler 执行中途崩溃，恢复时重跑的是 handler 而非节点
（langgraph 1.2.0 `error_handler=` parity）。

## TracePolicy

`NodePolicies.Trace` 在发射端变换被追踪的载荷——早于 LangSmith、OTel 桥或
任何 callback 观察到它们。输入脱敏、输出整体丢弃（`OmitPayload`），或两者
兼有：

```go
g.AddNodeWithPolicies("guarded_node", node, graphpkg.NodePolicies{
    Trace: &graphpkg.TracePolicy{
        ProcessInputs:  redactEmails,            // func(any) any
        ProcessOutputs: graphpkg.OmitPayload,    // 事件保留，载荷丢弃
    },
})
```

processor panic 时 **fail-closed**（运行中止，绝不泄漏原始载荷——刻意分歧，
见 DIVERGENCES.md）。智能体侧经 `TracePolicyProvider` 钩子对每个 middleware
提供同样控制（langchain 1.3.15）。

## 环境变量

零配置，策略即纯代码。观察脱敏后事件的 tracer 沿用常规环境变量
（`LANGSMITH_TRACING`、`LANGSMITH_API_KEY` 等）。

## 切换点

- **失败 vs 恢复**：不配 `ErrorHandler` 让错误使运行失败；配上则用更新或
  `Command` 改道挽回。
- **脱敏器**：任意 `func(any) any`——正则抹除、JSON 遍历，或载荷整体敏感时
  用 `OmitPayload`。
- **Middleware 追踪**：`CreateAgent` 图中，middleware 安装的
  `TracePolicyProvider` 与节点级策略可组合。
