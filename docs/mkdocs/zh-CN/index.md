# langchain-golang 文档

**Languages:** [English](../index.md) | 简体中文

[LangChain](https://github.com/langchain-ai/langchain) 和
[LangGraph](https://github.com/langchain-ai/langgraph) 的社区 **Go 语言端口**
——用纯 Go 构建生产级 LLM 智能体和应用。

> **未与 LangChain, Inc. 有关联或受其认可。** 预览质量；公开 API 在 `v1.0.0`
> 前可能变化。

## 使用指南

| 指南 | 内容 |
|------|------|
| [入门指南](getting-started.zh-CN.md) | 安装、配置 provider、运行你的第一个智能体 |
| [Runnable 组合（LCEL）](composition.zh-CN.md) | `Pipe` / `Parallel` / `Branch` / `Fallbacks` —— Go 版 LCEL |
| [智能体 — `CreateAgent`](agents.zh-CN.md) | 系统提示词、工具、中间件、结构化输出、中断 |
| [流式输出](streaming.zh-CN.md) | 逐 token 模型增量 + 工具/节点生命周期事件 |
| [图运行时](langgraph.zh-CN.md) | StateGraph、checkpoint、流模式、saver、函数式 API |
| [RAG 栈](rag-stack.zh-CN.md) | 过滤 DSL、pgvector、Redis 向量存储、重排器、高级检索器 |
| [MCP 工具](mcp.zh-CN.md) | 将 MCP 服务器接入智能体工具、elicitation、HITL 门控 |
| [Google Gemini](gemini.zh-CN.md) | 基于官方 genai SDK 的原生 Gemini 聊天模型 |
| [SQL 工具箱](sql-toolkit.zh-CN.md) | 基于 `database/sql` 的只读 SQL 智能体工具箱 |
| [容错与追踪隐私](fault-tolerance.zh-CN.md) | 重试策略、节点错误处理器、TracePolicy 脱敏 |
| [文档加载器](loaders.zh-CN.md) | HTML、Web、PDF 三件套加载器 |

## 示例

可运行的自包含示例位于仓库的
[`examples/`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples)
目录：quickstart、流式输出、人机交互中断、subgraph 恢复、容错、TracePolicy
脱敏、RAG 全链路（loader → split → embed → pgvector → retrieve → rerank →
agent）、MCP 工具、Gemini、SQL agent、高级检索器、middleware 套件。每个示例
默认用脚本化或 fake 模型离线运行，通过环境变量切换到真实 provider。

## API 参考

包文档见
[pkg.go.dev](https://pkg.go.dev/github.com/projanvil/langchain-golang)。
