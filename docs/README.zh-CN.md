# langchain-golang 文档

**Languages:** [English](README.md) | 简体中文

本目录存放 **langchain-golang** 的使用文档——它是
[LangChain](https://github.com/langchain-ai/langchain) 的社区 Go 移植版。

完整 API 参考见
[pkg.go.dev](https://pkg.go.dev/github.com/projanvil/langchain-golang) 上的包文档。
已支持的特性、范围与已知缺口，见仓库根目录的
[README](../README.md)。

## 使用指南

| 指南 | 内容 |
|------|----------------|
| [入门指南](usage/getting-started.md) | 安装、配置 provider、运行你的第一个 agent |
| [组合 runnables（LCEL）](usage/composition.md) | `Pipe` / `Pipe3-6` / `Parallel` / `Branch` / `Fallbacks` / `Retry` —— 等价于 Python 的 `prompt \| model \| parser` |
| [Agents —— `CreateAgent`](usage/agents.md) | system prompt、工具、middleware、结构化输出、interrupt、state/context schema |
| [图运行时（langgraph/）](usage/langgraph.md) | StateGraph、checkpoint、Stream 模式、saver、join edge、函数式 API |
| [流式输出](usage/streaming.md) | `Agent.StreamEvents`：逐 token 的模型增量 + 工具/节点生命周期事件 |

## 双语约定

本仓库的文档为中英双语（英文 + 简体中文）：

1. 英文文件为主文件；中文译文以同目录 `<name>.zh-CN.md` 姊妹文件的形式存放。
2. 两份文件顶部（标题行之下空一行后）互放语言切换行
   （`English | 简体中文`）。
3. 新增或修改的文档内容必须在同一次变更中双语同步落笔——不接受先落英文版、
   之后再补翻译。

## 仓库中的示例

可运行、经编译检查的示例以 `example_test.go` 文件的形式与代码放在一起
（`go test ./...` 会校验其 `// Output:` 块）：

- [agents 示例](../langchain/agents/example_test.go) —— 最小 `CreateAgent`、`provider:model` 字符串解析
- [chatmodels 示例](../langchain/chatmodels/example_test.go)
- [core/tools 示例](../core/tools/example_test.go)
- [core/language 示例](../core/language/example_test.go)

## 约定

- 所有示例默认离线运行——它们使用 `language.FakeChatModel`，无需 API key
  即可运行。实际使用时换成某个 partner `ChatModel`
  （`partners/openai`、`partners/anthropic`、`partners/ollama`）。
- 需要 Go 1.23+。
