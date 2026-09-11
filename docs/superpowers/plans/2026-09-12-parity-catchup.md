# 2026-09-12 Parity Catch-up Plan（审计后追齐 Python SDK）

**Goal**: 修复审计发现的所有过期文档/注释，然后按优先级实现剩余 parity 差距，追齐 Python SDK（参考：`../langchain/libs/{core,langchain_v1,partners}`、`../langgraph/libs/{langgraph,prebuilt}`）。

**Global Constraints**
- 刻意设计勿动：blank-import、shim 再导出、cache 短路 middleware、sync/async 合一。
- 实现语义以 Python 参考源码为准（file:line 见各任务）；Go 命名遵循仓库现有惯例。
- 每个代码任务必须 TDD：先写失败测试再实现；改后跑受影响包测试 + `go build ./...`。
- 实现者不改 `docs/usage/*.md`（用户文档在收尾统一同步）；godoc 注释随代码改。
- 不 commit/push（控制器统一提交）；不新增重型依赖（tiktoken 已在 go.mod）。
- 双语 README/usage 文档修改需中英同步。

**任务清单**（Wave1 并行：T1–T5 文件集不相交）

| # | 任务 | 关键文件 | Python 参考 |
|---|---|---|---|
| T1 | 文档/注释过期修复（不含 fn/doc.go、contentblocks.go） | docs/usage/*.md(+zh)、README.md(+zh)、core/indexing/indexing.go、core/prompts/loading.go、core/prompts/prompt.go | — |
| T2 | core token fallback FNV→tiktoken GPT-2 | core/language/tokens.go | core language_models/base.py:98-104 |
| T3 | 图级默认 retry + DefaultRetryOn 放宽 + fn/doc.go #4/#16-17 修正 | langgraph/graph/policy.go、graph.go(Compile)、fn/doc.go | langgraph state.py:275-292、_internal/_retry.py:11-27 |
| T4 | wrap_model_call update-only Command 应用 + Validate 接线 | langchain/agents/middleware/types.go、create_agent.go | langchain_v1 factory.py:216-232 |
| T5 | anthropic 非 base64 data URI 显式报错 + media_type 校验 | partners/anthropic/contentblocks.go | anthropic chat_models.py:195-239 |
| T6 | core BindTools 选项(tool_choice/parallel_tool_calls) + ToolStrategy 强制 "any" | core/language/chatmodel.go、partners/*、langchain/agents/invokeModel | core chat_models.py:2338、langchain_v1 factory.py:1388 |
| T7 | openai：多模态输入、Responses reasoning_effort、流式 usage、Capabilities 校真 | partners/openai/chatmodel.go、chat_completions.go、stream.go、chat_completions_stream.go | openai base.py:622-1060,4282、embeddings |
| T8 | InMemoryStore 语义检索（index= embed + cosine） | langgraph/store/memory.go、store.go | langgraph store/memory/__init__.py:183-292 |
| T9 | anthropic usage cache 细节 + stop_sequences | partners/anthropic/chatmodel.go | anthropic chat_models.py:2361-2382,67 |
| T10 | middleware 自动收集 tools/state_schema | langchain/agents/create_agent.go | factory.py:1005,1056-1075,1154 |
| T11 | 运行时 override 接线（WithResponseFormat/ToolChoice/ModelSettings）+ 动态重检 | langchain/agents/create_agent.go、structured_output.go | factory.py:1323-1404 |
| T12 | agents 杂项：write_todos 落 state、before/after_agent jump、重名校验、AI name | langchain/agents/todo.go、create_agent.go | todo.py:144-149、factory.py:1527-1766 |
| T13 | 流式 ProviderStrategy response_format 绑定 | langchain/agents/create_agent.go、core/language | factory.py:1356-1367 |
| T14+ | 延后大件：HITL interrupt 化、subgraph 中断恢复、astream_events/LangSmith、standardtests 深度、prompts Partial/from_messages、runnables bind/pick/map、durability per-run、subgraph ns node:task_id、openai 采样参数、embeddings 切分 | — | — |

**Wave 编排**: W1={T1..T5} 并行；W2={T6..T9} 并行（T6 先行 core 接口，内部顺序）；W3={T10..T13}（agents 层同文件，顺序或合并）；W4 按余量从 T14+ 取。
