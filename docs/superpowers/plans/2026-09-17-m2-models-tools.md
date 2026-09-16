# M2 模型与工具接入层 Implementation Plan（并入最终 v0.9.1 发布）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 spec §8——MCP adapter（langchain.mcp 蓝图）、Gemini 原生 provider、loaders 三件套。**本里程碑不单独发版**：全部完成后与 M1/M3 合并发布 v0.9.1（用户 2026-09-17 指示）。

**Architecture:** MCP 依赖 mark3labs/mcp-go（锁 minor），elicitation 能力首日 spike（不支持则降级 backlog）；Gemini 优先官方 genai SDK（结构化输出覆盖不足则降级手写 REST）；loaders 纯增量（goquery + ledongthuc/pdf）。

**Spec:** `docs/superpowers/specs/2026-09-16-parity-catchup-p0p1-design.md` §8。

## Global Constraints

继承 spec §10；新依赖须锁版本；MCP/Gemini 必须过 standardtests 相关 conformance；loaders golden 测试；modern-go 规范；conventional commits；**不 push 不发版**。

---

### Task A：MCP adapter（`partners/mcp/`，对标 langchain 1.4.0 `langchain.mcp` 快照）
- 首日 spike：mcp-go client 的 elicitation 能力（不支持→elicitation 进 backlog，其余照做，DIVERGENCES 记录）
- 连接两机制：MCPConfig（聚合）与 ClientGroup（独立连接/`{server}_{tool}` 命名空间）；stdio + streamable-http transports
- 工具发现：list_tools 带 cache_mode（use|refresh|bypass）
- 工具结果映射：text/image/file→content blocks；structured→ToolMessage artifact（MCPToolArtifact 等价）；isError→error ToolMessage（接 HandleToolErrors）；transport/session 失败→向上抛
- elicitation→LangGraph interrupt（需 checkpointer；request key；accept/decline/cancel）
- 工具级 HITL 门控：destructiveHint + InterruptOnConfig→human_in_the_loop.go
- 测试：mcp-go in-process server，无外部依赖

### Task B：Gemini 原生（`partners/gemini/`）
- 首日 spike：google.golang.org/genai 的 responseSchema/structured output、tool config、SSE 流覆盖度（不足→手写 REST 降级，范围不变）
- ChatModel 全接口：ToolCalling、ToolChoice 映射、StructuredOutput、SSE streaming、多模态输入、UsageMetadata、system instruction
- standardtests 分层 chatmodel conformance + 假服务器测试

### Task C：Loaders 三件套（`core/documentloaders/`）
- HTML：goquery；Web：http GET + 复用 HTML 抽取（正文对齐 Python readability 合理子集）；PDF：ledongthuc/pdf
- 全部接 LoadAndSplit；golden 文件测试

---

## 验收（合入 v0.9.1 前的门禁）

`go build/test` 全绿 + 嵌套模块全绿 + lint 0 + `make check-dep-alignment`；MCP in-process 测试全过；Gemini conformance 过；loaders golden 过。
