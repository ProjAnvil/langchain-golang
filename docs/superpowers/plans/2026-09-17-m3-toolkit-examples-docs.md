# M3 实用层+门面 Implementation Plan（并入最终 v0.9.1 发布）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 spec §9——SQL toolkit、examples/ 目录、mkdocs 双语文档站、README parity 刷新。与 M1/M2 合并发布 v0.9.1（不单独发版）。

**Architecture:** SQL toolkit 基于 `database/sql`（sqlite/postgres introspection 起步，只读护栏）；examples 是顶层可运行目录；文档站 mkdocs-material 双语 + GitHub Pages workflow；README parity 表刷新至本计划全部新能力。

**Spec:** `docs/superpowers/specs/2026-09-16-parity-catchup-p0p1-design.md` §9。

## Global Constraints

继承 spec §10；examples 全部 `go build` 可过（无 key 用 ollama/fake）；`mkdocs build` 可过；modern-go 规范；conventional commits；不 push 不发版。

---

### Task A：SQL toolkit（`langchain/toolkits/`）
- 四工具：Query（只读 SELECT）、ListTables、SchemaInfo、QueryChecker；基于 database/sql
- 只读护栏：单语句、仅 SELECT（拒绝 DML/DDL/多语句/分号拼接）
- introspection：sqlite + postgres（database/sql 元信息 + 方言查询）
- 现代 create_agent 组合（对齐上游推荐用法，非 create_sql_agent）
- 测试：sqlite 内存库全链路 + postgres env-gated

### Task B：examples/ 目录
- 12-15 个可运行示例（每例 main.go + 双语 README 简注）：quickstart、HITL、streaming、subgraph 断点恢复、容错（retry/timeout/error handler）、RAG 全链路（loader→split→embed→pgvector→retrieve→rerank→agent）、MCP tools、Gemini、SQL agent、常用 middleware、TracePolicy 脱敏、multiquery/ensemble
- 无 key 可跑的用 fake model/ollama；有 key 的读 env

### Task C：mkdocs 文档站 + README/parity 刷新
- mkdocs-material 双语：docs/usage 现有 10 文件迁入 + 新增（RAG 栈、MCP、Gemini、SQL toolkit、容错/TracePolicy 各一篇双语）
- `.github/workflows/docs.yml`：GitHub Pages 部署
- README 双语：parity 表加 pgvector/redisvector/rerank/高级 retrievers/MCP/Gemini/loaders/SQL toolkit；What's New 预留 v0.9.1；Partner 表加 Gemini/MCP/Cohere/Jina

---

## 验收（合入 v0.9.1 前门禁）

examples 全部 `go build ./examples/...` 过；`mkdocs build` 过（mkdocs 本地可用时）；全仓测试/lint/嵌套门禁绿。
