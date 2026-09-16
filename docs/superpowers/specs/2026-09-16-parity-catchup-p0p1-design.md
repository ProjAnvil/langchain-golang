# langchain-golang 全量补齐计划（P0+P1）设计文档

- 日期：2026-09-16（同日经 subagent 审计修订：采纳 P0×5、P1×6、P2 择优）
- 状态：已评审（用户批准设计 v2 与流水线：spec → 审计 → plan → 审计 → 实施 → 审计）
- 关联研究：两轮 PM 研究（版本策略 + 功能完整度）、上游生态审计（2026-09-16，Tavily 核实）、spec 审计（2026-09-16）
- 实施约束：全程使用 modern-go skill；每里程碑收尾前对上游 changelog 做一次 diff 复查

## 1. 背景与目标

Go 移植版（ProjAnvil/langchain-golang，当前 v0.7.1 准备中）的 runtime parity 已达上游水准
（core runtime 评级 A-），但作为开源项目完整度不足：integrations 广度 C-（RAG 实现层空、
MCP 缺失）、工程化 C（无 CI/CHANGELOG/CONTRIBUTING）。本计划以「全量补齐」口径追上
Python SDK 的功能完整度，同时补上审计发现的 6 项计划外高价值缺口。

上游基线（2026-09-16 核实）：langchain Python 1.4.0（2026-09-03）、langgraph 1.2.11、
langchain JS 1.5.11。Go 版 runtime 基线约 langgraph 1.2.10，落后两个 feature：
trace_policy（1.2.11）与 node error handler（1.2.0，Go 版漏做）。

## 2. 决策记录

用户已拍板：

1. P1 自研深度：**全量补齐**（五块全做，含文档站）
2. Vector store：**pgvector + redis** 两家
3. 原生 provider：**Gemini 先行**，Bedrock 进 backlog
4. 版本策略：**不发 1.0，继续 v0.x**；每里程碑一个 minor

用户未及时作答、由我按推荐代答（可推翻）：

5. 新缺口纳入：容错小项 2 件（node error handler + TracePolicy）、RAG 生产栈
   （reranker + 高级 retrievers）、SQL toolkit 进计划
6. LangSmith datasets/evaluate client：backlog（v0.12+ 之后候选）
7. MCP 对标基线：切换到 langchain 1.4.0 的 `langchain.mcp`（standalone
   langchain-mcp-adapters 已被官方弃用）

spec 审计引入的结构决策：

8. M0 拆分为 M0a（工程化）与 M0b（容错/合规 parity），后续里程碑版本号顺延
9. VectorStore filter 走**可选能力接口**（Go interface 封闭，加方法即 breaking；
   沿仓库 TextAdder/mmrSearcher 惯例）
10. TracePolicy 在**事件 emit 侧**应用（派发给任何 tracer 之前）；processor 出错
    **fail-closed**（刻意分歧，上游 fail-open，见 §5.5）
11. node error handler 按**上游完整语义**实现（含 checkpoint ERROR write 与同
    superstep 调度），run loop 改动若不成比例则降级 in-process 并记 DIVERGENCES
12. partners/pgvector 与 partners/redisvector 进 **root module**（pgx/go-redis 成为
    root 直接依赖），CI 增加与嵌套 checkpoint 模块的 pin 对齐检查

## 3. 范围

**做**：M0a 工程化、M0b 容错合规 parity、M1 RAG 生产栈、M2 模型与工具接入层、
M3 实用层+门面。

**不做（维持刻意分歧）**：LangGraph Platform/Studio/Server/CLI（商业线）、transformers
本地模型、Command-Send、remaining_steps、blank-import/shim 再导出设计、cache 短路
middleware。

**Backlog（需求/时机触发）**：LangSmith datasets/evaluate、Bedrock provider、RemoteGraph
client、run.subagents typed channel 投影、deepagents（上游仍 pre-1.0，最新 0.7.x，等 1.0）。

## 4. 里程碑总览

| 里程碑 | 内容 | 版本 | 预估 |
|---|---|---|---|
| M0a 工程化 | CI、四大文档、ParallelToolCalls provider 序列化、核销 | v0.8.0 | ~1 周 |
| M0b 容错合规 parity | node error handler、TracePolicy | v0.8.1 | ~1-1.5 周 |
| M1 RAG 生产栈 | VectorStore filter 能力接口、pgvector、redisvector、reranker 栈、高级 retrievers | v0.9.0 | ~2-2.5 周 |
| M2 模型与工具接入层 | MCP（langchain.mcp 蓝图）、Gemini、loaders 三件套 | v0.10.0 | ~2-2.5 周 |
| M3 实用层+门面 | SQL toolkit、examples/、mkdocs 文档站、README 刷新 | v0.11.0 | ~1.5-2 周 |

总计 ~8-9.5 周。每个里程碑独立走 plan → subagent 审计 → 实施（modern-go skill）→
subagent 审计 → 发版。

## 5. M0a：工程化（v0.8.0）

### 5.1 CI（`.github/workflows/ci.yml`）

- 触发：PR + main push
- 主模块：`go test ./...`（standardtests conformance 已在普通测试内）+
  golangci-lint（新增 `.golangci.yml`，pin lint 版本避免对最新 Go 滞后）；矩阵
  go 1.26（go.mod 下限）与 1.27（当前 stable）
- **嵌套模块**（langgraph/checkpoint/{postgres,redis,sqlite} 为独立 go module，
  `go test ./...` 不覆盖）：redis（miniredis，离线）与 sqlite（原生）的 Makefile
  测试目标进 CI；postgres 用 embedded-postgres（Makefile 已预留「cache that
  directory in CI」提示，非 docker），计划阶段定具体机制
- 不引入 docker/service 容器；外部服务依赖一律 env-gated（对齐现有 e2e 惯例）
- 发布自动化不做（维持 gh 手工发版流程）

### 5.2 ParallelToolCalls provider 序列化（core 已就绪）

现状核实：`core/language/chatmodel.go` 的 `BindToolsOptions` 已含
`ParallelToolCalls *bool`（nil=不发送）；`langchain/agents/create_agent.go` 已把
`ModelSettings["parallel_tool_calls"]` 映射到该选项。剩余工作只在 provider 层：

- openai：Chat Completions 与 Responses 两路径 payload struct 建模
  `parallel_tool_calls` 字段并序列化
- anthropic：合成 `disable_parallel_tool_use`（嵌在 tool_choice 对象内）；
  ToolChoice 未设而 ParallelToolCalls=false 时需合成
  `{"type":"auto","disable_parallel_tool_use":true}`；注意 anthropic 不支持
  ToolChoice=none（现行为报错），named-tool + disable 组合行为在测试中钉死
- openaicompat：经 `partners/openaicompat/providers.go` 直接构造 partners/openai
  ChatModel，天然继承，预计零改动，核验即可
- parity 测试：mock 服务器请求体断言 + 组合矩阵（ToolChoice × ParallelToolCalls，
  含上面两个边界组合）

### 5.3 审计 stale 项核销

对照 2026-09-12 审计清单逐条核验（tool_choice=any、多模态输入已被代码证实完成，
core 侧 ParallelToolCalls 已完成），核销记录写入 CHANGELOG/DIVERGENCES 说明，
不重复实现。

### 5.4 四大文档

- **CHANGELOG.md**：Keep a Changelog 格式；v0.7.x 详填（从 git log 提取），
  v0.5/v0.6 简表；此后每版必更
- **CONTRIBUTING.md**：扩 README 贡献段（代码风格、parity 约定、测试要求、PR 流程）
- **DIVERGENCES.md**：cache 短路 middleware、shim 再导出、transformers/Command-Send/
  remaining_steps 不做、Platform 永不对齐、deepagents 缓做（等 1.0）、RemoteGraph/
  Bedrock 需求触发、§5.5 的 TracePolicy fail-closed、§6.3 的 M0b 降级路径（若触发）——
  每项写 design decision 理由
- **SECURITY.md**：漏洞上报渠道与支持版本

## 6. M0b：容错合规 parity（v0.8.1；2026-09-17 用户指定由 v0.9.0 改号）

### 6.1 Node-level error handler（补齐容错三件套）

对齐 langgraph 1.2.0 `error_handler=` 完整语义：

- `langgraph/graph/policy.go` 的 `NodePolicies` 增加 `ErrorHandler` 字段（与
  RetryPolicy/TimeoutPolicy 同族；安装沿 `AddNodeWithPolicies`，compile 期默认机制
  不适用于 handler——防全局递归，计划阶段确认）
- `NodeError` typed 结构：node 名、attempt、原始 error（含 NodeTimeoutError）
- Handler 签名：`func(state map[string]any, err *NodeError) (any, error)`——接收
  当前 state 快照（Saga/补偿 handler 需要读状态）；返回普通 state update 或
  `types.Command`（走 `normalizeNodeResult` 现有路径）或 error
- 调度语义（上游对齐）：节点 retries 耗尽（或无 retry 直接失败）→ 失败节点的
  ERROR write **提交进 checkpoint** → handler 作为**同一 superstep 的新任务**调度；
  崩溃于 handler 中段时，resume 重跑 handler 而非原节点；返回值按正常写入路径生效
- 禁止给 handler 节点再挂 handler（防递归）
- 实现动 `langgraph/graph/graph.go` run loop（runTask 区域）与 checkpoint ERROR write
- **降级预案**：若 plan 阶段评估 run loop 改动不成比例 → in-process 同步调用 +
  DIVERGENCES.md 记录分歧（明确写缺崩溃恢复语义）
- 测试矩阵：耗尽后进入 handler；state 可读；plain update / Command 路由分别生效；
  handler 内失败传播（包装原错误）；与 RetryPolicy/TimeoutPolicy 组合（NodeTimeoutError
  触发 handler）；崩溃于 handler 中段后 resume 重跑 handler；handler 上挂 handler 被拒

### 6.2 TracePolicy（per-node / per-middleware trace 载荷控制）

对齐 langgraph 1.2.11（add_node，PR #8523）与 langchain 1.3.15（middleware，#38910），
但应用点按上游真实语义放在**事件 emit 侧**（派发给任何 tracer 之前），保证
langsmith/console/未来 OTel 一致生效；`core/tracers/langsmith.go` 不动：

- `TracePolicy{ProcessInputs, ProcessOutputs func(any) any}`；提供包级
  `trace.OmitPayload` helper（对齐上游 omit_payload 是 helper 而非 policy 方法）
- 挂点一：`add_node`（NodePolicies 扩展）
- 挂点二：AgentMiddleware（`langchain/agents/middleware`，控制该 middleware 包装层
  的 trace 载荷）
- 应用位置：`core/runnables/chain_events.go` 的 `EventChainStart.Input` /
  `EventChainEnd.Output` emit 处 + langgraph 节点事件发射 + agents middleware 包装点
- **processor 出错行为（刻意分歧）**：上游 fail-open（记日志并记录**未变换**载荷）；
  Go 版 **fail-closed**（丢弃该载荷 + OnError 注记）。理由：特性动机是 PII/合规，
  fail-open 静默泄漏原文。写入 DIVERGENCES.md
- 测试：omit 后脱敏；process 变换生效；langsmith 与 console 两个 tracer 都看到
  变换后载荷；processor 出错 fail-closed；与 PII redaction middleware 协同

## 7. M1：RAG 生产栈（v0.9.0）

### 7.1 前置：VectorStore 声明式 filter（可选能力接口）

现状：主接口 `VectorStore` 无声明式 filter；仓库已有两套局部机制——in-memory 的
客户端 `Filter func(doc) bool`、chroma 的服务端 `QueryOptions{Where, WhereDocument}`。
Go interface 封闭，加方法即破坏实现方，故采用**可选能力接口**模式（沿
`TextAdder`/`mmrSearcher` 的 type-assert 惯例）：

- 新增 `SearchOptions{K int, Filter map[string]any, ScoreThreshold float64, ...}` 与
  可选接口 `OptionSearcher`（`SimilaritySearchWithOptions` / `MMRSearchWithOptions`，
  命名计划定）；`VectorStore` 主接口**零改动**
- **声明式 Filter DSL**：对齐 Python `SearchArgs.filter` 的算子集
  （`$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$between/$exists/$like`，以 langchain-pgvector
  的支持集为基线）；各 store 翻译为服务端条件（pgvector→jsonb SQL、redisvector→
  RediSearch 条件、chroma→where、in-memory→客户端求值）；conformance 统一走 DSL
- `VectorStoreRetriever` 桥接层透传 filter（现 searchKwargs 仅认 k/fetch_k/
  lambda_mult/score_threshold，mmr 固定传 nil——需扩展，否则 filter 从 agent 侧不可达）
- in-memory 与 chroma 适配新接口（chroma 已有服务端 filter，做翻译即可）

### 7.2 pgvector（`partners/pgvector`，root module）

- 依赖 pgx v5（root module 新直接依赖，见 §11 依赖策略）
- 能力：collection≈表、自动 DDL（`vector(dim)` 列 + HNSW/IVFFlat 索引）、距离
  cosine/L2/IP、AddDocuments（embedding 回调）、DSL filter→jsonb SQL 条件
- 测试：SQL 构造层单测（pgxmock，计划定）+ `PGVECTOR_TEST_DSN` env-gated 真库
  conformance（standardtests.RunVectorStoreBasics + filter 用例）；CI 跳过 e2e

### 7.3 redis（`partners/redisvector`，root module）

- 命名说明：避开与 `langgraph/checkpoint/redis` 及 go-redis 客户端的包名冲突，
  目录与包名用 `redisvector`
- 依赖 go-redis v9 + RediSearch（FT.CREATE vector 字段、FT.SEARCH KNN、filter→
  查询条件）
- 与 pgvector 同接口同 conformance；`REDIS_TEST_ADDR` env-gated e2e（Redis Stack）
- 已知限制：miniredis 不支持向量查询 → 协议构造单测 + e2e 兜底

### 7.4 Reranker 栈

- `core/retrievers`：`DocumentCompressor` 抽象 + `ContextualCompressionRetriever`
  组合器（retriever 命中 → compressor 按 query 重排/压缩）；对 embeddings 无硬依赖
  （rerank API 收文本）
- `partners/cohere`、`partners/jina`：hosted rerank API（不违反不做 transformers 约束）
- 测试：假服务器（沿 chroma httptest 模式）+ conformance

### 7.5 高级 retrievers 四件（`core/retrievers`）

- MultiQueryRetriever：chat model 生成查询变体（可自定义 prompt），并发检索去重合并
- ParentDocumentRetriever：child 切片检索、按 parent id 回取全文（docstore 用
  `core/stores.BaseStore` 泛型，已核实适配）
- EnsembleRetriever：多 retriever RRF 融合，零新依赖
- ContextualCompressionRetriever：由 7.4 提供
- 测试：in-memory vector store + mock chat model 全链路单测

## 8. M2：模型与工具接入层（v0.10.0）

### 8.1 MCP adapter（`partners/mcp`，对标 langchain 1.4.0 `langchain.mcp` 快照）

- 依赖 mark3labs/mcp-go（锁 minor）
- **首日 spike**：验证 mcp-go client 的 elicitation 能力；不支持则 elicitation 降级
  进 backlog，其余照做
- API 面（锁 1.4.0 文档快照，不追后续版本）：
  - 连接两机制都对齐：**MCPConfig**（聚合连接、单一 protocol era、按 config key
    前缀）与 **ClientGroup**（独立连接/独立 auth/独立 era、`{server}_{tool}` 命名空间）
  - 工具发现：`list_tools` 带 `cache_mode: use|refresh|bypass`（Go 命名自由）
  - 工具结果映射：text/image/file → standardized content blocks；**structured
    output → ToolMessage artifact**（对应上游 MCPToolArtifact，不折入模型可见内容）；
    `isError=true` → error ToolMessage（接 `langchain/tools/tool_node.go` 的
    HandleToolErrors）；**transport/session 失败 → 向上抛**（模型无法处置断连，
    不转 ToolMessage）
  - elicitation → LangGraph interrupt：需 checkpointer 才能 resume；应答按 request
    key；动作 accept/decline/cancel；映射到已有 HITL interrupt 机制
  - 工具级 HITL 门控：`destructiveHint` + InterruptOnConfig → 映射到已有
    `langchain/agents/human_in_the_loop.go`
- 测试：mcp-go in-process server，无外部依赖

### 8.2 Gemini 原生（`partners/gemini`）

- 依赖 google.golang.org/genai（官方 SDK；**首日 spike** 验证 structured output 等
  覆盖度，不满足则降级手写 REST）
- ChatModel 全接口：ToolCalling、ToolChoice 映射、StructuredOutput（responseSchema）、
  SSE streaming、原生多模态输入、UsageMetadata、system instruction
- standardtests chatmodel conformance + 假服务器测试

### 8.3 Loaders 三件套（`core/documentloaders`）

- HTML：goquery
- Web：http GET + HTML 抽取（复用 goquery；正文提取对齐 Python readability 的合理子集）
- PDF：ledongthuc/pdf（纯 Go 轻量）
- 全部接 `LoadAndSplit` → 已有 textsplitters；golden 文件测试

## 9. M3：实用层 + 门面（v0.12.0）

### 9.1 SQL toolkit

- 位置：`langchain/toolkits`（Go 侧自定布局；上游 SQL toolkit 困在已冻结的
  langchain-community `agent_toolkits`，1.x 无对应物，不构成上游对齐理由）
- 基于 `database/sql`；四工具：Query、ListTables、SchemaInfo、QueryChecker
- 只读护栏：仅允许单语句 SELECT（拒绝 DML/DDL/多语句）
- schema introspection：sqlite + postgres 起步（database/sql 元信息 + 方言查询）
- 现代 `create_agent` 组合形态（对齐上游推荐用法，非已弃用的 create_sql_agent）
- 测试：sqlite 内存库全链路 + postgres env-gated

### 9.2 examples/ 目录

12-15 个可运行示例（每例 main.go + 双语简注）：quickstart、HITL、streaming、subgraph
断点恢复、容错（retry/timeout/error handler）、RAG 全链路（loader→split→embed→
pgvector→retrieve→rerank→agent）、MCP tools、Gemini、SQL agent、常用 middleware。
无 API key 可跑的示例用 ollama/mock。

### 9.3 mkdocs 文档站 + README 刷新

- mkdocs-material 双语；现有 docs/usage 5 个主题×双语共 10 个文件迁入 + M1/M2/M3
  新增各一篇
- GitHub Pages 部署 workflow（M0a 的 ci.yml 之外新增 docs.yml）
- README parity 表刷新：vector stores 2 家、MCP、loaders、Gemini、reranker、高级
  retrievers、SQL toolkit；scope 声明同步

## 10. 横切设计

- **测试策略**：所有新集成必须过 standardtests conformance；CI 零 docker；外部服务
  env-gated e2e（本地跑，附脚本）；假服务器/mock 模式沿 chroma 先例
- **兼容策略**：**零现有接口改动**——新能力一律走可选能力接口（type-assert + 桥接
  层降级）、新增包/选项；不动现有签名
- **依赖策略**：partners/pgvector、partners/redisvector 进 root module（pgx v5、
  go-redis v9 成为 root 直接依赖）；CI 增加 pin 对齐检查（与 langgraph/checkpoint/
  postgres、redis 嵌套模块的版本一致性，make 目标实现）
- **版本节奏**：M0a→v0.8.0、M0b→v0.8.1（用户改号）、M1→v0.9.0、M2→v0.10.0、M3→v0.11.0；
  嵌套 checkpoint 模块随需 bump（沿用现有 pin 流程）；CHANGELOG 每版必更
- **文档惯例**：新能力附双语 usage 文档进 docs/usage/；代码注释英文、Python parity
  出处标注（沿现有风格）
- **实施规范**：全程 modern-go skill；对齐仓库既有 Go 1.26 idioms（3a24fdf 已现代化）
- **上游漂移防线**：每里程碑收尾跑一次上游 changelog diff（langchain/langgraph 自
  上次基线以来），新 feature 记入 backlog 评估，不静默扩范围

## 11. 风险与对策

| 风险 | 对策 |
|---|---|
| mcp-go elicitation client 支持度未知 | M2 首日 spike；不支持则该项进 backlog，其余照做 |
| `langchain.mcp` 发布仅两周（1.4.0），API 面可能再变 | 设计锁 1.4.0 文档快照；实现锁行为不追版本 |
| genai SDK 对 structured output 等覆盖不足 | M2 首日 spike；降级手写 REST（范围不变） |
| error handler 的 run loop 改动超预期 | §6.1 降级预案（in-process + DIVERGENCES），触发条件在 plan 阶段量化 |
| RediSearch 无法 mock 向量查询 | 接受协议构造单测 + env-gated e2e 兜底 |
| filter DSL 算子集与各 store 能力不匹配 | 以 langchain-pgvector 算子集为基线，chroma/redis 不支持的算子显式报错 |
| PDF/网页抽取质量参差 | 验收线 = 对齐 Python 行为的合理子集，golden 测试 |
| pgx 单测方案（pgxmock vs 接口抽象）未定 | M1 计划阶段定，倾向 pgxmock（社区标准） |
| root 与嵌套模块的 pgx/go-redis 版本漂移 | §10 依赖策略的 CI pin 对齐检查 |
| 嵌套模块 CI 机制（embedded-postgres 缓存）未定 | M0a 计划阶段定（Makefile 已有预留提示） |
| 上游 patch 演进差累积 | 每里程碑收尾 changelog diff 复查（§10） |

## 12. 验收标准

- **M0a**：CI 双 Go 版本全绿（含嵌套模块离线测试）；ParallelToolCalls 两 provider
  请求体断言与组合矩阵通过；四文档落地；`go test ./...` 零失败；发 v0.8.0
- **M0b**：error handler 测试矩阵全过（含崩溃恢复，或降级预案触发并记录）；
  TracePolicy 测试全过（含多 tracer 一致与 fail-closed）；发 v0.9.0
- **M1**：pgvector/redisvector 通过 conformance 含 filter 用例（本地 env e2e）；
  reranker/高级 retrievers 单测全过；现有接口零改动（现有测试不动即过）
- **M2**：MCP in-process 测试全过（含 elicitation 若 spike 通过）；Gemini
  conformance 通过；loaders golden 测试通过
- **M3**：examples 全部 `go build` 通过且可运行；`mkdocs build` 通过；README/parity
  表/文档站一致；发 v0.12.0
- **通用**：每里程碑 CHANGELOG 更新、DIVERGENCES 如有新分歧同步、subagent 终审通过

## 13. 参考

- 上游版本核实：PyPI langchain 1.4.0（2026-09-03）、langgraph 1.2.11；docs.langchain.com
  changelog；langchain.com/blog "MCP in LangChain: Stateless Protocol, Elicitation,
  and More!"（2026-09-03）
- 容错三件套：langgraph 1.2.0 + 官方博客 "Fault Tolerance in LangGraph"（2026-06-04）
- TracePolicy：langgraph 1.2.11（PR #8523，emit 侧 `_trace_payload` 语义）、
  langchain 1.3.15（#38910）
- langchain.mcp：docs.langchain.com/oss/python/langchain/mcp（+ connections/tools 子页，
  1.4.0 快照）
- 本地审计：docs/superpowers/plans/2026-09-12-parity-catchup.md 及 2026-09-16 会话
  审计记录；spec 审计的代码证据（chatmodel.go / policy.go / graph.go runTask /
  vectorstore.go / retriever.go / chroma.go / providers.go 行号）已在本文各节内化
