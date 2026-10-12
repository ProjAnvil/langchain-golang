# langchain-golang parity wave 3（上游 2026-10-08 release 追齐 + Jev/TypeSafe 接入）设计文档

- 日期：2026-10-12（rev 2，已按 subagent 审计修订：B1–B3/M1–M4 及 minor 1–11 吸收）
- 状态：审计通过（rev 1 结论「需修改」，Blocker/Major 已全部修复）→ 进入 plan
- 关联研究：`reports/13-latest-release-port-gap.md`（2026-10-12 差距报告）+ TypeSafe/Jev 包勘察（`langchain/` @ `langchain==1.4.4` 的 `libs/partners/typesafe`，3 commits #40542/#40570/#40659，版本 0.0.1a3）+ exa/nomic 包勘察
- 上游基线：core 1.6.9 · langchain v1 1.4.4 · openai 1.7.0 · fireworks 1.7.1 · exa 1.1.0 · nomic 1.0.1 · typesafe 0.0.1a3 · langgraph 1.2.14（本地 checkout tag）
- 实施约束：全程 modern-go 规范；TDD（每任务先红后绿）；门禁 `go test ./...` + golangci-lint 0 issues + 受影响包 ≥95% 覆盖率；不 push、不打 tag

## 1. 背景

2026-10-12 差距报告（reports/13）确认本轮上游 delta 的可移植项集中在：SSRF IPv4-compatible
绕过修复（core #41151，已在 Go 侧实证漏洞存在）、SummarizationMiddleware 上下文溢出恢复
（#41159）、OpenAI Decisions API 及其两个 middleware（openai 1.7.0 #41125）。

用户追加指令（本轮）：接入 TypeSafe AI 的 **Jev**（System One 决策模型，2026-09-15 发布，
LangChain 已出官方 `langchain-typesafe` 集成，包内含 `TypeSafeClassifier` 与 experimental
middleware —— 即用户所称 "typesafeclassifier"），并扫尾其余 Python monorepo 中 Go 尚缺的
provider。

Provider 扫尾盘点（Python monorepo partners vs Go partners）：

| Python partner | Go 现状 | 本波处置 |
|---|---|---|
| typesafe（新，Jev） | 无 | **纳入（用户点名）** |
| exa（搜索 retriever+tools，472 行） | 无 | **纳入**（小包，REST 直连） |
| nomic（embeddings，242 行） | 无 | **纳入**（小包，REST 直连） |
| huggingface（4442 行） | 无 | **defer**（大包，独立立项） |
| qdrant（3912 行，vectorstore） | 无 | **defer**（Go 已有 chroma/pgvector/redisvector 三个 vectorstore，需求触发再立项） |
| deepseek/fireworks/groq/mistralai/openrouter/perplexity/xai | openaicompat registry 覆盖 | 无动作 |

## 2. 决策记录

用户已拍板（本轮指令）：

1. 按 reports/13 开始实施（spec→审计→plan→审计→实施）。
2. Jev/TypeSafe 接入本波完成。
3. 其余 provider 一并盘点接入（→ exa/nomic 纳入、huggingface/qdrant 延后，理由见 §1）。

按推荐代答（可推翻；rev 2 依审计调整）：

1. **OpenAI Decisions 与 TypeSafe Classifier 保持 1:1 平行移植，不做共享抽象**。两者
   wire 格式与问题词汇不同（Predicate/Choice/Score vs Noul/Choice/Score；choices 数组+value
   键 vs criteria 对象），Python 上游也是刻意的双份实现（partner 包自包含）。middleware 侧
   同理各放各的 partner 包。
2. **LangSmith Gateway 解析不实现**（reports/13 §3.13 仍未移植）：Decisions 客户端仅支持
   显式参数 + `OPENAI_API_KEY`/`OPENAI_API_BASE` 环境变量，记 DIVERGENCES；缺 key 报错
   文案裁剪掉 Python 文案中的 `LANGSMITH_GATEWAY` 提示句。
3. **run-tree usage 挂载不实现**：`core/tracers` 无 current-run 访问器（Python
   `get_current_run_tree`；grep 复核无 `CurrentRun`/`FromContext` 挂点），Usage 只随响应
   类型返回，记 DIVERGENCES。
4. **ModelRouter 状态值统一为 JSON-native map**（openai 侧 Python 本就是 plain map；
   typesafe 侧 Python 存 pydantic ChoiceAnswer，Go 存 `{"type","choice","probabilities",
   "confidence"}` map，durable-serde 安全），记 DIVERGENCES（仅 typesafe 侧）。
5. **typesafe 错误层不复制 Python 的双重继承类树**：Go 自定义 `APIError` 类型内嵌
   `lcerrors.ProviderError`（保持 `errors.As` 可读 status/body/retryAfter、`errors.Is`
   可命中 sentinel/Kind），**override `Error()`** 输出安全摘要（不含 body/headers，格式
   `{METHOD} {scheme}://{host[:port]}{path}: {status} {phrase} (request_id=...)`，对齐
   Python `_response_endpoint`+`__str__`）；响应校验错误用独立 `ResponseValidationError`
   （带 `FieldPath`，wrap `ErrSchemaValidation`）。Go 的 `ModelErrorKindForStatus` 会对
   400+溢出 marker 升级为 ContextOverflow（Python typesafe 一律 invalid_request）——
   附加行为，记 DIVERGENCES。
6. **exa 不依赖 Python 的 exa_py SDK 语义面**：Go 直接实现 Exa REST（search/findSimilar/
   contents），工具/检索器参数面与返回结构对齐 `langchain_exa` 1.1.0（见 §3E 修正后的
   参数清单）；Python 侧无 include_contents/url 交叉校验，Go 也不加。
7. **消息→OpenAI 形状映射自实现**（约 50 行，先例 `partners/openai/contentblocks.go` 的
   `responsesContentParts`/`imageDataURL`，未导出不可复用）：`core/messages.
   ConvertToOpenAIMessages` 输出 LangChain 形状图片块（`{"type":"image",...}`）而非 OpenAI
   `image_url` 形状，decisions 输入路径、base64 图片判定（等价 Python `_is_base64_image`
   的 `data:` 前缀判断）、typesafe `SerializeState` 的含图 wire 形状都以自实现映射为准。
   decisions 与 typesafe 各持一份（partner 自包含原则）。
8. **AutoMode 的 omit-payload trace policy 记 DIVERGENCES 不实现**：Go 的
   `applyMiddlewareTracePolicy` 仅在 wrap_model_call 两个调用点安装（create_agent.go:
   1926/1951），wrap-tool-call 层无 policy 安装点，实现 `TracePolicyProvider` 会被静默
   忽略。ModelRouter（wrap_model_call）照常实现 policy；AutoMode 分歧明示「工具层 payload
   未脱敏」。本波不改 create_agent 的 policy 安装范围。
9. **NAT64 两个前缀都整段封禁**：`64:ff9b::/96` 已封；补 `64:ff9b:1::/48`（Python 对两者
   都提取内嵌 IPv4，Go 统一整段封更严、实现更简单），维持 DIVERGENCES #7 的表述。
10. **nomic 对齐策略**：Go `Embeddings{Model, TaskType, Dimensionality, APIKey, ...}`；
    Python `model` 必填无默认，Go 允许空 `Model` 时默认 `nomic-embed-text-v1.5`（记
    DIVERGENCES）；`dimensionality`（Matryoshka 降维）纳入；`inference_mode local/dynamic`
    与 `embed_image`/`vision_model` 不做（remote 文本 only，记 DIVERGENCES）；Go 仓库无
    embeddings factory registry（仅 chatmodels 有 `RegisterProvider`），nomic 不注册任何
    registry，调用方直接构造。
11. huggingface/qdrant defer 记入 reports/13 追记（或 CHANGELOG Unreleased 备注）。

## 3. 范围与设计

### A. core/security：SSRF IPv4-compatible 修复（#41151，P1 安全）

现状（已实证）：`ValidateResolvedIP("::127.0.0.1")`、`("::10.0.0.5")`、
`("::169.254.169.254")` 在 `DefaultSSRFPolicy()` 下均放行；IPv4-mapped（`::ffff:x`）由 Go
`To4()` 语义天然覆盖，NAT64（`64:ff9b::/96`）已整段封禁。

设计（两处改动）：

1. `ValidateResolvedIP` 在既有检查前增加一步——地址为 16 字节 IPv6、`To4()==nil`、
   前 12 字节全零（`::/96`）且整数值 >1（排除 `::` 与 `::1`）时，取末 4 字节作为 IPv4
   继续走既有 IPv4 检查路径（loopback/private/metadata/AdditionalBlockedCIDRs）。
   `ValidateHostname` 的字面 IP 分支经由 `ValidateResolvedIP` 自动覆盖。
2. `isPrivateOrReserved` 的 ipv6Blocked 增加 `64:ff9b:1::/48`（NAT64 discovery prefix，
   Python #41151 对 /96 与 /48 都做内嵌提取）。

测试：Python 回归用例移植 + 本地探针固化（`::127.0.0.1`/`::10.0.0.5`/
`::169.254.169.254`/`64:ff9b:1::7f00:1` 拦截；`::ffff:127.0.0.1`、`64:ff9b::7f00:1`、
`::1`、`::2`→0.0.0.0/8 维持拦截；`::0102:0304` 公网地址不误伤）。

### B. SummarizationMiddleware 溢出重试（#41159）

现状：`BeforeModel` 直接把 `m.Summarize` 的错误向上抛。v0.10.0 起 openai/anthropic 客户端
已把 400 溢出分类为 `ErrModelContextOverflow`。

设计（`langchain/agents/middleware/summarization.go`，`SummarizerFunc` 契约不变）：

1. `Summarize` 返回错误且 `errors.Is(err, lcerrors.ErrModelContextOverflow)` 时进入溢出
   重试（sentinel 是真实客户端的唯一产出路径，`errors.As ContextOverflowError` 结构体
   目前无生产者，判断只走 sentinel）；
2. 重试 prompt：`tokens = m.countTokens([Human(formatted)])`；`budget = tokens/2`；若
   `maxInputTokens()` 已知则 `budget = min(budget, maxInput*2/5)`；`keepChars =
   len(formatted)*budget/max(tokens,1)`；`keepChars<=0` → 重抛原错误；`head=keepChars/4`、
   `tail=keepChars-head`，截断串 = `formatted[:head] + "\n\n[... middle of conversation
   omitted ...]\n\n" + formatted[len-tail:]`，最终 prompt 做 trailing-space 裁剪
   （对齐 Python `.rstrip()`）；
3. 只重试一次；第二次溢出原样上抛；
4. 移植 Python 语义文档（head 保留目标陈述、tail 保留最近上下文；40% 系数吸收 1.25x
   usage-metadata 缩放）。

### C. partners/openai/decisions + partners/openai/middleware（#41125）

**decisions 包**（package `decisions`，镜像 `langchain_openai.decisions`）：

- `types.go`：`Predicate{Instructions}`、`Choice{Instructions, Choices []ChoiceOption{Value
  any; Description *string}}`（有序切片保持 wire 数组顺序，`Value` 为 string/bool）、
  `Score{Instructions, Levels []ScoreLevel}`（≥2，`ScoreLevel{Label; Description *string}`，
  构造时接受裸字符串简写——对齐 Python `levels` 接受 `str | Level`）、`Question` 接口
  （sealed）；答案 `PredicateAnswer{Probability}`、`ChoiceAnswer{Choice any; Probabilities
  map[string]float64; Confidence}`、`ScoreAnswer{Score; Legend map[int]string;
  Probabilities map[int]float64; Confidence}`、`RefusalAnswer{Extra map[string]any}`
  （对齐 Python extra=allow 开放模型）、`Answer` 接口（sealed）；`Usage{InputTokens,
  OutputTokens *int}`、`Response{Model; Answers map[string]Answer; Usage; RequestID
  string}` + `Predicates()/Choices()/Scores()/Refusals()` 视图；`Request{Input any;
  Questions map[string]Question}`。
- `openai_shapes.go`（决策记录 #7 的自实现映射）：`messages.Message`（blocks 形态）→
  OpenAI 输入 part：text→`{"type":"input_text","text":...}`、base64 图→`{"type":
  "input_image","image_url":"data:..."}`、hosted URL 图报错、其它媒体报错；单值图片块
  判定 `isBase64Image(block) bool`（data: URL 前缀）。
- `input.go`（`_input.py` 语义）：Input 为 string → 原生；单条或全 Human 的
  `messages.Message` 序列 → 原生 user 消息（经 `openai_shapes.go`）；其余（任意含消息的
  JSON 结构）→ JSON 序列化为单条 user 消息，base64 图片保留为 image part，其它媒体报
  `ErrInvalidInput`。
- `client.go`：`Client{Model, APIKey, BaseURL, HTTPClient, Timeout}`（自包含配置；仓库
  惯例 `modelconfig.Config`+`httpclient.PostJSON` 面向 chat/embeddings，decisions 保持
  partner 自包含，错误映射自建）；env `OPENAI_API_KEY`（必填，缺失报错文案对齐 Python
  但裁剪 `LANGSMITH_GATEWAY` 句）、`OPENAI_API_BASE`（次选 `OPENAI_BASE_URL`），默认
  `https://api.openai.com/v1`；endpoint `{base}/decisions`；POST body
  `{"model","input","questions":[{"type","name",...}]}`（问题 name 从 map 键注入，
  Go map 无序 → 按 name 稳定排序序列化，与 Python 插入序不同，记 DIVERGENCES 并在金样
  测试按排序规则断言）；HTTP 错误 → lcerrors `NewProviderError`（400-body 溢出嗅探自动
  生效）；2xx 但 body 非法 → wrap `ErrSchemaValidation`（文案 "Invalid response from
  the OpenAI Decisions API."）；响应归一化：choice `probabilities:[{value,probability}]
  →map`、score 附 `legend`、未知 answer type 跳过（对齐 Python `_ANSWER_TYPES` 过滤）、
  `x-request-id`→RequestID。
- `Invoke(ctx, Request) (Response, error)`；ctx 取消传递；无 batch。

**middleware 包**（package `middleware`，import 别名 `openaimiddleware`）：

- `AutoMode`（Python `OpenAIAutoModeMiddleware`）：字段 `Tools []any`（接受名字符串或
  `tools.Tool`，归一化为集合，空集报错；裸字符串作为单元素在 Python 报错，Go `[]any`
  天然规避该误用形态）、`Model any`（string→`decisions.Client` 或注入客户端）、
  `Instructions`（默认值逐字对齐 Python `_DEFAULT_INSTRUCTIONS`）。语义：未列名工具
  直通；列名工具取 `state["messages"]` 末 30 条，媒体块替换 `[{type} omitted]` 占位
  （保留 base64 图）构建 with-images 变体，与 text-only 相同则只发一个请求；HTTP 400/413
  拒绝时降级 text-only 重试一次（其它状态上抛）；`is_risky` 概率 ≥0.5 → 错误 ToolMessage
  （文案逐字：``The tool call `{name}` was blocked because it was classified as risky
  (probability: {p:.2f}). The tool was not executed.``）；无 answer/refusal → ``The tool
  call `{name}` was blocked because its risk could not be assessed. The tool was not
  executed.``；分类错误上抛（fail-closed）。blocked ToolMessage 的 `status="error"` 在 Go
  `messages.Message` 无对应字段 → 写入 `AdditionalKwargs["status"]="error"`（记
  DIVERGENCES）。trace policy 不实现（决策记录 #8，记 DIVERGENCES）。
- `ModelRouter`（Python `OpenAIModelRouterMiddleware`）：字段 `Choices map[string]
  ModelChoice{Model any; Criteria string}`（string 模型经 `chatmodels.InitChatModel`
  解析，空集/空 instructions 报错）、`Instructions`、`Model any`。`BeforeAgent`：取末尾
  最近一条 Human 消息（无 → state 写 `model_route: nil`，走 agent 自身模型）；非 base64
  媒体块占位（`[{type} omitted]`，经 `openai_shapes.isBase64Image` 判定），400/413 降级
  text-only 一次；答案写 state 键 `model_route`（JSON-native map `{choice, probabilities,
  confidence}`；refusal/缺失 → nil）。`WrapModelCall`：有 route 则 `ModelRequestOverride`
  换模型（`WithModel`，先例 model_fallback.go:54；route 未知名字 → 报错）。state 声明走
  `StateSchemaContributor`（键 `model_route`）。trace policy 照常实现（wrap_model_call
  层有安装点）。

### D. partners/typesafe（用户点名，Jev 接入）

**classifier + types + client**（package `typesafe`，镜像 `langchain_typesafe`）：

- `types.go`：`Noul{Instructions any; Criteria *NoulCriteria{True, False any}}`、
  `Choice{Instructions any; Criteria []ChoiceCriterion{Label string; Description any}}`
  ——TypeSafe wire 的 choice criteria 是**对象**（label→description，非 openai 的数组），
  Go 用有序切片保序、marshal 为对象；值为 nil 的 description 键在 marshal 时省略
  （pydantic `exclude_none` 递归语义：`NoulCriteria` 的 true/false 为 nil 时同样省略该
  键——对齐 classifier.py:447 `model_dump(mode="json", exclude_none=True)`）。`Score
  {Instructions any; Criteria []any}`（≥2）。问题/答案 sealed 接口同 C。答案：
  `NoulAnswer{Noul float64}`（无独立 confidence）、`ChoiceAnswer{Choice string;
  Probabilities map[string]float64; Confidence}`、`ScoreAnswer{Score; Legend map[int]any;
  Probabilities map[int]float64; Confidence}`、`Usage`、`Response{Model; Answers; Usage;
  RequestID}` + `Nouls()/Choices()/Scores()` 视图、`Request{State any; Questions
  map[string]Question}`（序列化按 name 稳定排序，同 C 的 DIVERGENCES 项）。
- `state.go`（`_state.py` 语义）：`SerializeState(any) (any, error)` —— 根只接受
  string/`messages.Message`/slice/map（nil/数字/布尔根报错，文案对齐 Python）；消息
  （含嵌套于对象/数组任意深度）→ OpenAI role/content JSON（经决策记录 #7 的自实现
  映射，id 省略）；map 非 string 键报错；不可 JSON 化类型报错。
- `client.go`：`Classifier{Model, APIKey, BaseURL, HTTPClient, Timeout}`；默认模型
  `jev-latest`（strip、非空校验）、env `TYPESAFE_API_KEY`（必填）、`TYPESAFE_BASE_URL`
  （默认 `https://api.typesafe.ai`）、Timeout 默认 30s；endpoint `{base}/v1/systemone`；
  headers `Authorization: Bearer`、`Content-Type: application/json`、
  `User-Agent: langchain-typesafe-go/<ver>`；body `{"state","model","questions":{name:
  {"type","instructions","criteria"?}}}`。
- `errors.go`（决策记录 #5）：`APIError` 内嵌 `lcerrors.ProviderError`，override
  `Error()` 输出 `{METHOD} {scheme}://{host[:port]}{path}: {status} {phrase}
  (request_id=...)`（endpoint 由请求 URL sanitize：去 query/fragment/credentials，host
  含冒号加方括号——对齐 `_response_endpoint`；request_id 取 `x-typesafe-request-id`）；
  400/401/403/404/422/429/5xx → 对应 sentinel（`ErrModelInvalidRequest`/`ErrModelAuth`/
  `ErrModelPermissionDenied`/`ErrModelNotFound`/`ErrModelRateLimit`/`ErrModelServer`），
  retry-after 解析：`retry-after-ms`（×1ms）优先、`retry-after`（×1000ms，支持
  HTTP-date；空串当 0、负值/非法忽略 fall-through——对齐 `_parse_retry_after` 分支）；
  transport 错误 → `ErrModelConnection` wrap（timeout → `ErrModelTimeout`）；2xx 非法
  body → `ResponseValidationError{FieldPath}` wrap `ErrSchemaValidation`（**严格校验**，
  未知 answer type 报错——对齐 Python，与 openai decisions 的"跳过未知"相反）。
- `Invoke(ctx, Request) (Response, error)`。

**middleware**（`partners/typesafe/middleware`，import 别名 `typesafemiddleware`）：

- `AutoModeMiddleware`：Noul 问答；默认 instructions 与默认 criteria（true/false 文案逐字
  对齐 `_DEFAULT_INSTRUCTIONS`/`_DEFAULT_TRUE_CRITERIA`/`_DEFAULT_FALSE_CRITERIA`）；
  阈值 0.5；state 结构 `{"messages": 末30条, "tool_call": {id,name,args},
  "tool_description"?: 描述非空时}`；无媒体处理（JSON state）、无 400/413 降级；
  `is_risky` answer 缺失 → 报错上抛（对齐 Python KeyError 的 fail-closed 语义）；
  blocked 文案逐字对齐；ToolMessage `status="error"` 同 C 走 `AdditionalKwargs`。
- `ModelRouterMiddleware`：Choice 问答；`ModelChoice{Model any; Criteria any}`；
  BeforeAgent 取最近 Human 消息，**缺失 → 报 error**（Python `next()` 无默认即
  StopIteration；显式化为 error，与 openai 版 nil-route 语义不同，记 DIVERGENCES）；
  `model_route` 缺失 answer/refusal → 报错（对齐 Python KeyError）；state 键
  `model_route` 存 JSON-native map `{type, choice, probabilities, confidence}`；
  `WrapModelCall` 同 C。

### E. partners/exa + partners/nomic（provider 扫尾）

- **exa**（package `exa`，REST 直连 `https://api.exa.ai`，env `EXA_API_KEY`，
  header `x-api-key`）：
  - `SearchResults`（tool，POST `/search`）：参数 livecrawl/numResults/startPublishedDate/
    endPublishedDate/includeDomains/excludeDomains/text（contents 选项
    {maxCharacters}）/highlights（{query,numSentences}）；返回 `title/url/
    publishedDate/author/score/text/highlights`。
  - `FindSimilarResults`（tool，POST `/findSimilar`）：首参 url 必填 + 上述参数外加
    category；返回同上。
  - `SearchRetriever`（retriever）：包 search/findSimilar 调用，输出
    `documents.Document`（page_content=text、metadata 下划线键：title/url/
    published_date/author/score/highlights/id/summary/highlight_scores）。
  - Python 侧无 include_contents/url 交叉校验（审计 B2），Go 不加。
- **nomic**（package `nomic`，REST 直连 `https://api-atlas.nomic.ai/v1`，env
  `NOMIC_API_KEY`）：`Embeddings{Model, TaskType, Dimensionality *int, APIKey, BaseURL,
  HTTPClient, Timeout}`；POST `/embedding/text`，body `{"model","texts","task_type",
  "dimensionality"?}`；合法 `task_type`：`search_query`（embed_query）/
  `search_document`（embed_documents）/`classification`/`clustering`；空 Model 默认
  `nomic-embed-text-v1.5`；响应 `{"embeddings": [[...]], "usage": {...}}`。实现 Go
  embeddings 接口；不注册任何 registry（仓库无 embeddings factory 机制）。

## 4. 测试策略

- 每任务 TDD：先写失败测试（红）再实现（绿）；httptest.Server 桩按 wire 金样断言
  （请求 JSON 逐字段、响应解析含归一化）。
- A：Python 回归 + 探针固化（见 §3.A，含 `64:ff9b:1::` 前缀）。
- B：计数 fake Summarizer（首调返回溢出、次调成功/再溢出）、maxInputTokens 有/无两条
  路径、keepChars=0 重抛、重试 prompt rstrip。
- C/D：错误表测试（状态码→sentinel/Kind）、retry-after 全边界（ms 头、秒头、HTTP-date、
  空串→0、负值忽略、非法→无）、request-id、typesafe `APIError.Error()` 不含 body/
  headers、input/state 序列化递归与报错文案、答案归一化（choice/score/未知类型跳过 vs
  严格报错）、exclude_none 递归省略、middleware 全路径（直通/拦截/400-413 降级/缺
  answer/换模型/nil route vs error/未知 route/blocked ToolMessage kwargs）。
- E：wire 金样（search/findSimilar/embedding 请求体与响应解析）+ 错误映射 + 工具 schema
  暴露 + task_type 四值映射。
- 门禁：`go test ./...` 全绿；golangci-lint 0 issues；受影响包覆盖率 ≥95%；
  `make test-sqlite test-redis test-postgres` 不受影响（本波不触碰 langgraph 持久层）。

## 5. DIVERGENCES 新增（预计）

1. Decisions/TypeSafe：不接 LangSmith Gateway（`LANGSMITH_*` 解析缺失）；decisions 缺
   key 报错文案裁剪 gateway 提示句。
2. Decisions/TypeSafe：usage 不挂 run-tree（无 current-run 访问器），仅随响应返回。
3. typesafe ModelRouter：无 human 消息 / 缺 answer → error（Python StopIteration/KeyError
   崩溃语义显式化）；openai ModelRouter：nil route 继续（对齐 Python）。
4. typesafe ModelRouter 状态存 JSON-native map（Python 存 pydantic 模型）。
5. typesafe User-Agent 为 Go 客户端标识（`langchain-typesafe-go/...`）。
6. exa 直连 REST 而非 exa_py SDK（行为对齐，传输层实现不同）。
7. SSRF：NAT64 两个前缀（`64:ff9b::/96`、`64:ff9b:1::/48`）整段封禁（Python 为提取内嵌
   IPv4 后判断；Go 更严）。
8. questions 序列化按 name 稳定排序（Python dict 插入序；wire 为数组/对象键序差异）。
9. AutoMode（openai+typesafe）：wrap-tool-call 层 trace policy 不生效（Go 仅 wrap_model_
   call 有 policy 安装点），工具参数 payload 未脱敏。
10. blocked ToolMessage 的 `status="error"` 走 `AdditionalKwargs`（Go Message 无 status
    字段）。
11. typesafe 400+溢出 marker 会升级为 ContextOverflow（Python typesafe 一律
    invalid_request；Go 分类器统一行为）。
12. nomic：空 Model 默认 `nomic-embed-text-v1.5`（Python 必填无默认）；不做 local/dynamic
    inference 与 vision/embed_image。
13. 含图消息的 wire 形状以 §2 决策 #7 的自实现映射为准（OpenAI `image_url` 形状），与
    Python `convert_to_openai_messages` 输出在 hosted-URL 图等边缘形态的差异随实现记录。

## 6. 明确不做（本波）

- huggingface / qdrant partner（defer，独立立项）。
- langgraph 1.2.14（库本体零变更）。
- fireworks reasoning 流式 index 命名空间（audit-only，openaicompat 不受影响）。
- bedrock converse redacted_content（无 bedrock translator，defer）。
- model-profiles 数据刷新（openai/anthropic 无变化）。
- report-12 carry-over（ToolErrorMiddleware、modelprofiles CLI summarize、chat_loaders、
  llm-cache globals、langgraph_redis middleware 包）维持不动。
- create_agent 的 trace-policy 安装范围扩展（决策记录 #8 之反路径）。
