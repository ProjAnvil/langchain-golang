# Parity Wave 5 Design（上游 2026-11-04 追齐：core 1.7.6 / langgraph 1.2.20）

**Date**: 2026-11-08 · **Authority**: `reports/16-latest-release-port-gap.md`（逐项 file:line 核查）
**Python ground truth**: `langchain/` @ `langchain-typesafe==0.0.2`（71881c7eac；树携带 core 1.7.6 / langchain 1.4.9 / openai 1.7.4 / anthropic 1.7.10 + 未发布 #41715/#41678）· `langgraph/` @ `1.2.20`（a4d4187e8）
**Go target**: `langchain_golang/` 自 main（v0.12.5 之后）起，分支 `feat/parity-wave5`

本 spec 是 wave5 的唯一决策记录；plan 从本 spec 展开，实现冲突以本 spec 为准。范围 = report 16 的 P1 refusal 波次 + P2 strict resume + P3 anthropic 透传补齐 + P4 model-profiles 数据同步 + 分歧记录落地。

## 决策记录

### D1 — RefusalContentBlock + ModelRefusalError（core，#41582 核心半边）

**背景**：上游把各家"模型拒绝"统一为标准内容块。OpenAI 在响应里给 refusal 字段/块，Anthropic 给 `stop_reason="refusal"`，Gemini 给 SAFETY 类 finish reason。

**决策**：

- `core/messages/contentblocks.go` 新增 `RefusalBlock`：
  ```go
  // RefusalBlock 是标准化的模型拒绝块（Python RefusalContentBlock，core 1.7.5）。
  type RefusalBlock struct {
      Refusal string         // 人类可读拒绝文本；Anthropic 类无文本时为空串
      ID      string         // 可选
      Index   any            // 流式聚合用；int 或 string
      Extras  map[string]any // provider 细节，如 anthropic category、gemini finish_reason/block_reason
  }
  ```
  实现 sealed `ContentBlock` 接口；`BlockType()` 返回 `"refusal"`。
- 连带改动四处（缺一不可）：`KNOWN_BLOCK_TYPES` 增加 `"refusal"`；`ParseContentBlock` 增加 `case "refusal"`（从 map 构造，未知键进 Extras，对齐现有 extractExtras 惯例）；`BlockToMap` 增加序列化分支（`type`/`refusal` 必有，`id`/`index` 非零才写，**Extras 经 mergeExtras 摊平到顶层**——与现有 14 个块类型行为一致，不出 `extras` 键）；`CloneBlock`（contentblocks.go:606 的类型 switch）增加 case，否则走 default 浅拷贝导致 Extras map 共享。
- `core/lcerrors` 新增 `ModelRefusalError`：
  - 不嵌入 `ProviderError`（refusal 是 200 响应，不是 HTTP 错误）；独立 struct，实现 `error`。
  - 字段：`Message string`（默认 `"Model refused to produce the requested output"`）、`RawMessage *messages.Message`（nilable）。
  - 构造器 `ModelRefusalFromBlock(block messages.RefusalBlock, raw messages.Message) *ModelRefusalError`：消息体 = 默认文本 + ` (category='...')`（extras.category 存在时）+ `": <refusal文本>"`（非空时），对齐 Python `from_refusal`。
  - 方法名对齐现有惯例 `IsModelRetryable() bool`（参照 errors.go:194 ProviderError.IsModelRetryable），恒 false。
- `core/messages/convert.go` 的 `ConvertToOpenAIMessages`：AI 消息的 refusal 块 → `{"type":"refusal","refusal":...}`；`Refusal` 为空串的块不发送（对齐 utils.py 注释："Refusals without text have nothing to send"）。

**验收**：core 单测覆盖块 round-trip（map→typed→map）、KNOWN_BLOCK_TYPES 成员、`from_refusal` 三种消息组合（无 category 无文本 / 仅 category / 全量）、`Is`/`As` 语义、空文本 refusal 不出现在 OpenAI 转换结果。

### D2 — output parsers 拒绝即错（core，#41582 parser 半边）

**决策**：`core/outputparser` 新增 `RaiseIfRefused(msg messages.Message) error`（对齐 `_refusal.py`）：扫描 `msg.ContentBlocks`，遇 `refusal` 块返回 `*lcerrors.ModelRefusalError`（经 `ModelRefusalFromBlock`），否则 nil。接入点：

- `openai.go` 的 `JSONOutputToolsParser.ParseMessage`（openai.go:114）与 `JSONOutputKeyToolsParser.ParseMessage`（openai.go:137）：取到 message 后、解析 tool calls 前检查（对齐 openai_tools.py:190、255）。
- **json/pydantic 文本路径无消息可见性，不接**。Go 的 `JSONParser.ParseResult` 收 `[]outputs.Generation`，而 `Generation` 无 Message 字段（outputs.go:11-15），`ChatGeneration` 是独立类型、进不了 `[]Generation` 切片（无协变）。Python json.py 的检查点在 `parse_result` 的 ChatGeneration 分支，Go 无对应形状。Go 的消息级结构化输出检查统一放在消息通道：`core/language.InvokeStructured` 默认路径 + partner 的 StructuredCaller 实现（D3/D4 各自接入）。此取舍记入 DIVERGENCES。
- `pydantic.go` 不接：Go 的 `PydanticParser[T]` 只有文本路径 `Parse(ctx, text)`（pydantic.go:117），无 `PydanticToolsParser` 类型；上游 pydantic.py 也是靠继承获得检查，无独立 hook。

**验收**：上游 `tests/unit_tests/output_parsers/test_refusal.py`（112 行）逐条移植：refusal 消息进 parser → `ModelRefusalError`；非 refusal 消息不受影响；`include_raw` 等价路径（Go 侧若由 partner structured 输出实现则在 D3/D4 验收）。

### D3 — openai partner refusal（openai 1.7.4，#41582 openai 半边）

**现状**：Go openai 流式 responses API 已把 `response.refusal.done` 包成 `NonStandardContentBlock{Type:"refusal"}`（stream.go:390-394，此处的 Refusal 字段属于 responses API 的 streamEvent）；chat-completions 的 wire 结构**两路都没有 refusal 字段**（chat_completions.go:176-180、chat_completions_stream.go:281-289），需新增；非流式两路均未处理。

**决策**：

- chat-completions 非流式：wire message 结构补 `refusal` 字段 → `AdditionalKwargs["refusal"]` + `RefusalBlock`（对齐 base.py `additional_kwargs["refusal"] = refusal` + translator 转块）。
- chat-completions 流式：wire delta/终态结构补 refusal 字段；对齐上游形状——上游 CC 流式不逐 delta 捕获 refusal，只在 beta stream 的 `get_final_completion()` 终态消息里出现（test_refusal.py:189-197 钉死此路径），Go 同样在终态消息上产 `RefusalBlock`。**不设固定 index**，产出为无 index 块（与 D4 统一；上游当前的 `lc_refusal` 仅存于测试历史数据，无产出端）。
- responses API：`response.refusal.done` 的 NonStandard 占位改为正式 `RefusalBlock`；流式块只有 `{type, refusal}`（base.py:5640-5641），**`id`/`phase` 只在非流式产出**（base.py:5269-5277）。
- 回写方向（v1 → wire）：refusal 块在 `_compat` 等价层（Go 的 format/request 构造）——chat-completions 与 responses 两个方向都对**空文本 refusal 跳过不发送**（_compat.py:179-184、451-455 两处）；responses 方向非空时保留 `id` 与 `phase`（phase 优先取块顶层，其次 extras.phase）。
- 结构化输出路径：Go 的 `InvokeStructured` / with_structured_output 等价物在解析前做 `RaiseIfRefused`，refusal 时返回 `*ModelRefusalError`（对齐 base.py 的 `checked_parser` 接线；`include_raw` 等价模式下错误进 parsing_error 位而非抛出——按 Go 现有结构化输出错误形状对齐）。

**验收**：上游 `partners/openai/tests/unit_tests/chat_models/test_refusal.py`（197 行）与 test_base.py 新增 refusal 用例逐条移植。

### D4 — anthropic partner refusal（anthropic 1.7.10 + 未发布 #41715）

**现状**：Go anthropic `toMessage`（chatmodel.go:1205-1258）把 stop_reason 记入 ResponseMetadata，不产块；`messagePayload` 无 `stop_details` 字段。

**决策**：

- `messagePayload` 增加 `StopDetails *struct{ Category string; Explanation string }`（json `stop_details`）。
- `toMessage`：`StopReason == "refusal"` 时追加 `RefusalBlock{Refusal: StopDetails.Explanation（nil 时 ""）, Extras: {"category": ...}（非空时）}`；ResponseMetadata 增加 `stop_details`（存在时）。对齐 core translator `_refusal_from_stop_details`。**不**设固定 index——#41658 已在上游移除 `lc_refusal` 固定 index（translator 产出端）。
- 流式（stream.go）：`message_delta` 的 stop_reason=="refusal" 时同样向最终聚合消息追加 refusal 块。上游本波的相关变化是 stop_reason/stop_details 的 metadata 捕获**移出 `stream_usage` 门控**（chat_models.py:2510-2511；usage 本身仍受门控，2503-2505）。Go 侧有同类问题要一并修：Go 流式 message_delta 的 chunk 发射整体被 `s.streamUsage && !s.usage.isZero()` 门控（stream.go:237），`streamUsage=false` 时消费者永远看不到携带 stop_reason 的 chunk——改为「chunk 必发、usage 受门控」。
- 输入方向：`messageToAnthropicContent`/`formatContentBlock` 遇 refusal 块跳过（Messages API 无 refusal 输入块，对齐 _compat.py:125-127 与 _format_messages chat_models.py:789-791 两处）。
- **#41715**：结构化输出（`InvokeStructured`/function_calling 路径，现有"no tool_call"报错点在 chatmodel.go:478）：当 tool choice 非强制（Go 侧：未强制单工具）且响应含 refusal 块时，返回 `*ModelRefusalError` 而不是"no tool_call"解析错。对齐上游语义：refusal 优先于 schema 校验错误。

**验收**：上游 `partners/anthropic/tests/unit_tests/test_refusal.py`（144 行）逐条移植；含 category/explanation 透传、流式聚合单块、输入方向丢弃、#41715 的强制/非强制两分。

### D5 — gemini safety → refusal（core translator #41658，Go 落 gemini partner）

**背景**：Go 无独立 translator 包，partner 直接产 typed blocks；落点在 `partners/gemini`。

**决策**：

- 非流式 `responseToMessage`（content.go:234 起）：candidate `FinishReason ∈ {SAFETY, PROHIBITED_CONTENT, BLOCKLIST}` 或 `PromptFeedback.BlockReason` 非空且非 `BLOCKED_REASON_UNSPECIFIED` 时，追加 `RefusalBlock{Refusal: finish_message 或 "", Extras: {"finish_reason":..., "block_reason":...}}`。枚举名取尾段（对齐 `_reason_name` 的 `rsplit(".",1)`）。
- 现有"无 candidates 报错"（content.go:237-240）保留——那是 HTTP 层成功但零候选的硬错误；refusal 块针对**有候选但被 safety 终止**的情形。两路都要：`PromptFeedback` 阻断导致零候选时，错误信息保留，且若能构造消息则带 refusal 块——对齐上游 translator 只看 metadata 的形状，Go 侧以"返回消息 + refusal 块"优先，无法构造消息（零候选）时维持报错。
- 流式：finish reason 到达时同样追加。
- 输入方向：refusal 块不发给 Gemini（无对应 part）。

**验收**：上游 core `tests/unit_tests/messages/test_refusal.py` 的 gemini 段（210-284 行，约 75 行）逐条移植到 `partners/gemini` 测试。

### D6 — ModelRefusalMiddleware + trace 脱敏（langchain 1.4.9，#41582 + #41689）

**决策**：`langchain/agents/middleware/model_refusal.go` 新增 `ModelRefusalMiddleware`，实现 `WrapModelCall(ctx, ModelRequest, ModelHandler) (ModelResponse, error)`：

- 调 handler；若返回 `StructuredOutputValidationError` 类错误，取其携带的 AI 消息（Go 侧已携带：structured_output.go:41-52 的 `AIMessage` 字段）先 `RaiseIfRefused`。
- 对 `ModelResponse` 结果消息逐个 `RaiseIfRefused`，命中返回 `*ModelRefusalError`。
- **trace 脱敏（#41689）**：对齐 `trace_policy = TracePolicy(process_inputs=omit_payload)`——Go 已有 `TracePolicyProvider`/`TracePolicyConfig` 机制（middleware/types.go:210、226）与 ModelRouter 先例（typesafe/middleware/model_router.go:169、openai/middleware/model_router.go:199），在 wrap_model_call 层安装 omit-inputs 策略。
- 注册进 middleware 导出面。

**验收**：上游 `tests/unit_tests/agents/middleware/implementations/test_model_refusal.py`（164 行）逐条移植；另加 fake tracer 断言 inputs 被 omit（#41689 面）。

### D7 — langgraph strict resume（langgraph 1.2.20，#9574；#9589 核查后无需行动）

**背景**：resume map 指定了非 pending 的 interrupt id 时上游静默忽略，客户端无法发现答错了。新增 opt-in strict 模式。Go 的 interrupt id 已是 `nodeName-counter` 每调用唯一（graph.go:2841-2846），#9589 的共享 id bug 类在 Go 不存在——**#9589 不移植**，本条只做 #9574。

**决策**：

- 新错误类型（放 `langgraph/types` 与 GraphInterrupt 同处）：
  ```go
  type UnknownInterruptError struct {
      Reason        string   // "not_pending" | "empty_map" | "ambiguous"
      InterruptIDs  []string // not_pending 时：未知/已答的 id
      PendingIDs    []string // 当前 pending 的 id（排序）
  }
  ```
  实现 `error`，消息对齐上游文案。
- 开关：`graph.Options` 新增类型化字段 `StrictResume bool`（**注意：graph.Options 是纯类型化 struct，无 Configurable map**——`Configurable` 在 core 的 runnables.Config，graph 运行不经过它；字符串键 `__pregel_strict_resume` 仅作文档对齐说明）。仅 root 图生效：`AddSubgraph` 包装器逐字段拷贝 Options 时**不拷贝** `StrictResume`（对齐 Python `not self.is_nested`；参照 forceResume 内部字段的既有传播控制，graph.go:844-849 邻域）；resume 钉在已被超越的 checkpoint（wave4 D2 的 superseded 概念）时同样跳过检查。
- 检查点位（`langgraph/graph/resume.go` 的 `resumeFromTuple`/`planResume`，写任何 writes 之前）：
  - resume 为 map 且键全是 interrupt id 形态：空 map → `empty_map`；`keys - pending` 非空 → `not_pending`（带 ids）。**注意**：Go 支持 NS 寻址（resumeValuesFor 里 NS 优先），strict 检查的 pending 集合 = pending interrupts 的 ID 与 NS 的并集，未知键判定同此并集。
  - resume 为单值：pending > 1 → `ambiguous`（非 strict 时维持现有报错）；pending == 0 时单值合法（静态断点恢复，对齐上游）。
- Go 无 RemoteGraph（沿用 backlog 裁定），droplist 不适用，DIVERGENCES 记一笔。
- **#9590**（顺手）：现有"multiple pending interrupts require a map"报错文案补 docs URL `https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts`（#9590 后的上游现行文案，_loop.py:1060-1061）。

**验收**：上游 `tests/test_strict_resume.py`（423 行、16 个测试函数，参数化展开 44 用例）移植代表性子集：unknown id / 已答 id / 空 map / 多 pending 单值四拒绝（且断言无任何 checkpoint 写入）；subgraph interrupt、同节点第二次 interrupt、pinned head checkpoint、静态断点单值四通过；非 strict 行为不变回归。Go 侧用内存 + sqlite 两 saver 覆盖。

### D8 — anthropic passthrough 补 caller/content（#41678 + #41637 相邻面）

**决策**：`partners/anthropic` 的透传链补 caller/content 两步：

- `contentBlock` struct（chatmodel.go:1100 邻域）先加字段 `Caller map[string]any \`json:"caller,omitempty"\``——现有 struct 无此字段，不加则 `setContentBlockField` 无处可写。
- `setContentBlockField`（contentblocks.go:240-289）增两键：
  - `"caller"`：**按块类型分派门控**（对齐 Python 的非对称行为）——`tool_use`（client 工具）块无条件透传（chat_models.py:834-835）；server_tool_use/mcp_tool_use/tool_result 族仅当 `type != "direct"` 时透传（chat_models.py:853-855、965-967）。注意 Go map 迭代无序，不能依赖 Type 先设置——在 `passthroughBlock` 循环结束后按 `out.Type` 做 caller 归置。
  - `"content"`：任意值透传到 `contentBlock.Content`（修 compaction 块内容丢失；#41637 的 index 泄漏 Go 构造免疫——contentBlock struct 无 index 字段，本条只补 content）。

**验收**：上游 `test_format_messages_strips_index_from_compaction_block` 与 #41678 的 caller 用例（90 行测试）逐条移植；加 Go 侧断言：replay 的 compaction 块带 content 不带 index；direct caller 不透传、非 direct 透传。

### D9 — model-profiles 数据同步（#41556 + #41706）

**决策**：按既有生成流程刷新 `modelprofiles/data/openai`（+ anthropic 若 #41706 涉及）与 `profiles.json`，提交物化产物；测试保持离线 fixtures。对齐 wave4 D7 的做法。

### 分歧记录（DIVERGENCES.md 新增/更新）

- 非标准块的 `index`：Python 提升到块顶层，Go 留在 `NonStandardContentBlock.Value` 内（#40213 核查结论）。
- interrupt id 形态：Go 为 `node-N`（nodeName-counter），非 xxh3(ns) 哈希；跨 SDK 共享 checkpoint 时 id 不互通（#9589 核查结论）。
- `EventToolStart/End/Error` 枚举存在但无发射点（工具走 chain 事件）——#41660 核查时的附带事实，补记。
- strict resume 的 RemoteGraph droplist 不适用（Go 无 RemoteGraph）。
- JsonOutputParser 文本路径无消息可见性：Python 在 `parse_result` 的 ChatGeneration 分支查 refusal；Go 的 `[]outputs.Generation` 不带消息（无协变），消息级检查改放 `InvokeStructured` 与 partner StructuredCaller 路径（见 D2）。

### 已核查无需行动（本波次，证据见 report 16 §3）

#41467（parse_partial_json 无重试循环）、#41697（chunk 末尾一次合并）、#40213（cloneMap 免疫）、#41660（无 BaseException 分层，chain_error 关 span）、#41553（int 字段 null→0）、#41643（chunk 事件是超集）、#9589（id 已唯一）、typesafe #41657（WithClassifier 已有）、fireworks #41693（无此 partner）、langgraph sdk/cli（出范围）。

## 测试同步总原则（用户要求）

每个 PORT 项必须把上游对应单测**逐条移植**为 Go 测试（测试文件、用例名、断言语义对齐），放在与实现同包的 `*_test.go`。上游测试文件清单：

- core：`tests/unit_tests/messages/test_refusal.py`、`tests/unit_tests/output_parsers/test_refusal.py`、`tests/unit_tests/test_exceptions.py`（ModelRefusalError 两条）
- langchain：`tests/unit_tests/agents/middleware/implementations/test_model_refusal.py`
- openai：`tests/unit_tests/chat_models/test_refusal.py` + test_base.py 的 refusal 段
- anthropic：`tests/unit_tests/test_refusal.py` + test_chat_models.py 的 compaction/caller 段
- langgraph：`tests/test_strict_resume.py` 代表子集（见 D7）

## 门禁

每任务：`go build ./... && go vet ./... && go test -race ./...`（root）+ 触及嵌套模块时对应 saver 测试 + `golangci-lint run ./...` 零 issue + 触及 coverage 追踪包时 `make coverage` ≥95%。收尾：全模块全绿 + 全仓 lint + final whole-branch review + CHANGELOG（v0.13.0 草节）/DIVERGENCES/report 16 追记。
