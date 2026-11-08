# Parity Wave 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the upstream 2026-11-04 release wave (core 1.7.4–1.7.6 / langchain 1.4.8–1.4.9 / openai 1.7.4 / anthropic 1.7.7–1.7.10+ / langgraph 1.2.20) into `langchain_golang`: the standardized refusal content-block family, langgraph strict resume, anthropic passthrough completion, and a model-profiles data refresh.

**Architecture:** All changes follow the decisions in `docs/superpowers/specs/2026-11-08-parity-wave5-design.md` (D1–D9), which is the single source of truth. Core first (block type + error + parser hook + OpenAI message conversion), then partners (openai, anthropic, gemini), then the agent middleware, then langgraph strict resume, then the anthropic passthrough fix, then data sync, then docs/gates.

**Tech Stack:** Go 1.2x, `github.com/projanvil/langchain-golang` module, standard `go test`/`go vet`/`golangci-lint` toolchain.

## Global Constraints

- Spec authority: `docs/superpowers/specs/2026-11-08-parity-wave5-design.md`; any conflict resolves to the spec.
- Python ground truth checkouts: `../langchain` @ `langchain-typesafe==0.0.2`, `../langgraph` @ `1.2.20` (paths relative to the `langchain_golang` repo root: `../langchain`, `../langgraph`).
- Upstream unit tests are ported case-by-case into Go `*_test.go` files next to the implementation (spec「测试同步总原则」); each ported test names its upstream source in a comment.
- Every task ends green: `go build ./... && go vet ./... && go test -race ./...` from the repo root.
- `golangci-lint run ./...` zero issues; packages under coverage tracking keep `make coverage` ≥ 95%.
- Commit style: Conventional Commits (`feat(core): ...`, `fix(anthropic): ...`), matching repo history.
- No new third-party dependencies.
- Default behavior must not change for existing users: strict resume is opt-in; refusal blocks are additive; parsers only gain a new error where upstream added one.

---

### Task 1: core — `RefusalBlock` content block type

**Files:**
- Modify: `core/messages/contentblocks.go`（`KNOWN_BLOCK_TYPES` ~L317、`ParseContentBlock` ~L344、`BlockToMap` ~L382、`CloneBlock` ~L606、编译期检查 var 块 ~L305）
- Test: `core/messages/refusal_block_test.go`（新建）

**Interfaces:**
- Produces（后续所有任务依赖）:
  ```go
  // core/messages
  type RefusalBlock struct {
      Refusal string
      ID      string
      Index   any
      Extras  map[string]any
  }
  func (RefusalBlock) BlockType() string // "refusal"
  func (RefusalBlock) isContentBlock()
  ```

- [ ] **Step 1: 写失败测试**

新建 `core/messages/refusal_block_test.go`：

```go
package messages

import (
	"reflect"
	"testing"
)

// Ports core/tests/unit_tests/messages/test_refusal.py（#41582）的块级用例。

func TestRefusalBlockRoundTrip(t *testing.T) {
	m := map[string]any{
		"type":    "refusal",
		"refusal": "I cannot help with that.",
		"id":      "ref-1",
		"index":   "lc_0",
		"category": "safety", // 未知键 → Extras
	}
	block := ParseContentBlock(m)
	rb, ok := block.(RefusalBlock)
	if !ok {
		t.Fatalf("ParseContentBlock returned %T, want RefusalBlock", block)
	}
	if rb.Refusal != "I cannot help with that." || rb.ID != "ref-1" || rb.Index != "lc_0" {
		t.Fatalf("fields mismatch: %+v", rb)
	}
	if !reflect.DeepEqual(rb.Extras, map[string]any{"category": "safety"}) {
		t.Fatalf("extras mismatch: %+v", rb.Extras)
	}
	back := BlockToMap(rb)
	if back["type"] != "refusal" || back["refusal"] != "I cannot help with that." {
		t.Fatalf("BlockToMap lost fields: %v", back)
	}
	if back["id"] != "ref-1" || back["index"] != "lc_0" {
		t.Fatalf("BlockToMap lost id/index: %v", back)
	}
	// Extras 摊平到顶层（mergeExtras 惯例，无 "extras" 键）。
	if back["category"] != "safety" {
		t.Fatalf("BlockToMap must flatten extras, got: %v", back)
	}
	if _, hasExtrasKey := back["extras"]; hasExtrasKey {
		t.Fatalf("BlockToMap must not emit an \"extras\" key: %v", back)
	}
}

func TestRefusalBlockMinimalOmitsOptionalFields(t *testing.T) {
	back := BlockToMap(RefusalBlock{Refusal: ""})
	if back["type"] != "refusal" || back["refusal"] != "" {
		t.Fatalf("minimal block mismatch: %v", back)
	}
	if _, ok := back["id"]; ok {
		t.Fatalf("empty id must be omitted: %v", back)
	}
	if _, ok := back["index"]; ok {
		t.Fatalf("nil index must be omitted: %v", back)
	}
}

func TestRefusalBlockCloneIsDeep(t *testing.T) {
	orig := RefusalBlock{Refusal: "no", Extras: map[string]any{"category": "safety"}}
	cloned, ok := CloneBlock(orig).(RefusalBlock)
	if !ok {
		t.Fatalf("CloneBlock returned %T", CloneBlock(orig))
	}
	cloned.Extras["category"] = "mutated"
	if orig.Extras["category"] != "safety" {
		t.Fatal("CloneBlock must deep-copy Extras")
	}
}

func TestRefusalBlockKnownType(t *testing.T) {
	if !KNOWN_BLOCK_TYPES["refusal"] {
		t.Fatal("KNOWN_BLOCK_TYPES must include \"refusal\"")
	}
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/yuhaochen/Documents/codebase/projanvil/langchain_port/langchain_golang && go test ./core/messages/ -run TestRefusalBlock -v`
Expected: FAIL —— `undefined: RefusalBlock`。

- [ ] **Step 3: 实现**

在 `core/messages/contentblocks.go` 中：

1. 在 `ReasoningBlock` 定义之后新增：

```go
// RefusalBlock is the standardized model-refusal content block (Python's
// RefusalContentBlock, langchain-core 1.7.5, #41582). Providers report
// refusals differently — OpenAI returns refusal content, Anthropic reports a
// refusal stop reason, Gemini reports safety finish reasons — and each
// partner translates its shape into this block so callers detect refusals
// without provider-specific code. Refusals the model writes as ordinary text
// are not detected.
type RefusalBlock struct {
	// Refusal is the human-readable refusal message, or "" when the provider
	// gave none (e.g. Anthropic's stop_reason-only refusal).
	Refusal string
	// ID is the optional provider-side block identifier.
	ID string
	// Index positions the block during streaming aggregation.
	Index any
	// Extras carries provider-specific metadata, e.g. Anthropic's refusal
	// "category" or Gemini's finish_reason/block_reason.
	Extras map[string]any
}

// BlockType returns "refusal".
func (RefusalBlock) BlockType() string { return "refusal" }

// isContentBlock seals RefusalBlock.
func (RefusalBlock) isContentBlock() {}
```

2. 编译期检查 var 块（`_ ContentBlock = ReasoningBlock{}` 行后）加 `_ ContentBlock = RefusalBlock{}`。
3. `KNOWN_BLOCK_TYPES` map 加 `"refusal": true`。
4. `ParseContentBlock` switch 加：

```go
	case "refusal":
		return parseRefusalBlock(m)
```

并在 `parseReasoningBlock` 旁新增：

```go
func parseRefusalBlock(m map[string]any) RefusalBlock {
	b := RefusalBlock{}
	b.Refusal, _ = m["refusal"].(string)
	b.ID, _ = m["id"].(string)
	b.Index = m["index"]
	b.Extras = extractExtras(m, "type", "refusal", "id", "index")
	return b
}
```

5. `BlockToMap` switch 在 `case ReasoningBlock:` 后加：

```go
	case RefusalBlock:
		out := map[string]any{"type": "refusal", "refusal": v.Refusal}
		if v.ID != "" {
			out["id"] = v.ID
		}
		if v.Index != nil {
			out["index"] = v.Index
		}
		mergeExtras(out, v.Extras)
		return out
```

（先读 `BlockToMap` 现有 case 的确切写法——switch 是否 `switch v := b.(type)`——照抄其形态。）

6. `CloneBlock` switch 加：

```go
	case RefusalBlock:
		return RefusalBlock{
			Refusal: v.Refusal,
			ID:      v.ID,
			Index:   v.Index,
			Extras:  cloneMap(v.Extras),
		}
```

- [ ] **Step 4: 运行确认通过**

Run: `go test ./core/messages/ -run TestRefusalBlock -v && go build ./... && go vet ./core/...`
Expected: 4 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
cd /Users/yuhaochen/Documents/codebase/projanvil/langchain_port/langchain_golang
git add core/messages/contentblocks.go core/messages/refusal_block_test.go
git commit -m "feat(core): add standardized RefusalBlock content block (langchain-core 1.7.5, #41582)"
```

---

### Task 2: core — `ModelRefusalError` + `RaiseIfRefused` + parser/structured 接入

**Files:**
- Modify: `core/lcerrors/exceptions.go`
- Create: `core/outputparser/refusal.go`
- Modify: `core/outputparser/openai.go`（`JSONOutputToolsParser.ParseMessage` L114、`JSONOutputKeyToolsParser.ParseMessage` L137）
- Modify: `core/language/structured.go`（`InvokeStructured` 默认路径）
- Test: `core/lcerrors/refusal_error_test.go`（新建）、`core/outputparser/refusal_test.go`（新建）、`core/language/structured_refusal_test.go`（新建）

**Interfaces:**
- Consumes: Task 1 的 `messages.RefusalBlock`。
- Produces:
  ```go
  // core/lcerrors
  type ModelRefusalError struct {
      Message    string
      RawMessage *messages.Message
  }
  func (e *ModelRefusalError) Error() string
  func (e *ModelRefusalError) IsModelRetryable() bool // 恒 false
  func ModelRefusalFromBlock(block messages.RefusalBlock, raw messages.Message) *ModelRefusalError
  var ErrModelRefusal error // sentinel，errors.Is 可匹配

  // core/outputparser
  func RaiseIfRefused(msg messages.Message) error
  ```

- [ ] **Step 1: 写失败测试**

`core/lcerrors/refusal_error_test.go`：

```go
package lcerrors

import (
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Ports core/tests/unit_tests/test_exceptions.py::test_model_refusal_error{,_defaults}（#41582）。

func TestModelRefusalError(t *testing.T) {
	raw := messages.AI("")
	raw.ResponseMetadata = map[string]any{"stop_reason": "refusal"}
	err := &ModelRefusalError{Message: "Blocked.", RawMessage: &raw}
	if err.Error() != "Blocked." {
		t.Fatalf("Error() = %q", err.Error())
	}
	if err.IsModelRetryable() {
		t.Fatal("refusal must not be retryable")
	}
	if err.RawMessage == nil || err.RawMessage.ResponseMetadata["stop_reason"] != "refusal" {
		t.Fatal("RawMessage must be preserved")
	}
	if !errors.Is(err, ErrModelRefusal) {
		t.Fatal("errors.Is(err, ErrModelRefusal) must hold")
	}
}

func TestModelRefusalErrorDefaults(t *testing.T) {
	err := &ModelRefusalError{}
	if err.Error() != "Model refused to produce the requested output" {
		t.Fatalf("default message = %q", err.Error())
	}
	if err.RawMessage != nil {
		t.Fatal("RawMessage defaults to nil")
	}
}

func TestModelRefusalFromBlock(t *testing.T) {
	raw := messages.AI("")
	// 无 category 无文本
	e1 := ModelRefusalFromBlock(messages.RefusalBlock{}, raw)
	if e1.Error() != "Model refused to produce the requested output" {
		t.Fatalf("bare: %q", e1.Error())
	}
	// 仅 category
	e2 := ModelRefusalFromBlock(messages.RefusalBlock{Extras: map[string]any{"category": "safety"}}, raw)
	if e2.Error() != "Model refused to produce the requested output (category='safety')" {
		t.Fatalf("category: %q", e2.Error())
	}
	// 全量
	e3 := ModelRefusalFromBlock(messages.RefusalBlock{
		Refusal: "Cannot comply.",
		Extras:  map[string]any{"category": "safety"},
	}, raw)
	want := "Model refused to produce the requested output (category='safety'): Cannot comply."
	if e3.Error() != want {
		t.Fatalf("full: %q want %q", e3.Error(), want)
	}
	if e3.RawMessage == nil {
		t.Fatal("RawMessage must be set")
	}
}
```

`core/outputparser/refusal_test.go`：

```go
package outputparser

import (
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
)

// Ports core/tests/unit_tests/output_parsers/test_refusal.py（#41582）。

func refusalMessage() messages.Message {
	m := messages.AI("")
	m.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: "I can't do that."}}
	return m
}

func TestRaiseIfRefused(t *testing.T) {
	if err := RaiseIfRefused(refusalMessage()); err == nil {
		t.Fatal("refusal message must raise")
	} else {
		var mre *lcerrors.ModelRefusalError
		if !errors.As(err, &mre) {
			t.Fatalf("want *ModelRefusalError, got %T", err)
		}
	}
	plain := messages.AI("hello")
	plain.ContentBlocks = []messages.ContentBlock{messages.TextBlock{Text: "hello"}}
	if err := RaiseIfRefused(plain); err != nil {
		t.Fatalf("non-refusal message must pass, got %v", err)
	}
}

func TestJSONOutputToolsParserRefused(t *testing.T) {
	p := JSONOutputToolsParser{}
	_, err := p.ParseMessage(refusalMessage())
	var refusalErr *lcerrors.ModelRefusalError
	if !errors.As(err, &refusalErr) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}
}

func TestJSONOutputKeyToolsParserRefused(t *testing.T) {
	p := JSONOutputKeyToolsParser{KeyName: "anything"}
	_, err := p.ParseMessage(refusalMessage())
	var refusalErr *lcerrors.ModelRefusalError
	if !errors.As(err, &refusalErr) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}
}
```

`core/language/structured_refusal_test.go`：

```go
package language

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// 上游 base.py 的 checked_parser 语义（#41582）：结构化输出路径遇 refusal
// 抛 ModelRefusalError 而非 schema/parse 错误。Go 落点为 InvokeStructured
// 默认路径（partner StructuredCaller 各自接，见 wave5 spec D3/D4）。

// refusalModel 是一个不实现 StructuredCaller 的最小 ChatModel，恒返回
// refusal 块消息，强制走 InvokeStructured 默认路径。
type refusalModel struct{}

func (m *refusalModel) Invoke(_ context.Context, _ []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	out := messages.AI("")
	out.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: "no"}}
	return out, nil
}

func (m *refusalModel) Stream(_ context.Context, _ []messages.Message, _ ...runnables.Option) (runnables.Stream[messages.Message], error) {
	msg, _ := m.Invoke(context.Background(), nil)
	return runnables.NewSliceStream([]messages.Message{msg}), nil
}

func (m *refusalModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i := range inputs {
		msg, err := m.Invoke(ctx, inputs[i], opts...)
		if err != nil {
			return nil, err
		}
		out[i] = msg
	}
	return out, nil
}

func (m *refusalModel) InputSchema() schema.Schema  { return schema.Schema{"type": "array"} }
func (m *refusalModel) OutputSchema() schema.Schema { return schema.Schema{"type": "object"} }
func (m *refusalModel) ConfigSchema() schema.Schema { return schema.Schema{"type": "object"} }
func (m *refusalModel) BindTools(_ []tools.Tool) (ChatModel, error) {
	return m, nil
}
func (m *refusalModel) Capabilities() ChatModelCapabilities { return ChatModelCapabilities{} }

func TestInvokeStructuredRefused(t *testing.T) {
	_, err := InvokeStructured(context.Background(), &refusalModel{},
		[]messages.Message{messages.Human("hi")},
		schema.Schema{"type": "object"})
	var mre *lcerrors.ModelRefusalError
	if !errors.As(err, &mre) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}
}
```

注意：`refusalModel` 必须满足 `ChatModel` 接口（chatmodel.go:51-55，内嵌 `runnables.Runnable[[]messages.Message, messages.Message]`）；若 Runnable 接口还有别的方法（如 `WithConfig`），编译报错时按接口补齐空实现。若 `messages.Human` 构造函数名不同，按 messages 包现有构造器调整。

- [ ] **Step 2: 运行确认失败**

Run: `go test ./core/lcerrors/ -run TestModelRefusal -v; go test ./core/outputparser/ -run Refused -v`
Expected: FAIL —— undefined: ModelRefusalError / RaiseIfRefused。

- [ ] **Step 3: 实现**

1. `core/lcerrors/exceptions.go` 末尾新增：

```go
// ErrModelRefusal is the sentinel matched by ModelRefusalError.Is.
var ErrModelRefusal = errors.New("model refusal")

// ModelRefusalError mirrors Python's ModelRefusalError (langchain-core 1.7.5,
// #41582): the model declined to produce the requested output. Providers
// report refusals as successful responses (Anthropic stop_reason="refusal",
// OpenAI's refusal field), so plain model calls return the message as-is;
// paths that need the withheld content (structured output, output parsers,
// ModelRefusalMiddleware) surface this error instead.
//
// It is NOT a ProviderError: a refusal arrives with HTTP 200.
type ModelRefusalError struct {
	// Message is the error text; empty means the default.
	Message string
	// RawMessage is the model response that contained the refusal; its
	// ContentBlocks hold the refusal block including provider extras.
	RawMessage *messages.Message
}

// Error implements the error interface, defaulting the message.
func (e *ModelRefusalError) Error() string {
	if e.Message == "" {
		return "Model refused to produce the requested output"
	}
	return e.Message
}

// Is matches ErrModelRefusal.
func (e *ModelRefusalError) Is(target error) bool { return target == ErrModelRefusal }

// IsModelRetryable reports false — retrying the same request is unlikely to
// succeed (Python: ModelRefusalError.is_retryable = False).
func (e *ModelRefusalError) IsModelRetryable() bool { return false }

// ModelRefusalFromBlock builds a ModelRefusalError from a standard refusal
// block, mirroring Python's ModelRefusalError.from_refusal: the message
// carries the refusal's category and text when present.
func ModelRefusalFromBlock(block messages.RefusalBlock, raw messages.Message) *ModelRefusalError {
	msg := "Model refused to produce the requested output"
	if category, ok := block.Extras["category"].(string); ok && category != "" {
		msg += " (category='" + category + "')"
	}
	if block.Refusal != "" {
		msg += ": " + block.Refusal
	}
	return &ModelRefusalError{Message: msg, RawMessage: &raw}
}
```

文件头 import 增加 `"errors"`（若无）与 `"github.com/projanvil/langchain-golang/core/messages"`。core/messages 不 import lcerrors，无循环。

2. 新建 `core/outputparser/refusal.go`：

```go
package outputparser

import (
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
)

// RaiseIfRefused returns a *lcerrors.ModelRefusalError when msg carries a
// standard refusal content block, and nil otherwise. It mirrors Python's
// langchain_core.output_parsers._refusal.raise_if_refused (#41582), shared by
// the structured output parsers and the agents' ModelRefusalMiddleware.
func RaiseIfRefused(msg messages.Message) error {
	for _, block := range msg.ContentBlocks {
		if rb, ok := block.(messages.RefusalBlock); ok {
			return lcerrors.ModelRefusalFromBlock(rb, msg)
		}
	}
	return nil
}
```

3. `core/outputparser/openai.go`：两个 ParseMessage 入口处（取到 message 后、解析 tool calls 前）各加一行：

```go
	if err := RaiseIfRefused(message); err != nil {
		return nil, err
	}
```

即 `JSONOutputToolsParser.ParseMessage`（L114 起）与 `JSONOutputKeyToolsParser.ParseMessage`（L137 起）函数体第一行。

4. `core/language/structured.go` 的 `InvokeStructured`：native 路径返回后与默认路径 `m.Invoke` 成功后各插一行检查。改为：

```go
	if native, ok := m.(StructuredCaller); ok {
		msg, err := native.InvokeStructured(ctx, input, sch)
		if err != nil {
			return msg, err
		}
		if err := outputparser.RaiseIfRefused(msg); err != nil {
			return messages.Message{}, err
		}
		return msg, nil
	}

	response, err := m.Invoke(ctx, input)
	if err != nil {
		return messages.Message{}, fmt.Errorf("structured output: invoke model: %w", err)
	}
	if err := outputparser.RaiseIfRefused(response); err != nil {
		return messages.Message{}, err
	}
```

依赖方向已验证：`core/outputparser` 不依赖 `core/language`（`go list -deps ./core/outputparser` 无 language），`core/language` 可以安全 import `core/outputparser`。

- [ ] **Step 4: 运行确认通过**

Run: `go test -race ./core/lcerrors/ ./core/outputparser/ ./core/language/ -v 2>&1 | tail -30 && go build ./... && go vet ./core/...`
Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add core/lcerrors/ core/outputparser/ core/language/
git commit -m "feat(core): ModelRefusalError + RaiseIfRefused wired into tool parsers and InvokeStructured (#41582)"
```

---

### Task 3: core — `ConvertToOpenAIMessages` 发送 refusal 块

**Files:**
- Modify: `core/messages/convert.go`（`openAIContent` ~L256）
- Test: `core/messages/convert_refusal_test.go`（新建）

**Interfaces:**
- Consumes: Task 1 的 `messages.RefusalBlock`。
- Produces: 无新导出（行为修复）。

- [ ] **Step 1: 写失败测试**

`core/messages/convert_refusal_test.go`：

```go
package messages

import (
	"testing"
)

// Ports core/tests/unit_tests/messages/test_refusal.py 的 convert_to_openai_messages
// 用例（#41582，utils.py:1955-1959）：非空 refusal 文本发出，空文本不发。

func TestConvertToOpenAIMessagesRefusal(t *testing.T) {
	msg := AI("")
	msg.ContentBlocks = []ContentBlock{
		TextBlock{Text: "partial answer"},
		RefusalBlock{Refusal: "I cannot continue."},
	}
	out, err := ConvertToOpenAIMessages([]Message{msg})
	if err != nil {
		t.Fatal(err)
	}
	content, ok := out[0]["content"].([]map[string]any)
	if !ok {
		t.Fatalf("content is %T, want []map[string]any", out[0]["content"])
	}
	var refusal map[string]any
	for _, part := range content {
		if part["type"] == "refusal" {
			refusal = part
		}
	}
	if refusal == nil || refusal["refusal"] != "I cannot continue." {
		t.Fatalf("refusal part missing or wrong: %v", content)
	}
}

func TestConvertToOpenAIMessagesEmptyRefusalSkipped(t *testing.T) {
	msg := AI("")
	msg.ContentBlocks = []ContentBlock{
		TextBlock{Text: "ok"},
		RefusalBlock{Refusal: ""}, // Anthropic 式无文本 refusal
	}
	out, err := ConvertToOpenAIMessages([]Message{msg})
	if err != nil {
		t.Fatal(err)
	}
	content := out[0]["content"].([]map[string]any)
	for _, part := range content {
		if part["type"] == "refusal" {
			t.Fatalf("empty refusal must not be sent: %v", content)
		}
	}
}
```

注意：`openAIContent` 当前返回 `[]map[string]any`（经 `BlockToMap`）。若测试里类型断言与实际返回类型不符，按实现调整断言。

- [ ] **Step 2: 运行确认失败**

Run: `go test ./core/messages/ -run TestConvertToOpenAIMessagesRefusal -v`
Expected: FAIL —— refusal 块经 BlockToMap 已是 `{"type":"refusal",...}` 形状，非空用例可能直接通过；空文本用例必须 FAIL（当前无跳过逻辑）。若两例都过，说明 BlockToMap 输出意外符合——仍须加显式跳过逻辑保证语义（下面 Step 3），并把该发现记入 commit message。

- [ ] **Step 3: 实现**

修改 `core/messages/convert.go` 的 `openAIContent`：

```go
// openAIContent returns the OpenAI content value: the message's string content,
// or a list of block maps when structured blocks are present.
// Refusal blocks are emitted as bare {"type","refusal"} parts — id/index/extras
// are NOT forwarded (Python utils.py:1955-1959 builds a fresh two-key dict, and
// test_refusal.py:200-207 pins extras staying off the wire); empty-text
// refusals are dropped ("Refusals without text have nothing to send").
func openAIContent(message Message) any {
	if len(message.ContentBlocks) > 0 {
		blocks := make([]map[string]any, 0, len(message.ContentBlocks))
		for _, block := range message.ContentBlocks {
			if rb, ok := block.(RefusalBlock); ok {
				if rb.Refusal == "" {
					continue
				}
				blocks = append(blocks, map[string]any{"type": "refusal", "refusal": rb.Refusal})
				continue
			}
			blocks = append(blocks, BlockToMap(block))
		}
		return blocks
	}
	return message.Content
}
```

并在 `core/messages/convert_refusal_test.go` 追加 extras 丢弃断言（移植上游 test_refusal.py:200-207）：

```go
func TestConvertToOpenAIMessagesRefusalDropsExtras(t *testing.T) {
	msg := AI("")
	msg.ContentBlocks = []ContentBlock{
		RefusalBlock{Refusal: "no", ID: "r1", Extras: map[string]any{"category": "safety"}},
	}
	out, err := ConvertToOpenAIMessages([]Message{msg})
	if err != nil {
		t.Fatal(err)
	}
	content := out[0]["content"].([]map[string]any)
	if len(content) != 1 {
		t.Fatalf("want exactly 1 part, got %v", content)
	}
	if len(content[0]) != 2 || content[0]["type"] != "refusal" || content[0]["refusal"] != "no" {
		t.Fatalf("refusal part must carry exactly type+refusal, got %v", content[0])
	}
}
```

- [ ] **Step 4: 运行确认通过**

Run: `go test -race ./core/messages/ && go build ./... && go vet ./core/...`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add core/messages/convert.go core/messages/convert_refusal_test.go
git commit -m "feat(core): send non-empty refusal blocks in ConvertToOpenAIMessages (#41582)"
```

---

### Task 4: openai partner — refusal 双向支持

**Files:**
- Modify: `partners/openai/chat_completions.go`（`chatCompletionsMessage` L176、`toResponsesPayload` L303、`chatMessage` L71、`buildChatCompletionsRequest` 的 RoleAI 分支 L255）
- Modify: `partners/openai/chat_completions_stream.go`（`chatCompletionsStreamDelta` L281）
- Modify: `partners/openai/chatmodel.go`（`contentOutput` L643、`toMessage` L665、buildRequest 的 RoleAI 分支 L485）
- Modify: `partners/openai/stream.go`（`response.refusal.done` 分支 L390-394）
- Test: `partners/openai/refusal_test.go`（新建）

**Interfaces:**
- Consumes: Task 1 `messages.RefusalBlock`、Task 2 `outputparser.RaiseIfRefused`/`lcerrors.ModelRefusalError`。
- Produces: 无新导出（partner 内部行为）。

- [ ] **Step 1: 写失败测试**

`partners/openai/refusal_test.go`：

```go
package openai

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

// Ports partners/openai/tests/unit_tests/chat_models/test_refusal.py（#41582，
// openai 1.7.4）。

// --- chat completions 非流式：message.refusal → AdditionalKwargs + RefusalBlock ---

func TestChatCompletionsRefusalNonStream(t *testing.T) {
	raw := `{
		"id": "chatcmpl-1", "model": "gpt-x",
		"choices": [{"message": {"role": "assistant", "content": "", "refusal": "I can't help with that."}}]
	}`
	var resp chatCompletionsResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatal(err)
	}
	msg := resp.toResponsesPayload().toMessage()
	if msg.AdditionalKwargs["refusal"] != "I can't help with that." {
		t.Fatalf("additional_kwargs refusal missing: %v", msg.AdditionalKwargs)
	}
	var found bool
	for _, b := range msg.ContentBlocks {
		if rb, ok := b.(messages.RefusalBlock); ok {
			found = true
			if rb.Refusal != "I can't help with that." {
				t.Fatalf("refusal text = %q", rb.Refusal)
			}
		}
	}
	if !found {
		t.Fatalf("no RefusalBlock in %v", msg.ContentBlocks)
	}
}

// --- responses API 非流式：refusal content part → RefusalBlock ---
// id 挂在 message ITEM 上（上游 base.py:5269-5277 取 output.id），
// phase 同理（base.py:5250）；content part 自身不带 id。

func TestResponsesRefusalNonStream(t *testing.T) {
	raw := `{
		"id": "resp-1", "model": "gpt-x",
		"output": [{"type": "message", "role": "assistant", "id": "msg_1", "phase": "final_answer",
			"content": [{"type": "refusal", "refusal": "No."}]}]
	}`
	var resp responsePayload
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatal(err)
	}
	msg := resp.toMessage()
	var rb *messages.RefusalBlock
	for _, b := range msg.ContentBlocks {
		if r, ok := b.(messages.RefusalBlock); ok {
			cpy := r
			rb = &cpy
		}
	}
	if rb == nil {
		t.Fatalf("no RefusalBlock in %v", msg.ContentBlocks)
	}
	if rb.Refusal != "No." {
		t.Fatalf("refusal text = %q", rb.Refusal)
	}
	if rb.ID != "msg_1" {
		t.Fatalf("refusal ID must come from the message item, got %q", rb.ID)
	}
	if rb.Extras["phase"] != "final_answer" {
		t.Fatalf("phase must land in extras, got %v", rb.Extras)
	}
}

// --- 回写方向（responses）：part 只带 type+refusal，id/phase 提升到 item 级 ---
// 上游 test_refusal.py:39-47 钉死的 wire 形状。

func TestResponsesRefusalReplayWireShape(t *testing.T) {
	ai := messages.AI("")
	ai.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{
		Refusal: "No.", ID: "msg_1", Extras: map[string]any{"phase": "final_answer"},
	}}
	m := ChatModel{}
	req, err := m.buildRequest([]messages.Message{ai})
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(req.Input)
	var items []map[string]any
	_ = json.Unmarshal(encoded, &items)
	if len(items) != 1 {
		t.Fatalf("want 1 input item, got %s", encoded)
	}
	item := items[0]
	if item["type"] != "message" || item["role"] != "assistant" {
		t.Fatalf("item shape: %v", item)
	}
	if item["id"] != "msg_1" || item["phase"] != "final_answer" {
		t.Fatalf("id/phase must be lifted to the item: %v", item)
	}
	parts, _ := item["content"].([]any)
	if len(parts) != 1 {
		t.Fatalf("want 1 refusal part, got %v", item["content"])
	}
	part, _ := parts[0].(map[string]any)
	if len(part) != 2 || part["type"] != "refusal" || part["refusal"] != "No." {
		t.Fatalf("refusal part must carry exactly type+refusal: %v", part)
	}
}

// 空文本 refusal 不发送（_compat.py:451-455）。
func TestResponsesEmptyRefusalSkipped(t *testing.T) {
	ai := messages.AI("")
	ai.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: ""}}
	m := ChatModel{}
	req, err := m.buildRequest([]messages.Message{ai})
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(req.Input)
	if strings.Contains(string(encoded), "refusal") {
		t.Fatalf("empty refusal must not be sent: %s", encoded)
	}
}

// --- CC 回写方向：refusal 块 → content parts（上游 _compat.py:179-184 形状）---

func TestChatCompletionsRefusalReplayWireShape(t *testing.T) {
	ai := messages.AI("partial")
	ai.ContentBlocks = []messages.ContentBlock{
		messages.TextBlock{Text: "partial"},
		messages.RefusalBlock{Refusal: "declined"},
	}
	m := ChatModel{}
	req, err := m.buildChatCompletionsRequest([]messages.Message{ai})
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(req.Messages[0])
	var wire map[string]any
	_ = json.Unmarshal(encoded, &wire)
	parts, ok := wire["content"].([]any)
	if !ok {
		t.Fatalf("assistant content must become parts when refusal present: %s", encoded)
	}
	var sawText, sawRefusal bool
	for _, p := range parts {
		part, _ := p.(map[string]any)
		switch part["type"] {
		case "text":
			sawText = part["text"] == "partial"
		case "refusal":
			sawRefusal = part["refusal"] == "declined"
		}
	}
	if !sawText || !sawRefusal {
		t.Fatalf("want text+refusal parts: %s", encoded)
	}
}

// --- 结构化输出：refusal → ModelRefusalError（fake HTTPClient 端到端）---
// config.HTTPClient 可注入 fake RoundTripper（stream.go:54-57 先例）。

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func refusalResponseBody() string {
	return `{"id":"resp-r","model":"gpt-x","output":[{"type":"message","role":"assistant",
		"content":[{"type":"refusal","refusal":"I cannot provide that."}]}],
		"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
}

func TestInvokeStructuredRaisesModelRefusal(t *testing.T) {
	fake := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(refusalResponseBody)),
		}, nil
	})
	m := NewChatModel( /* 按包内现有测试的构造方式，注入 HTTPClient: fake */ )
	_ = fake
	_, err := m.InvokeStructured(t.Context(), []messages.Message{messages.Human("hi")},
		schema.Schema{"type": "object", "title": "Out"})
	var mre *lcerrors.ModelRefusalError
	if !errors.As(err, &mre) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}
}
```

实现注意：最后一个用例的模型构造与 HTTPClient 注入按本包 client_test.go / stream 测试的现有 fake 模式落地（配置字段名以 config struct 实际为准）。

- [ ] **Step 2: 运行确认失败**

Run: `go test ./partners/openai/ -run Refusal -v`
Expected: FAIL —— 编译失败（wire 结构无 refusal 字段）或断言失败。

- [ ] **Step 3: 实现**

1. `chat_completions.go`：
   - `chatCompletionsMessage` 加 `Refusal string \`json:"refusal"\``。
   - `toResponsesPayload`：tool calls 循环后加：
     ```go
     if msg.Refusal != "" {
     	out.Output = append(out.Output, outputItem{
     		Type:    "message",
     		Role:    "assistant",
     		Content: []contentOutput{{Type: "refusal", Refusal: msg.Refusal}},
     	})
     }
     ```
   - `buildChatCompletionsRequest` 的 `case messages.RoleAI:` 分支：refusal 回放对齐上游 `_convert_from_v1_to_chat_completions`（_compat.py:179-184）的 **content parts** 形状——AI 消息带非空 refusal 块时，Content 从纯字符串改为 parts 列表：
     ```go
     case messages.RoleAI:
     	cm.Role = "assistant"
     	var refusalParts []any
     	for _, block := range message.ContentBlocks {
     		if rb, ok := block.(messages.RefusalBlock); ok && rb.Refusal != "" {
     			refusalParts = append(refusalParts, map[string]any{"type": "refusal", "refusal": rb.Refusal})
     		}
     	}
     	if refusal, ok := message.AdditionalKwargs["refusal"].(string); ok && refusal != "" && len(refusalParts) == 0 {
     		refusalParts = append(refusalParts, map[string]any{"type": "refusal", "refusal": refusal})
     	}
     	if len(refusalParts) > 0 {
     		parts := make([]any, 0, 1+len(refusalParts))
     		if message.Content != "" {
     			parts = append(parts, map[string]any{"type": "text", "text": message.Content})
     		}
     		parts = append(parts, refusalParts...)
     		cm.Content = parts
     	}
     	// ……既有 tool_calls 循环保持不变
     ```
     注意：现有分支里 `cm.Content` 的默认赋值位置要读代码确认——保证 refusal 路径覆盖默认。
2. `chatmodel.go`：
   - `contentOutput` 加 `Refusal string \`json:"refusal,omitempty"\``（**不加 ID**——responses API 的 refusal part 不带 id）。
   - `outputItem`（chatmodel.go:616 邻域）加 `Phase string \`json:"phase,omitempty"\``（`ID` 字段若已有则复用）。
   - `toMessage` 的 `case "message":` 内层 content 循环加：
     ```go
     if content.Type == "refusal" {
     	// id 挂在 message item 上（base.py:5269-5277 取 output.id），
     	// phase 同理（base.py:5250）。
     	rb := messages.RefusalBlock{Refusal: content.Refusal, ID: output.ID}
     	if output.Phase != "" {
     		rb.Extras = map[string]any{"phase": output.Phase}
     	}
     	contentBlocks = append(contentBlocks, rb)
     	refusalText = content.Refusal
     }
     ```
     （`refusalText` 为新局部变量；内层循环当前能访问的 item 变量名以实际代码为准。）message 构造后：
     ```go
     if refusalText != "" {
     	if message.AdditionalKwargs == nil {
     		message.AdditionalKwargs = map[string]any{}
     	}
     	message.AdditionalKwargs["refusal"] = refusalText
     }
     ```
   - `buildRequest` 的 `case messages.RoleAI:` 分支（chatmodel.go:485 起）：refusal 回放对齐上游 `_construct_responses_api_input`——part 只带 `type`+`refusal`，id/phase 提升到 **item** 级：
     ```go
     // 收集非空 refusal 块
     var refusalParts []any
     var refusalID, refusalPhase string
     for _, block := range message.ContentBlocks {
     	if rb, ok := block.(messages.RefusalBlock); ok && rb.Refusal != "" {
     		refusalParts = append(refusalParts, map[string]any{"type": "refusal", "refusal": rb.Refusal})
     		if refusalID == "" {
     			refusalID = rb.ID
     		}
     		if refusalPhase == "" {
     			refusalPhase, _ = rb.Extras["phase"].(string)
     		}
     	}
     	if len(refusalParts) > 0 {
     		parts := make([]any, 0, 1+len(refusalParts))
     		if message.Content != "" {
     			parts = append(parts, map[string]any{"type": "output_text", "text": message.Content})
     		}
     		parts = append(parts, refusalParts...)
     		item := inputItem{Type: "message", Role: "assistant", Content: parts}
     		if refusalID != "" {
     			item.ID = refusalID
     		}
     		if refusalPhase != "" {
     			item.Phase = refusalPhase
     		}
     		payload.Input = append(payload.Input, item)
     	} else if message.Content != "" {
     		// 既有纯文本分支保持不变
     	}
     ```
     先读 `inputItem` 定义（chatmodel.go:576-588）确认 Content 字段类型接受 `[]any`、并给 `inputItem` 加 `Phase string \`json:"phase,omitempty"\``（ID 字段若已有则复用）。
3. `stream.go`：`response.refusal.done` 分支的 NonStandardContentBlock 改为正式块：
   ```go
   case "response.refusal.done":
   	chunk := messages.AI("")
   	chunk.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: event.Refusal}}
   	...
   ```
4. `chat_completions_stream.go`：`chatCompletionsStreamDelta` 加 `Refusal string \`json:"refusal"\``。**该文件的流聚合没有 `s.output` 字段**（不同于 anthropic stream）——读该文件找到流的聚合/终态位置（`[DONE]` 路径 / `finalToolCallChunk` ~L250 邻域），把累积的非空 refusal 挂到终态消息为 `RefusalBlock`（对齐上游"仅终态/beta 路径可见"的形状，不逐 delta 发块）。
5. `InvokeStructured`（chatmodel.go:347 起）：`return m.WithStructuredOutput(name, sch, true).Invoke(ctx, input)` 改为带检查：
   ```go
   msg, err := m.WithStructuredOutput(name, sch, true).Invoke(ctx, input)
   if err != nil {
   	return messages.Message{}, err
   }
   if err := outputparser.RaiseIfRefused(msg); err != nil {
   	return messages.Message{}, err
   }
   return msg, nil
   ```
   import `core/outputparser`。

- [ ] **Step 4: 运行确认通过**

Run: `go test -race ./partners/openai/ && go build ./... && go vet ./partners/openai/`
Expected: PASS（含既有全部测试无回归）。

- [ ] **Step 5: Commit**

```bash
git add partners/openai/
git commit -m "feat(openai): standardized refusal blocks both directions + structured-output refusal error (#41582)"
```

---

### Task 5: anthropic partner — refusal 块 + #41715 + 流式 chunk 门控修复

**Files:**
- Modify: `partners/anthropic/chatmodel.go`（`messagePayload` ~L1122 加 `StopDetails`、`toMessage` L1205、`InvokeStructured` L470-480 的 no-tool_call 报错点）
- Modify: `partners/anthropic/stream.go`（`message_delta` 分支 L217-248；stream delta 结构加 StopDetails）
- Modify: `partners/anthropic/contentblocks.go`（`formatContentBlock` refusal 跳过）
- Test: `partners/anthropic/refusal_test.go`（新建）

**Interfaces:**
- Consumes: Task 1 `messages.RefusalBlock`、Task 2 `lcerrors.ModelRefusalError`/`outputparser.RaiseIfRefused`。
- Produces: 无新导出。

- [ ] **Step 1: 写失败测试**

`partners/anthropic/refusal_test.go`：

```go
package anthropic

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
)

// Ports partners/anthropic/tests/unit_tests/test_refusal.py（144 行，
// anthropic 1.7.10 + 未发布 #41715）。

// --- stop_reason=refusal → RefusalBlock，category/explanation 进 extras/文本 ---

func TestRefusalStopReasonProducesBlock(t *testing.T) {
	raw := `{
		"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
		"content": [{"type": "text", "text": ""}],
		"stop_reason": "refusal",
		"stop_details": {"category": "safety", "explanation": "Cannot help with weapons."},
		"usage": {"input_tokens": 3, "output_tokens": 1}
	}`
	var payload messagePayload
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		t.Fatal(err)
	}
	msg := payload.toMessage()
	var rb *messages.RefusalBlock
	for _, b := range msg.ContentBlocks {
		if r, ok := b.(messages.RefusalBlock); ok {
			cpy := r
			rb = &cpy
		}
	}
	if rb == nil {
		t.Fatalf("no RefusalBlock in %v", msg.ContentBlocks)
	}
	if rb.Refusal != "Cannot help with weapons." {
		t.Fatalf("refusal text = %q", rb.Refusal)
	}
	if rb.Extras["category"] != "safety" {
		t.Fatalf("extras = %v", rb.Extras)
	}
	// 不设固定 index（#41658 后上游 translator 产出端无 index）。
	if rb.Index != nil {
		t.Fatalf("index must be nil, got %v", rb.Index)
	}
	if msg.ResponseMetadata["stop_reason"] != "refusal" {
		t.Fatalf("metadata = %v", msg.ResponseMetadata)
	}
	if msg.ResponseMetadata["stop_details"] == nil {
		t.Fatal("stop_details must be captured in metadata")
	}
}

// --- 输入方向：refusal 块不发往 API ---

func TestRefusalBlockDroppedOnInput(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.TextBlock{Text: "ok"},
		messages.RefusalBlock{Refusal: "no"},
	}
	blocks, err := formatContentBlocks(msg.ContentBlocks)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range blocks {
		if b.Type == "refusal" {
			t.Fatalf("refusal must not be sent to the Messages API: %+v", blocks)
		}
	}
	if len(blocks) != 1 || blocks[0].Type != "text" {
		t.Fatalf("want only the text block, got %+v", blocks)
	}
}

// --- #41715：结构化输出 no-tool_call + refusal → ModelRefusalError ---

func TestInvokeStructuredRefusalNotParseError(t *testing.T) {
	// 构造 stop_reason=refusal 且无 tool_use 的响应，走 InvokeStructured 的
	// 报错分支。无 HTTP fake 时直接测试判定函数（见 Step 3 提取的
	// refusalOrNoToolCallError）。
	payload := messagePayload{StopReason: "refusal"}
	msg := payload.toMessage()
	err := noToolCallError(msg)
	var mre *lcerrors.ModelRefusalError
	if !errors.As(err, &mre) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}

	payload2 := messagePayload{StopReason: "end_turn"}
	err2 := noToolCallError(payload2.toMessage())
	if err2 == nil || errors.As(err2, &mre) {
		t.Fatalf("non-refusal no-tool_call must stay a plain error, got %v", err2)
	}
}

// --- 流式：message_delta 的 stop chunk 不受 streamUsage 门控 ---

// 照 partners/anthropic/stream_usage_test.go 的现有 fixture 模式（内存 SSE
// 序列驱动 messageStream），以 streamUsage=false 构造 message_start +
// message_delta(stop_reason="end_turn") 序列，断言：消费者仍能收到携带
// stop_reason 的终止 chunk，且该 chunk 的 UsageMetadata 为零值（usage 仍受门控）。
// 再补一例 streamUsage=true 的对照（chunk 带 usage）。
func TestStreamStopChunkEmittedWithoutStreamUsage(t *testing.T) {
	// 实现者按 stream_usage_test.go 的 SSE fixture 构造：
	//   events: message_start(usage) → message_delta(stop_reason="end_turn", usage) → message_stop
	//   WithStreamUsage(false)
	// 断言：Next 收到的终止 chunk 的 ResponseMetadata["stop_reason"] == "end_turn"，
	//       且 chunk.UsageMetadata == messages.UsageMetadata{}。
}
```

- [ ] **Step 2: 运行确认失败**

Run: `go test ./partners/anthropic/ -run Refusal -v`
Expected: FAIL —— 编译失败（`StopDetails`/`noToolCallError` 不存在）。

- [ ] **Step 3: 实现**

1. `chatmodel.go`：
   - 新增 wire 类型并在 `messagePayload` 挂字段：
     ```go
     // stopDetailsPayload carries Anthropic's refusal details on a
     // stop_reason="refusal" response (both non-streaming message payloads
     // and streaming message_delta deltas).
     type stopDetailsPayload struct {
     	Category    string `json:"category"`
     	Explanation string `json:"explanation"`
     }
     ```
     `messagePayload` 加 `StopDetails *stopDetailsPayload \`json:"stop_details"\``。
   - `toMessage`：构造 `message.ResponseMetadata` 后、`return` 前加：
     ```go
     if r.StopReason == "refusal" {
     	explanation := ""
     	if r.StopDetails != nil {
     		explanation = r.StopDetails.Explanation
     		if r.StopDetails.Category != "" {
     			blocks = append(blocks, messages.RefusalBlock{
     				Refusal: explanation,
     				Extras:  map[string]any{"category": r.StopDetails.Category},
     			})
     		}
     	}
     	if r.StopDetails == nil || r.StopDetails.Category == "" {
     		blocks = append(blocks, messages.RefusalBlock{Refusal: explanation})
     	}
     	if r.StopDetails != nil {
     		message.ResponseMetadata["stop_details"] = map[string]any{
     			"category":    r.StopDetails.Category,
     			"explanation": r.StopDetails.Explanation,
     		}
     	}
     }
     ```
     （注意 `blocks` 是 toMessage 内局部变量；把 refusal 追加放在 `message.ContentBlocks = blocks` 赋值之前，或改为追加到 `message.ContentBlocks`。实现时按实际变量结构落位。）
   - 提取 `noToolCallError`：`InvokeStructured` 中 `len(response.ToolCalls) == 0` 分支（chatmodel.go:476-480）提取为下述函数。语义说明：Go 的 `InvokeStructured` 恒强制 `tool_choice` 到合成工具（chatmodel.go:470），而上游 #41715 的分支针对"未强制 tool 调用"路径；Go 无未强制路径，因此这里的行为是"no tool_call + refusal → ModelRefusalError"，即 #41715 的唯一可触发形态，实现注释里写明这一点。
     ```go
     // noToolCallError builds the structured-output failure for a response
     // without tool calls. A refusal surfaces as ModelRefusalError instead of
     // the generic "no tool_call" parse error (anthropic #41715: refusal
     // semantics win over schema enforcement). Go's InvokeStructured always
     // forces tool_choice to the synthesized tool, so #41715's "not forced"
     // gate has no Go counterpart — this branch is its only reachable form.
     func noToolCallError(response messages.Message) error {
     	if err := refusalErrorOf(response); err != nil {
     		return err
     	}
     	return fmt.Errorf(
     		"anthropic structured output: model returned no tool_call (stop_reason=%v)",
     		response.ResponseMetadata["stop_reason"])
     }

     // refusalErrorOf returns a *lcerrors.ModelRefusalError when the message
     // carries a refusal block or stop_reason="refusal", else nil.
     func refusalErrorOf(msg messages.Message) error {
     	if err := outputparser.RaiseIfRefused(msg); err != nil {
     		return err
     	}
     	if msg.ResponseMetadata["stop_reason"] == "refusal" {
     		return lcerrors.ModelRefusalFromBlock(messages.RefusalBlock{}, msg)
     	}
     	return nil
     }
     ```
     `InvokeStructured` 的报错点改调 `return messages.Message{}, noToolCallError(response)`。
2. `stream.go`：
   - stream delta 结构（`StopReason string \`json:"stop_reason"\`` 所在 struct，L296 邻域）加 `StopDetails *stopDetailsPayload \`json:"stop_details"\``。
   - `message_delta` 分支：stop_reason 落 metadata 的现有逻辑后追加 stop_details 捕获；refusal 块追加到 `s.output.ContentBlocks`：
     ```go
     if event.Delta.StopReason == "refusal" {
     	explanation, category := "", ""
     	if event.Delta.StopDetails != nil {
     		explanation = event.Delta.StopDetails.Explanation
     		category = event.Delta.StopDetails.Category
     		s.output.ResponseMetadata["stop_details"] = map[string]any{
     			"category": category, "explanation": explanation,
     		}
     	}
     	rb := messages.RefusalBlock{Refusal: explanation}
     	if category != "" {
     		rb.Extras = map[string]any{"category": category}
     	}
     	s.output.ContentBlocks = append(s.output.ContentBlocks, rb)
     }
     ```
   - **chunk 门控拆解**（spec D4）：当前 `if s.streamUsage && !s.usage.isZero()` 才发终止 chunk（stream.go:237）。改为：
     ```go
     hasStopInfo := event.Delta.StopReason != "" || event.Delta.StopSequence != ""
     if hasStopInfo || (s.streamUsage && !s.usage.isZero()) {
     	chunk := messages.AI("")
     	if s.streamUsage && !s.usage.isZero() {
     		chunk.UsageMetadata = s.output.UsageMetadata
     	}
     	chunk.ResponseMetadata = map[string]any{}
     	if event.Delta.StopReason != "" {
     		chunk.ResponseMetadata["stop_reason"] = event.Delta.StopReason
     	}
     	if event.Delta.StopSequence != "" {
     		chunk.ResponseMetadata["stop_sequence"] = event.Delta.StopSequence
     	}
     	if err := emitStream(ctx, s.cfg, chunk); err != nil {
     		return messages.Message{}, false, err
     	}
     	return chunk, true, nil
     }
     return messages.Message{}, false, nil
     ```
3. `contentblocks.go`：`formatContentBlock` 的 switch 加：
   ```go
   case "refusal":
   	// The Messages API has no refusal input block (#41582, _compat.py:125-127).
   	return contentBlock{}, errSkipBlock
   ```
   并定义包级 `var errSkipBlock = errors.New("anthropic: block not sendable")`，在 `formatContentBlocks` 循环里对该错误 `continue` 而不是 return。**实现前先检查 formatContentBlocks 的错误处理形状**——若调用方依赖逐块错误，改用 `(contentBlock, bool, error)` 三返回或静默 continue 的内部判定，保持其它错误行为不变。

- [ ] **Step 4: 运行确认通过**

Run: `go test -race ./partners/anthropic/ && go build ./... && go vet ./partners/anthropic/`
Expected: PASS（注意既有 stream 测试可能因 chunk 门控拆解出现期望差异——若有，核对新行为与上游一致后更新既有测试并在 commit message 说明）。

- [ ] **Step 5: Commit**

```bash
git add partners/anthropic/
git commit -m "feat(anthropic): refusal stop_reason becomes RefusalBlock; stop chunk emission ungated; ModelRefusalError in structured output (#41582, #41715)"
```

---

### Task 6: gemini partner — safety 阻断转 refusal 块

**Files:**
- Modify: `partners/gemini/content.go`（`responseToMessage` L234 起、无候选报错 L237-240、metadata 构造 L302-307）
- Modify: `partners/gemini/stream.go`（`handleResponse` 的 finish 分支 L202-217）
- Test: `partners/gemini/refusal_test.go`（新建）

**Interfaces:**
- Consumes: Task 1 `messages.RefusalBlock`。
- Produces: 无新导出（包内 helper `refusalFromMetadata`）。

- [ ] **Step 1: 写失败测试**

`partners/gemini/refusal_test.go`：

```go
package gemini

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Ports core/tests/unit_tests/messages/test_refusal.py 的 gemini 段
//（test_refusal.py:210-284，#41658）。

func TestRefusalFromSafetyFinishReason(t *testing.T) {
	meta := map[string]any{
		"finish_reason":  "SAFETY",
		"finish_message": "Blocked for safety.",
	}
	rb := refusalFromMetadata(meta)
	if rb == nil {
		t.Fatal("SAFETY finish reason must produce a refusal block")
	}
	if rb.Refusal != "Blocked for safety." {
		t.Fatalf("refusal text = %q", rb.Refusal)
	}
	if rb.Extras["finish_reason"] != "SAFETY" {
		t.Fatalf("extras = %v", rb.Extras)
	}
}

func TestRefusalFromPromptFeedback(t *testing.T) {
	meta := map[string]any{
		"prompt_feedback": map[string]any{"block_reason": "PROHIBITED_CONTENT"},
	}
	rb := refusalFromMetadata(meta)
	if rb == nil {
		t.Fatal("prompt block reason must produce a refusal block")
	}
	if rb.Refusal != "" {
		t.Fatalf("refusal text must be empty without finish_message, got %q", rb.Refusal)
	}
	if rb.Extras["block_reason"] != "PROHIBITED_CONTENT" {
		t.Fatalf("extras = %v", rb.Extras)
	}
}

func TestNoRefusalOnCleanFinish(t *testing.T) {
	for _, meta := range []map[string]any{
		{"finish_reason": "STOP"},
		{"finish_reason": "MAX_TOKENS"},
		{"prompt_feedback": map[string]any{"block_reason": "BLOCKED_REASON_UNSPECIFIED"}},
		{},
	} {
		if rb := refusalFromMetadata(meta); rb != nil {
			t.Fatalf("meta %v must not produce a refusal block", meta)
		}
	}
}

func TestRefusalReasonEnumTail(t *testing.T) {
	// _reason_name 对齐：枚举串取尾段（"FinishReason.SAFETY" → "SAFETY"）。
	meta := map[string]any{"finish_reason": "FinishReason.SAFETY"}
	rb := refusalFromMetadata(meta)
	if rb == nil || rb.Extras["finish_reason"] != "SAFETY" {
		t.Fatalf("enum-qualified reason must be normalized: %+v", rb)
	}
}
```

- [ ] **Step 2: 运行确认失败**

Run: `go test ./partners/gemini/ -run Refusal -v`
Expected: FAIL —— undefined: refusalFromMetadata。

- [ ] **Step 3: 实现**

1. `partners/gemini/content.go` 新增 helper：

```go
// refusalFinishReasons are the candidate finish reasons meaning Gemini
// blocked the response (Python google_genai.py #41658:
// _REFUSAL_FINISH_REASONS).
var refusalFinishReasons = map[string]bool{
	"SAFETY":             true,
	"PROHIBITED_CONTENT": true,
	"BLOCKLIST":          true,
}

// reasonName normalizes a Gemini finish/block reason to its enum member name
// tail (Python's _reason_name: str(getattr(reason, "name", reason)).rsplit(".", 1)[-1]).
func reasonName(reason any) string {
	s, _ := reason.(string)
	if s == "" {
		return ""
	}
	if i := strings.LastIndex(s, "."); i >= 0 {
		s = s[i+1:]
	}
	return s
}

// refusalFromMetadata builds a refusal content block from Gemini's finish or
// prompt-block reason metadata, or returns nil when neither signals a block
// (Python's _refusal_from_metadata, #41658).
func refusalFromMetadata(meta map[string]any) *messages.RefusalBlock {
	extras := map[string]any{}
	if fr := reasonName(meta["finish_reason"]); refusalFinishReasons[fr] {
		extras["finish_reason"] = fr
	}
	if feedback, ok := meta["prompt_feedback"].(map[string]any); ok {
		if br := reasonName(feedback["block_reason"]); br != "" && br != "BLOCKED_REASON_UNSPECIFIED" {
			extras["block_reason"] = br
		}
	}
	if len(extras) == 0 {
		return nil
	}
	refusal, _ := meta["finish_message"].(string)
	return &messages.RefusalBlock{Refusal: refusal, Extras: extras}
}
```

2. `responseToMessage`：`message.ResponseMetadata` 构造后（含 finish_message 捕获之后）加：

```go
	if rb := refusalFromMetadata(message.ResponseMetadata); rb != nil {
		blocks = append(blocks, *rb)
		message.ContentBlocks = blocks
	}
```

注意 PromptFeedback 目前只在报错路径读取（L237-238）；在 ResponseMetadata 中补记 `prompt_feedback`（含 block_reason）以便 helper 统一工作：

```go
	if response.PromptFeedback != nil && response.PromptFeedback.BlockReason != "" {
		message.ResponseMetadata["prompt_feedback"] = map[string]any{
			"block_reason": string(response.PromptFeedback.BlockReason),
		}
	}
```

把 refusal 检查放在这两段之后。无候选报错路径（L237-240）保持不变。

3. `stream.go` 的 `handleResponse`：`finish != ""` 分支（现有 metadata 记录后）加：

```go
		if rb := refusalFromMetadata(s.output.ResponseMetadata); rb != nil {
			s.output.ContentBlocks = append(s.output.ContentBlocks, *rb)
			chunk.ContentBlocks = append(chunk.ContentBlocks, *rb)
		}
```

并确认 `s.output.ResponseMetadata["finish_reason"]` 已先于该检查写入（现有代码 L205-208 已做）。PromptFeedback 的流式 metadata 若非现有字段，同 2 的方式补记。

- [ ] **Step 4: 运行确认通过**

Run: `go test -race ./partners/gemini/ && go build ./... && go vet ./partners/gemini/`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add partners/gemini/
git commit -m "feat(gemini): translate safety blocks to refusal content blocks (#41658)"
```

---

### Task 7: langchain — `ModelRefusalMiddleware` + trace 脱敏

**Files:**
- Create: `langchain/agents/middleware/model_refusal.go`
- Modify: `langchain/agents/structured_output.go`（加 getter，解开 import cycle——middleware 不能 import agents：create_agent.go:108 已是 agents→middleware 方向，types.go:175-179 注释明确记载此禁令）
- Test: `langchain/agents/middleware/model_refusal_test.go`（新建，**`package middleware_test`**——trace_policy_test.go:7-10 的先例：in-package 测试 import agents 同样构成 cycle）

**Interfaces:**
- Consumes: Task 2 的 `outputparser.RaiseIfRefused` / `lcerrors.ModelRefusalError`；`ModelHandler`（model_retry.go:11）、`ModelRequest`/`ModelResponse`（types.go:14/27）、`TracePolicyProvider`（types.go:226）、`agents.StructuredOutputValidationError`（structured_output.go:41-52，携带 `AIMessage messages.Message` 字段）——经 getter 间接消费，不 import agents 包。
- Produces:
  ```go
  // langchain/agents（本任务新增的 getter）
  func (e *StructuredOutputValidationError) RefusalMessage() messages.Message

  // langchain/agents/middleware
  type ModelRefusalMiddleware struct{}
  func NewModelRefusalMiddleware() *ModelRefusalMiddleware
  func (m *ModelRefusalMiddleware) WrapModelCall(ctx context.Context, request ModelRequest, handler ModelHandler) (ModelResponse, error)
  func (m *ModelRefusalMiddleware) TracePolicy() TracePolicyConfig
  ```

- [ ] **Step 1: 写失败测试**

`langchain/agents/middleware/model_refusal_test.go`（**`package middleware_test`**，import 同包用 `middleware.` 前缀）：

```go
package middleware_test

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents"
	middleware "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// Ports langchain tests/unit_tests/agents/middleware/implementations/
// test_model_refusal.py（164 行，#41582 + #41689）。

func refusedHandler(_ context.Context, _ middleware.ModelRequest) (middleware.ModelResponse, error) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: "Cannot answer."}}
	return middleware.ModelResponse{Result: []messages.Message{msg}}, nil
}

func cleanHandler(_ context.Context, _ middleware.ModelRequest) (middleware.ModelResponse, error) {
	return middleware.ModelResponse{Result: []messages.Message{messages.AI("fine")}}, nil
}

func TestModelRefusalMiddlewareRaises(t *testing.T) {
	mw := middleware.NewModelRefusalMiddleware()
	_, err := mw.WrapModelCall(context.Background(), middleware.ModelRequest{}, refusedHandler)
	var mre *lcerrors.ModelRefusalError
	if !errors.As(err, &mre) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}
}

func TestModelRefusalMiddlewarePassesClean(t *testing.T) {
	mw := middleware.NewModelRefusalMiddleware()
	resp, err := mw.WrapModelCall(context.Background(), middleware.ModelRequest{}, cleanHandler)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Result) != 1 {
		t.Fatalf("response lost: %+v", resp)
	}
}

// refusal 藏在 StructuredOutputValidationError 的 AI 消息里也要抛出（上游
// wrap_model_call 先查 exc.ai_message 再 re-raise 的分支）。
func TestModelRefusalMiddlewareChecksValidationErrorMessage(t *testing.T) {
	refused := messages.AI("")
	refused.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: "no"}}
	handler := func(_ context.Context, _ middleware.ModelRequest) (middleware.ModelResponse, error) {
		return middleware.ModelResponse{}, agents.NewStructuredOutputValidationError("out", errors.New("parse fail"), refused)
	}
	mw := middleware.NewModelRefusalMiddleware()
	_, err := mw.WrapModelCall(context.Background(), middleware.ModelRequest{}, handler)
	var mre *lcerrors.ModelRefusalError
	if !errors.As(err, &mre) {
		t.Fatalf("want ModelRefusalError from validation error's message, got %v", err)
	}
}

// 校验错误里的消息不含 refusal 时，原错误原样传播。
func TestModelRefusalMiddlewareReraisesUnrelatedValidationError(t *testing.T) {
	clean := messages.AI("not a refusal")
	handler := func(_ context.Context, _ middleware.ModelRequest) (middleware.ModelResponse, error) {
		return middleware.ModelResponse{}, agents.NewStructuredOutputValidationError("out", errors.New("parse fail"), clean)
	}
	mw := middleware.NewModelRefusalMiddleware()
	_, err := mw.WrapModelCall(context.Background(), middleware.ModelRequest{}, handler)
	var ve *agents.StructuredOutputValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("unrelated validation error must propagate, got %v", err)
	}
}

// #41689：trace policy omit inputs。
func TestModelRefusalMiddlewareTracePolicyOmitsInputs(t *testing.T) {
	mw := middleware.NewModelRefusalMiddleware()
	policy := mw.TracePolicy()
	if policy.ProcessInputs == nil {
		t.Fatal("ProcessInputs must be installed (#41689)")
	}
	scrubbed := policy.ProcessInputs(map[string]any{"messages": []string{"secret"}})
	m, ok := scrubbed.(map[string]any)
	if !ok || len(m) == 0 {
		t.Fatalf("scrubbed inputs must be an omitted marker, got %v", scrubbed)
	}
	if m["messages"] != nil {
		t.Fatalf("inputs must be omitted, got %v", m)
	}
}

// #41689 端到端：fake tracer 断言 wrap_model_call 层内 inputs 被 omit。
// 照本目录 trace_policy_test.go 的 invokeTracePolicyAgent fixture 模式
// 构一个装 ModelRefusalMiddleware 的 agent + fake tracer handler，
// 断言 tracer 收到的 model-start 输入为 omitted marker 而非原始消息。
// 若 fixture 构造超出本任务粒度，可将该用例挪进 trace_policy_test.go 同模式扩展。
```

注意：`agents.NewStructuredOutputValidationError` 的签名以 structured_output.go 实际为准（toolName, source, aiMessage）。

- [ ] **Step 2: 运行确认失败**

Run: `go test ./langchain/agents/middleware/ -run TestModelRefusal -v`
Expected: FAIL —— undefined: NewModelRefusalMiddleware。

- [ ] **Step 3: 实现**

1. `langchain/agents/structured_output.go` 给 `StructuredOutputValidationError` 加 getter（middleware 包不能 import agents，经接口间接取消息）：

```go
// RefusalMessage returns the AI message the validation error was built from.
// It exists for the middleware package (which agents imports, so middleware
// cannot import agents back): ModelRefusalMiddleware matches it through a
// local interface with errors.As.
func (e *StructuredOutputValidationError) RefusalMessage() messages.Message {
	return e.AIMessage
}
```

2. 新建 `langchain/agents/middleware/model_refusal.go`：

```go
package middleware

import (
	"context"
	"errors"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/outputparser"
)

// aiMessageCarrier matches agents.StructuredOutputValidationError (and any
// similar error) without importing the agents package — agents imports THIS
// package, so a direct import would cycle (see types.go's TracePolicy note).
type aiMessageCarrier interface {
	error
	RefusalMessage() messages.Message
}

// ModelRefusalMiddleware raises *lcerrors.ModelRefusalError for model
// responses containing a standard refusal content block, mirroring Python's
// ModelRefusalMiddleware (langchain 1.4.9, #41582). Add it to CreateAgent to
// treat a refusal as an error instead of a successful final answer; model
// calls without it keep returning refusal-carrying messages. List it after
// ModelFallbackMiddleware so the fallback can catch the error. Streaming
// content may already have been emitted before the refusal is known; the
// middleware neither infers refusals from text nor retries blocked requests.
type ModelRefusalMiddleware struct{}

// NewModelRefusalMiddleware creates the middleware.
func NewModelRefusalMiddleware() *ModelRefusalMiddleware { return &ModelRefusalMiddleware{} }

// WrapModelCall checks the completed model response for a refusal block.
func (m *ModelRefusalMiddleware) WrapModelCall(ctx context.Context, request ModelRequest, handler ModelHandler) (ModelResponse, error) {
	response, err := handler(ctx, request)
	if err != nil {
		// A structured-output validation failure may still carry the
		// refusing AI message (Python: wrap_model_call inspects
		// exc.ai_message before re-raising).
		var carrier aiMessageCarrier
		if errors.As(err, &carrier) {
			if refusalErr := outputparser.RaiseIfRefused(carrier.RefusalMessage()); refusalErr != nil {
				return ModelResponse{}, refusalErr
			}
		}
		return ModelResponse{}, err
	}
	for _, msg := range response.Result {
		if refusalErr := outputparser.RaiseIfRefused(msg); refusalErr != nil {
			return ModelResponse{}, refusalErr
		}
	}
	return response, nil
}

// TracePolicy implements TracePolicyProvider with omit-payload inputs
// (langchain #41689: trace_policy = TracePolicy(process_inputs=omit_payload)),
// matching the ModelRouter precedent (typesafe model_router.go:169). Python's
// omit_payload returns an empty dict; Go uses a non-empty marker so the
// scrubbed payload stays visibly intentional (see DIVERGENCES.md).
func (m *ModelRefusalMiddleware) TracePolicy() TracePolicyConfig {
	return TracePolicyConfig{
		ProcessInputs: func(any) any { return map[string]any{"omitted": true} },
	}
}
```

若编译报接口缺失（middleware 可能还需 `Name()`/`StateSchema()` 等），按 `ModelFallbackMiddleware` 同文件的接口集补齐。

- [ ] **Step 4: 运行确认通过**

Run: `go test -race ./langchain/agents/... && go build ./... && go vet ./langchain/...`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add langchain/agents/middleware/model_refusal.go langchain/agents/middleware/model_refusal_test.go
git commit -m "feat(langchain): ModelRefusalMiddleware with omit-inputs trace policy (#41582, #41689)"
```

---

### Task 8: langgraph — strict resume（#9574）+ 多中断报错 URL（#9590）

**Files:**
- Modify: `langgraph/types/types.go`（新增 `UnknownInterruptError`）
- Modify: `langgraph/graph/graph.go`（`Options` 新增 `StrictResume bool`；run 入口处把 strict 传入 resume 路径，L1342 邻域）
- Modify: `langgraph/graph/resume.go`（`resumeFromTuple` L232、`planResume` L333 签名与检查逻辑、多中断报错文案 L358-361）
- Test: `langgraph/graph/strict_resume_test.go`（新建）

**Interfaces:**
- Consumes: 无（独立于 T1-T7，可与 T6/T7 并行）。
- Produces:
  ```go
  // langgraph/types
  type UnknownInterruptError struct {
      Reason        string   // "not_pending" | "empty_map" | "ambiguous"
      InterruptIDs  []string // not_pending：未知/已答的 id
      PendingIDs    []string // 当前 pending 的 id，排序
  }
  func (e *UnknownInterruptError) Error() string

  // langgraph/graph
  // Options.StrictResume bool（opt-in，默认 false 行为完全不变）
  ```

- [ ] **Step 1: 写失败测试**

`langgraph/graph/strict_resume_test.go`。骨架说明（三处易错点已按本包真实 API 核实）：`NewStateGraph` 是**无参**构造（graph.go:180，签名以实际为准）；`runtime.Runtime` 自身实现 `context.Context`（runtime.go:279），调 `Interrupt(rt, "q1")` 而非 `Interrupt(rt.Context(), ...)`；终止边用 `types.END`（graph.go:372 用法）。先读本包一个现有 interrupt/resume 测试再定稿构造。必须覆盖的用例语义：

1. `TestStrictResumeUnknownID`：暂停后以 `Options{Resume: map[string]any{"bogus-id": 42}, StrictResume: true}` 恢复 → 返回 `*types.UnknownInterruptError`，`Reason == "not_pending"`，`InterruptIDs == ["bogus-id"]`，`PendingIDs` 为实际 pending id 排序；且 checkpoint 无新增写入（恢复前后 thread 的 checkpoint 数相同）。
2. `TestStrictResumeAnsweredID`：用真实 id 恢复一次成功；再用同一 id 恢复（已答）→ `not_pending`。
3. `TestStrictResumeEmptyMap`：`Resume: map[string]any{}` + StrictResume → `empty_map`，无写入。
4. `TestStrictResumeAmbiguousScalar`：两个 interrupt pending 时 `Resume: 42`（标量）+ StrictResume → `ambiguous`；非 strict 时同输入维持现有普通报错（非 UnknownInterruptError）。
5. `TestNonStrictResumeUnknownIDIgnored`：非 strict + 未知 id → 行为不变（静默忽略，interrupt 重触发）。
6. `TestStaticBreakpointScalarResumeAllowed`：interrupt_before 暂停的图，无 pending interrupt，strict 下标量 resume 合法（对齐上游"静态断点单值仍接受"）。
7. `TestStrictResumeNotPropagatedToSubgraph`：父子图，子图内 interrupt，父 run strict 恢复时子图运行不因 strict 误拒合法 resume（AddSubgraph 包装器不传播 StrictResume）。
8. `TestStrictResumePinnedHeadCheckpointAllowed`：strict + pinned 到 thread 最新 checkpoint（head）恢复，合法 resume 正常通过（pinned 跳过检查）。
9. **saver 覆盖**（spec D7 验收）：以上用例 1/3/4 用内存 saver + sqlite saver 各跑一遍（表驱动 `[]struct{name string; saver checkpoint.Saver}`；sqlite 用 `:memory:` 或临时文件，照本包现有 sqlite 测试 fixture）。

- [ ] **Step 2: 运行确认失败**

Run: `go test ./langgraph/graph/ -run StrictResume -v`
Expected: FAIL —— `Options` 无 `StrictResume` 字段 / `types.UnknownInterruptError` 未定义。

- [ ] **Step 3: 实现**

1. `langgraph/types/types.go`（GraphInterrupt 定义附近）新增：

```go
// UnknownInterruptError is raised by strict resume mode (langgraph 1.2.20,
// #9574, CONFIG_KEY_STRICT_RESUME="__pregel_strict_resume"): a resume map
// names an interrupt that is not pending, is empty, or a scalar resume meets
// multiple pending interrupts. Without strict mode these are silently
// ignored (or a plain error for the ambiguous case), which hides double- or
// mis-answered interrupts from clients.
type UnknownInterruptError struct {
	// Reason is "not_pending", "empty_map", or "ambiguous".
	Reason string
	// InterruptIDs carries the offending resume ids (Reason "not_pending").
	InterruptIDs []string
	// PendingIDs carries the pending interrupt ids at check time, sorted.
	PendingIDs []string
}

// Error implements the error interface.
func (e *UnknownInterruptError) Error() string {
	switch e.Reason {
	case "empty_map":
		return "The resume map names no interrupt"
	case "not_pending":
		if len(e.InterruptIDs) > 0 {
			return "Unknown or already-consumed interrupt: " + e.InterruptIDs[0]
		}
		return "Unknown or already-consumed interrupt"
	case "ambiguous":
		return "When there are multiple pending interrupts, you must specify the interrupt id when resuming. " +
			"Docs: https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts."
	default:
		return "unknown interrupt error: " + e.Reason
	}
}
```

2. `langgraph/graph/graph.go` 的 `Options` 加字段（放在 `Resume`/`Graph` 字段之后）：

```go
	// StrictResume opts into langgraph 1.2.20's strict resume mode (#9574,
	// CONFIG_KEY_STRICT_RESUME): a resume naming an interrupt that is not
	// pending (or an empty resume map, or a scalar with several interrupts
	// pending) fails with *types.UnknownInterruptError BEFORE anything is
	// written, instead of being silently ignored. Root runs only — the
	// AddSubgraph wrapper builds child Options field-by-field and does not
	// propagate this flag (Python: skipped for nested graphs). Also skipped
	// when resuming a checkpoint the thread has moved past (a finished task
	// can rerun and interrupt again there). Default false keeps today's
	// behavior.
	StrictResume bool
```

3. `resume.go` 改签名与检查。`resumeFromTuple` 增加两个参数 `strict bool` 与 `pinned bool`（pinned = 本 run 是否从 `Options.CheckpointID` 钉住的 checkpoint 恢复）并传给 `planResume`；调用点 graph.go:1342 传 `opts.StrictResume, opts.CheckpointID != ""`，L1347 的 nil-resume 路径传 `false, false`。`planResume` 相应加参，在 `pending` 计算与 graphNS 过滤之后、任务规划之前插入：

```go
	// Strict resume（#9574）：仅 root 非 pinned 恢复启用。pinned 恢复是
	// Go 中 superseded 的唯一入口（对齐 checkpoint_sink.go:684 的 exit 路径
	// flushOpts.CheckpointID != "" 门），此时已完成的 task 可能重跑并再次
	// interrupt，上游同样跳过检查（_loop.py:1029-1036 的
	// _addressed_checkpoint_superseded）。
	if strict && !pinned && resume != nil {
		pendingKeys := map[string]bool{}
		var pendingIDs []string
		for _, p := range pending {
			if p.ID != "" {
				pendingKeys[p.ID] = true
				pendingIDs = append(pendingIDs, p.ID)
			}
			// Go 支持 NS 寻址（resumeValuesFor NS 优先），strict 的合法键
			// 集合 = ID ∪ NS。
			if p.NS != "" {
				pendingKeys[p.NS] = true
			}
		}
		sort.Strings(pendingIDs)
		switch r := resume.(type) {
		case map[string]any:
			// Go 现状把任何 map[string]any 当 resume map（不同于 Python 的
			// xxh3 键检查），strict 沿用同一判定。
			if len(r) == 0 {
				return resumePlan{}, &types.UnknownInterruptError{
					Reason: "empty_map", PendingIDs: pendingIDs,
				}
			}
			var unknown []string
			for k := range r {
				if !pendingKeys[k] {
					unknown = append(unknown, k)
				}
			}
			if len(unknown) > 0 {
				sort.Strings(unknown)
				return resumePlan{}, &types.UnknownInterruptError{
					Reason: "not_pending", InterruptIDs: unknown, PendingIDs: pendingIDs,
				}
			}
		default:
			if len(pending) > 1 {
				return resumePlan{}, &types.UnknownInterruptError{
					Reason: "ambiguous", PendingIDs: pendingIDs,
				}
			}
			// pending == 0 时单值合法（静态断点恢复，上游同）。
		}
	}
```

4. **#9590**：planResume 现有的多中断报错文案（resume.go:358-361 邻域）末尾追加：

```go
" Docs: https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts."
```

5. DIVERGENCES 追加（本任务顺带）：「strict resume 的 map 判定：Python 要求 map 键全是 xxh3 interrupt id 才算 resume map；Go 无 xxh3 形态的 id（interrupt id 为 `node-N`），任何 map[string]any 都按 resume map 处理（与 Go 非 strict 路径一致）。」

- [ ] **Step 4: 运行确认通过**

Run: `go test -race ./langgraph/... && go build ./... && go vet ./langgraph/...`
Expected: PASS。若触及嵌套 saver 模块：`make test-sqlite`（或对应模块的 go test）。

- [ ] **Step 5: Commit**

```bash
git add langgraph/types/types.go langgraph/graph/ 
git commit -m "feat(langgraph): opt-in strict resume mode refusing stale interrupt answers (#9574, #9590)"
```

---

### Task 9: anthropic — passthrough 补 `caller`/`content`

**Files:**
- Modify: `partners/anthropic/chatmodel.go`（`contentBlock` struct ~L1085）
- Modify: `partners/anthropic/contentblocks.go`（`setContentBlockField` L240-289、`passthroughBlock` L232）
- Test: `partners/anthropic/passthrough_caller_test.go`（新建）

**Interfaces:**
- Consumes: 无（独立于 T1-T8）。
- Produces: 无新导出。

- [ ] **Step 1: 写失败测试**

`partners/anthropic/passthrough_caller_test.go`：

```go
package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Ports anthropic #41678 的 caller 用例与 #41637 的 compaction 用例
//（tests/unit_tests/test_chat_models.py 相应段）。

// compaction 块 replay 保留 content、不带 index（#41637；Go 构造无 index 字段，
// 本用例钉死 content 不丢）。
func TestCompactionBlockKeepsContentDropsIndex(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "compaction", Value: map[string]any{
			"content": "Summary of earlier turns.",
			"index":   0,
		}},
	}
	blocks, err := formatContentBlocks(msg.ContentBlocks)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %+v", blocks)
	}
	encoded, _ := json.Marshal(blocks[0])
	var wire map[string]any
	_ = json.Unmarshal(encoded, &wire)
	if wire["type"] != "compaction" {
		t.Fatalf("type lost: %v", wire)
	}
	if wire["content"] != "Summary of earlier turns." {
		t.Fatalf("compaction content dropped: %v", wire)
	}
	if _, has := wire["index"]; has {
		t.Fatalf("index must never reach the wire: %v", wire)
	}
	// 调用方 map 不被原地修改（#40213 同类保证）。
	if msg.ContentBlocks[0].(messages.NonStandardContentBlock).Value["index"] != 0 {
		t.Fatal("caller map must not be mutated")
	}
}

// 非 direct 的嵌套 caller 透传（#41678）；direct 被剥（server 族）。
func TestNestedCallerPreservedServerToolResult(t *testing.T) {
	caller := map[string]any{"type": "code_execution_20250825", "tool_id": "srvtool_1"}
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "server_tool_result", Value: map[string]any{
			"type":        "server_tool_result",
			"tool_use_id": "srvtoolu_1",
			"content":     "42",
			"caller":      caller,
		}},
	}
	blocks, err := formatContentBlocks(msg.ContentBlocks)
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(blocks[0])
	var wire map[string]any
	_ = json.Unmarshal(encoded, &wire)
	got, ok := wire["caller"].(map[string]any)
	if !ok || got["type"] != "code_execution_20250825" {
		t.Fatalf("nested caller must be preserved: %v", wire)
	}
}

func TestDirectCallerStrippedOnServerBlocks(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "server_tool_use", Value: map[string]any{
			"type":   "server_tool_use",
			"id":     "srvtoolu_2",
			"name":   "web_search",
			"caller": map[string]any{"type": "direct"},
		}},
	}
	blocks, err := formatContentBlocks(msg.ContentBlocks)
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(blocks[0])
	var wire map[string]any
	_ = json.Unmarshal(encoded, &wire)
	if _, has := wire["caller"]; has {
		t.Fatalf("direct caller must be stripped on server blocks: %v", wire)
	}
}

// client tool_use 的 caller 无条件透传（上游 chat_models.py:834-835 的非对称行为）。
func TestClientToolUseCallerPassesVerbatim(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "tool_use", Value: map[string]any{
			"type":   "tool_use",
			"id":     "toolu_1",
			"name":   "search",
			"input":  map[string]any{"q": "x"},
			"caller": map[string]any{"type": "direct"},
		}},
	}
	blocks, err := formatContentBlocks(msg.ContentBlocks)
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(blocks[0])
	var wire map[string]any
	_ = json.Unmarshal(encoded, &wire)
	if got, ok := wire["caller"].(map[string]any); !ok || got["type"] != "direct" {
		t.Fatalf("tool_use caller passes verbatim: %v", wire)
	}
}
```

- [ ] **Step 2: 运行确认失败**

Run: `go test ./partners/anthropic/ -run "Caller|Compaction" -v`
Expected: FAIL —— caller/content 被白名单丢弃。

- [ ] **Step 3: 实现**

1. `chatmodel.go` 的 `contentBlock` struct 加字段：

```go
	Caller  map[string]any `json:"caller,omitempty"`
```

（放在 `CacheControl` 字段后。）

2. `contentblocks.go`：
   - `setContentBlockField` 的 switch 加：
     ```go
     case "content":
     	out.Content = value
     case "caller":
     	// caller 的门控依赖块类型，而 map 迭代无序（Type 未必先设置），
     	// 所以这里暂存，由 passthroughBlock 循环后统一归置。
     	if isMap {
     		out.Caller = m
     	}
     ```
   - `passthroughBlock` 循环后加门置（对齐 #41678 的非对称：client `tool_use` 无条件透传；server 族剥 direct）：
     ```go
     func passthroughBlock(m map[string]any) contentBlock {
     	out := contentBlock{}
     	for key, value := range m {
     		setContentBlockField(&out, key, value)
     	}
     	// #41678: server-side tool blocks echoing a direct caller are
     	// rejected by the API; client tool_use passes its caller verbatim
     	// (chat_models.py:834-835 vs 853-855/965-967).
     	if out.Caller != nil && out.Type != "tool_use" {
     		if t, _ := out.Caller["type"].(string); t == "direct" {
     			out.Caller = nil
     		}
     	}
     	return out
     }
     ```
   - 检查 `contentBlock` 的 JSON 序列化是否还有其他引用点需要同步（如自定义 MarshalJSON）——`grep -n "MarshalJSON" partners/anthropic/*.go`，有则同步加 caller/content。

- [ ] **Step 4: 运行确认通过**

Run: `go test -race ./partners/anthropic/ && go build ./... && go vet ./partners/anthropic/`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add partners/anthropic/
git commit -m "fix(anthropic): pass through nested caller and compaction content on replay (#41678, #41637)"
```

---

### Task 10: model-profiles 数据同步（#41556/#41706）

**Files:**
- Modify: `modelprofiles/data/openai/profiles.json`、`modelprofiles/data/anthropic/profiles.json`（生成产物）、`modelprofiles/profiles.json`（物化 bundle）

**Interfaces:**
- Consumes: 无。**需要联网**（models.dev）。
- Produces: 无代码变更。

- [ ] **Step 1: 刷新 openai 与 anthropic 数据并物化**

```bash
cd /Users/yuhaochen/Documents/codebase/projanvil/langchain_port/langchain_golang
printf 'y\n' | go run ./cmd/langchain-profiles refresh --provider openai --data-dir modelprofiles/data/openai
printf 'y\n' | go run ./cmd/langchain-profiles refresh --provider anthropic --data-dir modelprofiles/data/anthropic
go run ./cmd/langchain-profiles materialize
```

- [ ] **Step 2: 核对 diff 并跑测试**

Run: `git diff --stat modelprofiles/ && go test ./modelprofiles/...`
Expected: profiles.json 有数据更新（或为空 diff——上游数据未变也属正常，commit message 注明）；测试 PASS。

- [ ] **Step 3: Commit**

```bash
git add modelprofiles/
git commit -m "chore(model-profiles): refresh openai/anthropic profile data (#41556, #41706)"
```

---

### Task 11: 收尾 — DIVERGENCES / CHANGELOG / 全量门禁

**Files:**
- Modify: `DIVERGENCES.md`
- Modify: `CHANGELOG.md`
- Modify: `../reports/16-latest-release-port-gap.md`（落地追记，参照 report 14 的追记格式。**注意：该文件在 langchain_golang 仓库之外**，若 langchain_golang 是独立 git 仓库，`git add ../reports/...` 会报 "outside repository"——reports 的改动在工作区顶层单独提交或不提交，与 Go 仓的提交分开。）

- [ ] **Step 1: DIVERGENCES.md 新增条目**（措辞落到文件「Known limitations / 分歧」相应节）：

```markdown
- **Non-standard block `index` stays inside Value** — Python lifts a non-standard block's `index` to the block top level (#40213 area); Go's `NonStandardContentBlock` has no Index field and keeps `index` inside `Value`. The caller-map mutation bug Python fixed there never existed in Go (the escape hatch clones via `cloneMap(stripType(m))`).
- **Interrupt IDs are `node-N`, not xxh3 hashes** — Python 1.2.20 keys an interrupt's id by `xxh3(ns)` plus the answer count (#9589); Go mints `nodeName-counter` per call, which already gives every interrupt in a task its own id (the shared-id stale-answer bug class #9589 fixes never existed in Go). Checkpoint files are not cross-SDK shareable anyway (existing divergence), so id format mismatch has no interop impact.
- **Strict resume map detection** — Python treats a resume dict as an id map only when every key is an xxh3 interrupt id; Go has no xxh3 ids, so any `map[string]any` resume is treated as a resume map (same as Go's non-strict path).
- **Strict resume on superseded checkpoints** — Python skips the strict check when the addressed checkpoint was superseded; Go skips it on pinned (`CheckpointID`) resumes, the only path where supersession occurs (mirrors the exit-path gate in checkpoint_sink.go).
- **`EventToolStart/End/Error` callback kinds are defined but never emitted** — Go tool executions surface as chain events; the tool-kind mapping exists only in the StreamEvents/LangSmith projection tables. (#41660 核查附带发现。)
- **`JSONParser` text path has no message visibility** — Python checks refusals in `parse_result`'s ChatGeneration branch; Go's `[]outputs.Generation` carries no message (no covariance), so message-level refusal checks live in `language.InvokeStructured` and partner StructuredCaller paths instead.
- **Strict resume RemoteGraph drop-list** — Go has no RemoteGraph (existing backlog), so Python's `CONFIG_KEY_STRICT_RESUME` forwarding drop does not apply.
- **ModelRefusalMiddleware's omit marker** — Python's `omit_payload` returns an empty dict; Go's TracePolicy for refusal/router middleware replaces inputs with `{"omitted": true}` so the scrub stays visibly intentional.
```

- [ ] **Step 2: CHANGELOG.md 新增 `[0.13.0]` 草节**（放在文件顶部既有节之前，格式照既有节）：

```markdown
## [0.13.0] - Unreleased

### Added

- core: standardized `RefusalBlock` content block + `lcerrors.ModelRefusalError` + `outputparser.RaiseIfRefused`, wired into the tool-call output parsers and `language.InvokeStructured` (langchain-core 1.7.5, #41582).
- core: `ConvertToOpenAIMessages` emits non-empty refusal blocks and skips empty ones (#41582).
- openai: refusal blocks both directions (chat-completions + responses APIs), including stream aggregation and structured-output refusal errors (#41582).
- anthropic: `stop_reason="refusal"` produces a `RefusalBlock` with category/explanation extras; refusal blocks are dropped on input; structured output raises `ModelRefusalError` on refusal instead of a no-tool-call parse error (#41582, #41715).
- anthropic: replayed compaction blocks keep their content and never leak `index`; nested non-direct `caller` passes through on server tool blocks, verbatim on client `tool_use` (#41637, #41678).
- gemini: safety-blocked responses (SAFETY / PROHIBITED_CONTENT / BLOCKLIST finish reasons, prompt block feedback) produce refusal blocks (#41658).
- langchain: `middleware.ModelRefusalMiddleware` with omit-inputs trace policy (#41582, #41689).
- langgraph: opt-in strict resume mode (`Options.StrictResume`) raising `types.UnknownInterruptError` for stale/empty/ambiguous resumes before any write (#9574, #9590).

### Fixed

- anthropic: streaming terminal chunk carrying stop_reason/stop_details is emitted regardless of the stream-usage gate (usage itself stays gated) (#41582 wave).
```

- [ ] **Step 3: report 16 追记**

在 `../reports/16-latest-release-port-gap.md` 顶部加落地追记（格式参照 report 14 顶部追记）：本波 P1–P4 全部落地于 v0.13.0，列出 commit 范围与本节链接的 spec/plan 路径；无需行动清单维持不变。

- [ ] **Step 4: 全量门禁**

```bash
cd /Users/yuhaochen/Documents/codebase/projanvil/langchain_port/langchain_golang
go build ./... && go vet ./... && go test -race ./...
golangci-lint run ./...
make coverage  # ≥95%（若仓库有此 target）
```

Expected: 全绿。任何失败回到对应任务修复后再过门禁。

- [ ] **Step 5: 最终提交**

```bash
git add DIVERGENCES.md CHANGELOG.md ../reports/16-latest-release-port-gap.md
git commit -m "docs: wave5 parity changelog, divergences, and report 16 follow-up"
```
