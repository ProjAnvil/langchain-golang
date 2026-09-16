# M0a 工程化里程碑 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 M0a——CI、ParallelToolCalls provider 序列化（openai 双路径 + anthropic 合成）、四大社区文档，并以 v0.8.0 发布。

**Architecture:** 纯增量变更：openai 在既有 `BindToolsWithOptions` → 请求构造链上加一个 `parallelToolCalls` 字段（Responses 与 Chat Completions 两个 payload struct 各加一列）；anthropic 在 payload 构建处把 flag 合成为 `tool_choice.disable_parallel_tool_use`；CI 覆盖 root + 三个嵌套 checkpoint 模块（零 docker）；四份社区文档按 Keep a Changelog / GitHub 惯例新增。

**Tech Stack:** Go 1.26（go.mod floor，CI 矩阵 1.26/1.27）、GitHub Actions、golangci-lint v2（pin 版本）、httptest 假服务器测试（既有模式）。

**Spec:** `docs/superpowers/specs/2026-09-16-parity-catchup-p0p1-design.md` §5（M0a）

## Global Constraints

- Go floor 1.26（go.mod）；测试与 CI 用 1.26 + 1.27 双版本
- 零现有接口改动：不修改任何既有导出签名；新行为只经既有 `language.BindToolsOptions.ParallelToolCalls`（已存在于 core/language/chatmodel.go:80-88）流入
- 代码注释英文，Python parity 处标注出处（Python 源码文件:行号），沿仓库既有风格
- CI 零 docker；外部服务一律 env-gated
- commit 用 conventional 前缀（feat/fix/docs/chore/test，见 git log 既有风格）
- 实施时启用 modern-go skill（用户指定）
- push、打 tag、gh release 是外发动作：执行前须用户确认
- 版本事实：v0.7.0 tag 已存在；HEAD 的嵌套 go.mod 已 pin 未发布的 v0.7.1（29cacb1）；v0.7.1 未发则嵌套模块无法解析依赖（见 Task 1）

---

### Task 1: 发布 v0.7.1（前置：让嵌套 pin 可解析）

**Files:** 无代码变更（git tag + gh release）

**Interfaces:** 无。产出 v0.7.1 tag 与 release，使 `langgraph/checkpoint/*/go.mod` 中 `langchain-golang v0.7.1` 引用可解析（否则 Task 5 的 `make test-sqlite/redis` 会因依赖 404 失败）。

- [ ] **Step 1: 验证嵌套模块当前确实被 v0.7.1 pin 卡住**

Run: `cd langgraph/checkpoint/redis && go list -m github.com/projanvil/langchain-golang`
Expected: 报错（v0.7.1 not found）——若能解析（如本地 module cache 已有），仍按本任务走，保持发版节奏

- [ ] **Step 2: 确认 v0.7.1 内容**

Run: `git log v0.7.0..29cacb1 --oneline`
Expected: 3a24fdf（Go 1.26 现代化）、91b4ac9（anthropic 流式 usage 修复，实际在 v0.7.0 内则此处只有 2 条）、2f4dbd9/29cacb1（pin commits）。以实际输出为准写 release notes

- [ ] **Step 3: 打 tag（在 29cacb1 上，不含其后的 spec 文档提交）**

```bash
git tag v0.7.1 29cacb1
```

- [ ] **Step 4: 【用户确认后】push tag 并发 release**

```bash
git push origin v0.7.1
gh release create v0.7.1 --title "v0.7.1" --notes "- refactor: modernize codebase to Go 1.26 idioms (3a24fdf)
- chore: pin langchain-golang v0.7.1 in nested checkpoint modules (29cacb1)"
```

- [ ] **Step 5: 嵌套模块补 tag（pin 变更即模块变更）**

```bash
git tag langgraph/checkpoint/sqlite/v0.3.2 29cacb1
git tag langgraph/checkpoint/redis/v0.3.2 29cacb1
git tag langgraph/checkpoint/postgres/v0.3.2 29cacb1
git push origin langgraph/checkpoint/sqlite/v0.3.2 langgraph/checkpoint/redis/v0.3.2 langgraph/checkpoint/postgres/v0.3.2
```

- [ ] **Step 6: 复验嵌套测试恢复**

Run: `make test-redis && make test-sqlite`
Expected: PASS

---

### Task 2: openai ParallelToolCalls 序列化（双路径）

**Files:**
- Create: `partners/openai/parallel_tool_calls_test.go`
- Modify: `partners/openai/chatmodel.go`（ChatModel struct ~:33 加字段；BindToolsWithOptions ~:160-175；requestPayload ~:551 加列；buildRequest ~:452 加赋值）
- Modify: `partners/openai/chat_completions.go`（chatCompletionsRequest ~:16 加列；CC 构建处 ~:206 加赋值）

**Interfaces:**
- Consumes: `language.BindToolsOptions.ParallelToolCalls *bool`（core/language/chatmodel.go:80-88，已存在）；`toolChoiceServer` / `toolChoiceResponsesBody` / `toolChoiceChatBody`（toolchoice_test.go，同包已存在）
- Produces: `ChatModel.parallelToolCalls *bool`（私有字段）；请求体新增 `"parallel_tool_calls"` 键（nil 时 omitempty 不发送）

- [ ] **Step 1: 写失败测试**

`partners/openai/parallel_tool_calls_test.go`：

```go
package openai

import (
	"context"
	"fmt"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

// Mirrors Python's BaseChatOpenAI default_params parallel_tool_calls
// (chat_models/base.py:1340-1350 exclude_if_none family): nil omits the
// field, a set value serializes on both the Responses and Chat Completions
// paths.

func parallelToolCallsModel(t *testing.T, baseURL string, chatCompletions bool, parallel *bool) ChatModel {
	t.Helper()
	tool, err := coretools.FromFunc("GenerateUsername", "Get a username.", func(ctx context.Context, args struct{ Name string }) (coretools.Result, error) {
		return coretools.Result{Content: args.Name}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	model := NewChatModel(
		modelconfig.WithBaseURL(baseURL),
		modelconfig.WithModel("gpt-test"),
	)
	if chatCompletions {
		model = model.WithChatCompletions()
	}
	bound, err := model.BindToolsWithOptions([]coretools.Tool{tool}, language.BindToolsOptions{ParallelToolCalls: parallel})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}
	return bound.(ChatModel)
}

func TestParallelToolCallsSerializedBothAPIs(t *testing.T) {
	for _, tc := range []struct {
		name            string
		chatCompletions bool
		body            string
	}{
		{"responses", false, toolChoiceResponsesBody},
		{"chat completions", true, toolChoiceChatBody},
	} {
		for _, want := range []bool{false, true} {
			t.Run(tc.name+"/"+fmt.Sprint(want), func(t *testing.T) {
				server, got := toolChoiceServer(t, tc.body)
				value := want
				model := parallelToolCallsModel(t, server.URL, tc.chatCompletions, &value)
				if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
					t.Fatalf("Invoke: %v", err)
				}
				if (*got)["parallel_tool_calls"] != want {
					t.Fatalf("parallel_tool_calls = %v, want %v", (*got)["parallel_tool_calls"], want)
				}
			})
		}
	}
}

func TestParallelToolCallsOmittedWhenNil(t *testing.T) {
	for _, tc := range []struct {
		name            string
		chatCompletions bool
		body            string
	}{
		{"responses", false, toolChoiceResponsesBody},
		{"chat completions", true, toolChoiceChatBody},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server, got := toolChoiceServer(t, tc.body)
			model := parallelToolCallsModel(t, server.URL, tc.chatCompletions, nil)
			if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
				t.Fatalf("Invoke: %v", err)
			}
			if _, present := (*got)["parallel_tool_calls"]; present {
				t.Fatalf("parallel_tool_calls must be omitted when nil, got %v", (*got)["parallel_tool_calls"])
			}
		})
	}
}
```

- [ ] **Step 2: 运行确认失败**

Run: `go test ./partners/openai/ -run TestParallelToolCalls -v`
Expected: FAIL —— payload 无 `parallel_tool_calls` 键

- [ ] **Step 3: 实现**

`partners/openai/chatmodel.go`——四处修改：

(a) ChatModel struct（`toolChoice *ToolChoice` 之后）加：

```go
	parallelToolCalls *bool
```

(b) `BindToolsWithOptions`（现有 `if opts.ToolChoice != ""` 块后）加，并同步改写 :160-166 的 godoc（现在写着 "parallel_tool_calls field is not modeled"，改为描述两路径序列化）：

```go
	if opts.ParallelToolCalls != nil {
		next.parallelToolCalls = opts.ParallelToolCalls
	}
```

(c) `requestPayload` struct（`ToolChoice any` 行后）加：

```go
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`
```

(d) `buildRequest`（`if m.toolChoice != nil` 块后）加：

```go
	if m.parallelToolCalls != nil {
		payload.ParallelToolCalls = m.parallelToolCalls
	}
```

`partners/openai/chat_completions.go`——两处：

(a) `chatCompletionsRequest` struct（`ToolChoice any` 行后）加：

```go
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`
```

(b) CC 构建处（`if m.toolChoice != nil { payload.ToolChoice = ... }` 后）加：

```go
	if m.parallelToolCalls != nil {
		payload.ParallelToolCalls = m.parallelToolCalls
	}
```

- [ ] **Step 4: 运行确认通过**

Run: `go test ./partners/openai/ -run TestParallelToolCalls -v && go test ./partners/openai/`
Expected: PASS（新旧全绿）

- [ ] **Step 5: Commit**

```bash
git add partners/openai/parallel_tool_calls_test.go partners/openai/chatmodel.go partners/openai/chat_completions.go
git commit -m "feat(openai): serialize parallel_tool_calls on both Responses and Chat Completions paths"
```

---

### Task 3: anthropic disable_parallel_tool_use 合成

**Files:**
- Create: `partners/anthropic/parallel_tool_calls_test.go`
- Modify: `partners/anthropic/chatmodel.go`（ChatModel struct :29 附近加字段；BindToolsWithOptions :142-160；payload 构建 :432-433 区域）

**Interfaces:**
- Consumes: `language.BindToolsOptions.ParallelToolCalls *bool`；`newTestServer(t, &request)` 与 `bindWithToolChoice` 模式（toolbinder_test.go，同包）；`cloneAnyMap`（chatmodel.go:203 使用过）
- Produces: 请求体 `tool_choice` 对象可能含 `disable_parallel_tool_use` 键（bool，语义取反）；`ParallelToolCalls != nil` 且 ToolChoice 未设时合成 `{"type":"auto",...}`

- [ ] **Step 1: 写失败测试**

`partners/anthropic/parallel_tool_calls_test.go`：

```go
package anthropic

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// bindWithParallel mirrors bindWithToolChoice (toolbinder_test.go) plus the
// parallel flag. Anthropic expresses parallel_tool_calls as
// tool_choice.disable_parallel_tool_use (inverted), synthesizing a default
// {"type":"auto"} tool_choice when none is set — mirroring
// langchain-anthropic's payload assembly.

func bindWithParallel(t *testing.T, choice language.ToolChoice, parallel *bool) map[string]any {
	t.Helper()
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	tool, err := tools.NewFunc(
		"get_weather",
		"gets weather",
		schema.Object(map[string]schema.Schema{
			"location": schema.String("location"),
		}, "location"),
		func(_ context.Context, _ map[string]any) (tools.Result, error) {
			return tools.Result{Content: "sunny"}, nil
		},
	)
	if err != nil {
		t.Fatalf("new tool: %v", err)
	}
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
	)
	bound, err := model.BindToolsWithOptions([]tools.Tool{tool}, language.BindToolsOptions{ToolChoice: choice, ParallelToolCalls: parallel})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}
	if _, err := bound.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	return request
}

func TestParallelToolCallsSynthesizesDisableFlag(t *testing.T) {
	yes, no := true, false
	cases := []struct {
		name       string
		choice     language.ToolChoice
		parallel   *bool
		wantChoice map[string]any
	}{
		{"disable with auto choice", language.ToolChoiceAuto, &no,
			map[string]any{"type": "auto", "disable_parallel_tool_use": true}},
		{"disable without choice", "", &no,
			map[string]any{"type": "auto", "disable_parallel_tool_use": true}},
		{"enable without choice", "", &yes,
			map[string]any{"type": "auto", "disable_parallel_tool_use": false}},
		{"disable with named tool", "get_weather", &no,
			map[string]any{"type": "tool", "name": "get_weather", "disable_parallel_tool_use": true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request := bindWithParallel(t, tc.choice, tc.parallel)
			got, ok := request["tool_choice"].(map[string]any)
			if !ok {
				t.Fatalf("tool_choice = %v, want object %v", request["tool_choice"], tc.wantChoice)
			}
			for k, v := range tc.wantChoice {
				if got[k] != v {
					t.Fatalf("tool_choice[%q] = %v, want %v (full: %v)", k, got[k], v, got)
				}
			}
		})
	}
}

func TestParallelToolCallsNilLeavesToolChoiceUntouched(t *testing.T) {
	request := bindWithParallel(t, language.ToolChoiceAuto, nil)
	got, ok := request["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice = %v, want object", request["tool_choice"])
	}
	if _, present := got["disable_parallel_tool_use"]; present {
		t.Fatalf("disable_parallel_tool_use must be absent when parallel unset: %v", got)
	}
}
```

- [ ] **Step 2: 运行确认失败**

Run: `go test ./partners/anthropic/ -run TestParallelToolCalls -v`
Expected: FAIL —— 当前 `ParallelToolCalls` 被 BindToolsWithOptions 显式忽略（chatmodel.go godoc "intentionally ignored"）

- [ ] **Step 3: 实现**

`partners/anthropic/chatmodel.go`：

(a) ChatModel struct（`toolChoice map[string]any` 后）加：

```go
	parallelToolCalls *bool
```

(b) `BindToolsWithOptions`：替换「intentionally ignored」注释与行为：

```go
	if opts.ParallelToolCalls != nil {
		next.parallelToolCalls = opts.ParallelToolCalls
	}
```

同步改写 :135-141 的 godoc：删去 "ParallelToolCalls is NOT supported ... silently ignored"，改为说明合成语义（disable_parallel_tool_use inside tool_choice, defaulting to {"type":"auto"}，Python langchain-anthropic 同构）。

(c) payload 构建（现有 `if m.toolChoice != nil { payload.ToolChoice = m.toolChoice }` 处，:432-433）改为：

```go
	if m.toolChoice != nil {
		payload.ToolChoice = m.toolChoice
	}
	if m.parallelToolCalls != nil {
		choice := payload.ToolChoice
		if choice == nil {
			choice = map[string]any{"type": "auto"}
		} else {
			choice = cloneAnyMap(choice)
		}
		choice["disable_parallel_tool_use"] = !*m.parallelToolCalls
		payload.ToolChoice = choice
	}
```

（cloneAnyMap 防止改写共享的构造期 toolChoice map。）

- [ ] **Step 4: 运行确认通过**

Run: `go test ./partners/anthropic/`
Expected: PASS（含既有 toolbinder/toolchoice 测试——nil 行为不变）

- [ ] **Step 5: Commit**

```bash
git add partners/anthropic/parallel_tool_calls_test.go partners/anthropic/chatmodel.go
git commit -m "feat(anthropic): synthesize tool_choice.disable_parallel_tool_use from ParallelToolCalls"
```

---

### Task 4: openaicompat 继承核验（test-only）

**Files:**
- Create: `partners/openaicompat/parallel_test.go`

**Interfaces:**
- Consumes: `chatmodels.ParseModelString` / `chatmodels.Resolve`（wire_test.go 模式）；`language.ToolBinder`（类型断言拿 BindToolsWithOptions）
- Produces: 无新接口——只证明 compat provider（经 providers.go 直接构造 partners/openai ChatModel）天然继承 ParallelToolCalls 序列化

- [ ] **Step 1: 写测试（预期直接通过；失败则说明继承断链，需修 providers.go 构造路径）**

`partners/openaicompat/parallel_test.go`：

```go
package openaicompat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// The compat factory builds partners/openai ChatModel (providers.go), so
// ParallelToolCalls must flow through to the Chat Completions payload with
// zero openaicompat changes.

func TestCompatInheritsParallelToolCalls(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-1","model":"llama-test",
			"choices":[{"message":{"role":"assistant","content":"ok"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()
	t.Setenv("GROQ_API_KEY", "test-key")
	t.Setenv("GROQ_API_BASE", server.URL)

	spec, err := chatmodels.ParseModelString("groq:llama-test")
	if err != nil {
		t.Fatalf("ParseModelString: %v", err)
	}
	model, err := chatmodels.Resolve(spec)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	tool, err := coretools.FromFunc("GetWeather", "Get weather.", func(ctx context.Context, args struct{ City string }) (coretools.Result, error) {
		return coretools.Result{Content: args.City}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	binder, ok := model.(language.ToolBinder)
	if !ok {
		t.Fatal("resolved model does not implement language.ToolBinder")
	}
	no := false
	bound, err := binder.BindToolsWithOptions([]coretools.Tool{tool}, language.BindToolsOptions{ParallelToolCalls: &no})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}
	if _, err := bound.Invoke(t.Context(), []messages.Message{messages.Human("hello")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if gotBody["parallel_tool_calls"] != false {
		t.Fatalf("parallel_tool_calls = %v, want false", gotBody["parallel_tool_calls"])
	}
}
```

（注意补 `context` import。）

- [ ] **Step 2: 运行**

Run: `go test ./partners/openaicompat/ -run TestCompatInheritsParallelToolCalls -v`
Expected: PASS（继承成立）。若 FAIL：排查 providers.go 构造是否绕过 BindToolsWithOptions 路径并修复

- [ ] **Step 3: Commit**

```bash
git add partners/openaicompat/parallel_test.go
git commit -m "test(openaicompat): prove ParallelToolCalls inherits via the openai adapter"
```

---

### Task 5: CI workflow + lint 配置

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.golangci.yml`

**Interfaces:** 无代码接口。产出 CI 门禁：root 测试 + 嵌套三模块 + lint，PR 与 main push 触发。

- [ ] **Step 1: 写 `.golangci.yml`（v2 配置，保守默认集）**

```yaml
version: "2"
linters:
  default: standard
  enable:
    - copyloopvar
    - misspell
    - unconvert
  exclusions:
    rules:
      - path: _test\.go
        linters:
          - errcheck
```

- [ ] **Step 2: 写 `.github/workflows/ci.yml`**

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    strategy:
      matrix:
        go-version: ['1.26.x', '1.27.x']
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: ${{ matrix.go-version }}
      - name: Root module tests
        run: go test ./...
      - name: Type-check integration tests (offline)
        run: make vet-integration
      - name: Nested sqlite saver
        run: make test-sqlite
      - name: Nested redis saver (miniredis)
        run: make test-redis

  postgres:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.26.x'
      - uses: actions/cache@v4
        with:
          path: ~/.embedded-postgres-go
          key: embedded-postgres-${{ runner.os }}
      - name: Nested postgres saver (embedded)
        run: make test-postgres

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.26.x'
      - uses: golangci/golangci-lint-action@v8
        with:
          version: v2.1.6 # pinned; check golangci-lint.run for the current v2 patch before merging
```

- [ ] **Step 3: 本地验证 lint 配置可运行**

Run: `go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.1.6 run ./partners/openai/ ./partners/anthropic/ ./core/language/ 2>&1 | tail -5`
Expected: 无输出（干净）或仅既有风格的告警；有新告警则修配置或修代码（不为此扩大 lint 集）

- [ ] **Step 4: 本地全量预演 CI 步骤**

Run: `go test ./... && make vet-integration && make test-sqlite && make test-redis && make test-postgres`
Expected: 全 PASS（postgres 首次下载 ~30MB 嵌入二进制）

- [ ] **Step 5: README 加 CI badge（标题行下）**

```markdown
[![CI](https://github.com/Projanvil/langchain-golang/actions/workflows/ci.yml/badge.svg)](https://github.com/Projanvil/langchain-golang/actions/workflows/ci.yml)
```

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml .golangci.yml README.md
git commit -m "ci: GitHub Actions for root + nested checkpoint modules, golangci-lint"
```

---

### Task 6: CHANGELOG.md

**Files:**
- Create: `CHANGELOG.md`

**Interfaces:** 无。v0.8.0 条目在 Task 9 发版时改为正式日期；v0.7.0 及以前为回填。

- [ ] **Step 1: 写 CHANGELOG.md**

```markdown
# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

### Added
- CI (GitHub Actions): root module tests on a go 1.26/1.27 matrix, nested checkpoint module suites (sqlite, redis via miniredis, postgres via embedded binaries), golangci-lint.
- openai: `parallel_tool_calls` serialization on both the Responses and Chat Completions paths via `BindToolsOptions.ParallelToolCalls`.
- anthropic: `ParallelToolCalls` now synthesizes `tool_choice.disable_parallel_tool_use`, defaulting tool_choice to `{"type":"auto"}` when unset.
- Community files: CONTRIBUTING, DIVERGENCES, SECURITY.

### Changed
- Modernized to Go 1.26 idioms throughout (3a24fdf).

## [0.7.0] - 2026-09-12

### Breaking
- Graph-level default retry: `DefaultRetryOn` now retries all errors unless a custom `RetryOn` is set (Python parity).
- Subgraph checkpoint namespaces changed to NS-routed values; interrupted subgraphs pause the parent and resume via the new namespace scheme.
- Anthropic streaming now yields a terminal usage-only chunk on `message_delta` (aligns usage accounting with invoke).

### Added
- langgraph: interrupted-subgraph pause/resume across processes; durable pause writes (async/exit); per-run durability override and per-task subgraph checkpoints; graph-level default retry policy; semantic search for InMemoryStore (`index=` parity); graph visualization parity (`get_graph`, Mermaid options, xray, ASCII).
- runnables/callbacks: StreamEvents (Runnable-level `astream_events`, v2 projection); chain lifecycle events across all combinators; Bind/Pick/Each/BatchAsCompleted and fallback error filtering.
- agents: interrupt-based human-in-the-loop with cross-process approval; middleware tool/state auto-collection; tool-returned Commands and agent-hook jumps; `write_todos` state; per-call response_format/tool_choice/model_settings overrides; apply update-only Commands from `wrap_model_call` middleware.
- tracers: LangSmith run-tree tracing with a batched background client.
- partners/openaicompat: OpenAI-compatible provider registry (groq, mistralai, deepseek, xai, openrouter, fireworks, perplexity).
- partners/openai: multimodal inputs, Responses reasoning effort, streaming usage, sampling params (top_p/stop/seed/penalties/logit_bias/n/logprobs), real tiktoken BPE token counting with the official image token formula, honest capability declarations.
- partners/anthropic: usage cache token details, stop_sequences, count_tokens API-backed message counting, explicit errors for non-base64 data URIs.
- language/agents: bind_tools options with tool_choice (ToolStrategy forces "any").
- standardtests: layered chat-model conformance suites (tool calling/choice, structured output, multimodal, streaming) wired to openai/anthropic/ollama.
- prompts: Partial and FewShotChatMessagePromptTemplate.
- core/language: real GPT-2 BPE (r50k_base) fallback token counting; messages usage-metadata scaling.

### Fixed
- openai: Chat Completions path serializes structured-output response_format; ToolStrategy narrowing override filters bindings and request tools.
- Durable pause writes, run-id tree, minted message ids, tracer payload ownership (final-review fixes).

## [0.6.5] - 2026-08
create_agent parity: return_direct, structured-output retry, routing, 9999 recursion default, dynamic model.

## [0.6.4] - 2026-08
Go floor raised to 1.26; tiktoken-go/tokenizer v0.8.1.

## [0.6.3] - 2026-08
Streaming robustness, bounded batch concurrency, constructor error returns (rectification batch).

## [0.6.0] - 2026-08
Initial public parity line: agents, graphs, checkpoint savers, partners (openai/anthropic/ollama/chroma), textsplitters, standardtests.

## [0.5.x] - 2026-07/08
Early development line preceding the parity baseline.

[Unreleased]: https://github.com/Projanvil/langchain-golang/compare/v0.7.1...HEAD
[0.7.0]: https://github.com/Projanvil/langchain-golang/compare/v0.6.5...v0.7.0
```

（v0.6.x 各条目日期以 `git log <tag> -1 --format=%as` 校正为准。）

- [ ] **Step 2: 校正历史日期**

Run: `for t in v0.5.0 v0.6.0 v0.6.3 v0.6.4 v0.6.5 v0.7.0; do echo "$t $(git log -1 --format=%as $t)"; done`
Expected: 输出各 tag 日期，替换文中占位日期

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG (Keep a Changelog; backfill 0.5-0.7.0, open 0.8.0 section)"
```

---

### Task 7: CONTRIBUTING.md

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: 写 CONTRIBUTING.md**

```markdown
# Contributing

Thanks for contributing! New partner integrations are especially welcome (Google Gemini, AWS Bedrock, more vector stores, ...).

## Getting started

1. Fork the repository and create a feature branch (`git checkout -b feat/your-feature`).
2. Make your change with tests.
3. Ensure the full gate passes locally:

   ```bash
   go build ./... && go vet ./... && go test -race ./...
   make test-sqlite test-redis   # nested checkpoint modules (offline)
   ```

4. Match the existing code style and Python-parity conventions.
5. Submit a pull request.

## Conventions

- **Python is authoritative**: this is a port of LangChain/LangGraph. When in doubt, check what the Python source does and cite the file:line in a comment.
- **Zero breaking changes to shipped interfaces** within a minor: add optional capability interfaces (see `core/vectorstores` `TextAdder` or `core/retrievers` searcher interfaces for the pattern) instead of widening existing ones.
- **Trust `go build/vet/test`**, not editor diagnostics (gopls may show false positives).
- Every package should have compile-checked examples in `example_test.go`.
- Bilingual docs: add both `guide.md` and `guide.zh-CN.md` under `docs/usage/` for new user-facing features.
- Tests use in-process fakes (`httptest`, miniredis, embedded postgres) — CI never needs docker or live API keys. Integration tests behind the `integration` build tag read keys from `.env` (see `.env.example`).
- Commit messages use conventional prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- New partner adapters must pass the `standardtests` conformance suites.

## Testing

- `make test` — offline unit suite (root module)
- `make test-sqlite` / `make test-redis` / `make test-postgres` — nested checkpoint saver modules
- `make test-integration` — live-provider tests (needs `.env`)
- `make vet-integration` — type-check integration tests without network

## Security

See SECURITY.md. Do not open public issues for vulnerabilities.
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING guide"
```

---

### Task 8: DIVERGENCES.md + SECURITY.md

**Files:**
- Create: `DIVERGENCES.md`
- Create: `SECURITY.md`

- [ ] **Step 1: 写 DIVERGENCES.md**

```markdown
# Divergences from Python LangChain/LangGraph

Deliberate design decisions where this port does not mirror Python. Each entry states why. Anything not listed here aims for parity; file an issue if you find a gap.

## Never planned

- **LangGraph Platform / Server / Studio / CLI** — commercial product lines, out of scope for an OSS port.
- **Local transformers models** — the Go ecosystem has no comparable runtime; use API providers (or ollama for local serving).
- **Command-Send / remaining_steps** — superseded upstream patterns whose Go equivalents (explicit routing, run control) already exist.

## Deliberate design differences

- **Cache short-circuit middleware** — Go's `cache` middleware short-circuits identical requests instead of Python's instrumentation-only behavior; Go idiom favors explicit memoization points.
- **Blank-import provider registration + shim re-exports** — partner packages self-register via `init()` and top-level `langchain/` shims re-export the stable surface, mirroring Go stdlib plugin patterns rather than Python's explicit imports.

## Deferred (upstream-triggered)

- **deepagents** — upstream is pre-1.0 (0.7.x) with an unstable API; re-evaluate when it reaches 1.0.
- **AWS Bedrock provider, RemoteGraph client** — demand-triggered; both are on the backlog.

## Notable per-adapter behaviors

- **anthropic `tool_choice=none`** — the Messages API has no `none` type; binding fails loudly instead of silently misrouting (bind no tools instead).
- **TracePolicy processor failures (M0b, planned)** — upstream records the untransformed payload when a trace processor errors; the Go port will drop the payload (fail-closed) since the feature's motivation is PII/compliance.

## Cleared as non-gaps (audited 2026-09-16)

- `tool_choice=any` (openai→required, anthropic→any) and openai multimodal inputs: verified implemented with tests.
- `BindToolsOptions.ParallelToolCalls` core plumbing (since the bind_tools options batch); provider serialization landed in v0.8.0.
```

- [ ] **Step 2: 写 SECURITY.md**

```markdown
# Security Policy

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/Projanvil/langchain-golang/security/advisories/new). Do not open public issues for security problems.

## Scope

- Vulnerabilities in this repository's code (prompt-injection-prone defaults, credential handling, SSRF in URL-accepting constructors, ...).
- Out of scope: vulnerabilities in upstream dependencies (report to the upstream project) and issues in models'/providers' hosted services.

## Supported versions

The latest minor release line receives security fixes.
```

- [ ] **Step 3: Commit**

```bash
git add DIVERGENCES.md SECURITY.md
git commit -m "docs: add DIVERGENCES (design decisions) and SECURITY policy"
```

---

### Task 9: 发布 v0.8.0

**Files:**
- Modify: `langgraph/checkpoint/sqlite/go.mod`、`langgraph/checkpoint/redis/go.mod`、`langgraph/checkpoint/postgres/go.mod`（pin v0.7.1 → v0.8.0）
- Modify: `CHANGELOG.md`（Unreleased → 0.8.0 + 日期）

- [ ] **Step 1: CHANGELOG 定稿（Unreleased → [0.8.0] - <发版日>，补 compare 链接）**

- [ ] **Step 2: 全量回归**

Run: `go test ./... && make test-sqlite && make test-redis && make test-postgres && make vet-integration`
Expected: 全 PASS

- [ ] **Step 3: 【用户确认后】tag v0.8.0 并 push，使嵌套 pin 可解析**

```bash
git tag v0.8.0
git push origin main --follow-tags  # 或按用户发版习惯
```

- [ ] **Step 4: 嵌套 pin bump + 嵌套 tag v0.3.3**

三个嵌套 `go.mod`：`github.com/projanvil/langchain-golang v0.7.1` → `v0.8.0`，随后：

```bash
go mod tidy 三个嵌套目录各自执行
git add langgraph/checkpoint/*/go.mod langgraph/checkpoint/*/go.sum
git commit -m "chore(langgraph/checkpoint): pin langchain-golang v0.8.0 in nested checkpoint modules"
git tag langgraph/checkpoint/sqlite/v0.3.3 && git tag langgraph/checkpoint/redis/v0.3.3 && git tag langgraph/checkpoint/postgres/v0.3.3
```

- [ ] **Step 5: 【用户确认后】gh release**

```bash
gh release create v0.8.0 --title "v0.8.0" --notes-file <从 CHANGELOG 0.8.0 段落生成>
```

- [ ] **Step 6: 验收对照 spec §12 M0a 行**

CI 双版本绿（push 后看 Actions）；ParallelToolCalls 断言/组合矩阵过；四文档落地；全测试零失败。

---

## Self-Review 记录

- **Spec 覆盖**：§5.1→Task 5；§5.2→Task 2/3/4；§5.3→Task 8（Cleared as non-gaps 节）；§5.4→Task 6/7/8；发版→Task 1/9。无遗漏。
- **占位符**：golangci-lint 版本号标注「merge 前核对」；CHANGELOG v0.6.x 日期有校正命令步骤。无 TBD。
- **类型一致性**：`parallelToolCalls *bool` 字段名、`cloneAnyMap`、`toolChoiceServer`、`newTestServer`、`bindWithParallel`/`parallelToolCallsModel` 均与仓库现有代码对齐。
