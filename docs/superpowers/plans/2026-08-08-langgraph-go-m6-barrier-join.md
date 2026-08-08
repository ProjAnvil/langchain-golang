# LangGraph Go Port M6: 多父 Barrier Join Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `langgraph/graph` 增加多父节点 barrier join（对齐 Python `add_edge((a, b), c)` + `NamedBarrierValue`）：builder 新增 `StateGraph.AddJoinEdge(from []string, to string)`，`langgraph/channels` 新增 `Barrier` channel 类型，执行器在父任务提交时隐式写 barrier、满员后恰好触发子节点一次、子节点提交后 Consume 复位，并把 join channel key 从一切用户可见面（snapshot/节点输入/stream values+updates/debug 事件）过滤。

**Architecture:** 忠实移植 Python 的 barrier channel 方案（spec M6 方案 A）：join 元数据在 Compile 时注册为 `join:a+b:c` 的 `channels.Barrier` channel prototype + 两份索引（barrier key→parents/child；parent→barrier keys）。执行器复用现有 channel/checkpoint 基建——隐式 write 搭进父任务的 update 批次，从而免费获得 `applyWrites` 的版本递增簿记与中断路径 `completedTaskWrites` 的 pending-writes 持久化（"父 A 已到达、父 B 中断→resume 补写"闭环）；触发判定复用 versions-seen 簿记保证"恰好一次"。不引入新的执行模型。

**Tech Stack:** Go 1.23，module `github.com/projanvil/langchain-golang`；零新增第三方依赖。

## Global Constraints

- 所有命令的工作目录：`langchain_golang/`。
- Go 1.23 下限；根 module 保持零第三方依赖（本里程碑不新增任何依赖，也不新增嵌套 module）。
- 底线全绿：`go test ./...` 与 `make test-sqlite`（不回归）。
- `langchain/agents` 零行为变化：M6 只新增 builder API 与执行器分支，不改动任何现有代码路径的行为；无 join 边的图执行轨迹逐字节不变。现有全量测试（含 `go test ./langchain/...`）不得修改即通过。
- 单元测试对标 Python：从 `libs/langgraph/tests/test_pregel.py` 的 waiting-edge 用例移植（见 Task 4），语义级对齐，不是象征性覆盖。
- 注释风格：充分的 doc comment，单反引号；代码与标识符保持英文原文。
- 提交信息：conventional commits，仿 `git log` 的 `feat(langgraph): ...` / `docs(langgraph): ...` 格式。
- 每个任务后的门禁：`go build ./... && go vet ./... && go test ./...`（在 `langchain_golang/` 下）。
- 不做：`defer=True` / `NamedBarrierValueAfterFinish`（依赖 PULL 循环的 finish 广播，边驱动无等价物——文档化声明，见 Task 5）。

## 移植基线（Python 引用语义，均已逐一核对）

- Builder：`add_edge((a, b), c)` 把 `(tuple(starts), end)` 存入 `waiting_edges`（`langgraph/graph/state.py:956-966`）；compile 时 `attach_edge` 注册 channel `join:a+b:c` = `NamedBarrierValue(str, set(starts))`，子节点订阅该 channel，每个父节点的 writers 追加一条 `ChannelWriteEntry(channel_name, start)`（`langgraph/graph/state.py:1546-1561`）。注意 Python 允许 `end_key == END`（`state.py:963-964`），Go 收紧为不允许（spec 已定，文档化分歧）。
- Channel（`langgraph/channels/named_barrier_value.py:13-81`）：`update` 幂等累计 `seen`（:56-67），收到非 `names` 成员的值抛 `InvalidUpdateError`；`get`/`is_available` 仅当 `seen == names`（:69-75），`get` 返回 `None`；`consume` 满员时清零返回 True，否则 False（:77-81）；`checkpoint` 返回 `seen` 集合、`from_checkpoint` 恢复（:46-54）——"父 A 已到达、父 B 中断"的暂停-恢复由此天然正确。
- 触发：barrier 满 → 子节点下一超步恰好运行一次（同超步多父完成也只一次）。
- 绕过语义（OR）：普通边/条件边（写 `branch:to:c`）、`Send`、`Command(goto=)` 指向 join 子节点时绕过 barrier 直接触发；混用时可能被触发多次——Python 既定行为，Go 复刻并在文档显著警告（`test_pregel.py:2710` 的 "silly edge" 用例即此语义）。
- 中断：父中断则其写入不执行；已到达记录随 checkpoint 持久化，resume 后中断父重跑补写。

## Go 侧现状锚点（均已核对行号）

- `channels.Channel` 接口五方法：`langgraph/channels/channel.go:11-26`；`InvalidUpdateError`：`channel.go:35-42`；现有 channel 实现风格参考 `channels/lastvalue.go`（全文件 55 行）。
- serde 注册表已含 `[]string`（envelope 名 `"[]string"`）：`langgraph/checkpoint/serde/json.go:40,173-174,303-317`——`Barrier.Checkpoint()` 返回 `[]string` 可直接往返。
- `runState.applyWrites`：`langgraph/graph/state.go:119-187`（其 bool 返回值**仅**被三处 `em.emitValues` 门控使用：`graph.go:683,698,995`；`resume.go:156` 与 `snapshot.go:119` 忽略该值）。
- `runState.snapshot`：`state.go:80-88`；`channelValues`：`state.go:92-100`（进 checkpoint，**不得**过滤 join key）。
- 提交点：`graph.go:941-948`（writes 批次 + `applyWrites`）；staticNext 循环：`graph.go:953-967`；`interrupt_after` 暂停：`graph.go:975-991`；loop checkpoint 保存：`graph.go:998-1002`。
- 中断路径 `completedTaskWrites` 持久化：`graph.go:911-934`（序列化器本体在 `resume.go:62-82`）；resume 重放 `applyWrites`：`resume.go:155-159`；重放 writes 的 updates 重发：`graph.go:710-712`。
- 结果收集/emission 循环：`graph.go:884-897`（`debugTaskResult` :895、`emitUpdate` :896）；debug checkpoint emission 实参 `rs.channelValues()`：`graph.go:639`。
- `staticNext`：`graph.go:1050-1062`（无出边节点当前报 error——join-only 父节点需要新分支）。
- `snapshotFromTuple`（GetState/GetStateHistory 的 Values 来源）：`snapshot.go:140-153`，调用点 `snapshot.go:53,73`。
- `Compile` 校验段与 CompiledGraph 构造：`graph.go:291-346`；当前直接把 builder 的 `g.channelProtos` 赋给 CompiledGraph（`graph.go:336`）——注册 barrier proto 时必须改为 clone，否则二次 Compile 会撞上已注册的 join key。
- 缓存 store pass：`graph.go:871-882`（隐式 write 注入点在其后——cache 条目不含 join key）；cache 命中重建：`resume.go:90-109`。
- 测试风格：白盒（`package graph`），参考 `graph/graph_test.go`、`graph/resume_test.go:18`（`checkpoint.NewMemorySaver()`）、`graph/cache_test.go:164-167`（`NodePolicies{Cache: &CachePolicy{}}` + `WithCache(checkpoint.NewInMemoryCache())`）；channels 测试 helper（`update`/`get`/`requireEmpty`/`requireRoundTrip`）在 `channels/channel_test.go:10-50`。

---

### Task 1: `channels.Barrier` 类型

**Files:**
- Create: `langgraph/channels/barrier.go`
- Test: `langchain_golang/langgraph/channels/barrier_test.go`（即 `langgraph/channels/barrier_test.go`，与现有 helper 同包）

**Interfaces:**
- Consumes: `channels.Channel` 接口（`channel.go:11-26`）、`InvalidUpdateError`（`channel.go:35-42`）。
- Produces:
  ```go
  // NewBarrier 构造等待 names 全部到达的 barrier channel（Python NamedBarrierValue）。
  func NewBarrier(names ...string) *Barrier
  // Names 返回期望到达名集合的排序副本（用于错误信息与测试断言）。
  func (c *Barrier) Names() []string
  // Channel 接口五方法：Update/Get/IsAvailable/Checkpoint/FromCheckpoint。
  // Consume 是接口外扩展点（对齐 Python BaseChannel.consume 可选方法），
  // 执行器经类型断言 interface{ Consume() bool } 调用。
  func (c *Barrier) Consume() bool
  ```
  后续任务依赖：Task 2 的 Compile 以 `channels.NewBarrier(parents...)` 注册 prototype；Task 3 的执行器对 channel 做 `interface{ Consume() bool }` 断言，并以 `rs.protos[key].(*channels.Barrier)` 判定 join key（因此 `NewBarrier` 必须返回具体类型 `*Barrier`，且 prototype 本身不被执行器 mutate——每个 run 经 `FromCheckpoint` 克隆）。

- [ ] **Step 1: 写失败测试** `langgraph/channels/barrier_test.go`，完整内容如下（复用 `channel_test.go` 的 `update`/`get`/`requireEmpty` helper）：

```go
package channels

import (
	"errors"
	"reflect"
	"testing"
)

// TestBarrierAccumulatesIdempotently mirrors NamedBarrierValue.update
// (named_barrier_value.py:56-67): arrivals accumulate, repeat arrivals are
// no-ops, and the channel reports change only on NEW arrivals.
func TestBarrierAccumulatesIdempotently(t *testing.T) {
	b := NewBarrier("a", "b")
	requireEmpty(t, b)

	if !update(t, b, "a") {
		t.Fatal("Update(a) changed = false, want true (new arrival)")
	}
	if update(t, b, "a") {
		t.Fatal("Update(a) changed = true, want false (idempotent)")
	}
	requireEmpty(t, b) // 1 of 2 arrivals: still unavailable

	if !update(t, b, "a", "b") { // mixed repeat + new in one batch
		t.Fatal("Update(a, b) changed = false, want true")
	}
	if !b.IsAvailable() {
		t.Fatal("IsAvailable() = false, want true (all names arrived)")
	}
	if got := get(t, b); got != nil {
		t.Fatalf("Get() = %v, want nil (Python get() returns None)", got)
	}
}

// TestBarrierRejectsUnknownName mirrors the InvalidUpdateError branch
// (named_barrier_value.py:63-66).
func TestBarrierRejectsUnknownName(t *testing.T) {
	b := NewBarrier("a", "b")
	_, err := b.Update([]any{"c"})
	var iu *InvalidUpdateError
	if !errors.As(err, &iu) {
		t.Fatalf("Update(c) error = %v, want *InvalidUpdateError", err)
	}
	if _, err := b.Update([]any{42}); !errors.As(err, &iu) {
		t.Fatalf("Update(42) error = %v, want *InvalidUpdateError (non-string)", err)
	}
	if b.IsAvailable() {
		t.Fatal("IsAvailable() = true after rejected update")
	}
}

// TestBarrierSurvivesStepBoundary: an empty Update (the applyWrites
// step-boundary notification) must NOT expire the barrier — unlike Ephemeral,
// arrivals persist across supersteps until Consume.
func TestBarrierSurvivesStepBoundary(t *testing.T) {
	b := NewBarrier("a", "b")
	update(t, b, "a", "b")
	if update(t, b /* no values: step boundary */) {
		t.Fatal("Update() changed = true, want false")
	}
	if !b.IsAvailable() {
		t.Fatal("IsAvailable() = false after step boundary, want true")
	}
}

// TestBarrierConsumeResets mirrors consume (named_barrier_value.py:77-81):
// no-op unless full; when full, resets so a loop round can re-accumulate.
func TestBarrierConsumeResets(t *testing.T) {
	b := NewBarrier("a", "b")
	if b.Consume() {
		t.Fatal("Consume() = true on partial barrier, want false")
	}
	update(t, b, "a", "b")
	if !b.Consume() {
		t.Fatal("Consume() = false on full barrier, want true")
	}
	requireEmpty(t, b)
	// Re-accumulation after reset (loop re-trigger).
	update(t, b, "b")
	requireEmpty(t, b)
	update(t, b, "a")
	if !b.IsAvailable() {
		t.Fatal("IsAvailable() = false after re-accumulation, want true")
	}
}

// TestBarrierCheckpointRoundTrip: Checkpoint omits an empty barrier, persists
// partial arrivals as a sorted []string (serde registry nameStrings,
// serde/json.go:40), and FromCheckpoint restores them — the "parent A
// arrived, parent B interrupted" pause/resume closure.
func TestBarrierCheckpointRoundTrip(t *testing.T) {
	proto := NewBarrier("a", "b")
	if _, ok := proto.Checkpoint(); ok {
		t.Fatal("Checkpoint() ok = true on empty barrier, want false (omit)")
	}

	b := NewBarrier("a", "b")
	update(t, b, "b")
	v, ok := b.Checkpoint()
	if !ok {
		t.Fatal("Checkpoint() ok = false on partial barrier, want true")
	}
	if !reflect.DeepEqual(v, []string{"b"}) {
		t.Fatalf("Checkpoint() = %v, want []string{\"b\"}", v)
	}

	restored := proto.FromCheckpoint(v)
	if restored.IsAvailable() {
		t.Fatal("restored IsAvailable() = true, want false (partial)")
	}
	if !update(t, restored.(*Barrier), "a") {
		t.Fatal("Update(a) on restored changed = false, want true")
	}
	if !restored.IsAvailable() {
		t.Fatal("restored IsAvailable() = false after completing arrival")
	}
	// The prototype is never mutated by FromCheckpoint.
	if proto.IsAvailable() {
		t.Fatal("prototype mutated by FromCheckpoint")
	}

	// Defensive branch: a JSON-decoded []any restore (serde without the
	// registry entry) still lands.
	def := proto.FromCheckpoint([]any{"a"})
	if !update(t, def.(*Barrier), "b") || !def.IsAvailable() {
		t.Fatal("[]any restore did not preserve arrival")
	}
	// FromCheckpoint(nil) yields an empty barrier that keeps the name set.
	empty := proto.FromCheckpoint(nil)
	requireEmpty(t, empty)
	if _, err := empty.Update([]any{"zzz"}); err == nil {
		t.Fatal("FromCheckpoint(nil) lost the name set: unknown name accepted")
	}
}
```

- [ ] **Step 2: 运行 `go test ./langgraph/channels/ -run TestBarrier -v`，确认全部失败**（`undefined: NewBarrier` 编译错误即预期失败形态）。
- [ ] **Step 3: 实现** `langgraph/channels/barrier.go`，完整内容：

```go
package channels

import (
	"fmt"
	"maps"
	"sort"
)

// Barrier mirrors Python's `NamedBarrierValue` channel
// (langgraph/channels/named_barrier_value.py): it accumulates a fixed set of
// string names — one per parent node of a waiting edge — and becomes
// available only once every name has arrived. The graph executor writes each
// parent's name when that parent commits, triggers the join child once the
// barrier is full, and calls Consume after the child commits so a looping
// graph re-arms the barrier for the next round.
//
// The barrier is control-plane state: the executor hides join channels from
// every user-visible state view (snapshot, node input, stream chunks), while
// Checkpoint/FromCheckpoint still persist partial arrivals so an interrupted
// run resumes with the arrival set intact.
type Barrier struct {
	names map[string]bool
	seen  map[string]bool
}

// NewBarrier returns a `Channel` equivalent to Python's
// `NamedBarrierValue(str, set(names))`. It returns the concrete *Barrier (not
// the Channel interface) so the executor can type-assert both the join-key
// marker and the optional Consume hook.
func NewBarrier(names ...string) *Barrier {
	b := &Barrier{names: make(map[string]bool, len(names)), seen: map[string]bool{}}
	for _, name := range names {
		b.names[name] = true
	}
	return b
}

// Names returns the barrier's expected arrival names, sorted.
func (c *Barrier) Names() []string {
	out := make([]string, 0, len(c.names))
	for name := range c.names {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// Update records arrivals. Each value must be one of the barrier's names
// (anything else is an *InvalidUpdateError, mirroring Python); repeat
// arrivals are idempotent no-ops. An empty update (the applyWrites
// step-boundary notification) never expires the barrier.
func (c *Barrier) Update(values []any) (bool, error) {
	changed := false
	for _, v := range values {
		name, ok := v.(string)
		if !ok || !c.names[name] {
			return false, &InvalidUpdateError{
				Channel: "NamedBarrierValue",
				Reason:  fmt.Sprintf("value %v is not one of the barrier names %v", v, c.Names()),
			}
		}
		if !c.seen[name] {
			c.seen[name] = true
			changed = true
		}
	}
	return changed, nil
}

// Get returns nil once all names have arrived (Python's get() returns None —
// the value carries no information, only availability matters) and
// ErrEmptyChannel before that.
func (c *Barrier) Get() (any, error) {
	if !c.IsAvailable() {
		return nil, ErrEmptyChannel
	}
	return nil, nil
}

// IsAvailable reports whether every name has arrived.
func (c *Barrier) IsAvailable() bool {
	return len(c.seen) == len(c.names)
}

// Checkpoint persists the partial arrival set as a sorted []string (the serde
// registry's "[]string" entry round-trips it); an empty barrier is omitted
// from the checkpoint, matching the other channels' empty-omit contract.
func (c *Barrier) Checkpoint() (any, bool) {
	if len(c.seen) == 0 {
		return nil, false
	}
	out := make([]string, 0, len(c.seen))
	for name := range c.seen {
		out = append(out, name)
	}
	sort.Strings(out)
	return out, true
}

// FromCheckpoint returns a fresh barrier with the receiver's name set and the
// checkpointed arrivals. The receiver (a registered prototype) is never
// mutated. []any is accepted defensively for JSON-decoded checkpoints that
// lost the []string registry typing.
func (c *Barrier) FromCheckpoint(value any) Channel {
	b := &Barrier{names: maps.Clone(c.names), seen: map[string]bool{}}
	switch v := value.(type) {
	case nil:
	case []string:
		for _, name := range v {
			b.seen[name] = true
		}
	case []any:
		for _, item := range v {
			if name, ok := item.(string); ok {
				b.seen[name] = true
			}
		}
	}
	return b
}

// Consume resets a full barrier for the next loop round, mirroring Python's
// `NamedBarrierValue.consume`. It is deliberately NOT part of the Channel
// interface — the executor reaches it via an `interface{ Consume() bool }`
// assertion, the same shape as Python's optional `BaseChannel.consume`.
func (c *Barrier) Consume() bool {
	if !c.IsAvailable() {
		return false
	}
	c.seen = map[string]bool{}
	return true
}
```

- [ ] **Step 4: 门禁 PASS**：`go build ./... && go vet ./... && go test ./langgraph/channels/ -v`（新测试全绿，现有 channel 测试不回归）。
- [ ] **Step 5: 提交**
  ```
  git add langgraph/channels/barrier.go langgraph/channels/barrier_test.go
  git commit -m "feat(langgraph/channels): Barrier channel (NamedBarrierValue port)"
  ```

---

### Task 2: builder `AddJoinEdge` + Compile 注册 join 元数据

**Files:**
- Modify: `langgraph/graph/graph.go` — `StateGraph` 结构体（:91-99）加 `joinEdges` 字段；新增 `AddJoinEdge`/`joinKey`；`Compile`（:291-346）加 join 校验与注册；`CompiledGraph` 结构体（:350-362）加 `joins`/`joinsByParent` 字段
- Test: `langgraph/graph/join_test.go`（本任务创建，只放 builder 测试；Task 3/4 续写同文件）

**Interfaces:**
- Consumes: `channels.NewBarrier`（Task 1）；现有 `setErr`（`graph.go:112-116`）、`types.START`/`types.END`。
- Produces:
  ```go
  // AddJoinEdge 注册 waiting edge：from 全部完成后 to 恰好触发一次。
  func (g *StateGraph) AddJoinEdge(from []string, to string) *StateGraph
  // joinKey 是 waiting edge 的 barrier channel key（"join:a+b:c"）。
  func joinKey(parents []string, child string) string

  type joinMeta struct {
      key     string   // barrier channel key
      parents []string // AddJoinEdge 顺序的父节点名
      child   string   // 满员后触发的子节点
  }
  // CompiledGraph 新字段（Task 3 消费）：
  //   joins         []joinMeta
  //   joinsByParent map[string][]string // parent -> 其参与的 barrier keys
  ```

- [ ] **Step 1: 写失败测试** `langgraph/graph/join_test.go`：

```go
package graph

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func joinNoop(_ context.Context, _ map[string]any) (any, error) { return nil, nil }

// joinBuilderBase returns a builder with nodes a, b, c registered and a as
// entry point, so AddJoinEdge validation failures are the only Compile errors.
func joinBuilderBase() *StateGraph {
	g := NewStateGraph()
	g.AddNode("a", joinNoop)
	g.AddNode("b", joinNoop)
	g.AddNode("c", joinNoop)
	g.AddEdge(types.START, "a")
	return g
}

// TestAddJoinEdgeValidation covers the call-time (count/dup/reserved-name)
// and Compile-time (node existence, duplicate join channel) checks. Python
// accepts a single-element start tuple and silently set-dedups (state.py:956);
// Go deliberately tightens both to errors (spec M6 documented divergence), and
// rejects END as a join child where Python allows it (state.py:963-964).
func TestAddJoinEdgeValidation(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(g *StateGraph)
		wantErr string
	}{
		{"zero parents", func(g *StateGraph) { g.AddJoinEdge(nil, "c") }, "at least 2"},
		{"single parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a"}, "c") }, "at least 2"},
		{"duplicate parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "a"}, "c") }, "duplicate join parent"},
		{"START parent", func(g *StateGraph) { g.AddJoinEdge([]string{types.START, "a"}, "c") }, "invalid join parent"},
		{"END parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a", types.END}, "c") }, "invalid join parent"},
		{"START child", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "b"}, types.START) }, "invalid join child"},
		{"END child", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "b"}, types.END) }, "invalid join child"},
		{"unknown parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "zzz"}, "c") }, "not a registered node"},
		{"unknown child", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "b"}, "zzz") }, "not a registered node"},
		{"duplicate join edge", func(g *StateGraph) {
			g.AddJoinEdge([]string{"a", "b"}, "c")
			g.AddJoinEdge([]string{"a", "b"}, "c")
		}, "duplicate join channel"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := joinBuilderBase()
			tc.mutate(g)
			_, err := g.Compile()
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Compile() error = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}

// TestCompileJoinRegistersBarrierChannel: Compile registers a Barrier
// prototype under "join:a+b:c" and the parent->barrier index, without
// polluting the builder's own channelProtos (Compile stays re-entrant).
func TestCompileJoinRegistersBarrierChannel(t *testing.T) {
	g := joinBuilderBase()
	g.AddJoinEdge([]string{"a", "b"}, "c")
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	proto, ok := cg.channelProtos["join:a+b:c"]
	if !ok {
		t.Fatal("channelProtos missing join:a+b:c")
	}
	b, ok := proto.(*channels.Barrier)
	if !ok {
		t.Fatalf("join:a+b:c proto = %T, want *channels.Barrier", proto)
	}
	if got := b.Names(); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("Names() = %v, want [a b]", got)
	}
	if got := cg.joinsByParent["a"]; len(got) != 1 || got[0] != "join:a+b:c" {
		t.Fatalf("joinsByParent[a] = %v", got)
	}
	if got := cg.joinsByParent["b"]; len(got) != 1 || got[0] != "join:a+b:c" {
		t.Fatalf("joinsByParent[b] = %v", got)
	}
	if _, polluted := g.channelProtos["join:a+b:c"]; polluted {
		t.Fatal("builder channelProtos polluted by Compile")
	}
	if _, err := g.Compile(); err != nil {
		t.Fatalf("second Compile() error = %v (join registration must be re-entrant)", err)
	}
}
```

- [ ] **Step 2: 运行 `go test ./langgraph/graph/ -run 'TestAddJoinEdge|TestCompileJoin' -v`，确认编译失败**（`AddJoinEdge` 未定义）。
- [ ] **Step 3: 实现**。`graph.go` 的 import 块（:42-54）加 `"slices"` 与 `"strings"`。`StateGraph` 结构体加字段与文档：

```go
type StateGraph struct {
	nodes         map[string]NodeFunc
	policies      map[string]NodePolicies
	channelProtos map[string]channels.Channel
	edges         map[string][]string
	conditional   map[string]ConditionalEdge
	// joinEdges holds the waiting edges registered via AddJoinEdge, in
	// registration order (Compile turns each into a joinMeta + barrier
	// channel prototype).
	joinEdges []joinEdge
	entry     string
	err       error
}

// joinEdge is one AddJoinEdge waiting edge: the child triggers once ALL
// parents have committed.
type joinEdge struct {
	parents []string
	child   string
}

// joinMeta is a compiled waiting edge (see StateGraph.AddJoinEdge).
type joinMeta struct {
	key     string   // barrier channel key ("join:a+b:c")
	parents []string // parent node names, in AddJoinEdge order
	child   string   // node dispatched once the barrier fills
}

// joinKey is the barrier channel key for a waiting edge, mirroring Python's
// `join:a+b:c` naming (langgraph/graph/state.py:1547): parent names in
// AddJoinEdge order, joined with "+".
func joinKey(parents []string, child string) string {
	return "join:" + strings.Join(parents, "+") + ":" + child
}
```

`AddJoinEdge`（放在 `AddEdge` 之后，`graph.go:186` 之后）：

```go
// AddJoinEdge adds a waiting edge: the child node is triggered exactly once
// after ALL parents have committed (Python's `add_edge((a, b), c)`, backed by
// a `NamedBarrierValue` channel). The parents' arrivals accumulate in a
// barrier channel named `join:a+b:c` registered at Compile time; the barrier
// is control-plane state and never appears in node inputs, snapshots, or
// stream output.
//
// WARNING (Python parity, OR semantics): a plain edge, conditional edge,
// types.Send, or Command.Goto targeting the join child BYPASSES the barrier
// and triggers the child directly. Mixing both edge kinds into one child can
// run it multiple times — that is Python's documented behavior, not a bug.
//
// Documented divergences from Python (state.py:956-966): Go requires at
// least 2 parents (Python accepts a single-element tuple as a degenerate
// waiting edge), rejects duplicate parents (Python silently set-dedups), and
// rejects types.END as the child (Python allows `add_edge((a, b), END)`).
// Node-existence is validated at Compile time, consistent with AddEdge.
func (g *StateGraph) AddJoinEdge(from []string, to string) *StateGraph {
	if len(from) < 2 {
		g.setErr(fmt.Errorf("graph: join edge into %q requires at least 2 parents, got %d", to, len(from)))
		return g
	}
	seen := make(map[string]bool, len(from))
	for _, name := range from {
		if name == "" || name == types.START || name == types.END {
			g.setErr(fmt.Errorf("graph: invalid join parent name %q", name))
			return g
		}
		if seen[name] {
			g.setErr(fmt.Errorf("graph: duplicate join parent %q", name))
			return g
		}
		seen[name] = true
	}
	if to == "" || to == types.START || to == types.END {
		g.setErr(fmt.Errorf("graph: invalid join child name %q", to))
		return g
	}
	g.joinEdges = append(g.joinEdges, joinEdge{parents: slices.Clone(from), child: to})
	return g
}
```

`Compile` 中，在 conditional 校验循环（:315-319）之后、`options` 构造之前插入节点存在性校验；并把 CompiledGraph 构造改为注册 join：

```go
	for _, je := range g.joinEdges {
		for _, p := range je.parents {
			if _, ok := g.nodes[p]; !ok {
				return nil, fmt.Errorf("graph: join edge parent %q is not a registered node", p)
			}
		}
		if _, ok := g.nodes[je.child]; !ok {
			return nil, fmt.Errorf("graph: join edge child %q is not a registered node", je.child)
		}
	}
```

CompiledGraph 构造段（:333-345）替换为（`channelProtos` 必须 clone 后再注册 barrier——否则 builder 被污染、二次 Compile 撞 key）：

```go
	// Register one barrier channel prototype per waiting edge (Python's
	// attach_edge, state.py:1546-1561). The clone keeps the builder's own
	// channelProtos untouched so Compile stays re-entrant.
	channelProtos := maps.Clone(g.channelProtos)
	joins := make([]joinMeta, 0, len(g.joinEdges))
	joinsByParent := map[string][]string{}
	for _, je := range g.joinEdges {
		key := joinKey(je.parents, je.child)
		if _, exists := channelProtos[key]; exists {
			return nil, fmt.Errorf("graph: duplicate join channel %q (identical AddJoinEdge calls, or a user-registered AddChannel/AddReducer collision)", key)
		}
		channelProtos[key] = channels.NewBarrier(je.parents...)
		joins = append(joins, joinMeta{key: key, parents: je.parents, child: je.child})
		for _, p := range je.parents {
			joinsByParent[p] = append(joinsByParent[p], key)
		}
	}

	return &CompiledGraph{
		nodes:           g.nodes,
		policies:        g.policies,
		channelProtos:   channelProtos,
		edges:           g.edges,
		conditional:     g.conditional,
		joins:           joins,
		joinsByParent:   joinsByParent,
		entry:           g.entry,
		checkpointer:    options.checkpointer,
		cache:           options.cache,
		recursionLimit:  options.recursionLimit,
		interruptBefore: options.interruptBefore,
		interruptAfter:  options.interruptAfter,
	}, nil
```

`CompiledGraph` 结构体加两字段：

```go
	// joins/joinsByParent are the compiled waiting edges (empty for graphs
	// without AddJoinEdge): the executor appends an implicit barrier write to
	// each parent task's commit batch and dispatches join children from the
	// commit path (see run).
	joins         []joinMeta
	joinsByParent map[string][]string
```

- [ ] **Step 4: 门禁 PASS**：`go build ./... && go vet ./... && go test ./langgraph/graph/ -run 'TestAddJoinEdge|TestCompileJoin' -v`，随后全量 `go test ./...`（现有测试零改动通过——此任务不触碰执行路径）。
- [ ] **Step 5: 提交**
  ```
  git add langgraph/graph/graph.go langgraph/graph/join_test.go
  git commit -m "feat(langgraph/graph): StateGraph.AddJoinEdge builder + compile-time barrier registration"
  ```

---

### Task 3: 执行器——隐式 write、恰好一次触发、Consume 复位、join key 过滤

**Files:**
- Modify: `langgraph/graph/graph.go` — 注入隐式 write（:882 后）、emission 过滤（:895-896、:711、:639）、Consume + 触发（:948 后、:967 后）、`staticNext`（:1058-1061）、`isJoinKey`/`dropJoinKeys` 辅助
- Modify: `langgraph/graph/state.go` — `snapshot`（:80-88）过滤、`applyWrites`（:119-187）bool 语义收窄、`isJoinKey` 自由函数
- Modify: `langgraph/graph/snapshot.go` — `snapshotFromTuple`（:140-153）改为方法并过滤，调用点 :53/:73 同步
- Test: `langgraph/graph/join_test.go` — 追加 `TestJoinBasicTrigger` 冒烟测试
- 不改：`subgraph.go`（子图运行自己的 CompiledGraph，join 元数据天然隔离）、`resume.go`（`completedTaskWrites`/`planResume` 对 join key 透明——它们把 join write 当普通 channel write 持久化/重放，这正是设计意图）

**Interfaces:**
- Consumes: Task 1 的 `*channels.Barrier`（`Consume` 断言、`rs.protos[key].(*channels.Barrier)` 判定）；Task 2 的 `CompiledGraph.joins`/`joinsByParent`/`joinMeta`。
- Produces（包内辅助，Task 4 测试间接覆盖）：
  ```go
  // state.go
  func isJoinKey(protos map[string]channels.Channel, key string) bool
  // graph.go
  func (g *CompiledGraph) dropJoinKeys(m map[string]any) map[string]any
  ```

设计要点（spec M6 执行器节逐条落实）：

1. **隐式 write 搭进任务 update 批次**：注入点在 cache store pass（:871-882）之后、结果收集循环（:884）之前。由此 ①走 `applyWrites` 的同一版本递增；②中断路径 `completedTaskWrites(o.update, o.cmd)`（:924）自动带上 join write，持久化为 pending writes；③cache 条目不含 join key（store 在注入之前），cache 命中的父任务在注入循环里照常补写 arrival。中断/出错的任务不注入（Python：父中断则其 ChannelWrite 不执行）。
2. **恰好一次**：`applyWrites` 提交后，对每个 `joinMeta`：barrier channel `IsAvailable()` 且 `rs.seen[child][key] < rs.versions[key]`（版本簿记未见过）→ 追加 `task{node: child}` 进 `nextTasks` 并立即记账 `rs.seen[child][key] = v`。同超步多父 writes 被 `applyWrites` 聚成一次 channel `Update`、只 bump 一个版本，天然一次；dispatch 时记账使 interrupt_before(child) 暂停 checkpoint 的 VersionsSeen 与 Next 自洽。触发循环插在 staticNext 循环（:953-967）之后、`interrupt_after` 检查（:975）之前——暂停 checkpoint 的 Next 因此包含 barrier 触发的子节点。
3. **Consume 复位**：同一提交块内、触发判定之前。关键不变量：**只有被 barrier 触发而运行的子节点才能 Consume 该 barrier**（Python 中 `consume` 只对任务实际读取过的 channel 调用；Send/普通边触发的 PUSH 任务没有读 join channel，不得消费它）。判定用版本簿记：barrier 触发 dispatch 时已记账 `seen[child][key] = v`，而子任务提交时 `applyWrites` 第 1 步会把 `seen[child][key]` 重写为**本超步写入前**的 barrier 版本——于是 `seen[child][key] >= versions[key]` 当且仅当子节点看到的是当前满员版本（即被 barrier 触发）；子节点经 Send/普通边与父同超步运行时，父的到达把 barrier 顶到新版本，`seen < versions`，Consume 跳过、barrier 保持待命，随后触发扫描正常 dispatch（`TestJoinSendBypassesBarrier` 锁死此路径）。非满 barrier 上 Consume 本身是 no-op；满员 barrier 上父的幂等重到达会被 consume 吞掉——Python 同样如此，不"改进"。
4. **过滤**：`isJoinKey` 以 prototype 类型断言判定控制面 key。四处过滤：`rs.snapshot()`（覆盖节点输入、`Result.Values`、values chunk、pause chunk、staticNext 路由输入）；`emitUpdate`/`debugTaskResult` 实参（:895-896）；resume 重放的 `emitUpdate`（:711）；`debugCheckpoint` 的 values 实参（:639——checkpoint 本体 `rs.channelValues()` 必须保留 join key，只过滤 emission 副本）。外加 `snapshotFromTuple`（GetState/GetStateHistory 的 Values）。
5. **`applyWrites` bool 语义收窄**：该返回值只被 values-emission 门控使用（:683/:698/:995），改为"至少一个**非 join** channel 版本递增"——Python 的 values 门控是 `updated_channels ∩ output_keys`，join channel 不在 output_keys，barrier-only 变化的超步不应产 values chunk。无 join 边时行为逐点不变（零回归）。
6. **`staticNext`**：节点无普通/条件出边但 `joinsByParent` 非空 → 返回 `nil, nil`（其后继由 barrier 触发），不再报 "no outgoing edge"。无 join 的图该分支不可达，报错路径不变。

- [ ] **Step 1: 写失败冒烟测试**（追加到 `langgraph/graph/join_test.go`）：

```go
// TestJoinBasicTrigger: two parents in the same superstep fill the barrier;
// the child runs exactly once; no join key leaks into the result.
func TestJoinBasicTrigger(t *testing.T) {
	var childCalls int
	g := NewStateGraph()
	g.AddNode("entry", joinNoop)
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"a_done": true}, nil
	})
	g.AddNode("b", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"b_done": true}, nil
	})
	g.AddNode("c", func(_ context.Context, state map[string]any) (any, error) {
		childCalls++
		if _, leaked := state["join:a+b:c"]; leaked {
			t.Error("join key leaked into node input")
		}
		return map[string]any{"c_done": true}, nil
	})
	g.AddEdge(types.START, "entry")
	g.AddEdge("entry", "a")
	g.AddEdge("entry", "b")
	g.AddJoinEdge([]string{"a", "b"}, "c")
	g.AddEdge("c", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if childCalls != 1 {
		t.Fatalf("childCalls = %d, want 1 (barrier triggers exactly once)", childCalls)
	}
	for _, k := range []string{"a_done", "b_done", "c_done"} {
		if res.Values[k] != true {
			t.Fatalf("Values[%q] = %v, want true (Values = %v)", k, res.Values[k], res.Values)
		}
	}
	for k := range res.Values {
		if strings.HasPrefix(k, "join:") {
			t.Fatalf("join key %q leaked into Result.Values", k)
		}
	}
}
```

- [ ] **Step 2: 运行 `go test ./langgraph/graph/ -run TestJoinBasicTrigger -v`，确认失败**（`c` 从不被触发：`childCalls = 0`）。
- [ ] **Step 3: 实现**，按以下六个编辑块：

**(a) `state.go`——`isJoinKey` 自由函数 + `snapshot` 过滤 + `applyWrites` bool 收窄。** 在 `cloneSeen` 前加：

```go
// isJoinKey reports whether key's registered channel prototype is a join
// *channels.Barrier — control-plane state hidden from every user-visible
// state view (snapshots, node inputs, stream chunks). Checkpoint persistence
// (channelValues) deliberately does NOT consult it.
func isJoinKey(protos map[string]channels.Channel, key string) bool {
	_, ok := protos[key].(*channels.Barrier)
	return ok
}
```

`snapshot`（:80-88）改为：

```go
// snapshot returns the externally visible graph state: the current value of
// every available channel, minus join barrier channels (control plane; Python
// likewise excludes them from output_keys). Keys whose channel is empty
// (never written, or expired) are absent.
func (rs *runState) snapshot() map[string]any {
	out := make(map[string]any, len(rs.channels))
	for key, ch := range rs.channels {
		if isJoinKey(rs.protos, key) {
			continue
		}
		if v, err := ch.Get(); err == nil {
			out[key] = v
		}
	}
	return out
}
```

`applyWrites` 两处版本 bump（:163-166 与 :181-184）都改为 join-aware：

```go
		if changed {
			rs.versions[key] = nextVersion
			if !isJoinKey(rs.protos, key) {
				anyChanged = true
			}
		}
```

并更新 `applyWrites` doc comment 末尾段（:116-118）为：

```go
// The bool result reports whether at least one NON-JOIN channel version was
// bumped; the stream emission layer gates `values` chunks on it (mirroring
// Python's `updated_channels ∩ output_keys` gate — join barrier channels are
// not in output_keys, so a superstep that only moved a barrier emits no
// values chunk). Join channels still get their version bump.
```

**(b) `graph.go`——隐式 write 注入。** 在 cache store pass 循环（:871-882）之后、`// Collect outcomes...` 注释（:884）之前插入：

```go
			// Join barrier arrivals: each completed parent task implicitly
			// writes its own name to every waiting-edge barrier it feeds
			// (Python attaches a ChannelWrite per parent at compile time,
			// state.py:1558-1561). Injecting into the task's update batch
			// (rather than a side channel) gives the write the same
			// applyWrites version bump AND the interrupt path's
			// completedTaskWrites persistence below — the "parent A arrived,
			// parent B interrupted, resume replays A's arrival" closure.
			// Interrupted/errored tasks write nothing (Python: the parent's
			// ChannelWrite never executes). Cache entries were stored BEFORE
			// this pass, so they stay free of control-plane keys; a cache-hit
			// parent still records its arrival here.
			for i, t := range active {
				if len(g.joinsByParent[t.node]) == 0 || outcomes[i].err != nil || outcomes[i].interrupted != nil {
					continue
				}
				if outcomes[i].update == nil {
					outcomes[i].update = map[string]any{}
				}
				for _, key := range g.joinsByParent[t.node] {
					outcomes[i].update[key] = t.node
				}
			}
```

**(c) `graph.go`——emission 过滤。** 收集循环（:889-897）改为过滤后发射（内部路径——`completedTaskWrites`、`applyWrites`——仍用未过滤的 `o.update`）：

```go
		var interrupts []types.Interrupt
		for i, o := range outcomes {
			var taskInterrupts []types.Interrupt
			if o.interrupted != nil {
				taskInterrupts = []types.Interrupt{*o.interrupted}
				interrupts = append(interrupts, *o.interrupted)
			}
			pub := g.dropJoinKeys(o.update)
			em.debugTaskResult(rs.step+1, active[i], pub, o.err, taskInterrupts)
			em.emitUpdate(active[i].node, pub)
		}
```

resume 重放 emission（:710-712）：

```go
	for _, w := range replayWrites {
		em.emitUpdate(w.node, g.dropJoinKeys(w.update))
	}
```

`save` 闭包里的 debugCheckpoint（:639）：

```go
		em.debugCheckpoint(md, cfg, *currentCfg, g.dropJoinKeys(rs.channelValues()), next)
```

并在 `staticNext` 前加辅助：

```go
// dropJoinKeys strips join barrier channels from a task update or
// channel-value map destined for user-visible emission (updates chunks,
// debug task_result/checkpoint payloads). Internal paths — applyWrites,
// completedTaskWrites, checkpoint saves — always keep the full map. Returns
// the input unchanged when the graph has no join edges.
func (g *CompiledGraph) dropJoinKeys(m map[string]any) map[string]any {
	if len(g.joins) == 0 || len(m) == 0 {
		return m
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		if !isJoinKey(g.channelProtos, k) {
			out[k] = v
		}
	}
	return out
}
```

**(d) `graph.go`——Consume + 触发。** 在 `rs.step++`（:949）之后插入 Consume 块：

```go
			// Join children that CONSUMED their barrier this superstep reset it
			// (Python calls NamedBarrierValue.consume only for channels the
			// task was actually triggered by). The seen>=versions check is
			// exactly that: a barrier-dispatched child carries the barrier's
			// current version in versions-seen (applyWrites re-records the
			// pre-write view at commit), while a child that ran via a Send/
			// plain edge in the superstep that FILLED the barrier holds an
			// older mark — its barrier stays armed so the trigger scan below
			// still dispatches the barrier task (OR semantics;
			// TestJoinSendBypassesBarrier locks this in). Consume itself is a
			// no-op on a non-full barrier. Must run BEFORE the trigger scan,
			// or a consumed barrier would re-dispatch its child.
			if len(g.joins) > 0 {
				ran := make(map[string]bool, len(active))
				for _, t := range active {
					ran[t.node] = true
				}
				for _, jm := range g.joins {
					if !ran[jm.child] {
						continue
					}
					if rs.seen[jm.child][jm.key] < rs.versions[jm.key] {
						continue // child ran via Send/edge, not via this barrier
					}
					if b, ok := rs.channels[jm.key].(interface{ Consume() bool }); ok {
						b.Consume()
					}
				}
			}
```

在 staticNext 的 nextTasks 循环（:953-967）之后、`// interrupt_after` 注释（:969）之前插入触发扫描：

```go
			// Waiting-edge triggers: a barrier filled by this superstep's
			// writes dispatches its child EXACTLY once. Versions-seen is the
			// dedup ledger: a barrier version the child already saw never
			// re-dispatches it (same-superstep multi-parent commits fold into
			// one channel update = one version bump = one dispatch). The mark
			// is recorded at dispatch time so a pause checkpoint's
			// VersionsSeen stays consistent with its planned Next. Send/
			// Command.Goto/normal-edge tasks for the same child are separate
			// entries in nextTasks on purpose (Python's OR semantics) — they
			// must NOT be deduped against this barrier task.
			for _, jm := range g.joins {
				ch, ok := rs.channels[jm.key]
				if !ok || !ch.IsAvailable() {
					continue
				}
				v := rs.versions[jm.key]
				if rs.seen[jm.child][jm.key] >= v {
					continue
				}
				if rs.seen[jm.child] == nil {
					rs.seen[jm.child] = map[string]int64{}
				}
				rs.seen[jm.child][jm.key] = v
				nextTasks = append(nextTasks, task{node: jm.child})
			}
```

**(e) `graph.go`——`staticNext` join-only 父分支**（:1058-1061 处）：

```go
	if edges, ok := g.edges[nodeName]; ok && len(edges) > 0 {
		return resolveDestinations(To(edges...))
	}
	if len(g.joinsByParent[nodeName]) > 0 {
		// Waiting-edge-only parent: its successors are dispatched by the
		// barrier trigger in the commit path, not per-parent edges.
		return nil, nil
	}
	return nil, fmt.Errorf("graph: node %q has no outgoing edge (add AddEdge/AddConditionalEdges, or return a *types.Command with Goto)", nodeName)
```

**(f) `snapshot.go`——`snapshotFromTuple` 改方法 + 过滤。** :53 与 :73 的调用点改为 `g.snapshotFromTuple(...)`，函数本体（:140-153）改为：

```go
// snapshotFromTuple projects a checkpoint tuple into its StateSnapshot view:
// the channel values (minus join barrier channels — control plane, excluded
// from Python's output_keys as well), the planned next nodes, and any pending
// interrupts (reconstructed from ReservedInterrupt pending writes).
func (g *CompiledGraph) snapshotFromTuple(tup *checkpoint.Tuple) StateSnapshot {
	values := maps.Clone(tup.Checkpoint.ChannelValues)
	for key := range values {
		if isJoinKey(g.channelProtos, key) {
			delete(values, key)
		}
	}
	next := make([]string, 0, len(tup.Checkpoint.Next))
	for _, pt := range tup.Checkpoint.Next {
		next = append(next, pt.Node)
	}
	return StateSnapshot{
		Values:       values,
		Next:         next,
		Config:       tup.Config,
		Metadata:     tup.Metadata,
		CreatedAt:    tup.Checkpoint.TS,
		ParentConfig: tup.ParentConfig,
		Interrupts:   interruptsFromWrites(tup.PendingWrites),
	}
}
```

- [ ] **Step 4: 门禁 PASS**：`go build ./... && go vet ./... && go test ./...` 全绿（`TestJoinBasicTrigger` 通过；全部现有测试——含 stream/snapshot/resume/cache/subgraph 与 `langchain/agents`——零修改通过）。另跑 `make test-sqlite` 确认嵌套 module 不回归。
- [ ] **Step 5: 提交**
  ```
  git add langgraph/graph/graph.go langgraph/graph/state.go langgraph/graph/snapshot.go langgraph/graph/join_test.go
  git commit -m "feat(langgraph/graph): barrier join execution — implicit parent writes, exactly-once trigger, consume reset, join-key filtering"
  ```

---

### Task 4: Python waiting-edge 测试移植 + Go 新增用例

**Files:**
- Test: `langgraph/graph/join_test.go`（追加；全部测试）

**Interfaces:**
- Consumes: Task 1–3 全部产出；`checkpoint.NewMemorySaver()`、`checkpoint.NewInMemoryCache()`、`NodePolicies{Cache: &CachePolicy{}}`、`WithCheckpointer`/`WithCache`/`WithInterruptAfter`、`Interrupt(ctx, value)`（`graph.go:1168`）、`Options{ThreadID, Resume}`、`Stream`/`StreamOptions`、`GetState`/`GetStateHistory`。
- Produces: 无新代码产物（纯测试任务）。

移植映射（Python `libs/langgraph/tests/test_pregel.py`）：

| Go 测试 | Python 基线 | 覆盖点 |
|---|---|---|
| `TestJoinSameSuperstepExactlyOnce` | `test_simple_multi_edge`（:3059-3106，`add_edge(["up","side"],"down")` 在 :3085） | 同超步多父→子恰好一次 |
| `TestJoinFanOutWaitingEdge` | `test_in_one_fan_out_state_graph_waiting_edge`（:1953-2086） | 跨超步等待齐后一次；updates chunk 序列；interrupt_after→resume |
| `TestJoinWaitingEdgePlusRegularEdge` | `..._plus_regular`（:2710-2804） | OR 语义：普通边直达 + barrier 触发，qa 恰好两次 |
| `TestJoinLoopReset` | `..._waiting_edge_multiple`（:2808-2921） | 循环中 Consume 复位重触发；cache 变体 |
| `TestJoinSendBypassesBarrier` | 绕过语义（spec M6；`attach_edge` vs Send，`state.py:1537-1545`） | Send 到 join 子节点绕过 barrier；PUSH 任务不被去重 |
| `TestJoinThreeParents` | Go 扩展（Python 无三父用例） | `join:a+b+c` key、三父齐后一次 |
| `TestJoinParentInterruptResume` | Go 新增 | 父中断→resume→补写触发；已完成父不重跑 |
| `TestJoinCheckpointPartialArrival` | Go 新增 | checkpoint 保留部分到达（[]string{"a"}）；snapshot 不泄漏 |
| `TestJoinKeyNotLeaked` | Go 新增（spec 风险项） | join key 不进 snapshot/节点输入/stream values+updates/debug |

顺序与时序说明：Go 执行器在超步内按 deterministic task order 收集 outcomes（M3 已文档化分歧），因此 updates chunk 顺序与 Python 的 sleep 编排不同——断言用 Go 的确定序（下方测试代码已按此写死），这不是语义偏差。docs 累积用 `sortedAddReducer`（对齐 Python 用例里的 `sorted_add`）使最终值逐点一致。

- [ ] **Step 1: 写失败测试**——向 `langgraph/graph/join_test.go` 追加（import 块需补 `"encoding/json"`、`"fmt"`、`"reflect"`、`"sort"`、`"sync"`、`"github.com/projanvil/langchain-golang/langgraph/checkpoint"`）：

```go
// sortedAddReducer mirrors the sorted_add reducer used by the Python
// waiting-edge tests (test_pregel.py:1956-1961): append then sort, so final
// accumulated values match Python's assertions exactly.
func sortedAddReducer(existing, update any) (any, error) {
	out, err := channels.AppendSliceReducer(existing, update)
	if err != nil {
		return nil, err
	}
	s, _ := out.([]string)
	sort.Strings(s)
	return s, nil
}

// newWaitingEdgeGraph builds the fan-out graph of test_pregel.py:1953:
// rewrite_query -> analyzer_one -> retriever_one, rewrite_query ->
// retriever_two, [retriever_one, retriever_two] -> qa -> END.
func newWaitingEdgeGraph(qaCalls *int) *StateGraph {
	g := NewStateGraph()
	g.AddReducer("docs", sortedAddReducer)
	g.AddNode("rewrite_query", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"query": "query: " + state["query"].(string)}, nil
	})
	g.AddNode("analyzer_one", func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"query": "analyzed: " + state["query"].(string)}, nil
	})
	g.AddNode("retriever_one", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc1", "doc2"}}, nil
	})
	g.AddNode("retriever_two", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc3", "doc4"}}, nil
	})
	g.AddNode("qa", func(_ context.Context, state map[string]any) (any, error) {
		*qaCalls++
		docs, _ := state["docs"].([]string)
		return map[string]any{"answer": strings.Join(docs, ",")}, nil
	})
	g.AddEdge(types.START, "rewrite_query")
	g.AddEdge("rewrite_query", "analyzer_one")
	g.AddEdge("analyzer_one", "retriever_one")
	g.AddEdge("rewrite_query", "retriever_two")
	g.AddJoinEdge([]string{"retriever_one", "retriever_two"}, "qa")
	g.AddEdge("qa", types.END)
	return g
}

var joinWantValues = map[string]any{
	"query":  "analyzed: query: what is weather in sf",
	"docs":   []string{"doc1", "doc2", "doc3", "doc4"},
	"answer": "doc1,doc2,doc3,doc4",
}

// TestJoinSameSuperstepExactlyOnce ports test_simple_multi_edge
// (test_pregel.py:3059): up and side complete in the same superstep; down
// (join child of [up, side]) still runs exactly once.
func TestJoinSameSuperstepExactlyOnce(t *testing.T) {
	var downCalls int
	g := NewStateGraph()
	g.AddReducer("my_key", func(existing, update any) (any, error) {
		ex, _ := existing.(string)
		u, _ := update.(string)
		return ex + u, nil
	})
	g.AddNode("up", joinNoop)
	g.AddNode("side", joinNoop)
	g.AddNode("other", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"my_key": "_more"}, nil
	})
	g.AddNode("down", func(_ context.Context, _ map[string]any) (any, error) {
		downCalls++
		return nil, nil
	})
	g.AddEdge(types.START, "up")
	g.AddEdge("up", "side")
	g.AddEdge("up", "other")
	g.AddJoinEdge([]string{"up", "side"}, "down")
	g.AddEdge("down", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), map[string]any{"my_key": "hello"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Values["my_key"] != "hello_more" {
		t.Fatalf("my_key = %v, want %q", res.Values["my_key"], "hello_more")
	}
	if downCalls != 1 {
		t.Fatalf("downCalls = %d, want 1 (same-superstep parents trigger once)", downCalls)
	}
}

// TestJoinFanOutWaitingEdge ports test_pregel.py:1953: the parents complete
// in DIFFERENT supersteps (retriever_two one step before retriever_one); qa
// waits for both, runs once, and sees the fully accumulated docs. The
// interrupt_after subtest mirrors :2018-2036.
func TestJoinFanOutWaitingEdge(t *testing.T) {
	var qaCalls int
	g := newWaitingEdgeGraph(&qaCalls)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), map[string]any{"query": "what is weather in sf"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if !reflect.DeepEqual(res.Values, joinWantValues) {
		t.Fatalf("Values = %v, want %v", res.Values, joinWantValues)
	}
	if qaCalls != 1 {
		t.Fatalf("qaCalls = %d, want 1", qaCalls)
	}

	// updates chunks in Go's deterministic task order (documented M3
	// divergence from Python's as-they-finish timing).
	var updates []any
	for c, err := range cg.Stream(context.Background(), map[string]any{"query": "what is weather in sf"},
		StreamOptions{Modes: []StreamMode{StreamUpdates}}) {
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		updates = append(updates, c.Payload)
	}
	wantUpdates := []any{
		map[string]any{"rewrite_query": map[string]any{"query": "query: what is weather in sf"}},
		map[string]any{"analyzer_one": map[string]any{"query": "analyzed: query: what is weather in sf"}},
		map[string]any{"retriever_two": map[string]any{"docs": []string{"doc3", "doc4"}}},
		map[string]any{"retriever_one": map[string]any{"docs": []string{"doc1", "doc2"}}},
		map[string]any{"qa": map[string]any{"answer": "doc1,doc2,doc3,doc4"}},
	}
	if !reflect.DeepEqual(updates, wantUpdates) {
		t.Fatalf("updates = %v, want %v", updates, wantUpdates)
	}

	// interrupt_after(retriever_one): pause after the second parent commits,
	// resume runs qa exactly once (test_pregel.py:2018-2036).
	var qaCalls2 int
	g2 := newWaitingEdgeGraph(&qaCalls2)
	cg2, err := g2.Compile(WithCheckpointer(checkpoint.NewMemorySaver()), WithInterruptAfter("retriever_one"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()
	paused, err := cg2.InvokeWithOptions(ctx, map[string]any{"query": "what is weather in sf"}, Options{ThreadID: "1"})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("run1 Interrupts = %v, want 1", paused.Interrupts)
	}
	if qaCalls2 != 0 {
		t.Fatalf("qaCalls after pause = %d, want 0", qaCalls2)
	}
	done, err := cg2.InvokeWithOptions(ctx, nil, Options{ThreadID: "1"})
	if err != nil {
		t.Fatalf("resume error = %v", err)
	}
	if !reflect.DeepEqual(done.Values, joinWantValues) {
		t.Fatalf("resume Values = %v, want %v", done.Values, joinWantValues)
	}
	if qaCalls2 != 1 {
		t.Fatalf("qaCalls after resume = %d, want 1", qaCalls2)
	}
}

// TestJoinWaitingEdgePlusRegularEdge ports test_pregel.py:2710: an extra
// plain edge rewrite_query -> qa bypasses the barrier (OR semantics) — qa
// runs once early with empty docs and once via the barrier; "having been
// triggered before doesn't break the semantics of the named barrier".
func TestJoinWaitingEdgePlusRegularEdge(t *testing.T) {
	var qaCalls int
	var answers []string
	g := newWaitingEdgeGraph(&qaCalls)
	// Wrap qa to record every invocation's answer, in run order.
	qaFn := g.nodes["qa"]
	g.nodes["qa"] = func(ctx context.Context, state map[string]any) (any, error) {
		out, err := qaFn(ctx, state)
		if m, ok := out.(map[string]any); ok {
			answers = append(answers, m["answer"].(string))
		}
		return out, err
	}
	g.AddEdge("rewrite_query", "qa") // the Python test's "silly edge" (:2759)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), map[string]any{"query": "what is weather in sf"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if qaCalls != 2 {
		t.Fatalf("qaCalls = %d, want 2 (plain edge + barrier)", qaCalls)
	}
	if !reflect.DeepEqual(answers, []string{"", "doc1,doc2,doc3,doc4"}) {
		t.Fatalf("answers = %v, want [\"\" \"doc1,doc2,doc3,doc4\"]", answers)
	}
	if !reflect.DeepEqual(res.Values, joinWantValues) {
		t.Fatalf("Values = %v, want %v", res.Values, joinWantValues)
	}
}

// TestJoinLoopReset ports test_pregel.py:2808 (waiting_edge_multiple): the
// join sits inside a decider loop; after each trigger the barrier is consumed
// and re-arms, so round 2 re-triggers exactly once. The withCache variant
// mirrors Python's cache parametrization and additionally puts a cache policy
// on retriever_one (a join PARENT) so the cache-hit injection path
// (graph.go: arrival write appended to a cache-injected outcome) is covered.
func TestJoinLoopReset(t *testing.T) {
	for _, withCache := range []bool{false, true} {
		t.Run(fmt.Sprintf("withCache=%v", withCache), func(t *testing.T) {
			var rewriteCalls int
			g := NewStateGraph()
			g.AddReducer("docs", sortedAddReducer)
			rewrite := func(_ context.Context, state map[string]any) (any, error) {
				rewriteCalls++
				return map[string]any{"query": "query: " + state["query"].(string)}, nil
			}
			cachePolicy := NodePolicies{}
			if withCache {
				cachePolicy = NodePolicies{Cache: &CachePolicy{}}
			}
			g.AddNodeWithPolicies("rewrite_query", rewrite, cachePolicy)
			g.AddNode("analyzer_one", func(_ context.Context, state map[string]any) (any, error) {
				return map[string]any{"query": "analyzed: " + state["query"].(string)}, nil
			})
			g.AddNodeWithPolicies("retriever_one", func(_ context.Context, _ map[string]any) (any, error) {
				return map[string]any{"docs": []string{"doc1", "doc2"}}, nil
			}, cachePolicy)
			g.AddNode("retriever_two", func(_ context.Context, _ map[string]any) (any, error) {
				return map[string]any{"docs": []string{"doc3", "doc4"}}, nil
			})
			g.AddNode("decider", joinNoop)
			g.AddNode("qa", func(_ context.Context, state map[string]any) (any, error) {
				docs, _ := state["docs"].([]string)
				return map[string]any{"answer": strings.Join(docs, ",")}, nil
			})
			g.AddEdge(types.START, "rewrite_query")
			g.AddEdge("rewrite_query", "analyzer_one")
			g.AddEdge("analyzer_one", "retriever_one")
			g.AddEdge("rewrite_query", "retriever_two")
			g.AddJoinEdge([]string{"retriever_one", "retriever_two"}, "decider")
			g.AddConditionalEdges("decider", func(_ context.Context, state map[string]any) ([]any, error) {
				if strings.Count(state["query"].(string), "analyzed") > 1 {
					return To("qa"), nil
				}
				return To("rewrite_query"), nil
			})
			g.AddEdge("qa", types.END)

			var cg *CompiledGraph
			var err error
			if withCache {
				cg, err = g.Compile(WithCache(checkpoint.NewInMemoryCache()))
			} else {
				cg, err = g.Compile()
			}
			if err != nil {
				t.Fatalf("Compile() error = %v", err)
			}
			want := map[string]any{
				"query":  "analyzed: query: analyzed: query: what is weather in sf",
				"docs":   []string{"doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"},
				"answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
			}
			// Two full runs, mirroring Python's invoke+stream count
			// assertions (rewrite_query_count == 4 uncached, 2 cached).
			for run := 0; run < 2; run++ {
				res, err := cg.Invoke(context.Background(), map[string]any{"query": "what is weather in sf"})
				if err != nil {
					t.Fatalf("run %d Invoke() error = %v", run, err)
				}
				if !reflect.DeepEqual(res.Values, want) {
					t.Fatalf("run %d Values = %v, want %v", run, res.Values, want)
				}
			}
			wantCalls := 4
			if withCache {
				wantCalls = 2
			}
			if rewriteCalls != wantCalls {
				t.Fatalf("rewriteCalls = %d, want %d", rewriteCalls, wantCalls)
			}
		})
	}
}

// TestJoinSendBypassesBarrier: a types.Send to the join child bypasses the
// barrier (Python OR semantics). The Send PUSH task (arg input) and the
// barrier task (shared state) are two legitimate independent dispatches —
// the barrier trigger must NOT dedup against the Send.
func TestJoinSendBypassesBarrier(t *testing.T) {
	var qaCalls int
	var answers []string
	g := NewStateGraph()
	g.AddReducer("docs", sortedAddReducer)
	g.AddNode("entry", joinNoop)
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc1"}}, nil
	})
	g.AddNode("b", func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc2"}}, nil
	})
	g.AddNode("qa", func(_ context.Context, state map[string]any) (any, error) {
		qaCalls++
		docs, _ := state["docs"].([]string)
		answers = append(answers, strings.Join(docs, ","))
		return map[string]any{"answer": strings.Join(docs, ",")}, nil
	})
	g.AddEdge(types.START, "entry")
	g.AddConditionalEdges("entry", func(_ context.Context, _ map[string]any) ([]any, error) {
		return []any{&types.Send{Node: "qa", Arg: map[string]any{"docs": []string{"sent"}}}, "a", "b"}, nil
	})
	g.AddJoinEdge([]string{"a", "b"}, "qa")
	g.AddEdge("qa", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if qaCalls != 2 {
		t.Fatalf("qaCalls = %d, want 2 (Send + barrier)", qaCalls)
	}
	// Superstep 2 runs the Send task (arg input, answer "sent"); the parents
	// fill the barrier in the same superstep, so superstep 3 runs the barrier
	// task on shared state.
	if !reflect.DeepEqual(answers, []string{"sent", "doc1,doc2"}) {
		t.Fatalf("answers = %v, want [sent doc1,doc2]", answers)
	}
	if res.Values["answer"] != "doc1,doc2" {
		t.Fatalf("final answer = %v, want doc1,doc2", res.Values["answer"])
	}
}

// TestJoinThreeParents (Go extension; Python has no three-parent waiting-edge
// case): join:a+b+c fires only after all three arrive.
func TestJoinThreeParents(t *testing.T) {
	var dCalls int
	g := NewStateGraph()
	g.AddNode("entry", joinNoop)
	for _, n := range []string{"a", "b", "c"} {
		g.AddNode(n, joinNoop)
		g.AddEdge("entry", n)
	}
	g.AddNode("d", func(_ context.Context, _ map[string]any) (any, error) {
		dCalls++
		return nil, nil
	})
	g.AddJoinEdge([]string{"a", "b", "c"}, "d")
	g.AddEdge("d", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, ok := cg.channelProtos["join:a+b+c:d"].(*channels.Barrier); !ok {
		t.Fatal("join:a+b+c:d barrier not registered")
	}
	if _, err := cg.Invoke(context.Background(), nil); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if dCalls != 1 {
		t.Fatalf("dCalls = %d, want 1", dCalls)
	}
}

// TestJoinParentInterruptResume (Go-new): parent a completes and parent b
// interrupts in the same superstep; a's barrier arrival is persisted as a
// pending write with its task batch; resume replays it (a does NOT re-run),
// b re-runs with the resume value, its arrival fills the barrier, and c runs
// exactly once.
func TestJoinParentInterruptResume(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var aCalls, bCalls, cCalls int
	g := NewStateGraph()
	g.AddNode("entry", joinNoop)
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		aCalls++
		return map[string]any{"a_done": true}, nil
	})
	g.AddNode("b", func(ctx context.Context, _ map[string]any) (any, error) {
		bCalls++
		Interrupt(ctx, "b needs input") // panics on run 1; returns "ok" on resume
		return map[string]any{"b_done": true}, nil
	})
	g.AddNode("c", func(_ context.Context, _ map[string]any) (any, error) {
		cCalls++
		return map[string]any{"c_done": true}, nil
	})
	g.AddEdge(types.START, "entry")
	g.AddEdge("entry", "a")
	g.AddEdge("entry", "b")
	g.AddJoinEdge([]string{"a", "b"}, "c")
	g.AddEdge("c", types.END)

	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()
	paused, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("run1 Interrupts = %v, want 1", paused.Interrupts)
	}
	if aCalls != 1 || bCalls != 1 || cCalls != 0 {
		t.Fatalf("after pause: a=%d b=%d c=%d, want 1/1/0", aCalls, bCalls, cCalls)
	}

	// a's arrival rode its task batch into the pause checkpoint's pending
	// writes (the interrupt-path completedTaskWrites closure).
	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple() = %v, %v", tup, err)
	}
	foundArrival := false
	for _, w := range tup.PendingWrites {
		if w.Channel == "join:a+b:c" && w.Value == "a" {
			foundArrival = true
		}
	}
	if !foundArrival {
		t.Fatal("pause checkpoint pending writes missing {join:a+b:c: a}")
	}

	res, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "ok"})
	if err != nil {
		t.Fatalf("resume error = %v", err)
	}
	if aCalls != 1 {
		t.Fatalf("aCalls = %d after resume, want 1 (replayed, not re-run)", aCalls)
	}
	if bCalls != 2 || cCalls != 1 {
		t.Fatalf("after resume: b=%d c=%d, want 2/1", bCalls, cCalls)
	}
	for _, k := range []string{"a_done", "b_done", "c_done"} {
		if res.Values[k] != true {
			t.Fatalf("Values[%q] = %v, want true", k, res.Values[k])
		}
	}
}

// TestJoinCheckpointPartialArrival (Go-new): with parents in DIFFERENT
// supersteps (a -> b), a's arrival is committed channel state; b's interrupt
// pauses with the partial barrier ([a]) inside the checkpoint, GetState
// filters the join key, and resume completes the barrier exactly once.
func TestJoinCheckpointPartialArrival(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var aCalls, bCalls, cCalls int
	g := NewStateGraph()
	g.AddNode("entry", joinNoop)
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		aCalls++
		return map[string]any{"a_done": true}, nil
	})
	g.AddNode("b", func(ctx context.Context, _ map[string]any) (any, error) {
		bCalls++
		Interrupt(ctx, "b needs input")
		return map[string]any{"b_done": true}, nil
	})
	g.AddNode("c", func(_ context.Context, _ map[string]any) (any, error) {
		cCalls++
		return nil, nil
	})
	g.AddEdge(types.START, "entry")
	g.AddEdge("entry", "a")
	g.AddEdge("a", "b")
	g.AddJoinEdge([]string{"a", "b"}, "c")
	g.AddEdge("c", types.END)

	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()
	if _, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("run1 error = %v", err)
	}

	// The pause checkpoint carries the partial barrier as committed channel
	// state (b's superstep never committed, so a's [a] arrival survives).
	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple() = %v, %v", tup, err)
	}
	if got := tup.Checkpoint.ChannelValues["join:a+b:c"]; !reflect.DeepEqual(got, []string{"a"}) {
		t.Fatalf("checkpoint join:a+b:c = %v, want []string{\"a\"} (partial arrival persisted)", got)
	}
	// ... while the user-visible snapshot filters it.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if _, leaked := snap.Values["join:a+b:c"]; leaked {
		t.Fatal("join key leaked into GetState Values")
	}
	if snap.Values["a_done"] != true {
		t.Fatalf("GetState Values = %v, want a_done committed", snap.Values)
	}

	if _, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "ok"}); err != nil {
		t.Fatalf("resume error = %v", err)
	}
	if aCalls != 1 || bCalls != 2 || cCalls != 1 {
		t.Fatalf("after resume: a=%d b=%d c=%d, want 1/2/1", aCalls, bCalls, cCalls)
	}
}

// TestJoinKeyNotLeaked (Go-new, spec risk item): the join channel is
// control-plane — it must not appear in any node's input, any stream chunk
// payload (values/updates/debug), Result.Values, or GetState/GetStateHistory.
func TestJoinKeyNotLeaked(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var mu sync.Mutex
	seenInputKeys := map[string]bool{}
	var qaCalls int

	g := NewStateGraph()
	g.AddReducer("docs", sortedAddReducer)
	wrap := func(fn NodeFunc) NodeFunc {
		return func(ctx context.Context, state map[string]any) (any, error) {
			mu.Lock()
			for k := range state {
				seenInputKeys[k] = true
			}
			mu.Unlock()
			return fn(ctx, state)
		}
	}
	g.AddNode("rewrite_query", wrap(func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"query": "query: " + state["query"].(string)}, nil
	}))
	g.AddNode("analyzer_one", wrap(func(_ context.Context, state map[string]any) (any, error) {
		return map[string]any{"query": "analyzed: " + state["query"].(string)}, nil
	}))
	g.AddNode("retriever_one", wrap(func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc1", "doc2"}}, nil
	}))
	g.AddNode("retriever_two", wrap(func(_ context.Context, _ map[string]any) (any, error) {
		return map[string]any{"docs": []string{"doc3", "doc4"}}, nil
	}))
	g.AddNode("qa", wrap(func(_ context.Context, state map[string]any) (any, error) {
		qaCalls++
		docs, _ := state["docs"].([]string)
		return map[string]any{"answer": strings.Join(docs, ",")}, nil
	}))
	g.AddEdge(types.START, "rewrite_query")
	g.AddEdge("rewrite_query", "analyzer_one")
	g.AddEdge("analyzer_one", "retriever_one")
	g.AddEdge("rewrite_query", "retriever_two")
	g.AddJoinEdge([]string{"retriever_one", "retriever_two"}, "qa")
	g.AddEdge("qa", types.END)

	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()
	for c, err := range cg.Stream(ctx, map[string]any{"query": "q"},
		StreamOptions{
			Options: Options{ThreadID: "t"},
			Modes:   []StreamMode{StreamValues, StreamUpdates, StreamDebug},
		}) {
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		payload, _ := json.Marshal(c.Payload)
		if strings.Contains(string(payload), "join:") {
			t.Fatalf("join key leaked into %s chunk: %s", c.Mode, payload)
		}
	}
	if qaCalls != 1 {
		t.Fatalf("qaCalls = %d, want 1", qaCalls)
	}
	for k := range seenInputKeys {
		if strings.HasPrefix(k, "join:") {
			t.Fatalf("join key %q leaked into a node input", k)
		}
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if _, leaked := snap.Values["join:retriever_one+retriever_two:qa"]; leaked {
		t.Fatal("join key leaked into GetState Values")
	}
	hist, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: "t"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("GetStateHistory() error = %v", err)
	}
	for i, s := range hist {
		for k := range s.Values {
			if strings.HasPrefix(k, "join:") {
				t.Fatalf("join key %q leaked into history snapshot %d", k, i)
			}
		}
	}
}
```

注意：`TestJoinWaitingEdgePlusRegularEdge` 直接改写 `g.nodes["qa"]`（白盒，同包）以记录每次 qa 的 answer；若实现期间觉得改 builder 内部 map 不妥，可改为在 `newWaitingEdgeGraph` 增加可选的记录参数——但以上写法对本计划即为定稿，不要再引入新 helper。

- [ ] **Step 2: 运行 `go test ./langgraph/graph/ -run TestJoin -v`，确认全部通过**（Task 3 已实现执行器，这些测试是对它的移植验证；若有失败，按 TDD 回到实现修 bug——不允许改测试断言来迁就错误行为，顺序/计数断言均以 Python 语义为准）。
- [ ] **Step 3: 门禁 PASS**：`go build ./... && go vet ./... && go test ./...` + `make test-sqlite`。
- [ ] **Step 4: 提交**
  ```
  git add langgraph/graph/join_test.go
  git commit -m "test(langgraph/graph): port Python waiting-edge tests + Go-specific join coverage"
  ```

---

### Task 5: 全量回归 + 文档 + spec 标记

**Files:**
- Modify: `langgraph/graph/graph.go` — package doc comment（:1-40）加 join edges 一条 scope 说明
- Modify: `docs/usage/langgraph.md` — 新增 join edges 小节
- Modify: `docs/superpowers/specs/2026-08-08-langgraph-go-m5-m8-design.md` — M6 节标题标记完成

**Interfaces:**
- Consumes: Task 2 的 `AddJoinEdge` 公开 API（文档示例以其签名为准）。
- Produces: 无代码产物。

- [ ] **Step 1: 回归验证**（工作目录 `langchain_golang/`）：
  ```
  go build ./... && go vet ./... && go test ./...
  make test-sqlite
  ```
  确认全绿。`go test ./langchain/...` 已包含在 `./...` 中——`langchain/agents` 零行为变化的验收标准是：**本里程碑未触碰 `langchain/` 任何文件**（`git diff --stat main...HEAD -- langchain/` 为空，或在合并基线上 `git status langchain/` 干净），且其测试零修改通过。
- [ ] **Step 2: 文档。** `graph.go` package doc 的 scope 列表（:21 "Stream modes..." 条之后）插入：

```go
//   - Multi-parent waiting edges (StateGraph.AddJoinEdge) are supported via
//     a barrier channel per edge (Python's NamedBarrierValue): the child
//     triggers exactly once after all parents commit; plain edges, Sends,
//     and Command.Goto into the child bypass the barrier (Python's OR
//     semantics). defer=True / NamedBarrierValueAfterFinish are NOT
//     supported (the edge-driven loop has no finish broadcast).
```

`docs/usage/langgraph.md` 在 Stream API 小节之前插入新小节（语言与既有文件一致——英文）：

```markdown
## Join edges (multi-parent barrier)

`AddJoinEdge([]string{"a", "b"}, "c")` mirrors Python's
`add_edge(("a", "b"), "c")`: `c` runs exactly once after ALL parents have
committed, whether they finish in the same superstep or several apart, and
re-arms on each loop round. Arrivals are checkpointed, so an interrupted
parent that resumes later still completes the barrier.

> **Warning (OR semantics, Python parity):** a plain edge, conditional edge,
> `types.Send`, or `Command.Goto` into the join child bypasses the barrier and
> triggers the child directly. Mixing both edge kinds into one child can run
> it multiple times.

Divergences from Python: Go requires >= 2 distinct parents (Python accepts a
single-element tuple and silently dedups) and rejects `types.END` as a join
child. `defer=True` / `NamedBarrierValueAfterFinish` are not supported. The
`join:a+b:c` barrier channel is control-plane state: it never appears in node
inputs, snapshots, or stream output.
```

- [ ] **Step 3: spec 标记**——把 spec 文件 M6 节标题改为 `## M6 多父 barrier join（已完成 2026-08-08）`。
- [ ] **Step 4: 提交**
  ```
  git add langgraph/graph/graph.go docs/usage/langgraph.md docs/superpowers/specs/2026-08-08-langgraph-go-m5-m8-design.md
  git commit -m "docs(langgraph): document M6 barrier join edges"
  ```

---

## Self-Review Notes

- Spec 覆盖：Barrier channel（五方法 + 接口外 Consume + ErrInvalidUpdate 对齐）→ Task 1；`AddJoinEdge` 校验与 Compile 注册（join key→parents/child、parent→keys 两份元数据）→ Task 2；隐式 write 搭批次 / 恰好一次 / Consume 时机 / 四面过滤 → Task 3 设计要点 1-6 逐条对应；Python 用例移植（:1953/:2710/:2808/:3059 + OR/Send）与 Go 新增（中断补写、部分到达往返、泄漏）→ Task 4 映射表逐行对应；`langchain/agents` 零变化 → Task 5 Step 1 验收；defer 不做 → Global Constraints + Task 5 文档声明。
- 类型一致性：Task 1 产 `*channels.Barrier`（具体类型）→ Task 2 注册进 `channelProtos` → Task 3 的 `isJoinKey` 类型断言与 `interface{ Consume() bool }` 断言依赖同一具体类型；Task 2 产 `joins`/`joinsByParent` → Task 3 注入/触发/Consume 消费；`joinMeta` 字段名三方一致。
- 行号核实：所有 `graph.go`/`state.go`/`snapshot.go`/`resume.go`/Python 行号均在写作时对源码逐一核对（Go 侧 @ M4 完成态，Python 侧 @ ea5f9cc9f）。
- 风险与缓解：①snapshot 泄漏——`snapshot()`/`snapshotFromTuple`/三处 emission 共五个过滤点，Task 4 `TestJoinKeyNotLeaked` 用 JSON 序列化扫全部 chunk payload；②Consume 时机与对象——提交块内"先 Consume 后触发扫描"，且 Consume 以 `seen[child][key] >= versions[key]` 判定"子节点确由本 barrier 触发"，Send/普通边触发的 PUSH 任务不会误消费同超步刚填满的 barrier（写作时逐超步推演 `TestJoinSendBypassesBarrier`/`...PlusRegularEdge` 捕获并修正了无条件 Consume 的初版错误）；③Send 去重误伤——触发扫描只 append 不去重，`TestJoinSendBypassesBarrier` 断言双触发；④cache 命中父节点——注入点在 cache store 之后、`outcomeFromCachedWrites` 重建的 outcome 同样过注入循环，`TestJoinLoopReset` withCache 变体把 cache policy 挂在 join 父 retriever_one 上覆盖此路径（Python 原用例无此配置，属有意的 Go 侧增强，已在测试注释说明）。
- 已知取舍：`applyWrites` bool 语义收窄（join-only 超步不产 values chunk）改变了该返回值的书面契约，但其全部读取点（三处 emitValues 门控）已核对，无 join 图行为不变；`snapshotFromTuple` 改为 `*CompiledGraph` 方法是包内重构，无公开 API 变化。
