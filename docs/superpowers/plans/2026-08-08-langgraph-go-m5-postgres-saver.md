# LangGraph Go Port M5: Postgres Checkpoint Saver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Go 版 langgraph 移植补齐 M5：Postgres checkpoint saver（对齐 `langgraph-checkpoint-postgres`）。含三部分工作：①根接口扩展（breaking）——`ListOptions.Filter`（metadata 包含过滤，对齐 Python `@>`）与 `PutWrites` 新增 `taskPath` 参数（`Write` 新增 `TaskPath` 字段）；②根 module 新增共享 saver 契约测试套件 `langgraph/checkpoint/savertest`，sqlite 测试重构为调用它；③嵌套 module `langgraph/checkpoint/postgres/`（pgx/v5 + pgxpool，四表 + 迁移表 schema 逐条照搬 Python MIGRATIONS v0–v9），并在 embedded-postgres 上跑同一契约套件。

**Architecture:** Postgres saver 复刻 sqlite saver 的包装模式（嵌套 module、`replace` 指回根 module），但采用 Python 的全对齐四表 schema（checkpoints / checkpoint_blobs / checkpoint_writes / checkpoint_migrations）：channel values 拆两类——仅 JSON 原语（nil/string/bool/float64）内联进 checkpoints JSONB，其余（int、map、slice、注册表类型）按 `(channel, version)` 进 blobs 表（per-version 去重）；读取拆 3 条查询（checkpoints / blobs / writes，scan-then-decode，与 sqlite 同风格）；metadata 过滤在 Postgres 服务端用 `metadata @> $n::jsonb`，memory/sqlite 加载 metadata 后进程内做 map 包含比较（语义相同）。

**Tech Stack:** Go 1.23，根 module `github.com/projanvil/langchain-golang` 保持零第三方依赖；嵌套 module `langgraph/checkpoint/postgres`（`module github.com/projanvil/langchain-golang/langgraph/checkpoint/postgres`）新增唯一运行时依赖 `github.com/jackc/pgx/v5`（纯 Go）；测试依赖 `github.com/fergusstrange/embedded-postgres`（进程内起真库，无需 Docker）。

## Global Constraints

- 所有命令的工作目录：`langchain_golang/`。嵌套 module（sqlite、postgres）有自己的 `go.mod`，其命令在各自目录下运行。
- Go 1.23 为最低版本；**根 module 保持零第三方依赖**（新依赖只允许出现在嵌套 module）。
- 验收底线：`go test ./...`（根 module）、`make test-sqlite`、新增 `make test-postgres` 全绿；`langchain/agents` 零改动（本 milestone 不触碰其行为）。
- **单元测试对标 Python**：契约用例从 Python `libs/checkpoint-postgres/tests/test_sync.py` 与 `libs/langgraph/tests/` 的 saver 契约用例移植，语义级对齐，不是象征性覆盖（见 Task 2 的移植映射）。
- 提交信息风格：仿 `git log` 的 conventional 格式——`feat(langgraph): ...` / `docs(langgraph): ...` / `test(langgraph): ...`。
- 注释风格：沿用现有代码的详尽 doc comment、单反引号引用标识符。
- 每个任务完成后过门禁：根 module `go build ./... && go vet ./... && go test ./...`；涉及嵌套 module 时另加其目录下的 `go build ./... && go vet ./... && go test ./...`。
- 不 reformat 与本任务无关的文件。
- 文档化分歧（spec 已定，必须写进 doc.go / 代码注释，不得静默）：
  - Python blobs 表 `version` 列为 TEXT（存十进制整数字符串，`base.py:564` 的 `cast(str, ver)`）；Go 的 channel version 是 `int64`，blobs 表 `version` 列用 **BIGINT**（MIGRATIONS v2 的唯一一行改动，见 Task 3）。
  - 跨语言共库不可行（serde 字节格式不互通：Go JSON envelope vs Python msgpack；叠加 version 列类型分歧）——doc.go 明确声明。
  - Go `checkpoint.Metadata` 是封闭结构体（`Source`/`Step`/`Parents`，`checkpoint/checkpoint.go:69-79`），无 Python 的任意 metadata 键；`Filter` 只作用于这三个键的 JSON 投影。
  - 不做 Shallow 变体与 delta channel history（Python 的 `ShallowPostgresSaver` 与 `_DeltaSnapshot` 快路径均无 Go 对应概念）。

## Locked Design Decisions (binding)

- **D1. 根接口扩展（breaking，spec 允许的 pre-1.0 break）**：
```go
// checkpoint/checkpoint.go
type ListOptions struct {
    Before *Config
    Limit  int
    // Filter 为 metadata 包含过滤（Python list(filter=...) 的 @> 语义）：
    // 每个键值对要求 checkpoint 的 metadata JSON 投影在该键上相等。
    // 键集合封闭为 source / step / parents（Go Metadata 是封闭结构体）。
    Filter map[string]any
}

type Write struct {
    TaskID   string
    // TaskPath 是写入时 task 的路径（Python PendingWrite 的 task_path，
    // 用于子图 task 命名空间），由 Saver.PutWrites 盖章；当前调用点均传 ""。
    TaskPath string
    Channel  string
    Value    any
}

type Saver interface {
    // ... GetTuple/List/Put/DeleteThread 不变 ...
    PutWrites(ctx context.Context, cfg Config, writes []Write, taskID, taskPath string) error
}
```
- **D2. 进程内 Filter 实现**（memory/sqlite 共用，根 module `checkpoint` 包导出）：`checkpoint.MetadataMatchesFilter(md Metadata, filter map[string]any) bool`，JSON 归一化后做顶层键包含比较（完整代码见 Task 1 Step 3）。
- **D3. sqlite schema 演进**：`writes` 表加 `task_path TEXT NOT NULL DEFAULT ''`；`New` 启动时用 `PRAGMA table_info(writes)` 检测列缺失则 `ALTER TABLE writes ADD COLUMN task_path TEXT NOT NULL DEFAULT ''`，兼容旧库（完整代码见 Task 1 Step 4）。
- **D4. savertest 套件**：根 module 新包 `langgraph/checkpoint/savertest`，导出 `func Run(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver)`；`newSaver` 必须返回**空存储**的 saver（postgres 工厂内部 TRUNCATE 四表）。套件对 MemorySaver（根 module 内）与 sqlite（嵌套 module）运行；Task 4 对 postgres 运行。
- **D5. postgres schema**：MIGRATIONS v0–v9 逐条照搬 Python（`libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py:43-91`），唯一改动是 v2 的 `version TEXT NOT NULL` → `version BIGINT NOT NULL`（spec 文档化分歧）。Setup 不在事务内执行（v6–v8 为 `CREATE INDEX CONCURRENTLY`）。
- **D6. inline 判定**：仅 `nil / string / bool / float64` 内联进 checkpoints JSONB；`int`/`int64`、`map[string]any`、`[]any` 及一切注册表类型进 blobs 表（对齐 Python `__init__.py:309-319`：只内联原语，dict/list 进 blobs；Go 的 int 走 serde envelope、非 JSON-native，故不内联）。判定函数集中一处（`isInline`）并配表驱动测试。

## Reference Semantics (from Python source)

- **MIGRATIONS v0–v9**：`libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py:43-91`（全文逐字抄入 Task 3 Step 3）。迁移机制（`__init__.py:85-110`）：先执行 `MIGRATIONS[0]` 建 `checkpoint_migrations`，`SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1` 取当前版本（无行则为 -1），逐条执行剩余 migration 并插入版本行；**不在事务内**（v6–v8 CONCURRENTLY 限制）。
- **写入路径**（`__init__.py:263-345`）：`put` 把 `channel_values` 拆两类——`v is None or isinstance(v, (str, int, float, bool))` 内联留在 checkpoint JSONB（:316-317），其余 pop 进 `blob_values`（:319）；blobs 行只覆盖 `new_versions` 中出现的 channel（:322-324），经 `_dump_blobs`（`base.py:549-572`）按 `(channel, cast(str, version))` 编码为 `(type, blob)`；blobs 用 `ON CONFLICT DO NOTHING`（不可变版本化行，`base.py:131-135`），checkpoints 用 `ON CONFLICT DO UPDATE`（`base.py:137-144`）。
- **writes**：`put_writes`（`__init__.py:347-379`）批次级选择 UPSERT vs INSERT...DO NOTHING——`all(w[0] in WRITES_IDX_MAP for w in writes)` 时 UPSERT；idx 经 `WRITES_IDX_MAP.get(channel, idx)`（`{ERROR:-1, SCHEDULED:-2, INTERRUPT:-3, RESUME:-4}`，`libs/checkpoint/langgraph/checkpoint/base/__init__.py:795`）；SQL 见 `base.py:146-159`（含 `task_path` 列）。Go sqlite 已实现同语义（`checkpoint/sqlite/sqlite.go:215-258`），postgres 照搬。
- **读取路径**：inline values 与 blobs 合并组装（`_load_checkpoint_tuple` 合并 `value["checkpoint"]["channel_values"]` 与 `_load_blobs`，`__init__.py:581-583`）；blob 行 `type == "empty"` 跳过（`base.py:375-384`）；metadata 过滤用 `metadata @> %s`（`base.py:653-656`）。
- **测试移植基线**：`libs/checkpoint-postgres/tests/test_sync.py` 的 `test_data`（:141-175）、`test_search`（:214-260，filter 单键/多键/空/无匹配四组查询）、`test_null_chars`（:262，JSONB 拒绝 `\u0000`）、`test_nonnull_migrations`（:277，migration 后约束检查）。`test_combined_metadata`（:189-211）依赖 Python config 任意 metadata 键（`run_id` 合并），**Go 无对应概念，不移植**（Go Metadata 封闭结构体，文档化分歧）。

---

### Task 1: 根接口扩展（breaking）——Filter + TaskPath

**Files:**
- Modify: `langgraph/checkpoint/checkpoint.go` — `ListOptions.Filter`（:159-166）、`Write.TaskPath`（:96-104）、`Saver.PutWrites` 签名（:189）
- Create: `langgraph/checkpoint/filter.go` — `MetadataMatchesFilter`
- Test: `langgraph/checkpoint/filter_test.go`、`langgraph/checkpoint/checkpoint_test.go`（更新 :166 的 `PutWrites` 调用；新增 Filter/TaskPath 用例）
- Modify: `langgraph/checkpoint/memory.go` — `List`（:49-69）加 Filter；`PutWrites`（:113-129）盖章 TaskPath
- Modify: `langgraph/checkpoint/sqlite/sqlite.go` — schema 加列 + 启动检测 ALTER、`PutWrites`/`loadWrites`/`List` 适配
- Test: `langgraph/checkpoint/sqlite/sqlite_test.go` — 全部 `PutWrites` 调用点（:242,:249,:254,:259,:296,:313,:361）加 `""` 实参；新增旧库迁移用例
- Modify: `langgraph/graph/resume.go` — :54 的 `saver.PutWrites(ctx, cfg, writes, taskID)` 改为 `saver.PutWrites(ctx, cfg, writes, taskID, "")`（`persistInterrupts` 服务 graph.go:752/:919/:985 三处边界/in-node 中断路径，签名不动）
- Modify: `langgraph/graph/graph.go` — :929 的 `PutWrites(ctx, *currentCfg, writes, taskID)` 追加 `""` 实参

**Interfaces:**
- Consumes: 现有 `checkpoint.Saver`（checkpoint.go:174-192）、`MemorySaver`（memory.go:24）、sqlite `Saver`（sqlite.go:70）。
- Produces: D1 的 `ListOptions.Filter` / `Write.TaskPath` / 新 `PutWrites(ctx, cfg, writes, taskID, taskPath string) error`；D2 的 `checkpoint.MetadataMatchesFilter(md Metadata, filter map[string]any) bool`。Task 2 的 savertest 与 Task 3 的 postgres saver 都建立在本任务的新签名上。

- [ ] **Step 1: 写失败测试（memory 侧）** — `checkpoint/checkpoint_test.go` 新增：
  - `TestListFilter`：向 thread `t` Put 三个 checkpoint，metadata 分别为 `{Source:"input", Step:-1}`、`{Source:"loop", Step:0}`、`{Source:"loop", Step:1}`；断言 `List(ctx, cfg, ListOptions{Filter: map[string]any{"source": "input"}})` 返回 1 条且为第一条；`Filter{"source": "loop"}` 返回 2 条（newest first）；`Filter{"step": 1}` 返回 1 条；`Filter{"source": "update"}` 返回 0 条；`Filter{}`（空）返回全部 3 条。移植自 Python `test_sync.py:214-260` 的 query_1/query_2/query_3/query_4 四组语义（键换成 Go Metadata 的 source/step）。
  - `TestPutWritesTaskPathRoundTrip`：Put 一个 checkpoint 后 `PutWrites(ctx, cfg, []Write{{Channel: "c", Value: "v"}}, "task-1", "path/a")`，断言 `GetTuple` 的 `PendingWrites[0].TaskPath == "path/a"`；再断言默认 `""` 实参时 `TaskPath == ""`。
  - 同步把 :166 的现有调用改为四实参形式（加 `""`）。
- [ ] **Step 2: 写失败测试（filter.go 单元测试 + sqlite 侧）** — `checkpoint/filter_test.go`：
  - 表驱动：`Metadata{Source:"loop", Step:1, Parents:map[string]string{"": "p1"}}` 对 `nil`、`{}`、`{"source":"loop"}`、`{"step":1}`（注意 filter 值写 `1` 即 int——归一化后必须匹配）、`{"step":1.0}`（float64 同样匹配，对齐 JSONB 数字相等语义）、`{"parents":map[string]string{"":"p1"}}` 返回 true；对 `{"source":"input"}`、`{"step":2}`、`{"missing":"x"}`、`{"parents":map[string]string{"":"other"}}` 返回 false。
  - sqlite 侧：`sqlite_test.go` 新增 `TestTaskPathColumnMigration`——先用 `database/sql` 直接建一个**旧 schema**（无 `task_path` 列，即当前 `setupSQL` 去掉该列）的 db 文件并插入一行 writes，关闭；再 `sqlite.New(path, ...)` 打开，断言旧行 `TaskPath == ""` 可读出、新 `PutWrites(..., "p")` 的 `TaskPath == "p"` 可往返。把既有 7 处 `PutWrites` 调用加 `""` 实参。
- [ ] **Step 3: 运行验证失败**（编译失败即失败——签名未改），然后实现根 module 部分。`checkpoint/checkpoint.go` 按 D1 改三处（含 doc comment：`Filter` 注明"键集合封闭为 source/step/parents"、`TaskPath` 注明 Python 对应与当前调用点传 `""`）。新建 `checkpoint/filter.go` 完整代码：
```go
package checkpoint

import (
	"encoding/json"
	"reflect"
)

// MetadataMatchesFilter reports whether md contains every key/value pair in
// filter, the in-process equivalent of Postgres's `metadata @> filter`
// containment used by persistent savers. The metadata document is md's JSON
// projection — {"source": ..., "step": ..., "parents": ...} — so filter keys
// are limited to source/step/parents (Metadata is a closed struct, unlike
// Python's free-form CheckpointMetadata). Both sides are normalized through
// JSON before comparison so `step` filters match whether written as int or
// float64 (JSONB numeric equality), and nested `parents` values compare by
// JSON equality.
func MetadataMatchesFilter(md Metadata, filter map[string]any) bool {
	if len(filter) == 0 {
		return true
	}
	doc := map[string]any{"source": md.Source, "step": md.Step}
	if md.Parents != nil {
		doc["parents"] = md.Parents
	}
	normDoc := normalizeJSON(doc)
	for k, v := range filter {
		got, ok := normDoc[k]
		if !ok || !reflect.DeepEqual(got, normalizeJSON(v)) {
			return false
		}
	}
	return true
}

// normalizeJSON round-trips v through encoding/json so numbers become
// float64 and maps/slices compare with JSON semantics.
func normalizeJSON(v any) any {
	raw, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out any
	if err := json.Unmarshal(raw, &out); err != nil {
		return v
	}
	return out
}
```
`memory.go` 的 `List` 重写 id 收集段为（Filter 必须先于 Limit 应用，对齐 Python WHERE→LIMIT 顺序）：
```go
	ns := s.threads[cfg.ThreadID][cfg.CheckpointNS]
	ids := make([]string, 0, len(ns))
	for id, st := range ns {
		if opts.Before != nil && opts.Before.CheckpointID != "" && id >= opts.Before.CheckpointID {
			continue
		}
		if !MetadataMatchesFilter(st.md, opts.Filter) {
			continue
		}
		ids = append(ids, id)
	}
```
（其余排序/Limit/tuple 组装逻辑不变。）`memory.go` 的 `PutWrites`：签名加 `taskPath string`，循环内改 `w.TaskID = taskID; w.TaskPath = taskPath`。
- [ ] **Step 4: 实现 sqlite 侧** — `sqlite/sqlite.go`：
  1. `setupSQL` 的 `writes` 表定义在 `value BLOB,` 后加一行 `task_path TEXT NOT NULL DEFAULT '',`。
  2. `New` 在 `db.Exec(setupSQL)` 之后调用新的 `ensureTaskPathColumn`（旧库兼容，D3）：
```go
// ensureTaskPathColumn adds the task_path column to writes tables created
// before the M5 schema evolution (Python added task_path in migration v9;
// Go sqlite savers created before M5 lack it).
func ensureTaskPathColumn(db *sql.DB) error {
	rows, err := db.Query(`PRAGMA table_info(writes)`)
	if err != nil {
		return fmt.Errorf("sqlite: inspect writes schema: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var cid int
		var name, typ string
		var notNull int
		var dflt any
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notNull, &dflt, &pk); err != nil {
			return fmt.Errorf("sqlite: inspect writes schema: %w", err)
		}
		if name == "task_path" {
			return rows.Err()
		}
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("sqlite: inspect writes schema: %w", err)
	}
	if _, err := db.Exec(`ALTER TABLE writes ADD COLUMN task_path TEXT NOT NULL DEFAULT ''`); err != nil {
		return fmt.Errorf("sqlite: add writes.task_path column: %w", err)
	}
	return nil
}
```
  3. SQL 常量：`insertOrIgnoreSQL`/`insertOrReplaceSQL` 列清单加 `task_path`、占位符 8→9 个；`selectWritesSQL` 改为 `SELECT task_id, task_path, channel, type, value FROM writes WHERE ... ORDER BY task_id, idx`。
  4. `PutWrites` 签名加 `taskPath string`，`stmt.ExecContext` 实参在 `taskID` 后插入 `taskPath`。
  5. `loadWrites` 的 `rows.Scan` 加 `&w.TaskPath`（在 TaskID 之后）。
  6. `List`：Filter 非空时 SQL 不加 `LIMIT`（过滤在进程内、必须先于 Limit）；循环改为：
```go
	out := make([]checkpoint.Tuple, 0, len(raw))
	for _, r := range raw {
		tup, err := s.finishTuple(ctx, cfg.CheckpointNS, r)
		if err != nil {
			return nil, err
		}
		if !checkpoint.MetadataMatchesFilter(tup.Metadata, opts.Filter) {
			continue
		}
		out = append(out, *tup)
		if opts.Limit > 0 && len(opts.Filter) > 0 && len(out) >= opts.Limit {
			break
		}
	}
```
  （Filter 为空时维持现有 SQL 级 Before/Limit 路径不变。）
  7. `graph/resume.go:54` 与 `graph/graph.go:929` 加 `""` 实参。
- [ ] **Step 5: 门禁 PASS** — 根 module `go build ./... && go vet ./... && go test ./...`；`cd langgraph/checkpoint/sqlite && go build ./... && go vet ./... && go test ./...`。
- [ ] **Step 6: Commit** —
```
git add langgraph/checkpoint/checkpoint.go langgraph/checkpoint/filter.go langgraph/checkpoint/filter_test.go langgraph/checkpoint/checkpoint_test.go langgraph/checkpoint/memory.go langgraph/checkpoint/sqlite/sqlite.go langgraph/checkpoint/sqlite/sqlite_test.go langgraph/graph/resume.go langgraph/graph/graph.go
git commit -m "feat(langgraph/checkpoint): ListOptions.Filter metadata filter and PutWrites task_path (breaking)"
```

---

### Task 2: `langgraph/checkpoint/savertest` 共享契约套件

**Files:**
- Create: `langgraph/checkpoint/savertest/savertest.go`（套件本体）、`langgraph/checkpoint/savertest/doc.go`
- Test: `langgraph/checkpoint/savertest/savertest_test.go` — 套件对 `MemorySaver` 自验（根 module 内）
- Modify: `langgraph/checkpoint/sqlite/sqlite_test.go` — 删除契约用例，改为调用 `savertest.Run`；保留 sqlite 专属用例
- Modify: `langgraph/checkpoint/checkpoint_test.go` — 删除被套件接管的通用契约用例（`TestPutGetRoundTrip`、`TestListNewestFirstWithBeforeAndLimit`、`TestPutWritesVisibleOnGetTuple`、`TestDeleteThread`、`TestParentConfigReflectsPutTimeConfig`），保留 MemorySaver 专属用例（`TestNewIDMonotonic`、`TestCopyOnRead`、`TestMemorySaverZeroValue`、`TestNamespacesAreIndependent`）与 Task 1 新增的 `TestListFilter`/`TestPutWritesTaskPathRoundTrip`（它们验证 D2/D3 的 memory 侧实现，套件用例不重复其断言粒度）

**Interfaces:**
- Consumes: Task 1 的新 `Saver` 接口（含 `Filter`/`TaskPath`）、`checkpoint.NewID`（`checkpoint/id.go:20`）、`serde.NewJSONSerializer()`（`checkpoint/serde/json.go:51`，由工厂闭包注入，套件本身不依赖 serde 包）。
- Produces:
```go
// Package savertest runs the shared checkpoint.Saver contract suite,
// ported from Python's saver contract tests
// (libs/checkpoint-postgres/tests/test_sync.py and the saver-facing cases
// in libs/langgraph/tests/). Every Saver implementation — MemorySaver,
// sqlite, postgres — runs the same suite.
package savertest

// Run executes the full contract suite as subtests of t. newSaver must
// return a Saver backed by EMPTY storage (factories wrapping a shared
// database must truncate all tables).
func Run(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver)
```
Task 3/4 的 postgres 测试以 `savertest.Run(t, factory)` 消费本接口。

- [ ] **Step 1: 写套件（先失败：包不存在时 sqlite/memory 调用方编译失败即验证）**。`savertest.go` 的 subtest 清单（每个 subtest 调一次 `newSaver(t)`，数据与断言精确规定；Python 移植映射见 Reference Semantics）：
  1. `put_get_round_trip`：checkpoint 含 `[]messages.Message`、`types.Interrupt{Value:"why?",ID:"int-1"}`、`[]string{"a","b"}`、`map[string]any{"k":"v"}` channel values + `VersionsSeen` + 带 typed Arg 的 `Next`；`Put`（`newVersions` 合并进 ChannelVersions 的断言同 sqlite 现 `TestPutGetRoundTrip`，sqlite_test.go:64-114）→ latest 与 by-ID 两条 `GetTuple` 路径 `reflect.DeepEqual` 全等（Checkpoint、Metadata、ParentConfig=nil）；未知 thread 返回 `(nil, nil)`。
  2. `int_values_survive_as_int`：`int(42)` 与 `int64(1<<62)` channel values 往返后类型不降级（sqlite_test.go:116-137 的逻辑）。
  3. `parent_links`：第二个 checkpoint 的 `ParentConfig` 指向第一个的 put-time config（sqlite_test.go:139-162）。
  4. `list_order_before_limit`：3 个 checkpoint + 1 个干扰 thread；newest-first 顺序、`Before`（严格小于）、`Limit`（sqlite_test.go:164-213）。
  5. `list_filter`：移植 Python `test_search`（test_sync.py:214-260）四组查询到 Go Metadata 键：`Filter{"source":"input"}`→1 条；`Filter{"step":1}`→1 条；`Filter{}`→全部；`Filter{"source":"update","step":1}`→0 条。数据：thread-1 一条 `{Source:"input", Step:2}`、thread-2 一条 `{Source:"loop", Step:1}`、thread-2 的 `CheckpointNS:"inner"` 一条 `{Source:"loop", Step:1}`；另断言 `List` 对 thread-2 不带 ns 时只返回 `CheckpointNS:""` 的 1 条（Go `Config.CheckpointNS` 精确匹配语义——Python 的"跨 ns 搜索"（test_sync.py:253-259）在 Go 接口中不存在，`Config.CheckpointNS` 空串即根 ns 精确匹配；此分歧写进套件注释）。
  6. `put_writes_round_trip`：`PutWrites` 后 `GetTuple.PendingWrites` 逐条可见，`TaskID` 盖章为传入 taskID、顺序保持插入序。
  7. `put_writes_batch_rule`：移植 sqlite_test.go:223-289（它本身是 Python `put_writes` 批次级规则的移植）——mixed 批次重复 `PutWrites` 同 task 被忽略（first-write-wins），all-reserved 批次 REPLACE；`PendingWrites` 恰好 3 行。
  8. `put_writes_task_path`：`PutWrites(..., "task-1", "path/a")` 后 `PendingWrites[i].TaskPath == "path/a"`。
  9. `put_writes_missing_checkpoint`：对未知 checkpoint `PutWrites` 返回非 nil error（MemorySaver 行为，sqlite_test.go:291-300）。
  10. `delete_thread`：两 thread 各 Put+PutWrites，`DeleteThread("t1")` 后 t1 的 `GetTuple`/`List` 为空、t2 不受影响（sqlite_test.go:302-336）。
  11. `concurrent_put`：8 goroutine × 5 次 `Put`+`PutWrites`（各自独立 thread），全部成功且每 thread `List` 为 5 条（sqlite_test.go:338-386 的逻辑，去 WAL 特定性）。
  12. `serde_round_trip`：channel values 覆盖全部注册表类型（`messages.Message`、`[]messages.Message`、`types.Send`、`types.Interrupt`、`time.Time`、`[]byte`、`int64`、`int`、`[]string`，注册表见 `checkpoint/serde/json.go:38-40`）+ planned task Arg + write values，Put/GetTuple/PutWrites 往返 DeepEqual。
  关键断言代码（batch rule，直接照搬现有逻辑改四实参签名）：
```go
	// all-reserved batch: re-writing the same task REPLACES the stored value.
	if err := s.PutWrites(ctx, cfg, []checkpoint.Write{
		{Channel: checkpoint.ReservedInterrupt, Value: types.Interrupt{Value: "v2", ID: "i1"}},
	}, "task-2", ""); err != nil {
		t.Fatalf("PutWrites reserved replace: %v", err)
	}
```
  `doc.go` 声明套件对标 Python `standardtests` 哲学与移植来源文件清单。
- [ ] **Step 2: 写 `savertest_test.go`**：
```go
package savertest_test

import (
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/savertest"
)

func TestMemorySaverContract(t *testing.T) {
	savertest.Run(t, func(t *testing.T) checkpoint.Saver {
		return checkpoint.NewMemorySaver()
	})
}
```
- [ ] **Step 3: 重构 sqlite 测试** — `sqlite_test.go` 删除 `TestPutGetRoundTrip`、`TestIntChannelValuesSurviveAsInt`、`TestParentLinks`、`TestListNewestFirstBeforeLimit`、`TestPutWritesBatchRule`、`TestPutWritesMissingCheckpoint`、`TestDeleteThread`，替换为：
```go
func TestSaverContract(t *testing.T) {
	savertest.Run(t, func(t *testing.T) checkpoint.Saver {
		s, err := sqlite.New(":memory:", serde.NewJSONSerializer())
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		t.Cleanup(func() { s.Close() })
		return s
	})
}
```
保留：`TestReducerChannelRoundTrip`（graph 级 typed-slice 往返，套件不覆盖）、`TestInMemoryDatabase`、`TestConcurrentAccess`（WAL 文件特定）、`TestTaskPathColumnMigration`（Task 1 新增，sqlite schema 演进特定）。同步精简 import（去掉 `messages`/`channels`/`graph` 等仅被删除用例使用的包；`TestReducerChannelRoundTrip` 仍需要 `channels`/`graph`）。
- [ ] **Step 4: 重构 memory 测试** — 按 Files 清单删/留 `checkpoint_test.go`。
- [ ] **Step 5: 门禁 PASS** — 根 module + sqlite 嵌套 module 的 `go build ./... && go vet ./... && go test ./...`。
- [ ] **Step 6: Commit** —
```
git add langgraph/checkpoint/savertest/ langgraph/checkpoint/checkpoint_test.go langgraph/checkpoint/sqlite/sqlite_test.go
git commit -m "test(langgraph/checkpoint): shared savertest contract suite; sqlite and memory tests call it"
```

---

### Task 3: 嵌套 module `langgraph/checkpoint/postgres/` — Saver 实现

**Files:**
- Create: `langgraph/checkpoint/postgres/go.mod`、`go.sum`
- Create: `langgraph/checkpoint/postgres/doc.go` — 包文档（跨语言不互通声明等，见 Step 2）
- Create: `langgraph/checkpoint/postgres/migrations.go` — MIGRATIONS 常量（v0–v9 逐字照搬）
- Create: `langgraph/checkpoint/postgres/postgres.go` — Saver 全部实现
- Test: `langgraph/checkpoint/postgres/postgres_test.go` — 无需数据库的单元测试（`isInline` 表驱动、checkpoint 投影 encode/decode 往返、List 查询构造、MIGRATIONS 内容断言）

**Interfaces:**
- Consumes: Task 1 的 `checkpoint.Saver`（含 `Filter`/`TaskPath`）、`checkpoint.Serializer`（checkpoint.go:125-133，由调用方注入 `serde.NewJSONSerializer()`）、Task 2 的 `savertest.Run`（Task 4 消费）。
- Produces:
```go
package postgres // github.com/projanvil/langchain-golang/langgraph/checkpoint/postgres

func New(pool *pgxpool.Pool, serde checkpoint.Serializer) *Saver
func NewFromConnString(ctx context.Context, dsn string, serde checkpoint.Serializer) (*Saver, error)
// Setup 执行未应用的 schema migrations；必须由用户显式首调一次。
// 不在事务内执行（v6–v8 的 CREATE INDEX CONCURRENTLY 不能在事务块内运行）。
func (s *Saver) Setup(ctx context.Context) error
func (s *Saver) Close()
// Saver 实现 checkpoint.Saver（Task 1 扩展后的接口）。
```

- [ ] **Step 1: 脚手架嵌套 module** — `mkdir langgraph/checkpoint/postgres`，写 `go.mod`：
```go
module github.com/projanvil/langchain-golang/langgraph/checkpoint/postgres

go 1.23.0

require (
	github.com/jackc/pgx/v5 v5.x.x
	github.com/projanvil/langchain-golang v0.0.0
)

replace github.com/projanvil/langchain-golang => ../../../
```
然后 `cd langgraph/checkpoint/postgres && go get github.com/jackc/pgx/v5@latest && go mod tidy`（需要网络；若 module fetch 失败，STOP 并报告 BLOCKED）。
- [ ] **Step 2: 写 `doc.go`**（包注释，英文，含全部文档化分歧）：
```go
// Package postgres implements checkpoint.Saver on PostgreSQL, the Go port
// of Python's langgraph-checkpoint-postgres (BasePostgresSaver). It uses
// pgx/v5 with pgxpool and mirrors Python's four-table schema
// (checkpoints / checkpoint_blobs / checkpoint_writes /
// checkpoint_migrations) with per-version channel blobs.
//
// Setup must be called explicitly once before first use; it applies pending
// migrations WITHOUT a transaction (migrations v6–v8 are CREATE INDEX
// CONCURRENTLY, which cannot run inside a transaction block).
//
// Documented divergences from Python:
//   - checkpoint_blobs.version is BIGINT (Go channel versions are int64);
//     Python stores the column as TEXT holding decimal strings
//     (base.py casts versions via cast(str, ver)).
//   - Cross-language database sharing is NOT possible: the serde byte
//     formats differ (Go JSON typed envelopes vs Python msgpack), on top of
//     the version column type divergence. A database written by one
//     language cannot be read by the other.
//   - Channel values inline only JSON primitives (nil/string/bool/float64)
//     into the checkpoints JSONB document; int/int64 (serde-enveloped, not
//     JSON-native), maps, slices and all registry types go to
//     checkpoint_blobs (Python inlines str/int/float/bool and sends
//     dict/list to blobs).
//   - checkpoint.Metadata is a closed struct (Source/Step/Parents), so
//     ListOptions.Filter keys are limited to those three.
//   - No Shallow saver variant and no delta channel history fast path:
//     Python's ShallowPostgresSaver and _DeltaSnapshot have no Go
//     counterparts.
package postgres
```
- [ ] **Step 3: 写失败测试**（无需数据库，`-short` 也运行）——`postgres_test.go`：
  - `TestIsInline` 表驱动：`nil`→true、`"s"`→true、`true`→true、`float64(1.5)`→true；`7`→false、`int64(7)`→false、`map[string]any{}`→false、`[]any{1}`→false、`[]string{"a"}`→false、`types.Send{}`→false、`time.Time{}`→false（后三类是注册表类型）。
  - `TestSplitChannelValues`：混合 map 拆分后 inline 恰含原语键、blobs 恰含其余键（并集=原键集、交集为空）。
  - `TestCheckpointProjectionRoundTrip`：含 inline 原语 + typed Next Arg 的 checkpoint 经 `encodeCheckpoint`→`decodeCheckpoint`（无 blobs 合并）后 DeepEqual（ChannelValues 只含 inline 键）。
  - `TestMigrations`：`len(Migrations) == 10`；`Migrations[6]`、`[7]`、`[8]` 含 `"CREATE INDEX CONCURRENTLY"`；`Migrations[2]` 含 `"version BIGINT NOT NULL"`（Go 分歧行）；`Migrations[9]` 含 `"task_path"`。
  - `TestListQuery`：`listQuery` 对空 Filter/有 Filter/Before/Limit 各组合的 SQL 文本与参数个数断言（`@> $N::jsonb` 片段存在）。
  运行 `go test ./...` 验证编译失败（包仅有 doc.go/go.mod）。
- [ ] **Step 4: 写 `migrations.go`** — 逐字照搬 Python `base.py:43-91`，唯一改动为 v2 的 version 列类型（行内注释标明分歧）：
```go
package postgres

// Migrations mirrors Python's MIGRATIONS
// (libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py:43-91)
// statement-for-statement; the slice index IS the schema version. The only
// deliberate deviation is v2's version column: BIGINT instead of TEXT,
// because Go channel versions are int64 (documented in doc.go — one of the
// two reasons cross-language database sharing is impossible).
//
// Setup executes these WITHOUT a transaction: v6–v8 are CREATE INDEX
// CONCURRENTLY, which Postgres forbids inside a transaction block.
var Migrations = []string{
	// v0
	`CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);`,
	// v1
	`CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);`,
	// v2 — deviation: version BIGINT (Python: version TEXT NOT NULL)
	`CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version BIGINT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);`,
	// v3
	`CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);`,
	// v4
	`ALTER TABLE checkpoint_blobs ALTER COLUMN blob DROP not null;`,
	// v5 — no-op migration, kept so migration table versions stay aligned
	// with Python (mirrors base.py:78-80).
	`SELECT 1;`,
	// v6
	`
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id);
    `,
	// v7
	`
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
    `,
	// v8
	`
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
    `,
	// v9
	`ALTER TABLE checkpoint_writes ADD COLUMN IF NOT EXISTS task_path TEXT NOT NULL DEFAULT '';`,
}
```
- [ ] **Step 5: 实现 `postgres.go`** — 完整代码（SQL 常量逐字对齐 Python `base.py:131-159`，占位符 `%s`→`$n`；读路径拆 3 条查询，对齐 spec 的"不照搬单条嵌套 array_agg SQL"）：
```go
package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

// Write-path SQL, mirroring Python base.py:131-159 with $n placeholders.
const (
	upsertBlobSQL = `
    INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING`
	upsertCheckpointSQL = `
    INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
    DO UPDATE SET
        checkpoint = EXCLUDED.checkpoint,
        metadata = EXCLUDED.metadata`
	upsertWritesSQL = `
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
        channel = EXCLUDED.channel,
        type = EXCLUDED.type,
        blob = EXCLUDED.blob`
	insertWritesSQL = `
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING`
)

// reservedWriteIdx maps reserved channels to their negative write idx,
// mirroring Python's WRITES_IDX_MAP {ERROR:-1, SCHEDULED:-2, INTERRUPT:-3,
// RESUME:-4}; Go has no SCHEDULED/RESUME writes (same comment as sqlite's).
var reservedWriteIdx = map[string]int{
	checkpoint.ReservedError:     -1,
	checkpoint.ReservedTasks:     -2,
	checkpoint.ReservedInterrupt: -3,
}

// Saver is a checkpoint.Saver backed by PostgreSQL. It is safe for
// concurrent use (pgxpool manages the connections). The caller must call
// Setup once before first use, and Close when done.
type Saver struct {
	pool  *pgxpool.Pool
	serde checkpoint.Serializer
}

var _ checkpoint.Saver = (*Saver)(nil)

// New returns a Saver on pool, persisting through serde.
func New(pool *pgxpool.Pool, serde checkpoint.Serializer) *Saver {
	if serde == nil {
		panic("postgres: New requires a non-nil checkpoint.Serializer")
	}
	return &Saver{pool: pool, serde: serde}
}

// NewFromConnString opens a pgxpool from dsn and returns a Saver on it.
func NewFromConnString(ctx context.Context, dsn string, serde checkpoint.Serializer) (*Saver, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("postgres saver: connect: %w", err)
	}
	return New(pool, serde), nil
}

// Close closes the underlying pool.
func (s *Saver) Close() { s.pool.Close() }

// Setup applies pending schema migrations. It MUST be called explicitly once
// before first use (Python parity: PostgresSaver.setup). It does NOT wrap
// migrations in a transaction — v6–v8 are CREATE INDEX CONCURRENTLY, which
// Postgres forbids inside a transaction block.
func (s *Saver) Setup(ctx context.Context) error {
	if _, err := s.pool.Exec(ctx, Migrations[0]); err != nil {
		return fmt.Errorf("postgres saver: setup migrations table: %w", err)
	}
	var version int
	err := s.pool.QueryRow(ctx,
		`SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1`).Scan(&version)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		version = -1
	case err != nil:
		return fmt.Errorf("postgres saver: read migration version: %w", err)
	}
	for v := version + 1; v < len(Migrations); v++ {
		if _, err := s.pool.Exec(ctx, Migrations[v]); err != nil {
			return fmt.Errorf("postgres saver: migration %d: %w", v, err)
		}
		if _, err := s.pool.Exec(ctx,
			`INSERT INTO checkpoint_migrations (v) VALUES ($1)`, v); err != nil {
			return fmt.Errorf("postgres saver: record migration %d: %w", v, err)
		}
	}
	return nil
}

// isInline reports whether a channel value stays inline in the checkpoints
// JSONB document. Only JSON-native primitives inline — nil, string, bool,
// float64; int/int64 are serde-enveloped (not JSON-native) and everything
// composite goes to checkpoint_blobs (Python parity: __init__.py:316-319
// inlines primitives, sends dict/list to blobs).
func isInline(v any) bool {
	switch v.(type) {
	case nil, string, bool, float64:
		return true
	default:
		return false
	}
}

// splitChannelValues partitions values into inline primitives (kept in the
// checkpoints JSONB document) and blob values (stored per-version in
// checkpoint_blobs).
func splitChannelValues(values map[string]any) (inline, blobs map[string]any) {
	inline = map[string]any{}
	blobs = map[string]any{}
	for k, v := range values {
		if isInline(v) {
			inline[k] = v
		} else {
			blobs[k] = v
		}
	}
	return inline, blobs
}

// Put implements checkpoint.Saver. Blob rows are written only for channels
// present in newVersions (Python parity: __init__.py:322-324) with
// ON CONFLICT DO NOTHING (immutable versioned rows); the checkpoints row
// upserts.
func (s *Saver) Put(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, newVersions map[string]int64) (checkpoint.Config, error) {
	stored := cp
	if len(newVersions) > 0 {
		stored.ChannelVersions = maps.Clone(cp.ChannelVersions)
		if stored.ChannelVersions == nil {
			stored.ChannelVersions = map[string]int64{}
		}
		maps.Copy(stored.ChannelVersions, newVersions)
	}
	inline, blobValues := splitChannelValues(stored.ChannelValues)
	cpJSON, err := s.encodeCheckpoint(stored, inline)
	if err != nil {
		return checkpoint.Config{}, fmt.Errorf("postgres saver: encode checkpoint %q: %w", cp.ID, err)
	}
	mdJSON, err := json.Marshal(storedMetadata{Source: md.Source, Step: md.Step, Parents: md.Parents})
	if err != nil {
		return checkpoint.Config{}, fmt.Errorf("postgres saver: encode metadata for %q: %w", cp.ID, err)
	}
	var parent *string
	if cfg.CheckpointID != "" {
		parent = &cfg.CheckpointID
	}
	batch := &pgx.Batch{}
	// Sorted for deterministic batch order.
	for _, channel := range slices.Sorted(maps.Keys(blobValues)) {
		ver, ok := newVersions[channel]
		if !ok {
			continue // only channels in newVersions get blob rows (Python parity)
		}
		typ, data, err := s.serde.DumpsTyped(blobValues[channel])
		if err != nil {
			return checkpoint.Config{}, fmt.Errorf("postgres saver: encode channel %q: %w", channel, err)
		}
		batch.Queue(upsertBlobSQL, cfg.ThreadID, cfg.CheckpointNS, channel, ver, typ, data)
	}
	batch.Queue(upsertCheckpointSQL, cfg.ThreadID, cfg.CheckpointNS, cp.ID, parent, string(cpJSON), string(mdJSON))
	br := s.pool.SendBatch(ctx, batch)
	defer br.Close()
	for range batch.Len() {
		if _, err := br.Exec(); err != nil {
			return checkpoint.Config{}, fmt.Errorf("postgres saver: put checkpoint %q: %w", cp.ID, err)
		}
	}
	if err := br.Close(); err != nil {
		return checkpoint.Config{}, fmt.Errorf("postgres saver: put checkpoint %q: %w", cp.ID, err)
	}
	return checkpoint.Config{ThreadID: cfg.ThreadID, CheckpointNS: cfg.CheckpointNS, CheckpointID: cp.ID}, nil
}

// PutWrites implements checkpoint.Saver with the Python BATCH-level insert
// rule (base.py:363-367): an all-reserved batch UPSERTs under the reserved
// negative idx (re-invocation overwrites); any other batch INSERTs with
// ON CONFLICT DO NOTHING at the positional idx (first write wins).
func (s *Saver) PutWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID, taskPath string) error {
	cpID, err := s.resolveCheckpointID(ctx, cfg)
	if err != nil {
		return err
	}
	query := insertWritesSQL
	if allReserved(writes) {
		query = upsertWritesSQL
	}
	batch := &pgx.Batch{}
	for i, w := range writes {
		idx := i
		if reserved, ok := reservedWriteIdx[w.Channel]; ok {
			idx = reserved
		}
		typ, data, err := s.serde.DumpsTyped(w.Value)
		if err != nil {
			return fmt.Errorf("postgres saver: encode write %d to channel %q: %w", i, w.Channel, err)
		}
		batch.Queue(query, cfg.ThreadID, cfg.CheckpointNS, cpID, taskID, taskPath, idx, w.Channel, typ, data)
	}
	br := s.pool.SendBatch(ctx, batch)
	defer br.Close()
	for range batch.Len() {
		if _, err := br.Exec(); err != nil {
			return fmt.Errorf("postgres saver: put writes for checkpoint %q: %w", cpID, err)
		}
	}
	if err := br.Close(); err != nil {
		return fmt.Errorf("postgres saver: put writes for checkpoint %q: %w", cpID, err)
	}
	return nil
}

// GetTuple implements checkpoint.Saver.
func (s *Saver) GetTuple(ctx context.Context, cfg checkpoint.Config) (*checkpoint.Tuple, error) {
	var row pgx.Row
	if cfg.CheckpointID != "" {
		row = s.pool.QueryRow(ctx,
			`SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2 AND checkpoint_id = $3`,
			cfg.ThreadID, cfg.CheckpointNS, cfg.CheckpointID)
	} else {
		row = s.pool.QueryRow(ctx,
			`SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2 ORDER BY checkpoint_id DESC LIMIT 1`,
			cfg.ThreadID, cfg.CheckpointNS)
	}
	var cpID string
	var parent *string
	var cpJSON, mdJSON []byte
	if err := row.Scan(&cpID, &parent, &cpJSON, &mdJSON); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("postgres saver: get checkpoint: %w", err)
	}
	return s.assemble(ctx, cfg.ThreadID, cfg.CheckpointNS, cpID, parent, cpJSON, mdJSON)
}

// List implements checkpoint.Saver: newest checkpoint ID first, with
// Before, metadata @> Filter (server-side JSONB containment) and Limit.
func (s *Saver) List(ctx context.Context, cfg checkpoint.Config, opts checkpoint.ListOptions) ([]checkpoint.Tuple, error) {
	query, args := listQuery(cfg, opts)
	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("postgres saver: list thread %q: %w", cfg.ThreadID, err)
	}
	type rawRow struct {
		cpID   string
		parent *string
		cpJSON []byte
		mdJSON []byte
	}
	// Scan every row BEFORE issuing the per-checkpoint blobs/writes queries.
	var raw []rawRow
	for rows.Next() {
		var r rawRow
		if err := rows.Scan(&r.cpID, &r.parent, &r.cpJSON, &r.mdJSON); err != nil {
			rows.Close()
			return nil, fmt.Errorf("postgres saver: list thread %q: %w", cfg.ThreadID, err)
		}
		raw = append(raw, r)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return nil, fmt.Errorf("postgres saver: list thread %q: %w", cfg.ThreadID, err)
	}
	rows.Close()

	out := make([]checkpoint.Tuple, 0, len(raw))
	for _, r := range raw {
		tup, err := s.assemble(ctx, cfg.ThreadID, cfg.CheckpointNS, r.cpID, r.parent, r.cpJSON, r.mdJSON)
		if err != nil {
			return nil, err
		}
		out = append(out, *tup)
	}
	return out, nil
}

// listQuery builds the checkpoints SELECT with Before / metadata @> Filter /
// Limit predicates. Filter marshals to a JSONB containment argument
// (Python's `metadata @> %s`, base.py:655).
func listQuery(cfg checkpoint.Config, opts checkpoint.ListOptions) (string, []any) {
	query := `SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2`
	args := []any{cfg.ThreadID, cfg.CheckpointNS}
	if opts.Before != nil && opts.Before.CheckpointID != "" {
		args = append(args, opts.Before.CheckpointID)
		query += fmt.Sprintf(` AND checkpoint_id < $%d`, len(args))
	}
	if len(opts.Filter) > 0 {
		filterJSON, _ := json.Marshal(opts.Filter)
		args = append(args, string(filterJSON))
		query += fmt.Sprintf(` AND metadata @> $%d::jsonb`, len(args))
	}
	query += ` ORDER BY checkpoint_id DESC`
	if opts.Limit > 0 {
		args = append(args, opts.Limit)
		query += fmt.Sprintf(` LIMIT $%d`, len(args))
	}
	return query, args
}

// assemble decodes one checkpoints row and merges in its blob channel
// values and pending writes — the read side splits into 3 queries
// (checkpoints / blobs / writes) instead of Python's single nested
// array_agg SELECT (base.py:93-118).
func (s *Saver) assemble(ctx context.Context, threadID, ns, cpID string, parent *string, cpJSON, mdJSON []byte) (*checkpoint.Tuple, error) {
	cp, err := s.decodeCheckpoint(cpJSON)
	if err != nil {
		return nil, fmt.Errorf("postgres saver: decode checkpoint %q: %w", cpID, err)
	}
	blobValues, err := s.loadBlobs(ctx, threadID, ns, cp.ChannelVersions)
	if err != nil {
		return nil, err
	}
	if cp.ChannelValues == nil && len(blobValues) > 0 {
		cp.ChannelValues = map[string]any{}
	}
	// Blob values win over the inline reading (base.py:581-583 merge order).
	maps.Copy(cp.ChannelValues, blobValues)
	var md checkpoint.Metadata
	var storedMd storedMetadata
	if len(mdJSON) > 0 {
		if err := json.Unmarshal(mdJSON, &storedMd); err != nil {
			return nil, fmt.Errorf("postgres saver: decode metadata for %q: %w", cpID, err)
		}
		md = checkpoint.Metadata{Source: storedMd.Source, Step: storedMd.Step, Parents: storedMd.Parents}
	}
	writes, err := s.loadWrites(ctx, threadID, ns, cpID)
	if err != nil {
		return nil, err
	}
	tup := &checkpoint.Tuple{
		Config:        checkpoint.Config{ThreadID: threadID, CheckpointNS: ns, CheckpointID: cpID},
		Checkpoint:    cp,
		Metadata:      md,
		PendingWrites: writes,
	}
	if parent != nil {
		tup.ParentConfig = &checkpoint.Config{ThreadID: threadID, CheckpointNS: ns, CheckpointID: *parent}
	}
	return tup, nil
}

// loadBlobs fetches the blob rows for exactly the (channel, version) pairs
// in versions. Rows with type "empty" are skipped (Python's _load_blobs,
// base.py:375-384).
func (s *Saver) loadBlobs(ctx context.Context, threadID, ns string, versions map[string]int64) (map[string]any, error) {
	if len(versions) == 0 {
		return nil, nil
	}
	channels := make([]string, 0, len(versions))
	vers := make([]int64, 0, len(versions))
	for _, ch := range slices.Sorted(maps.Keys(versions)) {
		channels = append(channels, ch)
		vers = append(vers, versions[ch])
	}
	rows, err := s.pool.Query(ctx,
		`SELECT channel, type, blob FROM checkpoint_blobs WHERE thread_id = $1 AND checkpoint_ns = $2 AND (channel, version) IN (SELECT * FROM unnest($3::text[], $4::bigint[]))`,
		threadID, ns, channels, vers)
	if err != nil {
		return nil, fmt.Errorf("postgres saver: load blobs: %w", err)
	}
	defer rows.Close()
	out := map[string]any{}
	for rows.Next() {
		var channel, typ string
		var blob []byte
		if err := rows.Scan(&channel, &typ, &blob); err != nil {
			return nil, fmt.Errorf("postgres saver: load blobs: %w", err)
		}
		if typ == "empty" {
			continue
		}
		v, err := s.serde.LoadsTyped(typ, blob)
		if err != nil {
			return nil, fmt.Errorf("postgres saver: decode channel %q: %w", channel, err)
		}
		out[channel] = v
	}
	return out, rows.Err()
}

// loadWrites returns pending writes ordered by (task_id, idx), Python parity.
func (s *Saver) loadWrites(ctx context.Context, threadID, ns, cpID string) ([]checkpoint.Write, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT task_id, task_path, channel, type, blob FROM checkpoint_writes WHERE thread_id = $1 AND checkpoint_ns = $2 AND checkpoint_id = $3 ORDER BY task_id, idx`,
		threadID, ns, cpID)
	if err != nil {
		return nil, fmt.Errorf("postgres saver: load writes for checkpoint %q: %w", cpID, err)
	}
	defer rows.Close()
	var out []checkpoint.Write
	for rows.Next() {
		var w checkpoint.Write
		var typ string
		var data []byte
		if err := rows.Scan(&w.TaskID, &w.TaskPath, &w.Channel, &typ, &data); err != nil {
			return nil, fmt.Errorf("postgres saver: load writes for checkpoint %q: %w", cpID, err)
		}
		v, err := s.serde.LoadsTyped(typ, data)
		if err != nil {
			return nil, fmt.Errorf("postgres saver: decode write to channel %q: %w", w.Channel, err)
		}
		w.Value = v
		out = append(out, w)
	}
	return out, rows.Err()
}

// DeleteThread implements checkpoint.Saver.
func (s *Saver) DeleteThread(ctx context.Context, threadID string) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("postgres saver: delete thread %q: %w", threadID, err)
	}
	defer tx.Rollback(ctx)
	for _, table := range []string{"checkpoints", "checkpoint_blobs", "checkpoint_writes"} {
		if _, err := tx.Exec(ctx, `DELETE FROM `+table+` WHERE thread_id = $1`, threadID); err != nil {
			return fmt.Errorf("postgres saver: delete thread %q: %w", threadID, err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("postgres saver: delete thread %q: %w", threadID, err)
	}
	return nil
}

// resolveCheckpointID resolves cfg to a stored checkpoint ID — cfg's own ID,
// or the latest for the thread/namespace when empty — and errors when no
// such checkpoint exists, matching MemorySaver's PutWrites behavior.
func (s *Saver) resolveCheckpointID(ctx context.Context, cfg checkpoint.Config) (string, error) {
	var row pgx.Row
	if cfg.CheckpointID != "" {
		row = s.pool.QueryRow(ctx,
			`SELECT checkpoint_id FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2 AND checkpoint_id = $3`,
			cfg.ThreadID, cfg.CheckpointNS, cfg.CheckpointID)
	} else {
		row = s.pool.QueryRow(ctx,
			`SELECT checkpoint_id FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2 ORDER BY checkpoint_id DESC LIMIT 1`,
			cfg.ThreadID, cfg.CheckpointNS)
	}
	var id string
	if err := row.Scan(&id); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return "", fmt.Errorf("postgres saver: PutWrites: no checkpoint %q for thread %q (ns %q)",
				cfg.CheckpointID, cfg.ThreadID, cfg.CheckpointNS)
		}
		return "", fmt.Errorf("postgres saver: PutWrites: %w", err)
	}
	return id, nil
}

// allReserved reports whether every write in the batch targets a reserved
// channel (the empty batch vacuously qualifies, as in Python's `all(...)`).
func allReserved(writes []checkpoint.Write) bool {
	for _, w := range writes {
		if _, ok := reservedWriteIdx[w.Channel]; !ok {
			return false
		}
	}
	return true
}

// storedCheckpoint is the JSONB projection of checkpoint.Checkpoint persisted
// in the checkpoints table's checkpoint column. Unlike the sqlite
// projection, ChannelValues holds only INLINE primitives as plain JSON;
// non-primitive values live in checkpoint_blobs. Next task args remain
// serde-typed envelopes (storedValue), exactly as in the sqlite saver.
type storedCheckpoint struct {
	V               int                         `json:"v"`
	ID              string                      `json:"id"`
	TS              time.Time                   `json:"ts"`
	ChannelValues   map[string]any              `json:"channel_values,omitempty"`
	ChannelVersions map[string]int64            `json:"channel_versions,omitempty"`
	VersionsSeen    map[string]map[string]int64 `json:"versions_seen,omitempty"`
	Next            []storedTask                `json:"next,omitempty"`
}

// storedValue is one serde-typed value embedded in the checkpoint document.
type storedValue struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

// storedTask is the projection of checkpoint.PlannedTask.
type storedTask struct {
	ID   string                 `json:"id"`
	Node string                 `json:"node"`
	Arg  map[string]storedValue `json:"arg,omitempty"`
}

// storedMetadata is the plain-JSON projection of checkpoint.Metadata.
type storedMetadata struct {
	Source  string            `json:"source"`
	Step    int               `json:"step"`
	Parents map[string]string `json:"parents,omitempty"`
}

// encodeCheckpoint marshals the projection with inline channel values.
func (s *Saver) encodeCheckpoint(cp checkpoint.Checkpoint, inline map[string]any) ([]byte, error) {
	proj := storedCheckpoint{
		V:               cp.V,
		ID:              cp.ID,
		TS:              cp.TS,
		ChannelVersions: cp.ChannelVersions,
		VersionsSeen:    cp.VersionsSeen,
	}
	if len(inline) > 0 {
		proj.ChannelValues = inline
	}
	if cp.Next != nil {
		proj.Next = make([]storedTask, len(cp.Next))
		for i, task := range cp.Next {
			st := storedTask{ID: task.ID, Node: task.Node}
			if task.Arg != nil {
				st.Arg = make(map[string]storedValue, len(task.Arg))
				for k, v := range task.Arg {
					typ, data, err := s.serde.DumpsTyped(v)
					if err != nil {
						return nil, fmt.Errorf("next task %q arg %q: %w", task.ID, k, err)
					}
					st.Arg[k] = storedValue{Type: typ, Data: data}
				}
			}
			proj.Next[i] = st
		}
	}
	return json.Marshal(proj)
}

// decodeCheckpoint restores a Checkpoint from its JSONB document; ChannelValues
// contains only the inline entries (blob values are merged by assemble).
func (s *Saver) decodeCheckpoint(blob []byte) (checkpoint.Checkpoint, error) {
	var proj storedCheckpoint
	if err := json.Unmarshal(blob, &proj); err != nil {
		return checkpoint.Checkpoint{}, err
	}
	cp := checkpoint.Checkpoint{
		V:               proj.V,
		ID:              proj.ID,
		TS:              proj.TS,
		ChannelValues:   proj.ChannelValues,
		ChannelVersions: proj.ChannelVersions,
		VersionsSeen:    proj.VersionsSeen,
	}
	if proj.Next != nil {
		cp.Next = make([]checkpoint.PlannedTask, len(proj.Next))
		for i, st := range proj.Next {
			task := checkpoint.PlannedTask{ID: st.ID, Node: st.Node}
			if st.Arg != nil {
				task.Arg = make(map[string]any, len(st.Arg))
				for k, sv := range st.Arg {
					v, err := s.serde.LoadsTyped(sv.Type, sv.Data)
					if err != nil {
						return checkpoint.Checkpoint{}, fmt.Errorf("next task %q arg %q: %w", st.ID, k, err)
					}
					task.Arg[k] = v
				}
			}
			cp.Next[i] = task
		}
	}
	return cp, nil
}
```
注意：Python checkpoints 表的 `type` 列在 Go 中始终不写（Python 的 `UPSERT_CHECKPOINTS_SQL` 同样不含 type 列，`base.py:137-144`）；blob 的 type tag 在 blobs/writes 表各自的 `type` 列。`Put` 中 serde 编码失败的值直接报错，不静默降级（spec 错误处理节）。
- [ ] **Step 6: 门禁 PASS** — `cd langgraph/checkpoint/postgres && go build ./... && go vet ./... && go test ./...`（Step 3 的免数据库测试通过）；根 module `go build ./... && go vet ./... && go test ./...`（根 module 不 import 嵌套 module，必须保持绿）。
- [ ] **Step 7: Commit** —
```
git add langgraph/checkpoint/postgres/go.mod langgraph/checkpoint/postgres/go.sum langgraph/checkpoint/postgres/doc.go langgraph/checkpoint/postgres/migrations.go langgraph/checkpoint/postgres/postgres.go langgraph/checkpoint/postgres/postgres_test.go
git commit -m "feat(langgraph/checkpoint/postgres): PostgreSQL checkpoint saver (nested module, pgx/v5, four-table schema)"
```

---

### Task 4: postgres 测试（embedded-postgres）+ Makefile

**Files:**
- Create: `langgraph/checkpoint/postgres/postgres_db_test.go` — embedded-postgres 上的契约套件 + postgres 专属用例
- Modify: `langgraph/checkpoint/postgres/go.mod` / `go.sum` — 新增测试依赖 `github.com/fergusstrange/embedded-postgres`
- Modify: `Makefile` — 新增 `test-postgres` target

**Interfaces:**
- Consumes: Task 2 的 `savertest.Run(t, newSaver)`；Task 3 的 `New`/`NewFromConnString`/`Setup`/`Close`。
- Produces: `make test-postgres`（`cd langgraph/checkpoint/postgres && go test ./...`）。CI 注意：embedded-postgres 首次运行联网下载约 30MB 到 `~/.embedded-postgres-go`（可用 `CachePath` 配置），CI 需缓存该目录；`go test -short` 完全跳过数据库测试（门禁的 `-short` 路径必须存在）。

- [ ] **Step 1: 加测试依赖** — `cd langgraph/checkpoint/postgres && go get github.com/fergusstrange/embedded-postgres@latest && go mod tidy`（需要网络；失败则 STOP 报告 BLOCKED）。
- [ ] **Step 2: 写测试（先失败：无 Makefile target 且新测试文件编译不过时验证）** — `postgres_db_test.go` 完整骨架：
```go
package postgres_test

import (
	"context"
	"flag"
	"fmt"
	"os"
	"testing"

	embeddedpostgres "github.com/fergusstrange/embedded-postgres"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/postgres"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/savertest"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
)

// testDSN points at the package-wide embedded Postgres instance started in
// TestMain. Port 55433 avoids clashing with a locally installed Postgres.
const testPort = 55433

var testDSN = fmt.Sprintf("postgres://postgres:postgres@localhost:%d/postgres?sslmode=disable", testPort)

func TestMain(m *testing.M) {
	flag.Parse()
	if testing.Short() {
		// No database in -short mode; every test below skips itself.
		os.Exit(m.Run())
	}
	db := embeddedpostgres.NewDatabase(embeddedpostgres.DefaultConfig().Port(testPort))
	if err := db.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "embedded postgres start: %v\n", err)
		os.Exit(1)
	}
	code := m.Run()
	_ = db.Stop()
	os.Exit(code)
}

// newPool returns a pool on the shared embedded instance; the caller
// truncates via newEmptySaver. Skips in -short mode.
func newPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping embedded-postgres test in -short mode")
	}
	pool, err := pgxpool.New(context.Background(), testDSN)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}

// newEmptySaver returns a Saver on EMPTY storage (the savertest factory
// contract): shared tables are truncated between subtests.
func newEmptySaver(t *testing.T) checkpoint.Saver {
	t.Helper()
	pool := newPool(t)
	s := postgres.New(pool, serde.NewJSONSerializer())
	ctx := context.Background()
	if err := s.Setup(ctx); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	if _, err := pool.Exec(ctx, `TRUNCATE checkpoints, checkpoint_blobs, checkpoint_writes`); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	return s
}

func TestPostgresSaverContract(t *testing.T) {
	savertest.Run(t, newEmptySaver)
}
```
postgres 专属用例（同文件继续）：
  1. `TestSetupIdempotent`：`NewFromConnString(ctx, testDSN, serde.NewJSONSerializer())` → `Setup` 两次均成功；SQL 断言 `SELECT count(*), max(v) FROM checkpoint_migrations` 为 `(10, 9)`；第三次 `NewFromConnString` + `Setup` 后 count 仍为 10（幂等，移植/扩展 `test_sync.py:277` `test_nonnull_migrations` 的约束验证意图）。
  2. `TestInlineBlobSplit`（inline/blobs 边界，D6）：Put 一个 checkpoint，`ChannelValues = {"s": "x", "b": true, "f": 1.5, "nilv": nil, "n": 7, "m": map[string]any{"a": 1}, "l": []any{1}}`，`newVersions` 全键 version 1。SQL 断言：`SELECT channel FROM checkpoint_blobs WHERE thread_id=$1 ORDER BY channel` 恰好 `["l","m","n"]`（int 进 blobs——Go 与 Python 的关键边界）；`SELECT checkpoint->'channel_values' FROM checkpoints ...` 的 JSON 键恰好 `["b","f","nilv","s"]` 且值正确。再 `GetTuple` 断言 `n` 还原为 `int(7)`、`m` 为 `map[string]any`。
  3. `TestMetadataContainmentFilter`（`@>` 服务端过滤）：两条 checkpoint（`{Source:"input",Step:-1}` / `{Source:"loop",Step:0,Parents:{"":"p1"}}`）；`List` 带 `Filter{"source":"loop"}` 返回 1 条；`Filter{"parents": map[string]string{"": "p1"}}` 返回 1 条（JSONB 嵌套包含）；`Filter{"step":2}` 返回 0 条。
  4. `TestPerVersionBlobDedup`（大 channel 值 per-version 去重）：cp1 `Put`（channel `big` = 大 `[]string`，version 1，在 newVersions 中）；cp2 以新 checkpoint ID `Put`，`big` 仍为 version 1 且再次出现在 newVersions——`ON CONFLICT DO NOTHING` 去重，`SELECT count(*) FROM checkpoint_blobs WHERE channel='big'` 为 1；cp3 把 `big` 推进到 version 2 后为 2；且三次 `GetTuple` 均读回正确值。
  5. `TestNullCharsRejected`（移植 `test_sync.py:262` `test_null_chars`）：channel value / metadata-adjacent 字符串含 `\x00` 时 `Put` 返回非 nil error（JSONB 拒绝 `\u0000`）——断言报错而非静默写坏。
  6. `TestPutWritesTaskPathStored`：直接 SQL `SELECT task_path FROM checkpoint_writes` 断言 `PutWrites(..., "path/a")` 落库为 `'path/a'`、默认 `''`（v9 列真实生效）。
- [ ] **Step 3: Makefile** — `.PHONY` 行加 `test-postgres`，追加：
```make
## Run the nested Postgres checkpoint saver module's tests (its own go.mod).
## Spins up an in-process embedded Postgres (first run downloads ~30MB to
## ~/.embedded-postgres-go — cache that directory in CI). Use -short to skip:
##   cd langgraph/checkpoint/postgres && go test -short ./...
test-postgres:
	cd langgraph/checkpoint/postgres && go test ./...
```
- [ ] **Step 4: 门禁 PASS** — `make test-postgres`（真库全量）；`cd langgraph/checkpoint/postgres && go test -short ./...`（跳过路径验证）；根 module `go test ./...` 与 `make test-sqlite` 回归。
- [ ] **Step 5: Commit** —
```
git add langgraph/checkpoint/postgres/go.mod langgraph/checkpoint/postgres/go.sum langgraph/checkpoint/postgres/postgres_db_test.go Makefile
git commit -m "test(langgraph/checkpoint/postgres): contract suite on embedded-postgres plus migration/inline/filter/dedup cases"
```

---

## Self-Review Notes

- **Spec 覆盖**：根接口扩展（Filter/TaskPath/sqlite 加列/调用点）→ Task 1；savertest 套件 + sqlite 重构 → Task 2；嵌套 module + MIGRATIONS + inline/blobs + 3 查询读路径 + `@>` + BIGINT version + doc.go 声明 → Task 3；embedded-postgres 套件 + migration 幂等/inline 边界/`@>`/per-version 去重 + Makefile → Task 4。spec M5 节每条要求均可指向某个任务；无越界内容。
- **类型一致性**：Task 1 产出新 `Saver` 接口与 `MetadataMatchesFilter`，Task 2 套件建立在该签名上，Task 3 实现同一签名，Task 4 消费 Task 2 的 `savertest.Run`；`Write.TaskPath` 贯穿 memory/sqlite/postgres 三实现的读写作业。
- **与任务描述/现状的核实偏差**（均已在计划内处理）：
  1. Go 内存 saver 实际名为 `MemorySaver`/`NewMemorySaver`（memory.go:24/31），不是任务描述中的 `InMemorySaver`。
  2. Go `Metadata` 是封闭结构体（Source/Step/Parents）——`Filter` 只能作用于这三个键；Python `test_search` 的 `score` 键用例与 `test_combined_metadata`（config metadata 合并）无 Go 对应，按计划中的替代键移植/放弃并注明。
  3. Python `test_search` 的"跨 namespace 搜索"（test_sync.py:253-259）在 Go 接口不存在（`Config.CheckpointNS` 精确匹配），套件注释中声明该分歧。
  4. MIGRATIONS 逐字照搬的唯一例外是 v2 的 `version` 列 TEXT→BIGINT（spec 已定分歧），在 migrations.go 行内注释与 doc.go 双重声明。
  5. `PutWrites` 调用点核实：`graph/graph.go:929`（中断 pending-writes 路径）与 `graph/resume.go:54`（`persistInterrupts` 内部，汇聚 graph.go:752/:919/:985 三处中断持久化）——共两个编辑点，均传 `""`。
- **风险**：(a) pgx/v5 与 embedded-postgres 的 `go get` 需要网络（Task 3 Step 1 / Task 4 Step 1 均标明 BLOCKED 处理）；(b) Setup 绝不包事务（CONCURRENTLY 限制），代码注释与测试双重固定；(c) embedded-postgres 首跑下载 ~30MB，`-short` 跳过路径在 TestMain 与每个测试双重保障；(d) 嵌套 module 对根 `go test ./...` 不可见——Task 4 的 Makefile target 与 Step 4 门禁分别运行。
