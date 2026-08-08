# LangGraph Go Port M8: 文档刷新与中英双语化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 M5（Postgres saver）、M6（barrier join）、M7（函数式 API）实现落地之后，把全部用户文档刷新到 M1–M7 全量现状，并完成中英双语化：`README.md` 的 Supported / Not supported 段与仓库布局更新、`docs/usage/langgraph.md` 补齐 join edges / Postgres saver / 函数式 API 完整 guide / 接口 breaking 变更说明、其余四篇 usage guides 对照代码校对刷新，所有文档配 `<name>.zh-CN.md` 中文版，双语约定写入 `docs/README.md`。

**Architecture:** 英文版为主文件，中文版为同目录 `<name>.zh-CN.md` 姊妹文件；两份文件顶部互放 `English | 简体中文` 语言切换行；段落结构一一对应，代码块、链接、标识符逐字节一致。纯文档里程碑——不新增、不修改任何 Go 代码。

**Tech Stack:** Markdown；验收工具为 bash（相对链接检查、结构对等 diff）+ `go test ./...`（守护各包 `example_test.go` 不受影响）。

## Global Constraints

- 所有命令的工作目录：`langchain_golang/`（仓库根）。
- Go 1.23 下限；根 module 保持零第三方依赖（新依赖只允许出现在嵌套 module）——本里程碑不改 `go.mod`，此约束是验收时必须仍然成立的底线。
- `go test ./...` 全绿是不可降级的底线；`make test-sqlite`、`make test-postgres`（M5 落地后存在）不被破坏。`langchain/agents` 零改动——本里程碑不触碰任何 `.go` 文件。
- 内容对标 Python 移植基线：文档中每一处语义声明都要能指回 spec（`docs/superpowers/specs/2026-08-08-langgraph-go-m5-m8-design.md`）或 Python 源码（`../langgraph/libs/` @ `ea5f9cc9f`），禁止凭印象描述行为；写每个文件前先读本计划"核实代码事实"步骤列出的代码。
- 提交信息遵循仓库 conventional 格式（参照 `git log`：`feat(langgraph): ...` / `docs(langgraph): ...`），每个任务一次提交，`git add` 只加本任务的具体文件。
- 双语同步落笔：同一任务内英文版与中文版一起写完一起提交，不接受"先英文后补翻译"。
- 中文版要求：段落结构与英文版一一对应（标题层级序列、列表项数、表格行数相同）；代码块（含围栏语言标记）原样保留不翻译；技术术语与代码标识符（`AddJoinEdge`、`checkpoint.Saver`、`iter.Seq2` 等）保持英文原文；正文行文为通顺中文。
- 语言切换行格式（每个双语文件第 2 行，即 `# 标题` 之下空一行后）：
  - 英文主文件：`**Languages:** English | [简体中文](<name>.zh-CN.md)`
  - 中文姊妹文件：`**Languages:** [English](<name>.md) | 简体中文`
  - 链接为同目录相对路径，不带子目录前缀。
- 写作风格对标现有 `docs/usage/agents.md`：先给签名再给最短可运行示例，divergence / 限制用 `>` 引述块显著标出，段落简短，术语首次出现附 Python 对应物。
- 不要顺手重排无关文件（`langchain/` 下存在已知的 gofmt 漂移，与本里程碑无关，不碰）。

## 前置条件（开工前必须确认）

本计划在 M5–M7 实现完成后执行。开工第一步运行：

```bash
cd langchain_golang
ls langgraph/fn langgraph/checkpoint/postgres langgraph/checkpoint/savertest
grep -n "func (g \*StateGraph) AddJoinEdge" langgraph/graph/state.go
grep -n "taskPath" langgraph/checkpoint/checkpoint.go
grep -n "Filter" langgraph/checkpoint/checkpoint.go
```

预期：三个目录存在且各有 `doc.go`；`AddJoinEdge`、`PutWrites` 的 `taskPath` 参数、`ListOptions.Filter` 均出现。任一缺失 → **STOP with BLOCKED**（说明哪项未落地），不得按 spec 签名凭空撰写文档。

---

### Task 1: 双语约定落地 —— `docs/README.md` 更新 + `docs/README.zh-CN.md`

**Files:**
- Modify: `docs/README.md`
- Create: `docs/README.zh-CN.md`

**Interfaces:**
- Consumes: 无（本任务是其余任务的格式基线）。
- Produces:
  - 语言切换行格式（见 Global Constraints，本任务在两个文件上首次落地，后续任务照抄）。
  - `docs/README.md` 新增一节（建议置于 "Usage guides" 表格之后、"Examples in the repo" 之前），标题 `## Bilingual convention`，内容须包含三条约定：① 英文为主文件、中文为同目录 `<name>.zh-CN.md` 姊妹文件；② 两份文件顶部互放 `English | 简体中文` 切换行；③ 新增/修改文档内容必须双语同步落笔，不接受先英文后补。
  - 修复已知缺口：现 `docs/README.md` 的 "Usage guides" 表格（第 13–18 行）**缺 `usage/langgraph.md` 一行**——本任务补上：`[Graph runtime (langgraph/)](usage/langgraph.md) | StateGraph, checkpoints, Stream modes, savers, join edges, functional API`（该描述在 Task 3 完成后如与 guide 实际内容不符，由 Task 5 终验时校正）。

- [ ] **Step 1: 核实现状** —— `grep -n "langgraph" docs/README.md` 确认表格确无 langgraph.md 行；`ls docs/*.zh-CN.md docs/usage/*.zh-CN.md 2>&1` 确认尚无任何中文版（预期 `No such file`）。
- [ ] **Step 2: 修改 `docs/README.md`** —— 第 2 行加语言切换行（指向 `README.zh-CN.md`）；Usage guides 表格补 langgraph.md 行；新增 `## Bilingual convention` 节（Produce 中的三条约定逐条写入）。其余内容不动。
- [ ] **Step 3: 创建 `docs/README.zh-CN.md`** —— 与英文版逐段对应的中文翻译：标题 `# langchain-golang 文档`；第 2 行切换行指回 `README.md`；"Usage guides" 表格的 Guide 列翻译为中文、链接目标保持 `usage/getting-started.md` 等英文文件名不变（usage 各篇的中文版由 Task 3/4 创建，docs/README 表格只链主文件）；Bilingual convention 节同步翻译。
- [ ] **Step 4: 验收** —— 运行全局验收脚本（见文末"全局验收命令"，下同）：相对链接检查无 BROKEN 输出；`diff <(grep -oE '^#+' docs/README.md) <(grep -oE '^#+' docs/README.zh-CN.md)` 无输出（标题层级序列一致）；`go test ./...` 全绿（本任务未碰代码，此步为底线守护）。
- [ ] **Step 5: Commit** —— `git add docs/README.md docs/README.zh-CN.md && git commit -m "docs(langgraph): establish bilingual docs convention, add docs/README.zh-CN.md"`

---

### Task 2: 顶层 `README.md` 更新 + `README.zh-CN.md` 新建

**Files:**
- Modify: `README.md`
- Create: `README.zh-CN.md`

**Interfaces:**
- Consumes（M5–M7 落地后的真实 API，写文档前逐一 grep 核实，签名以代码为准、语义以 spec 为准）：
  - `langgraph/checkpoint/postgres`：`postgres.New(pool *pgxpool.Pool, serde checkpoint.Serializer) *Saver`、`postgres.NewFromConnString(ctx context.Context, dsn string, serde checkpoint.Serializer) (*Saver, error)`、`(*Saver).Setup(ctx context.Context) error`。
  - `langgraph/checkpoint/savertest`：共享 saver 契约套件（对标 Python `libs/checkpoint-postgres/tests/` 与 `libs/langgraph/tests/test_checkpoint*.py` 的契约用例）。
  - M5 接口扩展（breaking）：`checkpoint.ListOptions` 新增 `Filter map[string]any`；`Saver.PutWrites` 新增 `taskPath string` 参数；`checkpoint.Write` 新增 `TaskPath string` 字段。
  - M6：`graph.(*StateGraph).AddJoinEdge(from []string, to string) error`、`channels.Barrier`。
  - M7：`langgraph/fn` 包（`NewTask` / `NewEntrypoint` / `NewEntrypointFinal` / `Final` / `AwaitAll`，完整签名见 Task 3 的 Interfaces 块）。
- Produces: 更新后的 `README.md` 与逐段对应的中文版。逐条要求：

  1. **Supported 段**（现有 langgraph 相关条目位于 "Not supported" 的 "A full `langgraph` port" bullet 第 74 行——注意：目前 langgraph 能力清单写在 Not supported 段的该 bullet 内）补三类条目：
     - M5：Postgres checkpoint saver（嵌套 module `langgraph/checkpoint/postgres`，`pgx/v5` 驱动，`New` / `NewFromConnString` / 显式 `Setup` 跑 migrations；`make test-postgres` 用 embedded-postgres 跑共享契约套件）；`savertest` 共享契约套件；Saver 接口扩展（`ListOptions.Filter` metadata 过滤、`PutWrites` taskPath）。
     - M6：`AddJoinEdge` 多父 barrier join（对齐 Python `add_edge((a, b), c)` 的 waiting edge）。
     - M7：`fn` 包函数式 API（`NewEntrypoint` / `NewTask`，对齐 Python `@entrypoint` / `@task`）。
  2. **Not supported 段**（第 69–76 行）：
     - 从 "A full `langgraph` port" bullet 的 "Intentionally absent" 列表中**删除** "the functional `@entrypoint`/`@task` API" 与 "a persistent Postgres checkpoint backend"；
     - **删除**独立的 "- **Functional `@entrypoint`/`@task` API** — see above." bullet（第 76 行整行）；
     - "Intentionally absent" 列表保留并补齐为：a graph-level default retry policy、the langgraph CLI/SDK、`defer=True` 节点（`NamedBarrierValueAfterFinish`）、Shallow saver 变体、delta channel history 快路径、函数式 API 的 `store` 参数（Python `@entrypoint(checkpointer=..., store=...)` 的跨线程 BaseStore 未移植）。同时在同 bullet 补一句 Postgres saver 的存在形式（嵌套 module、第二个第三方依赖 `pgx/v5`、经 `make test-postgres` 测试），与现有 SQLite 那句（第 74 行）句式对齐。
  3. **What this is 段**（第 19 行）：langgraph 能力枚举补 `join edges (AddJoinEdge)`、`Postgres checkpoint saver`、`functional API (fn package)` 三项。
  4. **测试计数**（第 21 行 "830+ tests across 56 packages"）：用下方命令实测更新（向下取整到十位，保留 `+`）。
  5. **Repository layout**（第 170–184 行）：`langgraph/` 行描述补 join edges / fn / postgres；`checkpoint/sqlite/` 行之后补两行：`│   ├── checkpoint/postgres/ # nested Go module: Postgres checkpoint saver (pgx/v5); test it with make test-postgres` 与 `│   └── fn/                # functional API: NewEntrypoint / NewTask (Python @entrypoint/@task)`（树形符号与现有行对齐，`postgres` 行用 `├──`、`fn` 行用 `└──`，`sqlite` 行的 `└──` 相应改 `├──`）。
  6. **Documentation 段**（第 164 行）：graph runtime guide 条目描述更新为含 join edges、Postgres saver、函数式 API。
  7. **不变**：Quick start、Installation、Acknowledgments、License、partner 段——逐字不动。

- [ ] **Step 1: 核实代码事实** —— 运行前置条件块的全部命令；另运行：
  ```bash
  go test ./... 2>/dev/null | grep -c '^ok'        # 包数（现 README 写 56）
  go test ./... -v -count=1 2>/dev/null | grep -c '^=== RUN'   # 测试数（现 README 写 830+）
  grep -n "func NewTask\|func NewEntrypoint" langgraph/fn/*.go
  grep -n "test-postgres" Makefile
  ```
  记录实测数字与签名；若 `make test-postgres` target 不存在 → STOP with BLOCKED。
- [ ] **Step 2: 改 `README.md` 英文版** —— 按 Produces 第 1–6 条逐条落笔；第 21 行计数替换为 Step 1 实测值。改完通读一遍确认 "Intentionally absent" 列表不再含 Postgres / 函数式 API。
- [ ] **Step 3: 创建 `README.zh-CN.md`** —— 全文逐段翻译（含 Supported / Not supported 全部条目、Quick start 代码块原样、Repository layout 代码块原样）。标题行 `# langchain-golang` 不变，第 2 行切换行指回 `README.md`。badge 行（shields.io 图片链接）原样保留。
- [ ] **Step 4: 验收** —— 全局验收脚本全过；结构对等 diff（标题层级 + 代码块 + 归一化链接）无输出；`go test ./...` 全绿。
- [ ] **Step 5: Commit** —— `git add README.md README.zh-CN.md && git commit -m "docs(langgraph): update README for M5-M7, add README.zh-CN.md"`

---

### Task 3: `docs/usage/langgraph.md` 刷新至 M1–M7 全量 + `langgraph.zh-CN.md`

**Files:**
- Modify: `docs/usage/langgraph.md`
- Create: `docs/usage/langgraph.zh-CN.md`

**Interfaces:**
- Consumes —— 现状文件已覆盖 M3（Stream / serde / SQLite）与 M4（retry / cache / `prebuilt.ToolNode`），这些章节只做校对不做重写；本任务**新增**四块内容，所引签名如下（M5–M7 已定稿于此，写文档前仍须按 Step 1 grep 核实实现与之一致）：

  Postgres saver（`langgraph/checkpoint/postgres`，嵌套 module）：

  ```go
  func New(pool *pgxpool.Pool, serde checkpoint.Serializer) *Saver
  func NewFromConnString(ctx context.Context, dsn string, serde checkpoint.Serializer) (*Saver, error)
  func (s *Saver) Setup(ctx context.Context) error // 必须显式首调；不在事务内执行
  ```

  Barrier join：

  ```go
  func (g *StateGraph) AddJoinEdge(from []string, to string) error // 校验: ≥2 个去重父节点、节点已注册、to 非 START/END
  // channels.Barrier 实现 channels.Channel; 执行器经类型断言 interface{ Consume() bool } 复位
  ```

  函数式 API（`langgraph/fn`）：

  ```go
  type TaskOpts struct {
      Retry   *graph.RetryPolicy
      Cache   *graph.CachePolicy
      Timeout time.Duration
  }
  func NewTask[I, O any](name string, f func(context.Context, I) (O, error), opts TaskOpts) *Task[I, O]
  func (t *Task[I, O]) Call(ctx context.Context, in I) *Future[O]
  func (f *Future[O]) Get(ctx context.Context) (O, error)
  func AwaitAll[T any](ctx context.Context, futs ...*Future[T]) ([]T, error)

  type EntrypointOpts struct {
      Checkpointer checkpoint.Saver
      Cache        checkpoint.Cache
      Retry        *graph.RetryPolicy
  }
  func NewEntrypoint[I, O, S any](opts EntrypointOpts,
      f func(ctx context.Context, in I, prev S, hasPrev bool) (O, error)) *Entrypoint[I, O, S]
  type Final[O, S any] struct{ Value O; Save S }
  func NewEntrypointFinal[I, O, S any](opts EntrypointOpts,
      f func(ctx context.Context, in I, prev S, hasPrev bool) (Final[O, S], error)) *Entrypoint[I, O, S]
  func (e *Entrypoint[I, O, S]) Invoke(ctx context.Context, in I, opts graph.Options) (O, error)
  func (e *Entrypoint[I, O, S]) Stream(ctx context.Context, in I, opts graph.Options) iter.Seq2[graph.StreamChunk, error]
  ```

  M5 接口 breaking 变更（before 为当前真实代码 `langgraph/checkpoint/checkpoint.go:159-166, 96-104, 189`；after 为 M5 落地形态，以代码为准）：

  ```go
  // before
  type ListOptions struct { Before *Config; Limit int }
  PutWrites(ctx context.Context, cfg Config, writes []Write, taskID string) error
  type Write struct { TaskID string; Channel string; Value any }
  // after
  type ListOptions struct { Before *Config; Limit int; Filter map[string]any }
  PutWrites(ctx context.Context, cfg Config, writes []Write, taskID string, taskPath string) error
  type Write struct { TaskID string; Channel string; Value any; TaskPath string }
  ```

- Produces —— `langgraph.md` 在现有章节基础上追加以下新章节（顺序接在 "`prebuilt.ToolNode`" 节之后、"`create_react_agent` ≡ `agents.CreateAgent`" 节之前），并把标题与开头段落更新为 M1–M7 全量口径：

  1. **Join edges (`AddJoinEdge`)**：语义 = 所有父节点都完成后子节点在下一超步**恰好运行一次**（同超步多父完成也只一次）；对齐 Python `add_edge((a, b), c)` → waiting edge（`../langgraph/libs/langgraph/langgraph/graph/state.py:956-966`）+ `NamedBarrierValue`（`../langgraph/libs/langgraph/langgraph/channels/named_barrier_value.py`）；到达记录进 checkpoint，"父 A 已到达、父 B 中断→resume 补写"天然正确；循环图中子节点写提交后 barrier 复位、可再次触发。最短示例（两父一子）须可对应到真实 API。校验规则：≥2 个去重父节点、父已注册、to 非 `types.START`/`types.END`，重复父报错——**文档化分歧**：Python 接受单元素起点并静默去重，Go 收紧。`>` 引述块显著警告**绕过语义**：普通边/条件边、`Send`、`Command.Goto` 指向 join 子节点时绕过 barrier 直接触发（OR 语义），同一节点混用两类边可能被触发多次——这是 Python 既定行为，Go 忠实复刻，不是 bug。另一 `>` 块：不支持 `defer=True` / `NamedBarrierValueAfterFinish`（边驱动模型无等价物）。
  2. **Postgres checkpoint saver**：`New` / `NewFromConnString` / `Setup` 示例（对照现有 SQLite 节的写法）；`Setup` 必须显式首调、执行 v0–v9 migrations、不包事务（`CREATE INDEX CONCURRENTLY` 限制，Python 基线 `../langgraph/libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py:43-91` 的四表 + 迁移表 schema）；依赖声明（嵌套 module，第二个第三方依赖 `github.com/jackc/pgx/v5`，纯 Go；根 module 仍零依赖）；测试方式：`make test-postgres` 用 `github.com/fergusstrange/embedded-postgres` 进程内起真库，`-short` 跳过，首次运行联网下载 ~30MB；`savertest` 共享契约套件（sqlite 与 postgres 跑同一套）。`>` 引述块**跨语言不互通声明**：Go serde 为 JSON + 封闭类型注册表（非 Python msgpack），且 blobs 表 `version` 列 Go 用 `BIGINT`（Python 为 TEXT 存十进制字符串）——Go/Python 各自建的库互不通用。另一 `>` 块：inline 判定分歧——仅 `nil`/`string`/`bool`/`float64` 内联进 checkpoint JSONB，`int`/`int64`/`map[string]any`/`[]any` 及一切其他类型进 blobs 表；无 Shallow saver 变体、无 delta channel history。
  3. **Functional API (`langgraph/fn`)**：完整 guide，章节顺序对齐 Python functional API 文档结构（`../langgraph/libs/langgraph/langgraph/func/__init__.py:576-596` 的单节点 Pregel 编译模型）：
     - 概述：普通 Go 控制流 + 任务级 checkpoint 重放；何时用函数式 API 而非 StateGraph。
     - `NewEntrypoint` 基础：最短 invoke 示例；三个保留 channel key `__start__`/`__end__`/`__previous__` 的实现事实一句话带过。
     - `previous` 跨轮状态：`prev` / `hasPrev` 语义；**分歧**：Python 无 checkpointer 时 `previous=None`，Go 用显式 `hasPrev bool` 防误读零值。
     - `NewEntrypointFinal` / `Final{Value, Save}`：解耦返回值与写入 `__previous__` 的保存值（对齐 `entrypoint.final`），value≠save 示例。
     - `NewTask` + `Call` + `Future.Get`：task 在 goroutine 立即执行（无 Python 的"下一 tick"调度——Go 边驱动无 tick 概念）；`Call` 仅可在 entrypoint 函数 / 其他 task / StateGraph 节点内调用，否则 panic。
     - `AwaitAll` 并发示例。
     - `TaskOpts` 的 retry / cache / timeout：retry/cache 复用 `graph.RetryPolicy` / `graph.CachePolicy` 与 `checkpoint.Cache`，cache key 为调用参数哈希、命名空间 `__fn_writes/<task-name>`；**分歧**：`Timeout` 只能做到 ctx 取消 + 放弃等待（goroutine 不可强杀；Python sync 函数同样不支持 timeout）。
     - Checkpoint 重放与 determinism 约束：确定性 task ID（hash 输入含恢复所用 checkpoint ID、step、task name、call idx）；resume 时 entrypoint 从头重跑、命中 pending writes 的 task 直接回填 future 不重跑（错误同样持久化并重抛）；per-run call counter 从零重放——**文档约束**：重跑时 task 调用顺序必须确定（非确定性逻辑放进 task 内），interrupt 的 resume 值按索引顺序匹配（同 Python determinism 一节）；interrupt 触发时已启动未完成的 task 被取消、已完成结果落 pending writes 后暂停；**不支持 `store`**（Python `@entrypoint` 的跨线程 BaseStore 未移植，`EntrypointOpts` 无此字段）；I/O/S 及 task 输入输出必须 JSON 可往返（serde 封闭注册表），未注册类型明确报错。
  4. **Breaking changes (M5 saver interface)**：一节列出 before/after 签名对照（用本任务 Interfaces 块的代码块原样呈现），并给自定义 `Saver` 实现者的迁移指引：实现体把 `taskPath` 透传进 `Write.TaskPath`（或忽略并传 `""`）；`List` 增加 `Filter` 的 map 包含过滤语义（postgres 服务端 `@>`，memory/sqlite 进程内 map 包含比较，语义相同）；sqlite 旧库兼容（启动时检测 `task_path` 列缺失则 `ALTER TABLE ADD COLUMN`）一句说明。标注这是 spec 允许的 pre-1.0 break。

  另：文件标题（第 1 行）更新为含 join edges / Postgres / functional API；开头段落（第 1–9 行）的 "covers the M3 additions ... and the M4 additions" 改为 M1–M7 全量导览口径。

- [ ] **Step 1: 核实代码事实** —— 除前置条件块外，运行：
  ```bash
  grep -n "func New\|func (s \*Saver) Setup" langgraph/checkpoint/postgres/*.go
  grep -n "func NewTask\|func NewEntrypoint\|func NewEntrypointFinal\|func AwaitAll\|type Final\|type TaskOpts\|type EntrypointOpts" langgraph/fn/*.go
  grep -n "type Barrier" langgraph/channels/*.go
  grep -n "func Run\|func RunAll\|func Suite" langgraph/checkpoint/savertest/*.go | head
  ```
  逐签名比对本任务 Interfaces 块；任何签名漂移 → 以代码为准修订文档文字，并在提交信息中注明。
- [ ] **Step 2: 校对现有章节** —— 重读 Stream / serde / SQLite / retry / cache / `prebuilt.ToolNode` 各节，对照 `langgraph/graph/stream.go`、`langgraph/checkpoint/serde/json.go`、`langgraph/checkpoint/sqlite/sqlite.go`、`langgraph/graph/policy.go`、`langgraph/prebuilt/` 现状核对每一条声明；只修正确证过时的表述，不做文风改写。
- [ ] **Step 3: 写英文版新章节** —— 按 Produces 1–4 落笔，更新标题与开头段落。
- [ ] **Step 4: 创建 `docs/usage/langgraph.zh-CN.md`** —— 全文（含原有 M3/M4 章节）逐段中文翻译；代码块原样；切换行指回 `langgraph.md`。
- [ ] **Step 5: 验收** —— 全局验收脚本全过；`diff` 结构对等（标题层级、代码块、归一化链接）无输出；`go test ./...` 全绿。
- [ ] **Step 6: Commit** —— `git add docs/usage/langgraph.md docs/usage/langgraph.zh-CN.md && git commit -m "docs(langgraph): refresh graph runtime guide for M5-M7, add zh-CN translation"`

---

### Task 4: 其余四篇 usage guides 校对刷新 + 中文版

**Files:**
- Modify: `docs/usage/getting-started.md`、`docs/usage/composition.md`、`docs/usage/agents.md`、`docs/usage/streaming.md`（仅修正确证过时的表述；无过时则不改）
- Create: `docs/usage/getting-started.zh-CN.md`、`docs/usage/composition.zh-CN.md`、`docs/usage/agents.zh-CN.md`、`docs/usage/streaming.zh-CN.md`

**Interfaces:**
- Consumes: Task 1 的语言切换行格式；Task 3 的 `langgraph.md` 新内容（agents.md 引用它）。
- Produces:
  1. **agents.md 必改项**：
     - "API note" 引述块（第 179–187 行）：Saver 接口列举更新为 M5 落地形态——`GetTuple` / `List`（`ListOptions` 含 `Filter`）/ `Put` / `PutWrites`（含 `taskPath`）/ `DeleteThread`；句末补一句指向 `langgraph.md` 的 Breaking changes 节。
     - "Durable checkpoints" 引述块（第 209–213 行）：在 SQLite 之后补 Postgres 选项一句——`postgres.NewFromConnString(ctx, dsn, serde.NewJSONSerializer())` + 显式 `Setup`，同样可用于 `WithAgentCheckpointer`。
  2. **getting-started.md 必改项**："Where to go next" 列表（第 134–140 行）补第四条：`[Graph runtime (langgraph/)](langgraph.md)` — checkpoints, stream modes, savers, join edges, functional API。
  3. **校对项（核实后仅在确证过时时修改）**：
     - streaming.md 的 "Note on caching" 节：`StreamEvents` 绕过 cache 的声明已与代码核实一致（`langchain/agents/create_agent.go:224`、`:328`），预期**不改**，仅复核。
     - composition.md：与 M5–M7 无关，复核 `core/runnables` 签名未被改动即可（`grep -n "func Pipe[0-9]*(\|func NewParallel\|func NewBranch\|func NewWithFallbacks\|func NewRetry" core/runnables/*.go`）。
     - agents.md 其余章节、getting-started.md 其余章节：对照 `langchain/agents/create_agent.go` 的 `WithAgent*` 选项集复核一遍。
  4. **四个 `.zh-CN.md`**：各自与英文版逐段对应；代码块原样；切换行互指。

- [ ] **Step 1: 核实代码事实** —— 运行上方各 grep；并确认 `agents.md` 引用的测试仍存在：`grep -n "TestCreateAgent_InterruptBeforeNode" langchain/agents/create_agent_test.go`（若改名则更新引用）。
- [ ] **Step 2: 修改英文版** —— 落笔 Produces 第 1、2 条的必改项与 Step 1 确证的过时表述；每篇加语言切换行（指向各自 `.zh-CN.md`）。
- [ ] **Step 3: 创建四个中文版** —— 逐段翻译；切换行指回英文主文件。
- [ ] **Step 4: 验收** —— 全局验收脚本全过；四组文件逐一跑结构对等 diff（标题层级、代码块、归一化链接）无输出；`go test ./...` 全绿（守护 `example_test.go`）。
- [ ] **Step 5: Commit** —— `git add docs/usage/getting-started.md docs/usage/getting-started.zh-CN.md docs/usage/composition.md docs/usage/composition.zh-CN.md docs/usage/agents.md docs/usage/agents.zh-CN.md docs/usage/streaming.md docs/usage/streaming.zh-CN.md && git commit -m "docs(langgraph): proofread usage guides against code, add zh-CN translations"`

---

### Task 5: 新包 doc.go 分歧审计 + spec 标记 + 全仓终验

**Files:**
- Audit（预期只读；缺漏时 Modify）: `langgraph/checkpoint/postgres/doc.go`、`langgraph/checkpoint/savertest/doc.go`、`langgraph/fn/doc.go`、`langgraph/channels/`（Barrier 的 doc comment）、`langgraph/graph/state.go`（`AddJoinEdge` 的 doc comment）
- Modify: `docs/superpowers/specs/2026-08-08-langgraph-go-m5-m8-design.md`（标记 M8 完成，写实际日期）

**Interfaces:**
- Consumes: spec 各节"文档化分歧"最低清单；Task 1–4 的双语产物。
- Produces: 审计结论（每条分歧 → 所在 doc.go/注释的 grep 命中）；spec 状态更新。

- [ ] **Step 1: doc.go 分歧清单审计** —— 逐条 grep，缺哪条补哪条（doc.go 用英文，遵循 Go 包注释惯例）：
  ```bash
  # postgres doc.go 须含：BIGINT version 分歧；仅原语 inline；无 Shallow/delta；跨语言不互通（serde + version 双重分歧）
  grep -n -i "bigint\|inline\|shallow\|delta\|interop\|msgpack" langgraph/checkpoint/postgres/doc.go
  # fn doc.go 须含：timeout 语义（ctx 取消 + 放弃等待）；interrupt 取消未完成 task；hasPrev 显式 bool；无 store；determinism 约束
  grep -n -i "timeout\|interrupt\|hasPrev\|store\|determinis" langgraph/fn/doc.go
  # Barrier / AddJoinEdge 注释须含：单元素起点收紧（≥2、去重、重复报错）；不支持 defer/NamedBarrierValueAfterFinish
  grep -n -i "defer\|barrier" langgraph/channels/*.go langgraph/graph/state.go | head
  ```
- [ ] **Step 2: spec 标记** —— 在 spec 的 M8 节末尾加一行完成标记（实际日期），参照 M1–M4 spec 的既有标记格式。
- [ ] **Step 3: 全仓终验** —— 完整运行全局验收脚本（相对链接检查、全部 7 组双语文件的结构对等 diff）；`go test ./...` 全绿；`make test-sqlite`、`make test-postgres` 不被破坏（docs-only，预期无影响，跑一次确认）；人工通读英文版与中文版各一遍（spec 的 M8 验收要求）。
- [ ] **Step 4: Commit** —— `git add docs/superpowers/specs/2026-08-08-langgraph-go-m5-m8-design.md`（若 Step 1 补了 doc.go 则一并 `git add` 对应文件）`&& git commit -m "docs(langgraph): audit M5-M7 doc.go divergence notes; mark M8 done"`

---

## 全局验收命令（每个任务的 Step "验收" 引用此块）

```bash
cd langchain_golang

# 1. 相对链接检查（预期：无任何 BROKEN 输出）
for f in README.md README.zh-CN.md docs/README.md docs/README.zh-CN.md docs/usage/*.md; do
  d=$(dirname "$f")
  grep -oE '\]\(([^)]+)\)' "$f" | sed -E 's/^\]\(//; s/\)$//; s/#.*$//' | sort -u | while read -r link; do
    case "$link" in ""|http://*|https://*|mailto:*) continue;; esac
    [ -e "$d/$link" ] || echo "BROKEN: $f -> $link"
  done
done

# 2. 双语结构对等（对每一对 <name>.md / <name>.zh-CN.md 运行；预期：三次 diff 均无输出）
en=docs/usage/langgraph.md; zh=docs/usage/langgraph.zh-CN.md
diff <(grep -oE '^#+' "$en") <(grep -oE '^#+' "$zh")                       # 标题层级序列一致
diff <(awk '/^```/{f=!f; print; next} f' "$en") \
     <(awk '/^```/{f=!f; print; next} f' "$zh")                           # 代码块逐字节一致
norm() { grep -oE '\]\(([^)#]+)' "$1" | sed -E 's/^\]\(//; s/\.zh-CN\.md$/.md/' | sort -u; }
diff <(norm "$en") <(norm "$zh")                                          # 链接集合一致（切换行归一化后）

# 3. 代码底线（守护 example_test.go 等编译检查型示例不受影响）
go test ./...
```

---

## Self-Review Notes

- Spec M8 覆盖：范围 1（README + 双语）→ Task 2；范围 2（docs/README + 双语约定）→ Task 1；范围 3（langgraph.md 全量刷新 + 双语）→ Task 3；范围 4（其余 guides 校对 + 双语）→ Task 4；范围 5（新包 doc.go 分歧清单）→ Task 5 Step 1。验收条款（双语对等、`example_test.go` 不受影响、链接检查）→ 全局验收命令块，每个任务引用，Task 5 Step 3 全仓跑一遍。无 spec 外扩张。
- 任务间一致性：Task 1 定语言切换行格式，Task 2–4 照抄；Task 3 的 fn/postgres/AddJoinEdge 签名是 Task 2 README 条目的引用源；Task 4 agents.md 的 "Durable checkpoints" 块引用 Task 3 的 langgraph.md；Task 5 依赖 Task 1–4 全部产物做终验。执行顺序即任务编号顺序。
- 风险：(a) M5–M7 未真正落地 → 前置条件块 STOP with BLOCKED，禁止按 spec 凭空写文档；(b) M5 落地签名与本计划 Interfaces 块漂移 → Task 3 Step 1 显式授权"以代码为准修订"并在提交信息注明；(c) 链接检查的 `while` 子 shell 无法回传计数，故以"BROKEN 行输出为空"为判定标准，已写入脚本注释语义；(d) docs/README.md 表格缺 langgraph.md 行是本次核实中发现的既有缺口，已在 Task 1 修复并在 Task 5 终验复核描述一致性。
