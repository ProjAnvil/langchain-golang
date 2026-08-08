# LangGraph Go 移植 M5–M8 设计：Postgres saver、多父 barrier join、函数式 API、文档双语化

日期：2026-08-08
状态：已批准（用户于 2026-08-08 确认方案选型）
上游基线：`langchain-ai/langgraph` main @ `ea5f9cc9f`（1.2.x 之后）
前置 spec：`2026-08-07-langgraph-go-port-design.md`（M1–M4 已完成）

## 目标

在已完成的 M1–M4（边驱动核心、版本化 checkpoint、stream modes、SQLite saver、prebuilt/retry/cache）之上，补齐三个被显式推迟的能力，并同步刷新用户文档：

- **M5**：Postgres checkpoint saver（对齐 `langgraph-checkpoint-postgres`）。
- **M6**：多父节点 barrier join（对齐 `add_edge((a, b), c)` + `NamedBarrierValue`）。
- **M7**：函数式 API（对齐 `langgraph.func` 的 `@entrypoint` / `@task`）。
- **M8**：文档刷新与中英双语化（README + 全部 usage guides）。

三者均忠实复用现有 channel/checkpoint 基建，不引入新的执行模型。

## 已否决的备选方案

- **M6-B 执行器内联 join 计数器**：不进 channel 体系。与版本簿记脱节（防重复触发、去重要重做），checkpoint 结构与 Python 分叉，"父中断→resume 重放"最易出错。否决。
- **M6-C 仅超步内去重**：跨超步 join 仍是静默错误语义，与 Python 不一致。否决（其去重效果由 M6-A 天然覆盖）。
- **M5-二 pgx stdlib 适配器走 `database/sql`**：丢 pgx 原生 JSONB/BYTEA 类型与 pgxpool 连接管理，是方案一的低配版。否决。
- **M5-三 两表整 blob 模型（沿用 sqlite schema）**：失去 per-version channel blob 去重（大 channel 值每次 checkpoint 全量重写）、堵死将来的 delta channel history 快路径、无法与 Python 生态工具对表。否决。
- **M7-B any 化签名 + Runtime 注入**：放弃编译期类型，与 `core` 已泛型化组件风格分裂。否决（其 `Runtime` 注入思路不采用；`prev` 走显式参数）。
- **M7-C 不做函数式 API**：放弃核心卖点（普通控制流 + 任务级 checkpoint 重放）。否决。

## 总体里程碑划分

每个里程碑独立可交付、可验证；实施顺序 **M5 → M6 → M7 → M8**（独立性从高到低、改动面从小到大）。所有里程碑共用的验收底线：

- 根 module 保持零第三方依赖（新依赖只允许出现在嵌套 module）。
- `go test ./...`、`make test-sqlite`、新增 `make test-postgres` 全绿；`langchain/agents` 零改动（M6 的 builder 新增 API 不触碰现有行为）。
- **单元测试对标 Python**：每个里程碑从 Python 对应测试文件中移植测试用例（见各节"测试移植"），语义级对齐，不是象征性覆盖。

---

## M5 Postgres checkpoint saver

### Python 侧语义（移植基线）

- **四表 + 迁移表**（`libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py:43-91`）：
  - `checkpoints(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint JSONB, metadata JSONB)`，主键三元组。
  - `checkpoint_blobs(thread_id, checkpoint_ns, channel, version, type, blob BYTEA)`，主键四元组——channel values 按版本单独成行（per-version 去重）。
  - `checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob BYTEA, task_path)`，主键五元组；`task_path` 为 v9 新增。
  - `checkpoint_migrations(v INTEGER PRIMARY KEY)`。
- **迁移机制**：`setup()` 用户显式首调；查 `max(v)` 后逐条执行剩余 migration。v6–v8 是 `CREATE INDEX CONCURRENTLY`，**不能在事务内执行**。
- **写入路径**：`channel_values` 拆两类——Python primitive（None/str/int/float/bool）内联留在 checkpoint JSONB；其余进 blobs 表，按 `(channel, version)` 经 `serde.dumps_typed` 编码成 `(type, blob)` 行。blobs 用 `ON CONFLICT DO NOTHING`（不可变版本化行），checkpoints 用 `ON CONFLICT DO UPDATE`。writes 的 idx 用保留 channel 负数映射（ERROR=-1, SCHEDULED=-2, INTERRUPT=-3, RESUME=-4），批次级选择 UPSERT vs INSERT...DO NOTHING（Go sqlite 已实现同语义，`checkpoint/sqlite/sqlite.go:215-258`）。
- **读取路径**：inline values 与 blobs 合并组装；metadata 过滤用 JSONB 包含运算 `metadata @> filter`。

### Go 设计（方案一：pgx 原生 + 全对齐四表 schema）

**包与依赖**：嵌套 module `langgraph/checkpoint/postgres/`（自己的 `go.mod`，`replace` 指回根 module），唯一运行时依赖 `github.com/jackc/pgx/v5`（纯 Go、活跃维护；lib/pq 已 maintenance mode，不选）。复刻 sqlite 的包装模式，根 module 零新增依赖。

**API**：

```go
package postgres // github.com/projanvil/langchain-golang/langgraph/checkpoint/postgres

func New(pool *pgxpool.Pool, serde checkpoint.Serializer) *Saver
func NewFromConnString(ctx context.Context, dsn string, serde checkpoint.Serializer) (*Saver, error)
// Setup 执行未应用的 schema migrations；必须由用户显式首调一次。
// 不在事务内执行（CREATE INDEX CONCURRENTLY 限制）。
func (s *Saver) Setup(ctx context.Context) error
// Saver 实现 checkpoint.Saver（含 M5 扩展后的接口，见下）。
```

**Schema**：逐条照搬 Python `MIGRATIONS`（v0–v9）为 Go 字符串常量，含 `checkpoint_migrations` 版本表与 `CREATE INDEX CONCURRENTLY`。**文档化分歧**：Python blobs 表 `version` 列为 TEXT、存十进制整数字符串（`get_next_version` 返回整数，`_dump_blobs` 仅 `cast(str, ver)`，postgres/base.py:549-570）；Go 的 channel version 是 `int64`，blobs 表 `version` 列用 `BIGINT`。Go 自建库不受影响，但与 Python 跨语言共库不可行（serde 字节格式本就不互通：Go JSON envelope vs Python msgpack——doc.go 明确声明）。

**inline 判定的文档化分歧**：Go 侧**仅原语内联**——`nil / string / bool / float64` 内联进 checkpoint JSONB；**`int/int64`、`map[string]any`、`[]any` 及一切其他类型一律进 blobs 表**（与 Python 对齐：Python 只内联原语，dict/list 进 blobs；Go 的 int 在 serde 中走 envelope、不是 JSON-native，故也不内联）。判定函数集中一处并配测试。

**读取实现**：拆 3 条查询（checkpoints / blobs / writes，scan-then-decode，sqlite.go 同风格），不照搬 Python 的单条嵌套 `array_agg` SQL。

**根接口扩展（breaking，spec 允许的 pre-1.0 break）**：

1. `ListOptions` 新增 `Filter map[string]any`——metadata 包含过滤（Python 的 `@>`）。postgres 服务端 `@>` 实现；memory/sqlite 加载 metadata 后进程内做 map 包含比较（语义相同，性能无关）。
2. `PutWrites` 签名新增 `taskPath string` 参数。postgres 存 `task_path` 列；InMemorySaver 存入 `Write` 记录（`Write` 结构新增 `TaskPath` 字段）；sqlite 表加列（`task_path TEXT NOT NULL DEFAULT ''`，启动时检测列缺失则 `ALTER TABLE ADD COLUMN`，兼容旧库）。

**Shallow 变体、delta channel history**：不做（Python 的 Shallow saver 与 `_DeltaSnapshot` 快路径均无 Go 对应概念），doc.go 声明。

**测试移植**：以共享契约套件为核心——在根 module 新增 `langgraph/checkpoint/savertest`（对标仓库 `standardtests/` 哲学），从 Python `libs/checkpoint-postgres/tests/` 与 `libs/langgraph/tests/test_checkpoint*.py` 的 saver 契约用例移植（put/get_tuple/list 顺序与过滤、pending writes 往返与覆盖语义、DeleteThread、serde 往返、并发 put）。sqlite 测试重构为调用 `savertest`；postgres 在 `embedded-postgres`（`github.com/fergusstrange/embedded-postgres`，进程内起真库，无需 Docker）上跑同一套件，`testing.Short()` 跳过。postgres 另补：migration 幂等（二次 Setup）、inline/blobs 拆分边界（int 进 blobs）、metadata `@>` 过滤、大 channel 值 per-version 去重验证。CI 需缓存 embedded-postgres 下载目录（Makefile `test-postgres` target 注明）。

**风险**：

- Setup 绝不包事务（CONCURRENTLY 限制）。
- 跨语言共库不可行（serde + version 格式双重分歧），doc.go 声明。
- embedded-postgres 首次运行联网下载 ~30MB；`-short` 必须有跳过路径。

---

## M6 多父 barrier join

### Python 侧语义（移植基线）

- Builder：`add_edge((a, b), c)` 存入 `waiting_edges`（`langgraph/graph/state.py:956-966`）；compile 时注册 channel `join:a+b:c` = `NamedBarrierValue(str, {a,b})`，c 订阅该 channel，每个父节点完成时把自己的名字写入（`state.py:1546-1561`）。
- Channel（`langgraph/channels/named_barrier_value.py`）：`update` 幂等累计 `seen`；`get`/`is_available` 仅当 `seen == names`；`consume` 在触发后清零（循环图每轮重新累计）；`seen` 集合作为普通 channel value 进 checkpoint——"父 A 已到达、父 B 中断"的暂停-恢复天然正确。
- 触发：barrier 满 → 子节点下一超步**恰好运行一次**（同超步多父完成也只一次）。
- **绕过语义（复刻并文档化）**：普通边/条件边（写 `branch:to:c`）、`Send`、`Command(goto=)` 指向 join 子节点时**绕过 barrier 直接触发**（OR 语义）。同一节点混用两类边时可能被触发多次——这是 Python 的既定行为，Go 复刻并在文档中显著警告，不做"改进"。
- 中断：父中断则其写入不执行；已到达记录随 checkpoint 持久化，resume 后中断父重跑补写，天然安全。

### Go 设计（方案 A：忠实移植 barrier channel）

**Builder**：新增 `AddJoinEdge(from []string, to string)`（校验：≥2 个去重父节点、节点已注册、to 非 START/END 保留名）。现有 `AddEdge` 不动。**文档化分歧**：Python 接受单元素起点序列（退化为普通等待边）并用 set 静默去重；Go 收紧为 ≥2 且重复父节点报错。Compile 时注册 `join:a+b:c` 的 `channels.Barrier` 实例，记录 join 元数据（barrier key → parents/child；parent → 参与的 barrier key 列表）。

**channels 包**：新增 `Barrier` 类型，实现现有 `channels.Channel` 接口（`channels/channel.go:11-26`）：`Update`（幂等累计，收到非父名写入报 `ErrInvalidUpdate`）、`Get`/`IsAvailable`（满才可用）、`Checkpoint`/`FromCheckpoint`（`seen` 为 `[]string`，serde 已支持，`checkpoint/serde/json.go:173`）。`Consume`（清零复位）**不在** `Channel` 接口内——它是接口外扩展点，执行器通过类型断言 `interface{ Consume() bool }` 调用（对齐 Python `BaseChannel.consume` 可选方法的形态）。

**执行器**（`graph/graph.go` / `graph/state.go`）：

- 父任务 commit 阶段，对其参与的每个 barrier 追加一条隐式 write（`{channel: barrierKey, value: parentName}`）。**该隐式 write 必须搭进父任务的 update 批次**（`taskWrites.update`），从而同时：①走 `applyWrites` 的同一版本递增路径；②在中断路径随 `completedTaskWrites` 序列化为 pending writes（`graph/graph.go:912-934`）——"父 A 已到达、父 B 中断→resume 补写"的持久化由此闭环，resume 重放时 barrier 到达记录不丢。
- `staticNext` 不为 waiting edge 产任务；当某 barrier 因本轮 writes 变为 available 且版本簿记判定未见过时，把 child 追加进 `nextTasks`（恰好一次）；child 任务的 writes 提交后对该 barrier `Consume` 复位（循环图语义）。
- join channel key 从一切用户可见面排除（控制面 channel，Python 中同样不在 output_keys）：`snapshot()`、节点输入、stream `values`/`updates` chunk、debug `task_result` 事件（隐式 write 搭在 update 批次里，后两者不过滤会泄漏 join key）。

**不做**：`defer=True` / `NamedBarrierValueAfterFinish`（依赖 PULL 循环"试探性最后超步"的 finish 广播，边驱动无等价物）——文档化。

**Send/Command.goto 到 join 子节点**：维持 PUSH 直发，绕过 barrier（Python 一致）。注意 Python 中 Send 触发的 c（带 arg）与 barrier 触发的 c（共享 state）是两个合法独立任务，Go 的去重不得误伤 Send 路径。

**测试移植**：从 Python `libs/langgraph/tests/` 中定位 waiting-edge 用例（grep `add_edge([` / `waiting_edges` / `NamedBarrierValue`；用例集中在 `test_pregel.py:1953-3085` 系列），移植：同超步多父→子恰好一次；跨超步多父→子等待齐后一次；三父 join；join 在循环中复位重触发；join 子节点被普通边/Send 同时触发（OR 语义、次数对齐 Python）；join channel 不进 snapshot/stream values。**Go 侧新增**（Python 无对应用例）：父中断→resume→补写后触发；checkpoint 往返后 barrier 部分到达状态保留。

**风险**：

- snapshot 泄漏（barrier 值泄进 state map）——必须过滤并配测试。
- Consume 时机：必须在 child 任务提交写之后、下一超步触发判定之前，否则循环中提前/漏触发。
- `langchain/agents` 行为零变化（其图无 join 边）——全量现有测试做回归。

---

## M7 函数式 API

### Python 侧语义（移植基线，`langgraph/func/__init__.py`）

- `@entrypoint(checkpointer=..., store=...)` 把函数编译成**单节点 Pregel 图**：三个保留 channel（`__start__`=EphemeralValue、`__end__`=LastValue、`__previous__`=LastValue，`func/__init__.py:576-596`）；`previous` 从上轮同线程 checkpoint 读入。
- `entrypoint.final(value, save)` 解耦"返回给调用者的值"与"写入 PREVIOUS 的保存值"；返回普通值时两者相同。
- `@task(retry=..., cache_policy=...)` 调用时不执行函数，返回 future；executor 把调用包成 `Call` 在下一 tick 调度。task 结果作为 `__return__`/`__error__` write 持久化到**当前 checkpoint**（不新建超步 checkpoint）。
- **恢复重放**：resume 时 entrypoint 从头重跑；`Call` 凭确定性 task ID = hash(checkpoint_id, checkpoint_ns, step, task_name, PUSH, parent_path, call_idx) 查 pending writes，命中即回填 future 不重跑；错误同样持久化并在 resume 时重抛。`call_counter` 保证同 task 循环调用区分。
- task 只能从 entrypoint / 另一个 task / StateGraph 节点内调用；输入输出须 JSON 可序列化；interrupt 的 resume 值按索引顺序匹配（非确定性逻辑必须放进 task）。
- 缓存是独立于 checkpoint 重放的第二套机制（`CachePolicy.key_func`，命名空间 `(__pregel_ns_writes, module.qualname)`）。

### Go 设计（方案 A：泛型 + 显式构造）

新包 `langgraph/fn`（`func` 是关键字；对标 JS 版 `task(name, fn)` 的显式命名形态）：

```go
package fn

type TaskOpts struct {
    Retry   *graph.RetryPolicy
    Cache   *graph.CachePolicy
    Timeout time.Duration
}

// NewTask 显式命名（name 替代 Python 的 module.qualname，用于缓存命名空间与 task ID）。
func NewTask[I, O any](name string, f func(context.Context, I) (O, error), opts TaskOpts) *Task[I, O]

// Call 仅可在 entrypoint 函数 / 其他 task / StateGraph 节点内调用（否则 panic，
// 对齐 Python 的运行期错误）；立即在 goroutine 中执行并返回 Future。
func (t *Task[I, O]) Call(ctx context.Context, in I) *Future[O]

// Get 阻塞取结果；resume 重放时命中 pending writes 直接回填（不重跑）。
func (f *Future[O]) Get(ctx context.Context) (O, error)

func AwaitAll[T any](ctx context.Context, futs ...*Future[T]) ([]T, error)

type EntrypointOpts struct {
    Checkpointer checkpoint.Saver
    Cache        checkpoint.Cache
    Retry        *graph.RetryPolicy // entrypoint 整体（节点级）重试
}

// prev 为上轮同线程 save 值；无 checkpointer 或首轮时 hasPrev=false、prev 为零值。
func NewEntrypoint[I, O, S any](opts EntrypointOpts,
    f func(ctx context.Context, in I, prev S, hasPrev bool) (O, error)) *Entrypoint[I, O, S]

// Final 变体：解耦返回值与保存值（对齐 entrypoint.final）。
type Final[O, S any] struct{ Value O; Save S }
func NewEntrypointFinal[I, O, S any](opts EntrypointOpts,
    f func(ctx context.Context, in I, prev S, hasPrev bool) (Final[O, S], error)) *Entrypoint[I, O, S]

func (e *Entrypoint[I, O, S]) Invoke(ctx context.Context, in I, opts graph.Options) (O, error)
func (e *Entrypoint[I, O, S]) Stream(ctx context.Context, in I, opts graph.Options) iter.Seq2[graph.StreamChunk, error]
```

**实现要点**：

- 内部编译为现有 `graph.StateGraph` 的单节点图，三个保留 channel key（`__start__`/`__end__`/`__previous__`，与 Python 对齐），节点函数跑用户函数并把返回拆 value/save 写两 channel——interrupt/resume/stream/time-travel 全部经由现有机制获得。保留 key 不进用户可见 state（同 M6 控制面过滤）。
- task dispatcher 经 `context.Context` 注入（与 `graph.Interrupt` 的 ctx 注入模式一致，`graph/graph.go:1127-1128`）。task 在 goroutine 立即执行（不引入"下一 tick"调度——Go 边驱动循环无 tick 概念，且 task 在节点函数内部执行，时序由用户控制流决定）。
- 确定性 task ID = fnv-1a(cpID, checkpoint_ns, step, taskName, parentPath, callIdx)，复用/扩展 `graph/taskid.go`；per-run call counter 每次 entrypoint 重跑从零重放（文档约束：重跑时调用顺序必须确定，同 Python determinism 一节）。task ID 中的 cpID 取**本次运行恢复所用的 checkpoint ID**（与 Python 哈希输入一致）。
- **task 结果持久化机制（闭环设计）**：`fn` 包是 StateGraph 的外层包装，checkpoint 保存在 graph 执行器内部、且无回调点；尤其 interrupt 以 panic 传播时节点内失去控制。因此：①**dispatcher 对象由 fn 包装层持有**（不挂在节点栈上），interrupt panic 后仍可访问，已缓冲的 task 结果不丢；②**每次运行开始时**，fn 层用 `Saver.GetTuple` 载入目标 checkpoint（ThreadID/CheckpointID 定位）的 pending writes，交给 dispatcher 供 `Call` 重放查询——命中即回填 future，不重跑；③**每次运行返回后**（正常完成、出错、或 interrupt 暂停），fn 层用 `GetTuple` 定位运行产生的最新 checkpoint，把本轮已完成 task 的 `__return__`/`__error__` 结果经 `PutWrites` 追加到该 checkpoint（pending writes 本就是惰性读取，事后追加语义正确）。
- retry/cache 复用 `graph.RetryPolicy`/`graph.CachePolicy` 与 `checkpoint.Cache` 接口；cache key 从"节点输入哈希"改为"调用参数哈希"，命名空间 `__fn_writes/<task-name>`。
- **文档化语义分歧**：①Timeout 只能做到"ctx 取消 + 放弃等待"（goroutine 不可强杀；Python sync 函数同样不支持 timeout）；②interrupt 触发时，已启动但未完成的 task 被取消（ctx cancel），已完成的结果落 pending writes 后暂停（Python 丢弃未完成 PUSH 任务，语义对应）；③无 checkpointer 时 `hasPrev=false`（Python 为 None，Go 用显式 bool 防误读零值）；④**不支持 `store`**（Python `@entrypoint(checkpointer=..., store=...)` 的跨线程 BaseStore 未移植，Go `EntrypointOpts` 无此字段）。
- serde：I/O/S 及 task 输入输出必须 JSON 可往返（封闭类型注册表），文档约束。

**测试移植**：定位 Python `libs/langgraph/tests/` 中函数式 API 测试文件（grep `@entrypoint` / `entrypoint.final`），移植：entrypoint 基本 invoke；previous 跨 invoke 累积；final 的 value≠save；task 并行 futures（AwaitAll）；task 结果 resume 重放（副作用计数器断言不重跑）；task 错误持久化与 resume 重抛；task retry；task cache 命中；entrypoint 内 interrupt→resume（resume 值按序匹配）；无 checkpointer 时 hasPrev=false；task 在 StateGraph 节点内调用；确定性 call counter（循环中同 task 多次调用结果各自正确）。

**风险**：

- 非确定性控制流导致重放对错值——只能靠文档约束（同 Python）。
- 错误重放缺失会静默重跑失败任务——`__error__` 路径必须配测试。
- serde 类型未注册时给出明确报错（不接受静默降级）。

---

## M8 文档刷新与双语化

现状问题：README 的 "Not supported" 段仍列着 Postgres/函数式 API 为缺失；usage guides 只有英文；M1–M4 新增能力在 guides 中覆盖不全。

**约定（大型开源项目写法）**：每个文档的英文版为主文件，中文版为同目录 `<name>.zh-CN.md` 姊妹文件；两份文件顶部互放语言切换行（`English | 简体中文`）。约定写入 `docs/README.md`（同样双语）。新增/修改内容必须双语同步落笔，不接受"先英文后补翻译"。

**范围**：

1. `README.md` + `README.zh-CN.md`：更新 "Not supported"（移除 Postgres、函数式 API；保留 CLI/SDK、barrier 之外仍缺的项如 defer 节点、Shallow saver）；Supported 段补 M5–M7 条目。
2. `docs/README.md` + 双语：文档地图更新 + 双语约定说明。
3. `docs/usage/langgraph.md` + 双语：刷新至 M1–M7 全量——join edges（含绕过语义警告）、Postgres saver（含 Setup/embedded 测试说明）、函数式 API 完整 guide（对齐 Python functional API 文档结构）、serde/跨语言不互通声明。
4. 其余 guides（`getting-started` / `composition` / `agents` / `streaming`）：内容校对刷新 + 补齐中文版。
5. 各新包 doc.go / 包注释：英文（代码注释跟随 Go 生态惯例），文档化分歧逐条写明（本 spec 各节的"文档化分歧"条目为最低清单）。

**验收**：双语文件内容对等（结构、代码块、链接一致）；`go test ./...` 的 example_test.go 不受影响；链接检查（相对路径可达）。

---

## 错误处理

- M5：所有 SQL 错误包装为 `fmt.Errorf("postgres saver: %w")` 风格；serde 不可往返值在 Put 时报错而非静默降级。
- M6：`AddJoinEdge` 校验失败（父节点不存在、重复、<2 个父）返回 error；barrier 收到非父名写入报 `ErrInvalidUpdate` 对齐 Python `InvalidUpdateError`。
- M7：`Task.Call` 在非法上下文调用 → panic（对齐 Python 运行期错误）；serde 失败、task panic 均转为 `__error__` write；entrypoint 函数 panic 传播为 graph 错误（与现有节点 panic 语义一致）。

## 验证计划

- 每里程碑：对应 Python 测试移植用例全绿 + 根 module 全量回归（`go test ./...`）+ 嵌套 module 测试（`make test-sqlite` / `make test-postgres`）。
- M6/M7 完成后跑 `langchain/agents` 全量测试确认零行为变化。
- M8：人工通读双语文档各一遍 + 相对链接检查。
