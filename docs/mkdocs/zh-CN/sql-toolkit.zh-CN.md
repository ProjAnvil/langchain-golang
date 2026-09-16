# SQL 工具箱

**Languages:** [English](../sql-toolkit.md) | 简体中文

`langchain/toolkits/sqltoolkit` 把任意 `database/sql` 连接适配成四个智能体
工具——langchain-community `SQLDatabaseToolkit` 的 Go 对应物，并采用现代组合
方式（`get_tools()` + `CreateAgent`，而非已弃用的 `create_sql_agent`）。可运行
示例：
[`examples/sql-agent`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/sql-agent)。

## 安装

```bash
go get github.com/projanvil/langchain-golang
```

驱动自备：`_ "modernc.org/sqlite"`（纯 Go）或 PostgreSQL 用
`_ "github.com/jackc/pgx/v5/stdlib"`。

## 构造工具箱

```go
import "github.com/projanvil/langchain-golang/langchain/toolkits/sqltoolkit"

db, _ := sql.Open("sqlite", ":memory:") // 任意 *sql.DB
toolkit, err := sqltoolkit.NewSQLToolkit(ctx, db, sqltoolkit.SQLite)
//                                     Postgres 用 sqltoolkit.Postgres

toolList, err := toolkit.Tools(ctx)
```

四个工具逐字对齐上游的名字与描述：

| 工具 | 用途 |
|---|---|
| `sql_db_query` | 执行单条**只读** SELECT 并返回行（封顶，默认 30 行） |
| `sql_db_schema` | 指定表（或全部表）的 DDL + 样例行 |
| `sql_db_list_tables` | 列出库中的表与视图 |
| `sql_db_query_checker` | 校验查询——经数据库 planner（EXPLAIN），而非额外一次 LLM 调用 |

## 基于工具的智能体

```go
agent, err := agents.CreateAgent(model, toolList,
    agents.WithAgentSystemPrompt(
        "Answer database questions with the sql_db_* tools. Always inspect "+
            "the schema before querying."))
out, err := agent.Invoke(ctx, []messages.Message{
    messages.Human("Which hardware product is the most expensive?"),
})
```

## 只读护栏（刻意分歧）

上游会执行模型发来的任何 SQL。本端口在到达数据库之前就拒绝一切非单条
只读语句：

- 先剥离注释，然后只允许**恰好一条**语句（拒绝分号拼接）；
- 语句必须以 `SELECT` 或 `WITH` 开头；
- 语句体内任何位置出现可写关键字（`INTO`、`UPDATE`、`DELETE`）即拒绝。

畸形查询返回 `"Error: ..."` 工具结果（上游 `run_no_throw` 语义），模型可在
下一轮自我纠正；真正非法的 SQL 由 checker 的 EXPLAIN 兜底。

## 调参

```go
toolkit, _ := sqltoolkit.NewSQLToolkit(ctx, db, sqltoolkit.Postgres,
    sqltoolkit.WithSampleRows(3),          // schema 输出中的样例行
    sqltoolkit.WithMaxQueryRows(30),       // 查询结果行数上限
    sqltoolkit.WithMaxSchemaLength(12000), // schema 输出的 rune 上限
)
```

这些上限让智能体提示词保持有界：查询结果被截断时附带说明，schema 输出
同样限长。

## 切换点

- **方言**：`sqltoolkit.SQLite`（经 `sqlite_master`）或
  `sqltoolkit.Postgres`（经 `information_schema`）——introspection 查询
  自动切换。
- **模型**：任意 `language.ChatModel`；示例默认用脚本化替身离线运行，设置
  `OLLAMA_BASE_URL` 即切到本地 Ollama 模型。
- **写权限**：设计上不可用——需要写入时，把可写的辅助程序指向受限数据库
  角色，而不是放松护栏。
