// Command sql-agent builds a modern create_agent-style SQL assistant over
// the sqltoolkit: an in-memory SQLite database is seeded with a small schema,
// the toolkit contributes its four read-only tools (sql_db_query,
// sql_db_schema, sql_db_list_tables, sql_db_query_checker), and the agent
// answers a question by calling the query tool.
//
// The default model is an offline scripted double that demonstrates the
// tool loop without any server; export OLLAMA_BASE_URL to drive the same
// agent with a local Ollama model instead.
//
// Required env: none. Optional:
//
//	OLLAMA_BASE_URL  base URL of a local Ollama server (e.g. http://localhost:11434);
//	                 when set, the real model replaces the scripted double
//
// Usage:
//
//	go run ./examples/sql-agent
package main

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"sync"

	_ "modernc.org/sqlite" // registers the database/sql driver "sqlite"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
	"github.com/projanvil/langchain-golang/langchain/toolkits/sqltoolkit"
	"github.com/projanvil/langchain-golang/partners/ollama"
)

// scriptedModel: offline ChatModel double, self-returning from BindTools
// (see examples/quickstart for the rationale). Its scripted responses make
// the agent run one query tool call and then answer.
type scriptedModel struct {
	mu          sync.Mutex
	responses   []messages.Message
	idx         int
	invocations [][]messages.Message
}

func (m *scriptedModel) Invoke(_ context.Context, input []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.invocations = append(m.invocations, append([]messages.Message(nil), input...))
	if m.idx >= len(m.responses) {
		return messages.Message{}, fmt.Errorf("scriptedModel: no more responses (call %d)", m.idx+1)
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp, nil
}

func (m *scriptedModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		var err error
		out[i], err = m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (m *scriptedModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	resp, err := m.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]messages.Message{resp}), nil
}

func (m *scriptedModel) InputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *scriptedModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}
func (m *scriptedModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) {
	return m, nil
}

func (m *scriptedModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
}

func seed(ctx context.Context, db *sql.DB) error {
	statements := []string{
		`CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL)`,
		`CREATE TABLE orders (id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER, created_at TEXT)`,
		`INSERT INTO products (name, category, price) VALUES
			('mechanical keyboard', 'hardware', 129.00),
			('usb-c dock', 'hardware', 89.50),
			('noise-cancelling headset', 'audio', 249.00),
			('desk mat', 'accessories', 25.00)`,
		`INSERT INTO orders (product_id, quantity, created_at) VALUES
			(1, 3, '2026-09-01'), (2, 1, '2026-09-02'), (3, 2, '2026-09-03')`,
	}
	for _, stmt := range statements {
		if _, err := db.ExecContext(ctx, stmt); err != nil {
			return err
		}
	}
	return nil
}

func main() {
	ctx := context.Background()

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		fmt.Println("open sqlite:", err)
		return
	}
	defer func() { _ = db.Close() }()
	if err := seed(ctx, db); err != nil {
		fmt.Println("seed db:", err)
		return
	}

	toolkit, err := sqltoolkit.NewSQLToolkit(ctx, db, sqltoolkit.SQLite)
	if err != nil {
		fmt.Println("new sql toolkit:", err)
		return
	}
	toolList, err := toolkit.Tools(ctx)
	if err != nil {
		fmt.Println("toolkit tools:", err)
		return
	}
	fmt.Println("sqltoolkit tools:")
	for _, t := range toolList {
		fmt.Printf("  %s — %s\n", t.Name(), firstLine(t.Description()))
	}

	// Model: a local Ollama server when OLLAMA_BASE_URL is exported,
	// otherwise the offline scripted double whose tool call mirrors what a
	// real model would emit for this question.
	var model language.ChatModel
	if baseURL := os.Getenv("OLLAMA_BASE_URL"); baseURL != "" {
		model = ollama.NewChatModel(
			modelconfig.WithBaseURL(baseURL),
			modelconfig.WithModel("llama3.1"),
		)
		fmt.Println("model: ollama (llama3.1) via OLLAMA_BASE_URL")
	} else {
		model = &scriptedModel{responses: []messages.Message{
			{
				Role: messages.RoleAI,
				ToolCalls: []messages.ToolCall{
					{
						ID:   "call_1",
						Name: sqltoolkit.QueryToolName,
						Args: map[string]any{
							"query": "SELECT name, price FROM products WHERE category = 'hardware' ORDER BY price DESC",
						},
					},
				},
			},
			messages.AI("The two hardware products are the noise-cancelling headset ($249.00) " +
				"and the mechanical keyboard ($129.00); the usb-c dock ($89.50) is third."),
		}}
		fmt.Println("model: offline scripted double (set OLLAMA_BASE_URL for a real model)")
	}

	agent, err := agents.CreateAgent(model, toolList,
		agents.WithAgentSystemPrompt(
			"You are a SQL analyst over a SQLite database (dialect: "+string(toolkit.Dialect())+"). "+
				"Inspect schemas with sql_db_schema, validate queries with sql_db_query_checker, "+
				"and fetch data with sql_db_query. Only read-only SELECT queries are allowed."),
	)
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}

	out, err := agent.Invoke(ctx, []messages.Message{
		messages.Human("List hardware products from most to least expensive."),
	})
	if err != nil {
		fmt.Println("invoke agent:", err)
		return
	}
	fmt.Println("agent run:")
	for _, m := range out {
		if m.Role == messages.RoleTool {
			fmt.Printf("  %-8s (call %s) -> %s\n", m.Role, m.ToolCallID, messages.Text(m))
			continue
		}
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s calls %s(%v)\n", m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Args)
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}
}

func firstLine(s string) string {
	for i, r := range s {
		if r == '\n' {
			return s[:i]
		}
	}
	return s
}
