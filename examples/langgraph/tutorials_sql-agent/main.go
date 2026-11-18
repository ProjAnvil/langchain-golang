// Command tutorials_sql-agent is the Go port of examples/tutorials/sql-agent.ipynb.
//
// The notebook builds a ReAct-style SQL agent and then a hand-written
// LangGraph workflow: list_tables -> call_get_schema -> get_schema ->
// generate_query -> (check_query -> run_query -> generate_query)* -> END.
// Tool calls go through langgraph.prebuilt.ToolNode and the model is an
// OpenAI chat model. This port keeps that graph and its tool plumbing with
// offline doubles: a function-backed ChatModel and core/tools tools backed by
// string maps. NO SQL statement is ever sent to a database and NO DML is
// executed: "queries" are parsed as strings and answered from fixed rows.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	lctools "github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	checkpoint "github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/prebuilt"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// scriptedModel is a deterministic offline ChatModel double. Its answer
// function reads the conversation and decides which tool call to emit; it
// never contacts a provider.
type scriptedModel struct {
	answer func(ctx context.Context, input []messages.Message) (messages.Message, error)
}

func (m *scriptedModel) Invoke(ctx context.Context, input []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	return m.answer(ctx, input)
}
func (m *scriptedModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		msg, err := m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
		out[i] = msg
	}
	return out, nil
}
func (m *scriptedModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	msg, err := m.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]messages.Message{msg}), nil
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

// offlineSchema is the in-memory stand-in for SQLDatabase.get_table_info.
var offlineSchema = map[string]string{
	"Artist": "Artist(ArtistId INTEGER, Name TEXT)",
	"Album":  "Album(AlbumId INTEGER, Title TEXT, ArtistId INTEGER)",
	"Track":  "Track(TrackId INTEGER, Name TEXT, AlbumId INTEGER, GenreId INTEGER, Milliseconds INTEGER)",
	"Genre":  "Genre(GenreId INTEGER, Name TEXT)",
}

// safeQueryResult emulates sql_db_query without a database. It only inspects
// the SQL string and returns canned rows; DML keywords are rejected.
func safeQueryResult(query string) (string, error) {
	upper := strings.ToUpper(query)
	for _, forbidden := range []string{"INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE "} {
		if strings.Contains(upper, forbidden) {
			return "", fmt.Errorf("only SELECT statements are allowed (offline guard): %s", forbidden)
		}
	}
	if !strings.HasPrefix(strings.TrimSpace(upper), "SELECT") {
		return "", fmt.Errorf("only SELECT statements are allowed (offline guard)")
	}
	if strings.Contains(upper, "AVG") && strings.Contains(upper, "GENRE") {
		return "Genre | avg_ms\nRock | 321456\nJazz | 287123\nMetal | 265400", nil
	}
	return "Name | Milliseconds\nTrack A | 210000\nTrack B | 195000", nil
}

func buildTools() (listNode, schemaNode, queryNode *lctools.ToolNode) {
	listTool, err := coretools.NewSimple("sql_db_list_tables", "List available tables.",
		func(_ context.Context, _ string) (coretools.Result, error) {
			names := []string{"Artist", "Album", "Track", "Genre"}
			return coretools.Result{Content: strings.Join(names, ", ")}, nil
		})
	if err != nil {
		log.Fatal(err)
	}
	schemaTool, err := coretools.NewSimple("sql_db_schema", "Show the schema for tables.",
		func(_ context.Context, input string) (coretools.Result, error) {
			var parts []string
			for _, name := range strings.Split(input, ",") {
				name = strings.TrimSpace(name)
				if ddl, ok := offlineSchema[name]; ok {
					parts = append(parts, ddl)
				} else {
					parts = append(parts, name+": <unknown table>")
				}
			}
			return coretools.Result{Content: strings.Join(parts, "\n\n")}, nil
		})
	if err != nil {
		log.Fatal(err)
	}
	queryTool, err := coretools.NewSimple("sql_db_query", "Run a read-only SELECT query (offline emulation).",
		func(_ context.Context, query string) (coretools.Result, error) {
			out, err := safeQueryResult(query)
			if err != nil {
				return coretools.Result{}, err
			}
			return coretools.Result{Content: out}, nil
		})
	if err != nil {
		log.Fatal(err)
	}

	listNode, err = lctools.NewToolNode([]lctools.Tool{listTool})
	if err != nil {
		log.Fatal(err)
	}
	schemaNode, err = lctools.NewToolNode([]lctools.Tool{schemaTool})
	if err != nil {
		log.Fatal(err)
	}
	queryNode, err = lctools.NewToolNode([]lctools.Tool{queryTool})
	if err != nil {
		log.Fatal(err)
	}
	return listNode, schemaNode, queryNode
}

func msgsOf(state map[string]any) []messages.Message {
	msgs, _ := state["messages"].([]messages.Message)
	return msgs
}

// queryResultSeen reports whether run_query has produced a sql_db_query
// tool result, mirroring the notebook's `state["messages"][-1]` inspection
// after run_query. Schema and list-table results do not end the loop.
func queryResultSeen(msgs []messages.Message) bool {
	for _, m := range msgs {
		if m.Role == messages.RoleTool && m.Name == "sql_db_query" {
			return true
		}
	}
	return false
}

func buildGraph() *graphpkg.CompiledGraph {
	listNode, schemaNode, queryNode := buildTools()

	// generateQueryModel emits a SELECT tool call until run_query has
	// produced a tool result, then answers naturally (ending the loop).
	generateModel := &scriptedModel{answer: func(_ context.Context, input []messages.Message) (messages.Message, error) {
		if queryResultSeen(input) {
			return messages.AI("Rock has the longest average track length (about 321456 ms)."), nil
		}
		return messages.Message{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
			ID: "call_query", Name: "sql_db_query",
			Args: map[string]any{"tool_input": "SELECT g.Name, AVG(t.Milliseconds) FROM Track t JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.Name LIMIT 5;"},
		}}}, nil
	}}

	// checkQueryModel forces the same query through a tool call again, as the
	// notebook's `check_query` node does with tool_choice="any".
	checkModel := &scriptedModel{answer: func(_ context.Context, input []messages.Message) (messages.Message, error) {
		q := "SELECT g.Name, AVG(t.Milliseconds) FROM Track t JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.Name LIMIT 5;"
		for i := len(input) - 1; i >= 0; i-- {
			if len(input[i].ToolCalls) > 0 {
				if v, ok := input[i].ToolCalls[0].Args["tool_input"].(string); ok {
					q = v
				}
				break
			}
		}
		return messages.Message{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
			ID: "call_check", Name: "sql_db_query", Args: map[string]any{"tool_input": q},
		}}}, nil
	}}

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)

	g.AddNode("list_tables", func(rt runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---LIST TABLES---")
		call := messages.Message{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
			ID: "call_list", Name: "sql_db_list_tables", Args: map[string]any{"tool_input": ""},
		}}}
		results, err := listNode.Invoke(rt, []messages.Message{call}, state)
		if err != nil {
			return nil, err
		}
		available := ""
		if len(results) > 0 {
			available = messages.Text(results[0])
		}
		updates := append([]messages.Message{call}, results...)
		updates = append(updates, messages.AI("Available tables: "+available))
		fmt.Println("  " + available)
		return map[string]any{"messages": updates}, nil
	})

	g.AddNode("call_get_schema", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		fmt.Println("---CALL GET SCHEMA---")
		return map[string]any{"messages": []messages.Message{{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
			ID: "call_schema", Name: "sql_db_schema", Args: map[string]any{"tool_input": "Track,Genre"},
		}}}}}, nil
	})

	prebuilt.AddToolNode(g, "get_schema", schemaNode)

	g.AddNode("generate_query", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---GENERATE QUERY---")
		resp, err := generateModel.Invoke(context.Background(), msgsOf(state))
		if err != nil {
			return nil, err
		}
		return map[string]any{"messages": []messages.Message{resp}}, nil
	})

	g.AddNode("check_query", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---CHECK QUERY---")
		resp, err := checkModel.Invoke(context.Background(), msgsOf(state))
		if err != nil {
			return nil, err
		}
		return map[string]any{"messages": []messages.Message{resp}}, nil
	})

	prebuilt.AddToolNode(g, "run_query", queryNode)

	g.AddEdge(types.START, "list_tables")
	g.AddEdge("list_tables", "call_get_schema")
	g.AddEdge("call_get_schema", "get_schema")
	g.AddEdge("get_schema", "generate_query")
	g.AddConditionalEdges("generate_query", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		if len(lctools.PendingToolCalls(msgsOf(state))) > 0 {
			return graphpkg.To("check_query"), nil
		}
		return graphpkg.To(types.END), nil
	})
	g.AddEdge("check_query", "run_query")
	g.AddEdge("run_query", "generate_query")

	cg, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		log.Fatalf("compile sql agent graph: %v", err)
	}
	return cg
}

func main() {
	ctx := context.Background()
	cg := buildGraph()
	question := "Which genre on average has the longest tracks?"
	result, err := cg.InvokeWithOptions(ctx, map[string]any{
		"messages": []messages.Message{messages.Human(question)},
	}, graphpkg.Options{ThreadID: "sql-thread"})
	if err != nil {
		log.Fatalf("invoke sql agent: %v", err)
	}
	msgs := msgsOf(result.Values)
	fmt.Println("\n===== FINAL ANSWER =====")
	if len(msgs) > 0 {
		fmt.Println(messages.Text(msgs[len(msgs)-1]))
	}
	fmt.Printf("total messages: %d\n", len(msgs))
}
