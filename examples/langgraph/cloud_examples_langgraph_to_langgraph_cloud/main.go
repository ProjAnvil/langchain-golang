// Command cloud_examples_langgraph_to_langgraph_cloud is the Go port of
// langgraph/examples/cloud_examples/langgraph_to_langgraph_cloud.ipynb. The
// notebook compares running a local graph with running the same graph through
// the LangGraph Cloud SDK. Here the "deployment" and its client are in-memory
// objects wrapping the local compiled graph, so the comparison runs offline.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
)

// ---------------------------------------------------------------------------
// Offline model and tool
// ---------------------------------------------------------------------------

// contextModel is a deterministic ChatModel double whose reply depends on the
// conversation so that multi-turn, thread-scoped runs behave consistently.
type contextModel struct {
	mu      sync.Mutex
	invokes int
}

func (m *contextModel) Invoke(_ context.Context, input []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	m.invokes++
	m.mu.Unlock()
	if len(input) == 0 {
		return messages.AI("Hello."), nil
	}
	last := input[len(input)-1]
	text := strings.ToLower(messages.Text(last))

	// A prior tool result gives the model a memory of the city even when the
	// current user message does not repeat it.
	knownCity := ""
	for _, msg := range input {
		if msg.Role == messages.RoleTool && strings.Contains(strings.ToLower(msg.Content), "nyc") {
			knownCity = "nyc"
		} else if msg.Role == messages.RoleTool && strings.Contains(strings.ToLower(msg.Content), "sf") {
			knownCity = "sf"
		}
	}
	if knownCity != "" && !strings.Contains(text, "weather") {
		if knownCity == "nyc" {
			return messages.AI("NYC is known for Broadway and Central Park."), nil
		}
		return messages.AI("San Francisco is known for the Golden Gate Bridge."), nil
	}

	switch {
	case last.Role == messages.RoleTool:
		if strings.Contains(strings.ToLower(last.Content), "nyc") {
			return messages.AI("It might be cloudy in nyc. NYC is known for Broadway and Central Park."), nil
		}
		return messages.AI("It's always sunny in sf. San Francisco is known for the Golden Gate Bridge."), nil
	case strings.Contains(text, "weather"):
		city := "sf"
		if strings.Contains(text, "nyc") {
			city = "nyc"
		}
		resp := messages.AI("")
		resp.ToolCalls = []messages.ToolCall{{
			ID: fmt.Sprintf("call_%d", m.invokes), Name: "get_weather",
			Args: map[string]any{"city": city},
		}}
		return resp, nil
	default:
		return messages.AI("I'm an offline demo agent. How can I help?"), nil
	}
}

func (m *contextModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		var err error
		if out[i], err = m.Invoke(ctx, in, opts...); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (m *contextModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	resp, err := m.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]messages.Message{resp}), nil
}

func (m *contextModel) InputSchema() schema.Schema  { return schema.Object(map[string]schema.Schema{}) }
func (m *contextModel) OutputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *contextModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) {
	return m, nil
}
func (m *contextModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true, Streaming: true}
}

func weatherTool() (coretools.Tool, error) {
	return coretools.NewFunc(
		"get_weather",
		"Use this to get weather information.",
		schema.Object(map[string]schema.Schema{"city": schema.String("nyc or sf")}, "city"),
		func(_ context.Context, input map[string]any) (coretools.Result, error) {
			city, _ := input["city"].(string)
			if city == "nyc" {
				return coretools.Result{Content: "It might be cloudy in nyc"}, nil
			}
			return coretools.Result{Content: "It's always sunny in sf"}, nil
		},
	)
}

func newAgent(checkpointer checkpoint.Saver, interruptBefore bool) *agents.Agent {
	tool, err := weatherTool()
	if err != nil {
		log.Fatalf("weather tool: %v", err)
	}
	opts := []agents.AgentOption{}
	if checkpointer != nil {
		opts = append(opts, agents.WithAgentCheckpointer(checkpointer))
	}
	if interruptBefore {
		opts = append(opts, agents.WithAgentInterruptBefore(agents.ToolsNodeName))
	}
	agent, err := agents.CreateAgent(&contextModel{}, []coretools.Tool{tool}, opts...)
	if err != nil {
		log.Fatalf("create agent: %v", err)
	}
	return agent
}

func printMessages(label string, msgs []messages.Message) {
	fmt.Println(label)
	for _, m := range msgs {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-6s [tool call %s]\n", m.Role, m.ToolCalls[0].Name)
			continue
		}
		fmt.Printf("  %-6s %s\n", m.Role, messages.Text(m))
	}
	fmt.Println()
}

// ---------------------------------------------------------------------------
// In-memory LangGraph Cloud client analogue
// ---------------------------------------------------------------------------

type cloudClient struct {
	agent      *agents.Agent
	nextThread int
	calls      []string
}

func newCloudClient(agent *agents.Agent) *cloudClient {
	return &cloudClient{agent: agent}
}

func (c *cloudClient) threadsCreate() string {
	c.nextThread++
	id := fmt.Sprintf("thread-%d", c.nextThread)
	c.calls = append(c.calls, "threads.create -> "+id)
	return id
}

func (c *cloudClient) runsWait(threadID *string, assistant string, input map[string]any) map[string]any {
	id := ""
	if threadID == nil {
		id = c.threadsCreate()
	} else {
		id = *threadID
	}
	c.calls = append(c.calls, fmt.Sprintf("runs.wait(%s, %s)", id, assistant))
	result, err := c.agent.Graph.InvokeWithOptions(context.Background(), input,
		graphpkg.Options{ThreadID: id})
	if err != nil {
		log.Fatalf("runs.wait: %v", err)
	}
	return result.Values
}

func (c *cloudClient) runsStream(threadID *string, assistant string, input map[string]any) []map[string]any {
	id := ""
	if threadID == nil {
		id = c.threadsCreate()
	} else {
		id = *threadID
	}
	c.calls = append(c.calls, fmt.Sprintf("runs.stream(%s, %s, values)", id, assistant))
	states := []map[string]any{}
	for chunk, err := range c.agent.Graph.Stream(context.Background(), input,
		graphpkg.StreamOptions{
			Options: graphpkg.Options{ThreadID: id},
			Modes:   []graphpkg.StreamMode{graphpkg.StreamValues},
		}) {
		if err != nil {
			log.Fatalf("runs.stream: %v", err)
		}
		if state, ok := chunk.Payload.(map[string]any); ok {
			states = append(states, state)
		}
	}
	return states
}

func (c *cloudClient) threadsGetState(threadID string) map[string]any {
	c.calls = append(c.calls, "threads.get_state("+threadID+")")
	snap, err := c.agent.Graph.GetState(context.Background(), checkpoint.Config{ThreadID: threadID})
	if err != nil {
		log.Fatalf("threads.get_state: %v", err)
	}
	return snap.Values
}

func humanInput(text string) map[string]any {
	return map[string]any{"messages": []messages.Message{messages.Human(text)}}
}

func lastMessageText(values map[string]any) string {
	msgs, _ := values["messages"].([]messages.Message)
	if len(msgs) == 0 {
		return ""
	}
	last := msgs[len(msgs)-1]
	if len(last.ToolCalls) > 0 {
		return "[tool call " + last.ToolCalls[0].Name + "]"
	}
	return messages.Text(last)
}

// ---------------------------------------------------------------------------
// Demonstrations
// ---------------------------------------------------------------------------

func localVsCloudInvoke(client *cloudClient, local *agents.Agent) {
	fmt.Println("=== 1. Invoke: local graph vs cloud run (new thread) ===")
	input := humanInput("what's the weather in sf")
	localOut, err := local.Graph.Invoke(context.Background(), input)
	if err != nil {
		log.Fatalf("local invoke: %v", err)
	}
	msgs, _ := localOut.Values["messages"].([]messages.Message)
	printMessages("--- local graph.ainvoke ---", msgs)

	cloudOut := client.runsWait(nil, "agent", input)
	msgs, _ = cloudOut["messages"].([]messages.Message)
	printMessages("--- cloud client.runs.wait(None, \"agent\") ---", msgs)
}

func localVsCloudStream(client *cloudClient, local *agents.Agent) {
	fmt.Println("=== 2. Stream values: local graph vs cloud run ===")
	input := humanInput("what's the weather in sf")
	fmt.Println("--- local graph.astream(stream_mode=values) ---")
	for chunk, err := range local.Graph.Stream(context.Background(), input,
		graphpkg.StreamOptions{Modes: []graphpkg.StreamMode{graphpkg.StreamValues}}) {
		if err != nil {
			log.Fatalf("local stream: %v", err)
		}
		if state, ok := chunk.Payload.(map[string]any); ok {
			fmt.Printf("  %s\n", lastMessageText(state))
		}
	}
	fmt.Println("--- cloud client.runs.stream(None, \"agent\", values) ---")
	for i, state := range client.runsStream(nil, "agent", input) {
		fmt.Printf("  [%d] %s\n", i, lastMessageText(state))
	}
	fmt.Println()
}

func threadMemory(client *cloudClient, saver checkpoint.Saver) {
	fmt.Println("=== 3. Thread-scoped memory: local checkpointer vs cloud threads ===")
	local := newAgent(saver, false)

	thread1 := checkpoint.Config{ThreadID: "1"}
	for _, text := range []string{"what's the weather in nyc", "what's it known for?"} {
		out, err := local.Graph.InvokeWithOptions(context.Background(), humanInput(text),
			graphpkg.Options{ThreadID: thread1.ThreadID})
		if err != nil {
			log.Fatalf("local thread 1: %v", err)
		}
		fmt.Printf("  local thread 1, %q -> %s\n", text, lastMessageText(out.Values))
	}
	thread2 := checkpoint.Config{ThreadID: "2"}
	out, err := local.Graph.InvokeWithOptions(context.Background(),
		humanInput("what's it known for?"), graphpkg.Options{ThreadID: thread2.ThreadID})
	if err != nil {
		log.Fatalf("local thread 2: %v", err)
	}
	fmt.Printf("  local thread 2, fresh question -> %s\n", lastMessageText(out.Values))
	snap, err := local.Graph.GetState(context.Background(), thread2)
	if err != nil {
		log.Fatalf("local get_state: %v", err)
	}
	thread2Msgs, _ := snap.Values["messages"].([]messages.Message)
	fmt.Printf("  local checkpointer.get(thread 2): %d messages\n\n", len(thread2Msgs))

	cloudThread := client.threadsCreate()
	for _, text := range []string{"what's the weather in nyc", "what's it known for?"} {
		state := client.runsWait(&cloudThread, "agent", humanInput(text))
		fmt.Printf("  cloud %s, %q -> %s\n", cloudThread, text, lastMessageText(state))
	}
	other := client.threadsCreate()
	state := client.runsWait(&other, "agent", humanInput("what's it known for?"))
	fmt.Printf("  cloud %s, fresh question -> %s\n", other, lastMessageText(state))
	state = client.threadsGetState(cloudThread)
	cloudMsgs, _ := state["messages"].([]messages.Message)
	fmt.Printf("  cloud threads.get_state(%s): %d messages\n\n", cloudThread, len(cloudMsgs))
}

func interruptBeforeTools() {
	fmt.Println("=== 4. interrupt_before tools: local vs cloud ===")
	saver := checkpoint.NewMemorySaver()
	local := newAgent(saver, true)
	first, err := local.Graph.InvokeWithOptions(context.Background(),
		humanInput("what's the weather in sf"), graphpkg.Options{ThreadID: "3"})
	if err != nil {
		log.Fatalf("local interrupt invoke: %v", err)
	}
	fmt.Printf("  local graph paused with %d interrupt(s)\n", len(first.Interrupts))
	resumed, err := local.Graph.InvokeWithOptions(context.Background(), nil,
		graphpkg.Options{ThreadID: "3"})
	if err != nil {
		log.Fatalf("local resume: %v", err)
	}
	fmt.Printf("  local after resume -> %s\n", lastMessageText(resumed.Values))

	cloud := newCloudClient(newAgent(checkpoint.NewMemorySaver(), true))
	thread := cloud.threadsCreate()
	cloud.runsStream(&thread, "agent", humanInput("what's the weather in sf"))
	fmt.Printf("  cloud %s streamed up to the tool boundary\n", thread)
	cloud.runsStream(&thread, "agent", nil)
	fmt.Printf("  cloud %s resumed -> %s\n", thread,
		lastMessageText(cloud.threadsGetState(thread)))
	fmt.Println()
}

func streamModes() {
	fmt.Println("=== 5. Message and event streams ===")
	agent := newAgent(nil, false)
	fmt.Println("--- local graph.astream(stream_mode=messages) ---")
	for chunk, err := range agent.Graph.Stream(context.Background(),
		humanInput("what's the weather in sf"),
		graphpkg.StreamOptions{Modes: []graphpkg.StreamMode{graphpkg.StreamMessages}}) {
		if err != nil {
			log.Fatalf("messages stream: %v", err)
		}
		if part, ok := chunk.Payload.(graphpkg.MessageChunk); ok {
			text := messages.Text(part.Message)
			if text != "" {
				fmt.Printf("  %s\n", text)
			}
		}
	}

	fmt.Println("--- cloud client.runs.stream(stream_mode=events) analogue ---")
	events, err := agent.StreamEvents(context.Background(),
		[]messages.Message{messages.Human("what's the weather in sf")})
	if err != nil {
		log.Fatalf("stream events: %v", err)
	}
	for {
		event, ok, err := events.Next(context.Background())
		if err != nil {
			log.Fatalf("stream events: %v", err)
		}
		if !ok {
			break
		}
		if event.Type == agents.StreamNodeStart || event.Type == agents.StreamNodeEnd {
			fmt.Printf("  event=%s node=%s\n", event.Type, event.Node)
		}
	}
	events.Close()
	fmt.Println()
}

func main() {
	client := newCloudClient(newAgent(checkpoint.NewMemorySaver(), false))
	local := newAgent(nil, false)

	localVsCloudInvoke(client, local)
	localVsCloudStream(client, local)
	threadMemory(client, checkpoint.NewMemorySaver())
	interruptBeforeTools()
	streamModes()

	fmt.Println("--- cloud client API calls (offline replay) ---")
	for _, call := range client.calls {
		fmt.Println("  " + call)
	}
}
