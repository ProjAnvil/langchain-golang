// Command streaming shows Agent.StreamEvents end to end: token-level
// model_delta events, balanced node_start/node_end pairs, tool_start/
// tool_end dispatch events, and the single terminal "end" event carrying the
// final state. The model is an offline double that emits its responses as
// multi-chunk streams, so deltas can be observed without any API key.
//
// Usage:
//
//	go run ./examples/streaming
package main

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
)

// chunkedModel is an offline ChatModel double whose Stream splits each
// scripted response into chunks, standing in for a token-streaming provider
// model. It returns itself from BindTools so its cursor advances across the
// agent loop's per-iteration re-binds (see examples/quickstart).
type chunkedModel struct {
	mu         sync.Mutex
	responses  []messages.Message
	idx        int
	streamText [][]string // per-response chunk text; len == len(responses)
}

func (m *chunkedModel) next() (messages.Message, []string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.idx >= len(m.responses) {
		return messages.Message{}, nil, fmt.Errorf("chunkedModel: no more responses (call %d)", m.idx+1)
	}
	resp, chunks := m.responses[m.idx], m.streamText[m.idx]
	m.idx++
	return resp, chunks, nil
}

func (m *chunkedModel) Invoke(_ context.Context, input []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	resp, _, err := m.next()
	return resp, err
}

func (m *chunkedModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
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

func (m *chunkedModel) Stream(_ context.Context, _ []messages.Message, _ ...runnables.Option) (runnables.Stream[messages.Message], error) {
	resp, chunks, err := m.next()
	if err != nil {
		return nil, err
	}
	messagesOut := make([]messages.Message, 0, len(chunks))
	for _, text := range chunks {
		if text == "" {
			continue
		}
		messagesOut = append(messagesOut, messages.AI(text))
	}
	if len(messagesOut) == 0 {
		messagesOut = []messages.Message{resp}
	}
	return runnables.NewSliceStream(messagesOut), nil
}

func (m *chunkedModel) InputSchema() schema.Schema  { return schema.Object(map[string]schema.Schema{}) }
func (m *chunkedModel) OutputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *chunkedModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) {
	return m, nil
}

func (m *chunkedModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true, Streaming: true}
}

func main() {
	ctx := context.Background()

	weather, err := coretools.NewSimple("get_weather", "returns current weather for a city",
		func(_ context.Context, city string) (coretools.Result, error) {
			return coretools.Result{Content: city + ": 18C, light rain"}, nil
		})
	if err != nil {
		fmt.Println("build weather tool:", err)
		return
	}

	// Response 1 requests the weather tool (streamed as a tool-call chunk);
	// response 2 streams the final answer in four text chunks.
	model := &chunkedModel{
		responses: []messages.Message{
			{
				Role: messages.RoleAI,
				ToolCalls: []messages.ToolCall{
					{ID: "call_1", Name: "get_weather", Args: map[string]any{"tool_input": "Tokyo"}},
				},
			},
			messages.AI("Tokyo is at 18C with light rain. Bring an umbrella."),
		},
		streamText: [][]string{
			{""}, // the tool-call response carries no text chunks
			{"Tokyo ", "is at 18C ", "with light rain. ", "Bring an umbrella."},
		},
	}

	agent, err := agents.CreateAgent(model, []coretools.Tool{weather})
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}

	stream, err := agent.StreamEvents(ctx, []messages.Message{messages.Human("what's the weather in Tokyo?")})
	if err != nil {
		fmt.Println("stream events:", err)
		return
	}
	defer func() { _ = stream.Close() }()

	var assembled strings.Builder
	for {
		ev, ok, err := stream.Next(ctx)
		if err != nil {
			fmt.Println("stream next:", err)
			return
		}
		if !ok {
			break
		}
		switch ev.Type {
		case agents.StreamNodeStart:
			fmt.Printf("[node_start] %s\n", ev.Node)
		case agents.StreamNodeEnd:
			fmt.Printf("[node_end]   %s\n", ev.Node)
		case agents.StreamModelDelta:
			// Non-text deltas (tool-call chunks) carry an empty Text; only
			// the text deltas are printed, but all are accumulated.
			if ev.Text != "" {
				fmt.Printf("[delta]      %q\n", ev.Text)
			}
			assembled.WriteString(ev.Text)
		case agents.StreamModelEnd:
			fmt.Printf("[model_end]  assembled=%q\n", ev.Message.Content)
		case agents.StreamToolStart:
			fmt.Printf("[tool_start] %s(%v)\n", ev.ToolName, ev.ToolArgs)
		case agents.StreamToolEnd:
			fmt.Printf("[tool_end]   %s -> %v\n", ev.ToolName, ev.ToolResult)
		case agents.StreamEnd:
			if ev.Err != nil {
				fmt.Printf("[end]        error=%v\n", ev.Err)
				return
			}
			fmt.Printf("[end]        final message=%q\n", ev.Message.Content)
		}
	}
	fmt.Printf("concatenated deltas == final message: %v\n",
		assembled.String() == "Tokyo is at 18C with light rain. Bring an umbrella.")
}
