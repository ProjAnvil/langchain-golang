//go:build integration

package integration

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
)

func TestAnthropicChatModel_Invoke(t *testing.T) {
	model := newAnthropicModel(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := model.Invoke(ctx, []messages.Message{
		messages.Human("Reply with the single word: pong"),
	})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if strings.TrimSpace(resp.Content) == "" {
		t.Fatalf("empty response: %+v", resp)
	}
	t.Logf("Anthropic Invoke reply: %q", resp.Content)
}

func TestAnthropicChatModel_Stream(t *testing.T) {
	model := newAnthropicModel(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	stream, err := model.Stream(ctx, []messages.Message{
		messages.Human("Count from 1 to 5."),
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	chunks := 0
	for {
		_, ok, err := stream.Next(ctx)
		if err != nil {
			t.Fatalf("stream next: %v", err)
		}
		if !ok {
			break
		}
		chunks++
	}
	if chunks == 0 {
		t.Fatalf("stream produced no chunks")
	}
	t.Logf("Anthropic Stream received %d chunks", chunks)
}

func TestAnthropic_CreateAgent_ToolLoop(t *testing.T) {
	model := newAnthropicModel(t)
	agent, err := agents.CreateAgent(model, []tools.Tool{echoTool(t)},
		agents.WithAgentSystemPrompt("You are a test assistant. When asked to echo, call the echo tool."),
		agents.WithAgentRecursionLimit(10),
	)
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	reply, err := agent.Invoke(ctx, []messages.Message{
		messages.Human("Use the echo tool to echo: hello world"),
	})
	if err != nil {
		t.Fatalf("agent Invoke: %v", err)
	}
	if len(reply) == 0 {
		t.Fatalf("agent returned no messages")
	}
	t.Logf("Anthropic agent final reply: %q", reply[len(reply)-1].Content)
}

func TestAnthropic_StreamEvents(t *testing.T) {
	model := newAnthropicModel(t)
	agent, err := agents.CreateAgent(model, nil)
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	stream, err := agent.StreamEvents(ctx, []messages.Message{
		messages.Human("Say hello in one short sentence."),
	})
	if err != nil {
		t.Fatalf("StreamEvents: %v", err)
	}
	deltas := 0
	for {
		ev, ok, err := stream.Next(ctx)
		if err != nil {
			t.Fatalf("stream next: %v", err)
		}
		if !ok {
			break
		}
		if ev.Type == agents.StreamModelDelta && ev.Text != "" {
			deltas++
		}
	}
	if deltas == 0 {
		t.Fatalf("no model_delta events received")
	}
	t.Logf("Anthropic StreamEvents received %d text deltas", deltas)
}
