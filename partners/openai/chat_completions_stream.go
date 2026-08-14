package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/httpclient"
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// Chat Completions streaming. The Chat Completions API streams newline-
// delimited SSE `data:` events whose `choices[0].delta` carries incremental
// `content` and `tool_calls` fields.

func (m ChatModel) createChatCompletionsStream(
	ctx context.Context,
	input []messages.Message,
	cfg runnables.Config,
) (*chatCompletionsStream, error) {
	requestPayload := m.buildChatCompletionsRequest(input)
	requestPayload.Stream = true

	body, err := json.Marshal(requestPayload)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, m.config.Timeout)
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		strings.TrimRight(m.config.BaseURL, "/")+"/chat/completions",
		bytes.NewReader(body),
	)
	if err != nil {
		cancel()
		return nil, err
	}
	configureRequest(req, m.config)

	client := m.config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		cancel()
		return nil, lcerrors.WrapTransport(err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		cancel()
		return nil, httpclient.ResponseError(providerName, "/chat/completions", resp)
	}

	return &chatCompletionsStream{
		ctx:       ctx,
		cancel:    cancel,
		body:      resp.Body,
		scanner:   bufio.NewScanner(resp.Body),
		cfg:       cfg,
		toolCalls: make(map[int]*streamToolCall),
	}, nil
}

type chatCompletionsStream struct {
	ctx       context.Context
	cancel    context.CancelFunc
	body      io.Closer
	scanner   *bufio.Scanner
	cfg       runnables.Config
	done      bool
	toolCalls map[int]*streamToolCall
}

func (s *chatCompletionsStream) Next(ctx context.Context) (messages.Message, bool, error) {
	for {
		if s.done {
			return messages.Message{}, false, nil
		}
		if err := ctx.Err(); err != nil {
			return messages.Message{}, false, err
		}
		if !s.scanner.Scan() {
			if err := s.scanner.Err(); err != nil {
				return messages.Message{}, false, err
			}
			s.done = true
			_ = emit(ctx, s.cfg, callbacks.EventChatModelEnd, nil, nil, nil)
			return messages.Message{}, false, nil
		}

		line := s.scanner.Text()
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			s.done = true
			chunk, ok := s.finalToolCallChunk()
			if ok {
				if err := emitStream(ctx, s.cfg, chunk); err != nil {
					return messages.Message{}, false, err
				}
				_ = emit(ctx, s.cfg, callbacks.EventChatModelEnd, nil, chunk, nil)
				return chunk, true, nil
			}
			_ = emit(ctx, s.cfg, callbacks.EventChatModelEnd, nil, nil, nil)
			return messages.Message{}, false, nil
		}

		var event chatCompletionsStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return messages.Message{}, false, fmt.Errorf("decode openai chat completions stream event: %w", err)
		}
		if len(event.Choices) == 0 {
			continue
		}
		delta := event.Choices[0].Delta

		if delta.Content != "" {
			chunk := messages.AI(delta.Content)
			if err := emitStream(ctx, s.cfg, chunk); err != nil {
				return messages.Message{}, false, err
			}
			return chunk, true, nil
		}
		for _, tc := range delta.ToolCalls {
			call := s.toolCalls[tc.Index]
			if call == nil {
				call = &streamToolCall{}
				s.toolCalls[tc.Index] = call
			}
			if tc.ID != "" {
				call.ID = tc.ID
			}
			if tc.Function.Name != "" {
				call.Name = tc.Function.Name
			}
			call.Arguments += tc.Function.Arguments
		}
	}
}

func (s *chatCompletionsStream) Close() error {
	s.done = true
	s.cancel()
	return s.body.Close()
}

// finalToolCallChunk assembles any streamed tool calls into a single message
// chunk, in index order.
func (s *chatCompletionsStream) finalToolCallChunk() (messages.Message, bool) {
	if len(s.toolCalls) == 0 {
		return messages.Message{}, false
	}
	chunk := messages.AI("")
	for i := 0; i < len(s.toolCalls); i++ {
		call := s.toolCalls[i]
		if call == nil {
			continue
		}
		toolCall, ok := call.toToolCall()
		if ok {
			chunk.ToolCalls = append(chunk.ToolCalls, toolCall)
		} else {
			chunk.InvalidToolCalls = append(chunk.InvalidToolCalls, messages.ToolCall{ID: call.ID, Name: call.Name})
		}
	}
	if len(chunk.ToolCalls) == 0 && len(chunk.InvalidToolCalls) == 0 {
		return messages.Message{}, false
	}
	return chunk, true
}

type chatCompletionsStreamEvent struct {
	Choices []chatCompletionsStreamChoice `json:"choices"`
}

type chatCompletionsStreamChoice struct {
	Delta chatCompletionsStreamDelta `json:"delta"`
}

type chatCompletionsStreamDelta struct {
	Content   string               `json:"content"`
	ToolCalls []chatStreamToolCall `json:"tool_calls"`
}

type chatStreamToolCall struct {
	Index    int          `json:"index"`
	ID       string       `json:"id"`
	Function chatFunction `json:"function"`
}
