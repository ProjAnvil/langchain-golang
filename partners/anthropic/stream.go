package anthropic

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
	"github.com/projanvil/langchain-golang/core/streamevents"
)

func (m ChatModel) createMessageStream(
	ctx context.Context,
	input []messages.Message,
	cfg runnables.Config,
) (*messageStream, error) {
	requestPayload, err := m.buildRequest(input)
	if err != nil {
		return nil, err
	}
	requestPayload.Stream = true

	body, err := json.Marshal(requestPayload)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, m.config.Timeout)
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		strings.TrimRight(m.config.BaseURL, "/")+"/messages",
		bytes.NewReader(body),
	)
	if err != nil {
		cancel()
		return nil, err
	}
	setHeaders(req, m.config)

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
		return nil, httpclient.ResponseError(providerName, "/messages", resp)
	}

	return &messageStream{
		ctx:            ctx,
		cancel:         cancel,
		body:           resp.Body,
		scanner:        bufio.NewScanner(resp.Body),
		cfg:            cfg,
		textBlocks:     make(map[int]*streamTextBlock),
		toolBlocks:     make(map[int]*streamToolBlock),
		thinkingBlocks: make(map[int]*streamThinkingBlock),
	}, nil
}

type messageStream struct {
	ctx             context.Context
	cancel          context.CancelFunc
	body            io.Closer
	scanner         *bufio.Scanner
	cfg             runnables.Config
	done            bool
	eventName       string
	data            []string
	textBlocks      map[int]*streamTextBlock
	toolBlocks      map[int]*streamToolBlock
	thinkingBlocks  map[int]*streamThinkingBlock
	output          messages.Message
	protocolStarted bool
}

type streamTextBlock struct {
	started bool
	text    string
}

type streamToolBlock struct {
	id        string
	name      string
	arguments string
}

// streamThinkingBlock accumulates an extended-thinking content block. Redacted
// thinking blocks carry opaque data instead of reasoning text/signature.
type streamThinkingBlock struct {
	text      string
	signature string
	redacted  bool
	data      string
	id        string
}

func (s *messageStream) Next(ctx context.Context) (messages.Message, bool, error) {
	for {
		if s.done {
			return messages.Message{}, false, nil
		}
		if err := ctx.Err(); err != nil {
			_ = s.emitError(ctx, err)
			return messages.Message{}, false, err
		}
		if !s.scanner.Scan() {
			if err := s.scanner.Err(); err != nil {
				_ = s.emitError(ctx, err)
				return messages.Message{}, false, err
			}
			s.done = true
			if err := emit(ctx, s.cfg, callbacks.EventChatModelEnd, nil, s.output, nil); err != nil {
				return messages.Message{}, false, err
			}
			return messages.Message{}, false, nil
		}

		line := s.scanner.Text()
		if line == "" {
			chunk, ok, err := s.consumeEvent(ctx)
			if err != nil || ok {
				return chunk, ok, err
			}
			continue
		}
		if strings.HasPrefix(line, "event:") {
			s.eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if strings.HasPrefix(line, "data:") {
			s.data = append(s.data, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
}

func (s *messageStream) Close() error {
	s.done = true
	s.cancel()
	return s.body.Close()
}

func (s *messageStream) consumeEvent(ctx context.Context) (messages.Message, bool, error) {
	defer func() {
		s.eventName = ""
		s.data = nil
	}()
	if len(s.data) == 0 {
		return messages.Message{}, false, nil
	}
	data := strings.Join(s.data, "\n")
	if data == "[DONE]" {
		s.done = true
		if err := emit(ctx, s.cfg, callbacks.EventChatModelEnd, nil, s.output, nil); err != nil {
			return messages.Message{}, false, err
		}
		return messages.Message{}, false, nil
	}

	var event streamEvent
	if err := json.Unmarshal([]byte(data), &event); err != nil {
		err = fmt.Errorf("decode anthropic stream event: %w", err)
		_ = s.emitError(ctx, err)
		return messages.Message{}, false, err
	}
	eventType := event.Type
	if eventType == "" {
		eventType = s.eventName
	}

	switch eventType {
	case "message_start":
		s.output = event.Message.toMessage()
		return messages.Message{}, false, nil
	case "content_block_start":
		return s.contentBlockStart(ctx, event)
	case "content_block_delta":
		return s.contentBlockDelta(ctx, event)
	case "content_block_stop":
		return s.contentBlockStop(ctx, event.Index)
	case "message_delta":
		if event.Usage.OutputTokens != 0 {
			s.output.UsageMetadata.OutputTokens = event.Usage.OutputTokens
			s.output.UsageMetadata.TotalTokens = s.output.UsageMetadata.InputTokens + event.Usage.OutputTokens
		}
		if event.Delta.StopReason != "" {
			if s.output.ResponseMetadata == nil {
				s.output.ResponseMetadata = map[string]any{}
			}
			s.output.ResponseMetadata["stop_reason"] = event.Delta.StopReason
		}
		return messages.Message{}, false, nil
	case "message_stop":
		s.done = true
		if err := s.finishOpenBlocks(ctx); err != nil {
			return messages.Message{}, false, err
		}
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event:  streamevents.EventMessageFinish,
			Output: s.output,
		}); err != nil {
			return messages.Message{}, false, err
		}
		if err := emit(ctx, s.cfg, callbacks.EventChatModelEnd, nil, s.output, nil); err != nil {
			return messages.Message{}, false, err
		}
		return messages.Message{}, false, nil
	case "error":
		err := fmt.Errorf("anthropic stream error: %s", event.Error.Message)
		if event.Error.Message == "" {
			err = fmt.Errorf("anthropic stream error")
		}
		_ = s.emitError(ctx, err)
		return messages.Message{}, false, err
	default:
		return messages.Message{}, false, nil
	}
}

type streamEvent struct {
	Type         string         `json:"type"`
	Index        int            `json:"index"`
	Message      messagePayload `json:"message"`
	ContentBlock contentBlock   `json:"content_block"`
	Delta        streamDelta    `json:"delta"`
	Usage        usagePayload   `json:"usage"`
	Error        streamError    `json:"error"`
}

type streamDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	PartialJSON string `json:"partial_json"`
	Thinking    string `json:"thinking"`
	Signature   string `json:"signature"`
	StopReason  string `json:"stop_reason"`
}

type streamError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

func (s *messageStream) contentBlockStart(ctx context.Context, event streamEvent) (messages.Message, bool, error) {
	switch event.ContentBlock.Type {
	case "text":
		block := s.upsertTextBlock(event.Index)
		block.started = true
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockStart,
			Index: event.Index,
			Content: messages.ContentBlock{
				"type": "text",
				"text": "",
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
	case "tool_use":
		block := s.upsertToolBlock(event.Index)
		block.id = event.ContentBlock.ID
		block.name = event.ContentBlock.Name
		if event.ContentBlock.Input != nil {
			raw, _ := json.Marshal(event.ContentBlock.Input)
			block.arguments = string(raw)
		}
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockStart,
			Index: event.Index,
			Content: messages.ContentBlock{
				"type": "tool_call",
				"id":   event.ContentBlock.ID,
				"name": event.ContentBlock.Name,
				"args": map[string]any{},
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
		chunk := messages.AI("")
		chunk.ContentBlocks = []messages.ContentBlock{{
			"type":  "tool_use",
			"id":    event.ContentBlock.ID,
			"name":  event.ContentBlock.Name,
			"input": event.ContentBlock.Input,
			"index": event.Index,
		}}
		if err := emitStream(ctx, s.cfg, chunk); err != nil {
			return messages.Message{}, false, err
		}
		return chunk, true, nil
	case "thinking":
		block := s.upsertThinkingBlock(event.Index)
		if event.ContentBlock.Thinking != "" {
			block.text = event.ContentBlock.Thinking
		}
		if event.ContentBlock.Signature != "" {
			block.signature = event.ContentBlock.Signature
		}
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockStart,
			Index: event.Index,
			Content: messages.ContentBlock{
				"type":      "reasoning",
				"reasoning": "",
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
	case "redacted_thinking":
		block := s.upsertThinkingBlock(event.Index)
		block.redacted = true
		block.data = event.ContentBlock.Data
		block.id = event.ContentBlock.ID
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockStart,
			Index: event.Index,
			Content: messages.ContentBlock{
				"type": "reasoning",
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
	}
	return messages.Message{}, false, nil
}

func (s *messageStream) contentBlockDelta(ctx context.Context, event streamEvent) (messages.Message, bool, error) {
	switch event.Delta.Type {
	case "text_delta":
		block := s.upsertTextBlock(event.Index)
		block.text += event.Delta.Text
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockDelta,
			Index: event.Index,
			Delta: messages.ContentBlock{
				"type": "text-delta",
				"text": event.Delta.Text,
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
		s.output.Content += event.Delta.Text
		chunk := messages.AI(event.Delta.Text)
		if err := emitStream(ctx, s.cfg, chunk); err != nil {
			return messages.Message{}, false, err
		}
		return chunk, true, nil
	case "input_json_delta":
		block := s.upsertToolBlock(event.Index)
		block.arguments += event.Delta.PartialJSON
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockDelta,
			Index: event.Index,
			Delta: messages.ContentBlock{
				"type": "tool_call_chunk",
				"args": event.Delta.PartialJSON,
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
		chunk := messages.AI("")
		chunk.ContentBlocks = []messages.ContentBlock{{
			"type":  "tool_use",
			"input": event.Delta.PartialJSON,
			"index": event.Index,
		}}
		if err := emitStream(ctx, s.cfg, chunk); err != nil {
			return messages.Message{}, false, err
		}
		return chunk, true, nil
	case "thinking_delta":
		block := s.upsertThinkingBlock(event.Index)
		block.text += event.Delta.Thinking
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockDelta,
			Index: event.Index,
			Delta: messages.ContentBlock{
				"type":      "reasoning-delta",
				"reasoning": event.Delta.Thinking,
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
		chunk := messages.AI("")
		chunk.ContentBlocks = []messages.ContentBlock{{
			"type":      "reasoning",
			"reasoning": event.Delta.Thinking,
			"index":     event.Index,
		}}
		if err := emitStream(ctx, s.cfg, chunk); err != nil {
			return messages.Message{}, false, err
		}
		return chunk, true, nil
	case "signature_delta":
		block := s.upsertThinkingBlock(event.Index)
		block.signature += event.Delta.Signature
		return messages.Message{}, false, nil
	default:
		return messages.Message{}, false, nil
	}
}

func (s *messageStream) contentBlockStop(ctx context.Context, index int) (messages.Message, bool, error) {
	if block := s.textBlocks[index]; block != nil {
		delete(s.textBlocks, index)
		s.output.ContentBlocks = append(s.output.ContentBlocks, messages.ContentBlock{
			"type": "text",
			"text": block.text,
		})
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockFinish,
			Index: index,
			Content: messages.ContentBlock{
				"type": "text",
				"text": block.text,
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
		return messages.Message{}, false, nil
	}
	if block := s.toolBlocks[index]; block != nil {
		delete(s.toolBlocks, index)
		call := messages.ToolCall{ID: block.id, Name: block.name}
		if block.arguments != "" {
			_ = json.Unmarshal([]byte(block.arguments), &call.Args)
		}
		s.output.ToolCalls = append(s.output.ToolCalls, call)
		s.output.ContentBlocks = append(s.output.ContentBlocks, messages.ContentBlock{
			"type": "tool_call",
			"id":   call.ID,
			"name": call.Name,
			"args": call.Args,
		})
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockFinish,
			Index: index,
			Content: messages.ContentBlock{
				"type": "tool_call",
				"id":   call.ID,
				"name": call.Name,
				"args": call.Args,
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
		chunk := messages.AI("")
		chunk.ToolCalls = []messages.ToolCall{call}
		if err := emitStream(ctx, s.cfg, chunk); err != nil {
			return messages.Message{}, false, err
		}
		return chunk, true, nil
	}
	if block := s.thinkingBlocks[index]; block != nil {
		delete(s.thinkingBlocks, index)
		var contentBlock messages.ContentBlock
		if block.redacted {
			contentBlock = messages.ContentBlock{
				"type": "reasoning",
				"data": block.data,
			}
			if block.id != "" {
				contentBlock["id"] = block.id
			}
		} else {
			contentBlock = messages.ContentBlock{
				"type":      "reasoning",
				"reasoning": block.text,
				"signature": block.signature,
			}
		}
		s.output.ContentBlocks = append(s.output.ContentBlocks, contentBlock)
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event:   streamevents.EventContentBlockFinish,
			Index:   index,
			Content: contentBlock,
		}); err != nil {
			return messages.Message{}, false, err
		}
		return messages.Message{}, false, nil
	}
	return messages.Message{}, false, nil
}

func (s *messageStream) finishOpenBlocks(ctx context.Context) error {
	for index := range s.textBlocks {
		if _, _, err := s.contentBlockStop(ctx, index); err != nil {
			return err
		}
	}
	for index := range s.toolBlocks {
		if _, _, err := s.contentBlockStop(ctx, index); err != nil {
			return err
		}
	}
	for index := range s.thinkingBlocks {
		if _, _, err := s.contentBlockStop(ctx, index); err != nil {
			return err
		}
	}
	return nil
}

func (s *messageStream) upsertTextBlock(index int) *streamTextBlock {
	block := s.textBlocks[index]
	if block == nil {
		block = &streamTextBlock{}
		s.textBlocks[index] = block
	}
	return block
}

func (s *messageStream) upsertToolBlock(index int) *streamToolBlock {
	block := s.toolBlocks[index]
	if block == nil {
		block = &streamToolBlock{}
		s.toolBlocks[index] = block
	}
	return block
}

func (s *messageStream) upsertThinkingBlock(index int) *streamThinkingBlock {
	block := s.thinkingBlocks[index]
	if block == nil {
		block = &streamThinkingBlock{}
		s.thinkingBlocks[index] = block
	}
	return block
}

func (s *messageStream) emitError(ctx context.Context, err error) error {
	return emit(ctx, s.cfg, callbacks.EventChatModelError, nil, nil, err)
}

func emitStream(ctx context.Context, cfg runnables.Config, chunk messages.Message) error {
	if cfg.Callbacks.Empty() {
		return nil
	}
	return cfg.Callbacks.Emit(ctx, callbacks.Event{
		Kind:     callbacks.EventChatModelStream,
		Name:     cfg.Name,
		RunID:    cfg.RunID,
		ParentID: cfg.ParentID,
		Tags:     append([]string(nil), cfg.Tags...),
		Metadata: cloneMetadata(cfg.Metadata),
		Chunk:    chunk,
	})
}

func (s *messageStream) emitProtocol(ctx context.Context, event streamevents.Event) error {
	if s.cfg.Callbacks.Empty() {
		return nil
	}
	if !s.protocolStarted {
		s.protocolStarted = true
		if err := s.cfg.Callbacks.Emit(ctx, callbacks.Event{
			Kind:     callbacks.EventChatModelProtocol,
			Name:     s.cfg.Name,
			RunID:    s.cfg.RunID,
			ParentID: s.cfg.ParentID,
			Tags:     append([]string(nil), s.cfg.Tags...),
			Metadata: cloneMetadata(s.cfg.Metadata),
			Chunk:    streamevents.Event{Event: streamevents.EventMessageStart},
		}); err != nil {
			return err
		}
	}
	return s.cfg.Callbacks.Emit(ctx, callbacks.Event{
		Kind:     callbacks.EventChatModelProtocol,
		Name:     s.cfg.Name,
		RunID:    s.cfg.RunID,
		ParentID: s.cfg.ParentID,
		Tags:     append([]string(nil), s.cfg.Tags...),
		Metadata: cloneMetadata(s.cfg.Metadata),
		Chunk:    event,
	})
}
