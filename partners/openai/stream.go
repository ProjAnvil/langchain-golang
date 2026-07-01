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
	"github.com/projanvil/langchain-golang/core/streamevents"
)

func (m ChatModel) createResponseStream(
	ctx context.Context,
	input []messages.Message,
	cfg runnables.Config,
) (*responseStream, error) {
	requestPayload := m.buildRequest(input)
	requestPayload.Stream = true

	body, err := json.Marshal(requestPayload)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, m.config.Timeout)
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		strings.TrimRight(m.config.BaseURL, "/")+"/responses",
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
		return nil, httpclient.ResponseError(providerName, "/responses", resp)
	}

	return &responseStream{
		ctx:             ctx,
		cancel:          cancel,
		body:            resp.Body,
		scanner:         bufio.NewScanner(resp.Body),
		cfg:             cfg,
		toolCalls:       make(map[int]*streamToolCall),
		textBlocks:      make(map[int]*streamTextBlock),
		reasoningBlocks: make(map[int]*streamReasoningBlock),
	}, nil
}

type responseStream struct {
	ctx             context.Context
	cancel          context.CancelFunc
	body            io.Closer
	scanner         *bufio.Scanner
	cfg             runnables.Config
	done            bool
	eventName       string
	data            []string
	toolCalls       map[int]*streamToolCall
	textBlocks      map[int]*streamTextBlock
	reasoningBlocks map[int]*streamReasoningBlock
	protocolStarted bool
}

func (s *responseStream) Next(ctx context.Context) (messages.Message, bool, error) {
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
			if err := emit(ctx, s.cfg, callbacks.EventChatModelEnd, nil, nil, nil); err != nil {
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

func (s *responseStream) Close() error {
	s.done = true
	s.cancel()
	return s.body.Close()
}

func (s *responseStream) consumeEvent(ctx context.Context) (messages.Message, bool, error) {
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
		if err := emit(ctx, s.cfg, callbacks.EventChatModelEnd, nil, nil, nil); err != nil {
			return messages.Message{}, false, err
		}
		return messages.Message{}, false, nil
	}

	var event streamEvent
	if err := json.Unmarshal([]byte(data), &event); err != nil {
		err = fmt.Errorf("decode openai stream event: %w", err)
		_ = s.emitError(ctx, err)
		return messages.Message{}, false, err
	}
	eventType := event.Type
	if eventType == "" {
		eventType = s.eventName
	}

	switch eventType {
	case "response.output_text.delta":
		if event.Delta == "" {
			return messages.Message{}, false, nil
		}
		if err := s.emitTextDelta(ctx, event.OutputIndex, event.Delta); err != nil {
			return messages.Message{}, false, err
		}
		chunk := messages.AI(event.Delta)
		if err := emitStream(ctx, s.cfg, chunk); err != nil {
			return messages.Message{}, false, err
		}
		return chunk, true, nil
	case "response.output_text.done":
		if err := s.finishTextBlock(ctx, event.OutputIndex); err != nil {
			return messages.Message{}, false, err
		}
		return messages.Message{}, false, nil
	case "response.output_item.added":
		if event.Item.Type != "function_call" {
			if isProtocolOutputItem(event.Item.Type) {
				if event.Item.Type == "reasoning" {
					s.upsertReasoningBlock(event.OutputIndex).item = event.Item
				}
				return s.emitOutputItemStart(ctx, event.OutputIndex, event.Item)
			}
			return messages.Message{}, false, nil
		}
		call := s.upsertToolCall(event.OutputIndex)
		call.ID = event.Item.CallID
		call.Name = event.Item.Name
		call.Arguments = event.Item.Arguments
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockStart,
			Index: event.OutputIndex,
			Content: messages.ContentBlock{
				"type": "tool_call",
				"id":   event.Item.CallID,
				"name": event.Item.Name,
				"args": map[string]any{},
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
		chunk := messages.AI("")
		chunk.ContentBlocks = []messages.ContentBlock{{
			"type":      "function_call",
			"id":        event.Item.ID,
			"call_id":   event.Item.CallID,
			"name":      event.Item.Name,
			"arguments": event.Item.Arguments,
			"index":     event.OutputIndex,
		}}
		if err := emitStream(ctx, s.cfg, chunk); err != nil {
			return messages.Message{}, false, err
		}
		return chunk, true, nil
	case "response.function_call_arguments.delta":
		call := s.upsertToolCall(event.OutputIndex)
		call.Arguments += event.Delta
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockDelta,
			Index: event.OutputIndex,
			Delta: messages.ContentBlock{
				"type": "tool_call_chunk",
				"args": event.Delta,
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
		chunk := messages.AI("")
		chunk.ContentBlocks = []messages.ContentBlock{{
			"type":      "function_call",
			"arguments": event.Delta,
			"index":     event.OutputIndex,
		}}
		if err := emitStream(ctx, s.cfg, chunk); err != nil {
			return messages.Message{}, false, err
		}
		return chunk, true, nil
	case "response.function_call_arguments.done":
		call := s.upsertToolCall(event.OutputIndex)
		if event.Name != "" {
			call.Name = event.Name
		}
		if event.Arguments != "" {
			call.Arguments = event.Arguments
		}
		return messages.Message{}, false, nil
	case "response.output_item.done":
		if event.Item.Type != "function_call" {
			if isProtocolOutputItem(event.Item.Type) {
				if err := s.emitOutputItemFinish(ctx, event.OutputIndex, event.Item); err != nil {
					return messages.Message{}, false, err
				}
				chunk := messages.AI("")
				chunk.ContentBlocks = []messages.ContentBlock{contentBlockFromOutputItem(event.Item, event.OutputIndex)}
				if err := emitStream(ctx, s.cfg, chunk); err != nil {
					return messages.Message{}, false, err
				}
				return chunk, true, nil
			}
			return messages.Message{}, false, nil
		}
		call := s.upsertToolCall(event.OutputIndex)
		if event.Item.CallID != "" {
			call.ID = event.Item.CallID
		}
		if event.Item.Name != "" {
			call.Name = event.Item.Name
		}
		if event.Item.Arguments != "" {
			call.Arguments = event.Item.Arguments
		}
		chunk := messages.AI("")
		toolCall, ok := call.toToolCall()
		if ok {
			chunk.ToolCalls = []messages.ToolCall{toolCall}
			if err := s.emitProtocol(ctx, streamevents.Event{
				Event: streamevents.EventContentBlockFinish,
				Index: event.OutputIndex,
				Content: messages.ContentBlock{
					"type": "tool_call",
					"id":   toolCall.ID,
					"name": toolCall.Name,
					"args": toolCall.Args,
				},
			}); err != nil {
				return messages.Message{}, false, err
			}
		} else {
			chunk.InvalidToolCalls = []messages.ToolCall{{
				ID:   call.ID,
				Name: call.Name,
			}}
			if err := s.emitProtocol(ctx, streamevents.Event{
				Event: streamevents.EventContentBlockFinish,
				Index: event.OutputIndex,
				Content: messages.ContentBlock{
					"type": "invalid_tool_call",
					"id":   call.ID,
					"name": call.Name,
				},
			}); err != nil {
				return messages.Message{}, false, err
			}
		}
		if err := emitStream(ctx, s.cfg, chunk); err != nil {
			return messages.Message{}, false, err
		}
		return chunk, true, nil
	case "response.reasoning_summary_text.delta":
		block := s.upsertReasoningBlock(event.OutputIndex)
		block.text += event.Delta
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockDelta,
			Index: event.OutputIndex,
			Delta: messages.ContentBlock{
				"type":      "reasoning-delta",
				"reasoning": event.Delta,
			},
		}); err != nil {
			return messages.Message{}, false, err
		}
		chunk := messages.AI("")
		chunk.ContentBlocks = []messages.ContentBlock{{
			"type":      "reasoning",
			"reasoning": event.Delta,
			"index":     event.OutputIndex,
		}}
		if err := emitStream(ctx, s.cfg, chunk); err != nil {
			return messages.Message{}, false, err
		}
		return chunk, true, nil
	case "response.reasoning_summary_text.done":
		block := s.upsertReasoningBlock(event.OutputIndex)
		if event.Text != "" {
			block.text = event.Text
		}
		return messages.Message{}, false, nil
	case "response.reasoning_summary_part.done":
		block := s.upsertReasoningBlock(event.OutputIndex)
		if event.Part.Text != "" {
			block.text = event.Part.Text
		}
		return messages.Message{}, false, nil
	case "response.refusal.done":
		block := messages.ContentBlock{
			"type":    "refusal",
			"refusal": event.Refusal,
		}
		chunk := messages.AI("")
		chunk.ContentBlocks = []messages.ContentBlock{block}
		if err := emitStream(ctx, s.cfg, chunk); err != nil {
			return messages.Message{}, false, err
		}
		return chunk, true, nil
	case "response.completed":
		s.done = true
		output := messages.Message{}
		if event.Response != nil {
			output = event.Response.toMessage()
		}
		if err := s.finishOpenTextBlocks(ctx); err != nil {
			return messages.Message{}, false, err
		}
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event:  streamevents.EventMessageFinish,
			Output: output,
		}); err != nil {
			return messages.Message{}, false, err
		}
		if err := emit(ctx, s.cfg, callbacks.EventChatModelEnd, nil, output, nil); err != nil {
			return messages.Message{}, false, err
		}
		return messages.Message{}, false, nil
	case "error", "response.failed":
		err := fmt.Errorf("openai stream error: %s", event.Error.Message)
		if event.Error.Message == "" {
			err = fmt.Errorf("openai stream error")
		}
		_ = s.emitError(ctx, err)
		return messages.Message{}, false, err
	default:
		return messages.Message{}, false, nil
	}
}

func (s *responseStream) emitError(ctx context.Context, err error) error {
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

type streamEvent struct {
	Type         string           `json:"type"`
	Delta        string           `json:"delta"`
	Text         string           `json:"text"`
	OutputIndex  int              `json:"output_index"`
	ContentIndex int              `json:"content_index"`
	SummaryIndex int              `json:"summary_index"`
	ItemID       string           `json:"item_id"`
	Refusal      string           `json:"refusal"`
	Item         outputItem       `json:"item"`
	Part         reasoningPart    `json:"part"`
	Name         string           `json:"name"`
	Arguments    string           `json:"arguments"`
	Response     *responsePayload `json:"response"`
	Error        streamError      `json:"error"`
}

type streamError struct {
	Message string `json:"message"`
	Code    string `json:"code"`
}

type streamToolCall struct {
	ID        string
	Name      string
	Arguments string
}

type streamTextBlock struct {
	started bool
	text    string
}

type streamReasoningBlock struct {
	item outputItem
	text string
}

type reasoningPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func (s *responseStream) emitProtocol(ctx context.Context, event streamevents.Event) error {
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

func (s *responseStream) emitTextDelta(ctx context.Context, index int, text string) error {
	block := s.textBlocks[index]
	if block == nil {
		block = &streamTextBlock{}
		s.textBlocks[index] = block
	}
	if !block.started {
		block.started = true
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event: streamevents.EventContentBlockStart,
			Index: index,
			Content: messages.ContentBlock{
				"type": "text",
				"text": "",
			},
		}); err != nil {
			return err
		}
	}
	block.text += text
	return s.emitProtocol(ctx, streamevents.Event{
		Event: streamevents.EventContentBlockDelta,
		Index: index,
		Delta: messages.ContentBlock{
			"type": "text-delta",
			"text": text,
		},
	})
}

func (s *responseStream) finishTextBlock(ctx context.Context, index int) error {
	block := s.textBlocks[index]
	if block == nil || !block.started {
		return nil
	}
	delete(s.textBlocks, index)
	return s.emitProtocol(ctx, streamevents.Event{
		Event: streamevents.EventContentBlockFinish,
		Index: index,
		Content: messages.ContentBlock{
			"type": "text",
			"text": block.text,
		},
	})
}

func (s *responseStream) finishOpenTextBlocks(ctx context.Context) error {
	for index := range s.textBlocks {
		if err := s.finishTextBlock(ctx, index); err != nil {
			return err
		}
	}
	return nil
}

func (s *responseStream) emitOutputItemStart(ctx context.Context, index int, item outputItem) (messages.Message, bool, error) {
	block := contentBlockFromOutputItem(item, index)
	if err := s.emitProtocol(ctx, streamevents.Event{
		Event:   streamevents.EventContentBlockStart,
		Index:   index,
		Content: block,
	}); err != nil {
		return messages.Message{}, false, err
	}
	chunk := messages.AI("")
	chunk.ContentBlocks = []messages.ContentBlock{block}
	if err := emitStream(ctx, s.cfg, chunk); err != nil {
		return messages.Message{}, false, err
	}
	return chunk, true, nil
}

func (s *responseStream) emitOutputItemFinish(ctx context.Context, index int, item outputItem) error {
	block := contentBlockFromOutputItem(item, index)
	if item.Type == "reasoning" {
		if reasoning := s.reasoningBlocks[index]; reasoning != nil {
			if reasoning.text != "" {
				block["reasoning"] = reasoning.text
			}
			delete(s.reasoningBlocks, index)
		}
	}
	return s.emitProtocol(ctx, streamevents.Event{
		Event:   streamevents.EventContentBlockFinish,
		Index:   index,
		Content: block,
	})
}

func contentBlockFromOutputItem(item outputItem, index int) messages.ContentBlock {
	block := messages.ContentBlock{}
	for key, value := range item.Raw {
		block[key] = value
	}
	if block["type"] == nil {
		block["type"] = item.Type
	}
	block["index"] = index
	return block
}

func isProtocolOutputItem(itemType string) bool {
	switch itemType {
	case "reasoning",
		"web_search_call",
		"file_search_call",
		"code_interpreter_call",
		"custom_tool_call",
		"computer_call",
		"mcp_call",
		"mcp_list_tools",
		"mcp_approval_request",
		"image_generation_call",
		"tool_search_call",
		"tool_search_output",
		"apply_patch_call",
		"apply_patch_call_output",
		"compaction":
		return true
	default:
		return false
	}
}

func (s *responseStream) upsertReasoningBlock(index int) *streamReasoningBlock {
	block := s.reasoningBlocks[index]
	if block == nil {
		block = &streamReasoningBlock{}
		s.reasoningBlocks[index] = block
	}
	return block
}

func (s *responseStream) upsertToolCall(index int) *streamToolCall {
	call := s.toolCalls[index]
	if call == nil {
		call = &streamToolCall{}
		s.toolCalls[index] = call
	}
	return call
}

func (c streamToolCall) toToolCall() (messages.ToolCall, bool) {
	toolCall := messages.ToolCall{
		ID:   c.ID,
		Name: c.Name,
	}
	if c.Arguments == "" {
		return toolCall, true
	}
	if err := json.Unmarshal([]byte(c.Arguments), &toolCall.Args); err != nil {
		return messages.ToolCall{}, false
	}
	return toolCall, true
}
