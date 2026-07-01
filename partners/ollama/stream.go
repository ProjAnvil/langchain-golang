package ollama

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

const (
	textBlockIndex     = 0
	reasoningBlockIndex = 1
)

func (m ChatModel) createChatStream(
	ctx context.Context,
	input []messages.Message,
	cfg runnables.Config,
) (*chatStream, error) {
	requestPayload := m.buildRequest(input, true)

	body, err := json.Marshal(requestPayload)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, m.config.Timeout)
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		strings.TrimRight(m.config.BaseURL, "/")+"/api/chat",
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
		return nil, httpclient.ResponseError(providerName, "/api/chat", resp)
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	return &chatStream{
		ctx:        ctx,
		cancel:     cancel,
		body:       resp.Body,
		scanner:    scanner,
		cfg:        cfg,
		reasoning:  m.reasoning,
		textBlock:  &streamTextBlock{},
	}, nil
}

type chatStream struct {
	ctx             context.Context
	cancel          context.CancelFunc
	body            io.Closer
	scanner         *bufio.Scanner
	cfg             runnables.Config
	reasoning       any
	pending         []streamStep
	textBlock       *streamTextBlock
	reasoningBlock  *streamTextBlock
	toolIndex       int
	output          messages.Message
	done            bool
	protocolStarted bool
}

type streamStep func(ctx context.Context) (messages.Message, bool, error)

type streamTextBlock struct {
	started bool
	text    string
}

// Next returns the next streamed message chunk. Ollama emits newline-delimited
// JSON; each line may carry a text delta, a reasoning delta, tool calls, or the
// terminal done payload. Each line is decomposed into at most a few ordered
// steps so a single Next call maps to one user-visible chunk.
func (s *chatStream) Next(ctx context.Context) (messages.Message, bool, error) {
	for {
		if s.done {
			return messages.Message{}, false, nil
		}
		if err := ctx.Err(); err != nil {
			_ = s.emitError(ctx, err)
			return messages.Message{}, false, err
		}
		if len(s.pending) > 0 {
			step := s.pending[0]
			s.pending = s.pending[1:]
			return step(ctx)
		}
		if !s.scanner.Scan() {
			if err := s.scanner.Err(); err != nil {
				_ = s.emitError(ctx, err)
				return messages.Message{}, false, err
			}
			// Stream ended without an explicit done payload; finalize gracefully.
			s.done = true
			if err := s.finalize(ctx); err != nil {
				return messages.Message{}, false, err
			}
			return messages.Message{}, false, nil
		}

		line := strings.TrimSpace(s.scanner.Text())
		if line == "" {
			continue
		}
		var chunk chatResponse
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			err = fmt.Errorf("decode ollama stream chunk: %w", err)
			_ = s.emitError(ctx, err)
			return messages.Message{}, false, err
		}
		s.planChunk(ctx, chunk)
	}
}

// Close releases stream resources.
func (s *chatStream) Close() error {
	s.done = true
	s.cancel()
	return s.body.Close()
}

// planChunk inspects one Ollama NDJSON chunk and queues ordered steps that emit
// v3 protocol events and yield message chunks.
func (s *chatStream) planChunk(ctx context.Context, chunk chatResponse) {
	if chunk.Done && chunk.DoneReason == "load" && strings.TrimSpace(chunk.Message.Content) == "" && len(chunk.Message.ToolCalls) == 0 {
		// Model loaded but produced no generation; skip like the Python adapter.
		return
	}

	if delta := chunk.Message.Content; delta != "" {
		text := delta
		s.pending = append(s.pending, func(ctx context.Context) (messages.Message, bool, error) {
			if err := s.startTextBlock(ctx); err != nil {
				return messages.Message{}, false, err
			}
			s.textBlock.text += text
			s.output.Content += text
			if err := s.emitProtocol(ctx, streamevents.Event{
				Event: streamevents.EventContentBlockDelta,
				Index: textBlockIndex,
				Delta: messages.ContentBlock{
					"type": "text-delta",
					"text": text,
				},
			}); err != nil {
				return messages.Message{}, false, err
			}
			out := messages.AI(text)
			if err := emitStream(ctx, s.cfg, out); err != nil {
				return messages.Message{}, false, err
			}
			return out, true, nil
		})
	}

	if reasoningEnabled(s.reasoning) {
		if thinking := chunk.Message.Thinking; thinking != "" {
			text := thinking
			s.pending = append(s.pending, func(ctx context.Context) (messages.Message, bool, error) {
				if err := s.startReasoningBlock(ctx); err != nil {
					return messages.Message{}, false, err
				}
				s.reasoningBlock.text += text
				if err := s.emitProtocol(ctx, streamevents.Event{
					Event: streamevents.EventContentBlockDelta,
					Index: reasoningBlockIndex,
					Delta: messages.ContentBlock{
						"type":      "reasoning-delta",
						"reasoning": text,
					},
				}); err != nil {
					return messages.Message{}, false, err
				}
				out := messages.AI("")
				out.ContentBlocks = []messages.ContentBlock{{
					"type":      "reasoning",
					"reasoning": text,
					"index":     reasoningBlockIndex,
				}}
				if err := emitStream(ctx, s.cfg, out); err != nil {
					return messages.Message{}, false, err
				}
				return out, true, nil
			})
		}
	}

	toolCalls, invalidToolCalls := parseToolCalls(chunk.Message.ToolCalls)
	if len(toolCalls) > 0 || len(invalidToolCalls) > 0 {
		calls := toolCalls
		invalid := invalidToolCalls
		s.pending = append(s.pending, func(ctx context.Context) (messages.Message, bool, error) {
			chunk := messages.AI("")
			chunk.ToolCalls = append([]messages.ToolCall(nil), calls...)
			chunk.InvalidToolCalls = append([]messages.ToolCall(nil), invalid...)
			for _, call := range calls {
				index := s.nextToolIndex()
				if err := s.emitToolCallFinish(ctx, index, call); err != nil {
					return messages.Message{}, false, err
				}
			}
			s.output.ToolCalls = append(s.output.ToolCalls, calls...)
			if err := emitStream(ctx, s.cfg, chunk); err != nil {
				return messages.Message{}, false, err
			}
			return chunk, true, nil
		})
	}

	if chunk.Done {
		usage := usageFromChunk(chunk)
		done := chunk
		s.pending = append(s.pending, func(ctx context.Context) (messages.Message, bool, error) {
			s.applyDoneMetadata(done)
			if usage.TotalTokens > 0 {
				s.output.UsageMetadata = usage
			}
			if err := s.finalize(ctx); err != nil {
				return messages.Message{}, false, err
			}
			s.done = true
			return messages.Message{}, false, nil
		})
	}
}

func (s *chatStream) emitToolCallFinish(ctx context.Context, index int, call messages.ToolCall) error {
	if err := s.beginProtocol(ctx); err != nil {
		return err
	}
	block := messages.ContentBlock{
		"type": "tool_call",
		"id":   call.ID,
		"name": call.Name,
		"args": call.Args,
	}
	if err := s.emitProtocol(ctx, streamevents.Event{
		Event:   streamevents.EventContentBlockStart,
		Index:   index,
		Content: block,
	}); err != nil {
		return err
	}
	s.output.ContentBlocks = append(s.output.ContentBlocks, block)
	return s.emitProtocol(ctx, streamevents.Event{
		Event:   streamevents.EventContentBlockFinish,
		Index:   index,
		Content: block,
	})
}

// nextToolIndex returns a strictly increasing content-block index for the next
// streamed tool call. Text and reasoning blocks reserve indices 0 and 1.
func (s *chatStream) nextToolIndex() int {
	if s.toolIndex < firstToolBase {
		s.toolIndex = firstToolBase
	}
	index := s.toolIndex
	s.toolIndex++
	return index
}

// finalize closes any open text/reasoning blocks and emits message-finish.
func (s *chatStream) finalize(ctx context.Context) error {
	if err := s.finishTextBlock(ctx); err != nil {
		return err
	}
	if err := s.finishReasoningBlock(ctx); err != nil {
		return err
	}
	if s.protocolStarted {
		if err := s.emitProtocol(ctx, streamevents.Event{
			Event:  streamevents.EventMessageFinish,
			Output: s.output,
		}); err != nil {
			return err
		}
	}
	return emit(ctx, s.cfg, callbacks.EventChatModelEnd, nil, s.output, nil)
}

// beginProtocol emits the v3 message-start event exactly once, before any
// content-block event is emitted.
func (s *chatStream) beginProtocol(ctx context.Context) error {
	if s.protocolStarted {
		return nil
	}
	s.protocolStarted = true
	return s.emitProtocol(ctx, streamevents.Event{Event: streamevents.EventMessageStart})
}

func (s *chatStream) startTextBlock(ctx context.Context) error {
	if err := s.beginProtocol(ctx); err != nil {
		return err
	}
	if s.textBlock.started {
		return nil
	}
	s.textBlock.started = true
	return s.emitProtocol(ctx, streamevents.Event{
		Event: streamevents.EventContentBlockStart,
		Index: textBlockIndex,
		Content: messages.ContentBlock{
			"type": "text",
			"text": "",
		},
	})
}

func (s *chatStream) finishTextBlock(ctx context.Context) error {
	if s.textBlock == nil || !s.textBlock.started {
		return nil
	}
	if !s.protocolStarted {
		return nil
	}
	text := s.textBlock.text
	s.textBlock.started = false
	block := messages.ContentBlock{
		"type": "text",
		"text": text,
	}
	s.output.ContentBlocks = append(s.output.ContentBlocks, block)
	return s.emitProtocol(ctx, streamevents.Event{
		Event:   streamevents.EventContentBlockFinish,
		Index:   textBlockIndex,
		Content: block,
	})
}

func (s *chatStream) startReasoningBlock(ctx context.Context) error {
	if err := s.beginProtocol(ctx); err != nil {
		return err
	}
	if s.reasoningBlock == nil {
		s.reasoningBlock = &streamTextBlock{}
	}
	if s.reasoningBlock.started {
		return nil
	}
	s.reasoningBlock.started = true
	return s.emitProtocol(ctx, streamevents.Event{
		Event: streamevents.EventContentBlockStart,
		Index: reasoningBlockIndex,
		Content: messages.ContentBlock{
			"type":      "reasoning",
			"reasoning": "",
		},
	})
}

func (s *chatStream) finishReasoningBlock(ctx context.Context) error {
	if s.reasoningBlock == nil || !s.reasoningBlock.started {
		return nil
	}
	text := s.reasoningBlock.text
	s.reasoningBlock.started = false
	block := messages.ContentBlock{
		"type":      "reasoning",
		"reasoning": text,
	}
	s.output.ContentBlocks = append(s.output.ContentBlocks, block)
	return s.emitProtocol(ctx, streamevents.Event{
		Event:   streamevents.EventContentBlockFinish,
		Index:   reasoningBlockIndex,
		Content: block,
	})
}

func (s *chatStream) applyDoneMetadata(chunk chatResponse) {
	if s.output.ResponseMetadata == nil {
		s.output.ResponseMetadata = map[string]any{}
	}
	s.output.ResponseMetadata["model"] = chunk.Model
	s.output.ResponseMetadata["model_name"] = chunk.Model
	s.output.ResponseMetadata["created_at"] = chunk.CreatedAt
	s.output.ResponseMetadata["done_reason"] = chunk.DoneReason
	s.output.ResponseMetadata["model_provider"] = "ollama"
}

func (s *chatStream) emitProtocol(ctx context.Context, event streamevents.Event) error {
	if s.cfg.Callbacks.Empty() {
		return nil
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

func (s *chatStream) emitError(ctx context.Context, err error) error {
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

// reasoningEnabled reports whether the think request parameter requests
// reasoning content (true or a non-empty string level).
func reasoningEnabled(reasoning any) bool {
	switch value := reasoning.(type) {
	case bool:
		return value
	case string:
		return value != "" && value != "false"
	default:
		return false
	}
}

func usageFromChunk(chunk chatResponse) messages.UsageMetadata {
	return messages.UsageMetadata{
		InputTokens:  chunk.PromptEvalCount,
		OutputTokens: chunk.EvalCount,
		TotalTokens:  chunk.PromptEvalCount + chunk.EvalCount,
	}
}

// firstToolBase is the first content-block index reserved for tool calls. Text
// and reasoning blocks reserve indices 0 and 1 when present.
const firstToolBase = 2
