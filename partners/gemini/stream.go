package gemini

import (
	"context"
	"fmt"
	"sync"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/partners/internal/providerutil"
	"google.golang.org/genai"
)

// createStream starts the SSE generateContent call (alt=sse) and adapts the
// SDK's iterator to runnables.Stream. A pump goroutine drains the iterator
// into a channel so Next honors its own context and Close cancels the
// underlying HTTP request.
func (m ChatModel) createStream(
	ctx context.Context,
	input []messages.Message,
	cfg runnables.Config,
) (runnables.Stream[messages.Message], error) {
	client, err := m.ensureClient()
	if err != nil {
		return nil, err
	}
	systemInstruction, contents, err := buildContents(input)
	if err != nil {
		return nil, err
	}
	config, err := m.buildGenerateContentConfig(systemInstruction)
	if err != nil {
		return nil, err
	}

	streamCtx, cancel := context.WithCancel(ctx)
	iterator := client.Models.GenerateContentStream(streamCtx, m.config.Model, contents, config)

	items := make(chan streamItem, 4)
	pumpDone := make(chan struct{})
	go func() {
		defer close(items)
		for response, err := range iterator {
			select {
			case items <- streamItem{response: response, err: err}:
			case <-pumpDone:
				return
			}
		}
	}()

	return &genaiStream{
		model:  m.config.Model,
		items:  items,
		pump:   pumpDone,
		cancel: cancel,
		cfg:    cfg,
		output: messages.AI(""),
	}, nil
}

type streamItem struct {
	response *genai.GenerateContentResponse
	err      error
}

// genaiStream yields one messages.Message chunk per SSE response. Chunks with
// neither text nor tool calls are skipped (the API emits bookkeeping-only
// responses), and usage metadata is attached solely to the terminal chunk:
// the API reports cumulative counts on every response, so re-emitting them
// per chunk would overcount aggregating consumers.
type genaiStream struct {
	model  string
	items  <-chan streamItem
	pump   chan struct{}
	cancel context.CancelFunc
	cfg    runnables.Config

	finished      bool
	ended         bool
	lastUsage     *messages.UsageMetadata
	usageAttached bool
	pending       []messages.Message
	output        messages.Message
	shutdown      sync.Once
}

// Next returns the next stream chunk. Iteration ends (ok=false) after the
// terminal chunk; a canceled context or transport error surfaces as an error
// after the error callback event is emitted.
func (s *genaiStream) Next(ctx context.Context) (messages.Message, bool, error) {
	for {
		if s.ended {
			return messages.Message{}, false, nil
		}
		if len(s.pending) > 0 {
			chunk := s.pending[0]
			s.pending = s.pending[1:]
			if err := emitStream(ctx, s.cfg, chunk); err != nil {
				s.ended = true
				return messages.Message{}, false, err
			}
			return chunk, true, nil
		}
		if s.finished {
			// Iterator drained: flush any usage that arrived after the last
			// finish-carrying chunk as a terminal usage-only chunk (the
			// anthropic adapter's stream_usage shape), then end.
			if s.lastUsage != nil && !s.usageAttached {
				s.usageAttached = true
				chunk := messages.AI("")
				chunk.UsageMetadata = *s.lastUsage
				s.pending = append(s.pending, chunk)
				continue
			}
			s.ended = true
			s.releaseResources()
			if err := emit(ctx, s.cfg, callbacks.EventChatModelEnd, nil, s.output, nil); err != nil {
				return messages.Message{}, false, err
			}
			return messages.Message{}, false, nil
		}

		select {
		case <-ctx.Done():
			s.finished = true
			s.ended = true
			err := fmt.Errorf("gemini %s: stream canceled: %w", s.model, ctx.Err())
			_ = emit(ctx, s.cfg, callbacks.EventChatModelError, nil, nil, err)
			return messages.Message{}, false, err
		case item, ok := <-s.items:
			if !ok {
				s.finished = true
				continue
			}
			if item.err != nil {
				s.finished = true
				s.ended = true
				err := fmt.Errorf("gemini %s: stream: %w", s.model, item.err)
				_ = emit(ctx, s.cfg, callbacks.EventChatModelError, nil, nil, err)
				return messages.Message{}, false, err
			}
			if chunk, yield := s.handleResponse(item.response); yield {
				if err := emitStream(ctx, s.cfg, chunk); err != nil {
					s.ended = true
					return messages.Message{}, false, err
				}
				return chunk, true, nil
			}
		}
	}
}

// handleResponse folds one SSE response into the aggregate output message and
// decides whether it yields a chunk now. finishReason-carrying responses mark
// the terminal chunk and take the usage; responses without finishReason defer
// usage to whatever terminates later.
func (s *genaiStream) handleResponse(response *genai.GenerateContentResponse) (messages.Message, bool) {
	if response == nil {
		return messages.Message{}, false
	}
	if usage := usageToMetadata(response.UsageMetadata); usage != (messages.UsageMetadata{}) {
		s.lastUsage = &usage
		s.output.UsageMetadata = usage
	}

	chunk := messages.AI("")
	var finish string
	if len(response.Candidates) > 0 {
		candidate := response.Candidates[0]
		finish = string(candidate.FinishReason)
		if candidate.Content != nil {
			callIndex := len(s.output.ToolCalls)
			for _, part := range candidate.Content.Parts {
				if part == nil {
					continue
				}
				switch {
				case part.FunctionCall != nil:
					call := messages.ToolCall{
						ID:   orSyntheticCallID(part.FunctionCall.ID, callIndex),
						Name: part.FunctionCall.Name,
						Args: part.FunctionCall.Args,
					}
					callIndex++
					chunk.ToolCalls = append(chunk.ToolCalls, call)
				case part.Text != "" && !part.Thought:
					chunk.Content += part.Text
				}
			}
		}
	}
	hasContent := chunk.Content != "" || len(chunk.ToolCalls) > 0

	// Aggregate for the chat-model-end event.
	s.output.Content += chunk.Content
	s.output.ToolCalls = append(s.output.ToolCalls, chunk.ToolCalls...)
	if response.ModelVersion != "" {
		s.output.ID = response.ResponseID
		if s.output.ResponseMetadata == nil {
			s.output.ResponseMetadata = map[string]any{}
		}
		s.output.ResponseMetadata["model"] = response.ModelVersion
		s.output.ResponseMetadata["model_provider"] = providerName
	}

	if finish != "" {
		if s.output.ResponseMetadata == nil {
			s.output.ResponseMetadata = map[string]any{}
		}
		s.output.ResponseMetadata["finish_reason"] = finish
		if s.lastUsage != nil && !s.usageAttached {
			chunk.UsageMetadata = *s.lastUsage
			s.usageAttached = true
		}
		if !hasContent && chunk.UsageMetadata == (messages.UsageMetadata{}) {
			// A finish-only response with nothing to say: end silently.
			return messages.Message{}, false
		}
		return chunk, true
	}
	if !hasContent {
		return messages.Message{}, false
	}
	return chunk, true
}

// Close terminates the stream: the underlying HTTP request is canceled (which
// ends the SDK iterator and its response body) and further Next calls end
// without emitting the chat-model-end event, mirroring the anthropic
// adapter's Close. It is idempotent, like every other Stream implementation.
func (s *genaiStream) Close() error {
	s.finished = true
	s.ended = true
	s.releaseResources()
	return nil
}

// releaseResources stops the pump goroutine and cancels the stream context
// exactly once, whether the stream is Closed early or ends naturally. Without
// it a fully drained stream would leak its context.WithCancel registration
// until the parent context ends.
func (s *genaiStream) releaseResources() {
	s.shutdown.Do(func() {
		close(s.pump)
		s.cancel()
	})
}

func emitStream(ctx context.Context, cfg runnables.Config, chunk messages.Message) error {
	return providerutil.EmitStream(ctx, cfg, chunk)
}
