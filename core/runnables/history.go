package runnables

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/projanvil/langchain-golang/core/chathistory"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

const defaultSessionIDKey = "session_id"

// MessageHistoryFactory returns the chat history selected by configurable
// runtime fields. It mirrors Python's get_session_history hook.
type MessageHistoryFactory func(context.Context, map[string]any) (chathistory.History, error)

// RunnableWithMessageHistory wraps a runnable and maintains chat message
// history around each successful invocation.
type RunnableWithMessageHistory struct {
	Runnable           Runnable[any, any]
	GetSessionHistory  MessageHistoryFactory
	InputMessagesKey   string
	OutputMessagesKey  string
	HistoryMessagesKey string
	HistoryFactoryKeys []string
}

// NewRunnableWithMessageHistory creates a history-aware runnable. When
// factoryKeys is empty, "session_id" is required in Configurable fields.
func NewRunnableWithMessageHistory(
	runnable Runnable[any, any],
	getSessionHistory MessageHistoryFactory,
	factoryKeys ...string,
) (RunnableWithMessageHistory, error) {
	if runnable == nil {
		return RunnableWithMessageHistory{}, fmt.Errorf("runnable is required")
	}
	if getSessionHistory == nil {
		return RunnableWithMessageHistory{}, fmt.Errorf("session history factory is required")
	}
	if len(factoryKeys) == 0 {
		factoryKeys = []string{defaultSessionIDKey}
	}
	return RunnableWithMessageHistory{
		Runnable:           runnable,
		GetSessionHistory:  getSessionHistory,
		HistoryFactoryKeys: append([]string(nil), factoryKeys...),
	}, nil
}

// Invoke injects history into the wrapped runnable input and appends the new
// input/output messages to history after the wrapped runnable succeeds.
func (r RunnableWithMessageHistory) Invoke(ctx context.Context, input any, opts ...Option) (any, error) {
	history, err := r.history(ctx, opts...)
	if err != nil {
		return nil, err
	}
	runnableInput, newInputMessages, err := r.prepareInput(ctx, history, input)
	if err != nil {
		return nil, err
	}
	output, err := r.Runnable.Invoke(ctx, runnableInput, opts...)
	if err != nil {
		return nil, err
	}
	outputMessages, err := r.outputMessages(output)
	if err != nil {
		return nil, err
	}
	if err := history.AddMessages(ctx, append(newInputMessages, outputMessages...)); err != nil {
		return nil, err
	}
	return output, nil
}

// Batch invokes the wrapper for all inputs while preserving order.
func (r RunnableWithMessageHistory) Batch(ctx context.Context, inputs []any, opts ...Option) ([]any, error) {
	outputs := make([]any, len(inputs))
	errs := make([]error, len(inputs))
	for i, input := range inputs {
		outputs[i], errs[i] = r.Invoke(ctx, input, opts...)
	}
	return outputs, errors.Join(errs...)
}

// Stream updates history when the wrapped stream is exhausted successfully.
func (r RunnableWithMessageHistory) Stream(ctx context.Context, input any, opts ...Option) (Stream[any], error) {
	history, err := r.history(ctx, opts...)
	if err != nil {
		return nil, err
	}
	runnableInput, newInputMessages, err := r.prepareInput(ctx, history, input)
	if err != nil {
		return nil, err
	}
	stream, err := r.Runnable.Stream(ctx, runnableInput, opts...)
	if err != nil {
		return nil, err
	}
	return &messageHistoryStream{
		stream:            stream,
		history:           history,
		inputMessages:     newInputMessages,
		outputMessages:    r.outputMessages,
		outputMessagesKey: r.OutputMessagesKey,
	}, nil
}

// InputSchema returns the wrapped runnable input schema.
func (r RunnableWithMessageHistory) InputSchema() schema.Schema { return r.Runnable.InputSchema() }

// OutputSchema returns the wrapped runnable output schema.
func (r RunnableWithMessageHistory) OutputSchema() schema.Schema { return r.Runnable.OutputSchema() }

// ConfigSchema returns the wrapped runnable config schema plus required message
// history factory keys.
func (r RunnableWithMessageHistory) ConfigSchema() schema.Schema {
	cfg := GetConfigSchema(r.Runnable)
	configurable, _ := configurableSchema(cfg)
	props := schemaProperties(configurable)
	if props == nil {
		props = map[string]schema.Schema{}
	}
	requiredSet := map[string]bool{}
	for _, key := range schemaRequired(configurable) {
		requiredSet[key] = true
	}
	for _, key := range r.HistoryFactoryKeys {
		props[key] = schema.String("message history factory key")
		requiredSet[key] = true
	}
	required := make([]string, 0, len(requiredSet))
	for key := range requiredSet {
		required = append(required, key)
	}
	sort.Strings(required)
	return configurableConfigSchema(props, required...)
}

func (r RunnableWithMessageHistory) history(ctx context.Context, opts ...Option) (chathistory.History, error) {
	cfg := NewConfig(opts...)
	if existing, ok := cfg.Configurable["message_history"].(chathistory.History); ok {
		return existing, nil
	}
	values := make(map[string]any, len(r.HistoryFactoryKeys))
	for _, key := range r.HistoryFactoryKeys {
		value, ok := cfg.Configurable[key]
		if !ok {
			return nil, fmt.Errorf("missing configurable key %q for message history", key)
		}
		values[key] = value
	}
	history, err := r.GetSessionHistory(ctx, values)
	if err != nil {
		return nil, err
	}
	if history == nil {
		return nil, fmt.Errorf("message history factory returned nil")
	}
	return history, nil
}

func (r RunnableWithMessageHistory) prepareInput(
	ctx context.Context,
	history chathistory.History,
	input any,
) (any, []messages.Message, error) {
	historic, err := history.Messages(ctx)
	if err != nil {
		return nil, nil, err
	}
	newInputMessages, err := r.inputMessages(input)
	if err != nil {
		return nil, nil, err
	}
	if r.HistoryMessagesKey != "" {
		inputMap, ok := input.(map[string]any)
		if !ok {
			return nil, nil, fmt.Errorf("history_messages_key requires map input")
		}
		out := cloneMap(inputMap)
		out[r.HistoryMessagesKey] = append([]messages.Message(nil), historic...)
		return out, newInputMessages, nil
	}

	allMessages := append(append([]messages.Message(nil), historic...), newInputMessages...)
	if r.InputMessagesKey == "" {
		return allMessages, newInputMessages, nil
	}
	inputMap, ok := input.(map[string]any)
	if !ok {
		return nil, nil, fmt.Errorf("input_messages_key requires map input")
	}
	out := cloneMap(inputMap)
	out[r.InputMessagesKey] = allMessages
	return out, newInputMessages, nil
}

func (r RunnableWithMessageHistory) inputMessages(input any) ([]messages.Message, error) {
	value := input
	if inputMap, ok := input.(map[string]any); ok {
		key := r.InputMessagesKey
		if key == "" {
			key = mapMessageKey(inputMap, "input")
		}
		value = inputMap[key]
	}
	return asInputMessages(value)
}

func (r RunnableWithMessageHistory) outputMessages(output any) ([]messages.Message, error) {
	value := output
	if outputMap, ok := output.(map[string]any); ok {
		key := r.OutputMessagesKey
		if key == "" {
			key = mapMessageKey(outputMap, "output")
		}
		value = outputMap[key]
	}
	return asOutputMessages(value)
}

func mapMessageKey(values map[string]any, fallback string) string {
	if len(values) == 1 {
		for key := range values {
			return key
		}
	}
	return fallback
}

func asInputMessages(value any) ([]messages.Message, error) {
	switch typed := value.(type) {
	case string:
		return []messages.Message{messages.Human(typed)}, nil
	case messages.Message:
		return []messages.Message{typed}, nil
	case []messages.Message:
		return append([]messages.Message(nil), typed...), nil
	case []any:
		out := make([]messages.Message, 0, len(typed))
		for _, item := range typed {
			message, ok := item.(messages.Message)
			if !ok {
				return nil, fmt.Errorf("expected message in slice, got %T", item)
			}
			out = append(out, message)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("expected string, message, or message slice, got %T", value)
	}
}

func asOutputMessages(value any) ([]messages.Message, error) {
	switch typed := value.(type) {
	case string:
		return []messages.Message{messages.AI(typed)}, nil
	case messages.Message:
		return []messages.Message{typed}, nil
	case []messages.Message:
		return append([]messages.Message(nil), typed...), nil
	case []any:
		out := make([]messages.Message, 0, len(typed))
		for _, item := range typed {
			message, ok := item.(messages.Message)
			if !ok {
				return nil, fmt.Errorf("expected message in slice, got %T", item)
			}
			out = append(out, message)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("expected string, message, or message slice, got %T", value)
	}
}

type messageHistoryStream struct {
	stream            Stream[any]
	history           chathistory.History
	inputMessages     []messages.Message
	outputMessages    func(any) ([]messages.Message, error)
	outputMessagesKey string
	chunks            []any
	done              bool
}

func (s *messageHistoryStream) Next(ctx context.Context) (any, bool, error) {
	value, ok, err := s.stream.Next(ctx)
	if err != nil || ok {
		if ok {
			s.chunks = append(s.chunks, value)
		}
		return value, ok, err
	}
	if s.done {
		return nil, false, nil
	}
	s.done = true
	outputValue := streamOutputValue(s.chunks)
	outputMessages, err := s.outputMessages(outputValue)
	if err != nil {
		return nil, false, err
	}
	if err := s.history.AddMessages(ctx, append(s.inputMessages, outputMessages...)); err != nil {
		return nil, false, err
	}
	return nil, false, nil
}

func (s *messageHistoryStream) Close() error {
	return s.stream.Close()
}

func streamOutputValue(chunks []any) any {
	if len(chunks) == 1 {
		return chunks[0]
	}
	messagesOut := make([]messages.Message, 0, len(chunks))
	text := ""
	for _, chunk := range chunks {
		switch typed := chunk.(type) {
		case string:
			text += typed
		case messages.Message:
			messagesOut = append(messagesOut, typed)
		default:
			return chunks
		}
	}
	if text != "" && len(messagesOut) == 0 {
		return text
	}
	if len(messagesOut) > 0 && text == "" {
		return messagesOut
	}
	return chunks
}
