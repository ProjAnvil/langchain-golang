package streamevents

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/projanvil/langchain-golang/core/messages"
)

// EventName identifies a chat stream protocol event. The names match the
// LangChain v3 content-block streaming protocol.
type EventName string

const (
	EventMessageStart       EventName = "message-start"
	EventContentBlockStart  EventName = "content-block-start"
	EventContentBlockDelta  EventName = "content-block-delta"
	EventContentBlockFinish EventName = "content-block-finish"
	EventMessageFinish      EventName = "message-finish"
)

// Event is a provider-neutral chat streaming protocol event.
type Event struct {
	Event   EventName             `json:"event"`
	Index   int                   `json:"index,omitempty"`
	Content messages.ContentBlock `json:"content,omitempty"`
	Delta   messages.ContentBlock `json:"delta,omitempty"`
	Output  messages.Message      `json:"output,omitempty"`
	Extra   map[string]any        `json:"extra,omitempty"`
}

// ChatModelStream projects v3 content-block protocol events into typed views.
//
// It is the Go equivalent of Python's ChatModelStream projection surface:
// text and reasoning accumulate from content-block-delta events, tool calls
// finalize from content-block-finish events, and Output returns the assembled
// AI message after message-finish. The input is only the provider-neutral
// streamevents.Event protocol; provider-native payload normalization belongs in
// adapters.
type ChatModelStream struct {
	events []Event

	textDeltas       []string
	reasoningDeltas  []string
	textByBlock      map[int]string
	reasoningByBlock map[int]string
	blocks           map[int]messages.ContentBlock

	toolChunks       map[int]toolChunk
	toolCalls        []messages.ToolCall
	invalidToolCalls []messages.ToolCall

	output messages.Message
	done   bool
	err    error
}

type toolChunk struct {
	ID   string
	Name string
	Args string
}

// NewChatModelStream creates an empty typed projection accumulator.
func NewChatModelStream() *ChatModelStream {
	return &ChatModelStream{
		textByBlock:      make(map[int]string),
		reasoningByBlock: make(map[int]string),
		blocks:           make(map[int]messages.ContentBlock),
		toolChunks:       make(map[int]toolChunk),
	}
}

// Dispatch records and applies one v3 protocol event.
func (s *ChatModelStream) Dispatch(event Event) {
	if s.done {
		return
	}
	s.events = append(s.events, cloneEvent(event))
	switch event.Event {
	case EventContentBlockDelta:
		s.pushDelta(event.Index, event.Delta)
	case EventContentBlockFinish:
		s.finishBlock(event.Index, event.Content)
	case EventMessageFinish:
		s.finish(event.Output)
	}
}

// Fail marks the stream complete with an error.
func (s *ChatModelStream) Fail(err error) {
	s.err = err
	s.done = true
}

// Done reports whether the stream has reached message-finish or failed.
func (s *ChatModelStream) Done() bool {
	return s.done
}

// Events returns a replayable copy of all raw protocol events seen so far.
func (s *ChatModelStream) Events() []Event {
	out := make([]Event, len(s.events))
	for i, event := range s.events {
		out[i] = cloneEvent(event)
	}
	return out
}

// TextDeltas returns a copy of streamed text deltas.
func (s *ChatModelStream) TextDeltas() []string {
	return append([]string(nil), s.textDeltas...)
}

// Text returns the final text projection, or the current delta sum mid-stream.
func (s *ChatModelStream) Text() string {
	if len(s.textByBlock) > 0 {
		return joinIndexedStrings(s.textByBlock)
	}
	return joinStrings(s.textDeltas)
}

// ReasoningDeltas returns a copy of streamed reasoning deltas.
func (s *ChatModelStream) ReasoningDeltas() []string {
	return append([]string(nil), s.reasoningDeltas...)
}

// Reasoning returns the final reasoning projection, or current delta sum.
func (s *ChatModelStream) Reasoning() string {
	if len(s.reasoningByBlock) > 0 {
		return joinIndexedStrings(s.reasoningByBlock)
	}
	return joinStrings(s.reasoningDeltas)
}

// ToolCalls returns finalized client-side tool calls.
func (s *ChatModelStream) ToolCalls() []messages.ToolCall {
	return cloneToolCalls(s.toolCalls)
}

// InvalidToolCalls returns finalized invalid client-side tool calls.
func (s *ChatModelStream) InvalidToolCalls() []messages.ToolCall {
	return cloneToolCalls(s.invalidToolCalls)
}

// Usage returns the UsageMetadata from the final assembled message.
// It is only meaningful after Done() returns true (i.e., after message-finish).
// This is the Go equivalent of Python's ChatModelStream.usage projection.
func (s *ChatModelStream) Usage() messages.UsageMetadata {
	return s.output.UsageMetadata
}

// Output returns the assembled AI message after message-finish.
func (s *ChatModelStream) Output() (messages.Message, error) {
	if s.err != nil {
		return messages.Message{}, s.err
	}
	if !s.done {
		return messages.Message{}, fmt.Errorf("stream is not done")
	}
	return cloneMessage(s.output), nil
}

func (s *ChatModelStream) pushDelta(index int, delta messages.ContentBlock) {
	switch delta["type"] {
	case "text-delta":
		text, _ := delta["text"].(string)
		if text == "" {
			return
		}
		s.textDeltas = append(s.textDeltas, text)
		s.textByBlock[index] += text
	case "reasoning-delta":
		reasoning, _ := delta["reasoning"].(string)
		if reasoning == "" {
			return
		}
		s.reasoningDeltas = append(s.reasoningDeltas, reasoning)
		s.reasoningByBlock[index] += reasoning
	case "tool_call_chunk":
		chunk := s.toolChunks[index]
		if id, _ := delta["id"].(string); id != "" && chunk.ID == "" {
			chunk.ID = id
		}
		if name, _ := delta["name"].(string); name != "" && chunk.Name == "" {
			chunk.Name = name
		}
		if args, _ := delta["args"].(string); args != "" {
			chunk.Args += args
		}
		s.toolChunks[index] = chunk
	}
}

func (s *ChatModelStream) finishBlock(index int, block messages.ContentBlock) {
	if block == nil {
		return
	}
	block = cloneBlock(block)
	switch block["type"] {
	case "text":
		if text, _ := block["text"].(string); text != "" {
			s.textByBlock[index] = text
		}
		block["text"] = s.textByBlock[index]
		s.blocks[index] = block
	case "reasoning":
		if reasoning, _ := block["reasoning"].(string); reasoning != "" {
			s.reasoningByBlock[index] = reasoning
		}
		if reasoning := s.reasoningByBlock[index]; reasoning != "" {
			block["reasoning"] = reasoning
		}
		s.blocks[index] = block
	case "tool_call":
		call := toolCallFromBlock(block)
		s.toolCalls = append(s.toolCalls, call)
		delete(s.toolChunks, index)
		s.blocks[index] = block
	case "invalid_tool_call":
		call := toolCallFromBlock(block)
		s.invalidToolCalls = append(s.invalidToolCalls, call)
		delete(s.toolChunks, index)
		s.blocks[index] = block
	default:
		s.blocks[index] = block
	}
}

func (s *ChatModelStream) finish(output messages.Message) {
	s.sweepToolChunks()
	message := output
	if message.Role == "" {
		message = messages.AI("")
	}
	// Preserve usage metadata and response metadata from the provider message;
	// overwrite content-level fields from accumulated stream projections.
	message.Content = s.Text()
	message.ContentBlocks = s.orderedBlocks()
	message.ToolCalls = cloneToolCalls(s.toolCalls)
	message.InvalidToolCalls = cloneToolCalls(s.invalidToolCalls)
	s.output = message
	s.done = true
}

func (s *ChatModelStream) sweepToolChunks() {
	indexes := make([]int, 0, len(s.toolChunks))
	for index := range s.toolChunks {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	for _, index := range indexes {
		chunk := s.toolChunks[index]
		call := messages.ToolCall{ID: chunk.ID, Name: chunk.Name}
		if chunk.Args != "" {
			if err := json.Unmarshal([]byte(chunk.Args), &call.Args); err != nil {
				s.invalidToolCalls = append(s.invalidToolCalls, call)
				s.blocks[index] = messages.ContentBlock{
					"type": "invalid_tool_call",
					"id":   chunk.ID,
					"name": chunk.Name,
					"args": chunk.Args,
				}
				continue
			}
		}
		s.toolCalls = append(s.toolCalls, call)
		s.blocks[index] = messages.ContentBlock{
			"type": "tool_call",
			"id":   chunk.ID,
			"name": chunk.Name,
			"args": call.Args,
		}
	}
	clear(s.toolChunks)
}

func (s *ChatModelStream) orderedBlocks() []messages.ContentBlock {
	if len(s.blocks) == 0 {
		return nil
	}
	indexes := make([]int, 0, len(s.blocks))
	for index := range s.blocks {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	blocks := make([]messages.ContentBlock, 0, len(indexes))
	for _, index := range indexes {
		blocks = append(blocks, cloneBlock(s.blocks[index]))
	}
	return blocks
}

func toolCallFromBlock(block messages.ContentBlock) messages.ToolCall {
	call := messages.ToolCall{}
	call.ID, _ = block["id"].(string)
	call.Name, _ = block["name"].(string)
	if args, ok := block["args"].(map[string]any); ok {
		call.Args = cloneMap(args)
	}
	return call
}

func joinIndexedStrings(values map[int]string) string {
	indexes := make([]int, 0, len(values))
	for index := range values {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	var out string
	for _, index := range indexes {
		out += values[index]
	}
	return out
}

func joinStrings(values []string) string {
	var out string
	for _, value := range values {
		out += value
	}
	return out
}

func cloneEvent(event Event) Event {
	event.Content = cloneBlock(event.Content)
	event.Delta = cloneBlock(event.Delta)
	event.Output = cloneMessage(event.Output)
	if event.Extra != nil {
		event.Extra = cloneMap(event.Extra)
	}
	return event
}

func cloneMessage(message messages.Message) messages.Message {
	message.ContentBlocks = cloneBlocks(message.ContentBlocks)
	message.ToolCalls = cloneToolCalls(message.ToolCalls)
	message.InvalidToolCalls = cloneToolCalls(message.InvalidToolCalls)
	message.ResponseMetadata = cloneMap(message.ResponseMetadata)
	message.AdditionalKwargs = cloneMap(message.AdditionalKwargs)
	message.ProviderNativeEvent = cloneMap(message.ProviderNativeEvent)
	return message
}

func cloneBlocks(blocks []messages.ContentBlock) []messages.ContentBlock {
	if blocks == nil {
		return nil
	}
	out := make([]messages.ContentBlock, len(blocks))
	for i, block := range blocks {
		out[i] = cloneBlock(block)
	}
	return out
}

func cloneBlock(block messages.ContentBlock) messages.ContentBlock {
	if block == nil {
		return nil
	}
	return messages.ContentBlock(cloneMap(map[string]any(block)))
}

func cloneToolCalls(calls []messages.ToolCall) []messages.ToolCall {
	if calls == nil {
		return nil
	}
	out := make([]messages.ToolCall, len(calls))
	for i, call := range calls {
		out[i] = call
		out[i].Args = cloneMap(call.Args)
	}
	return out
}

func cloneMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}
