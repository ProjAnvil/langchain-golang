package chathistory

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/projanvil/langchain-golang/core/messages"
)

var ErrNotImplemented = errors.New("chat history method is not implemented")

// History stores the message interactions in a chat.
type History interface {
	Messages(ctx context.Context) ([]messages.Message, error)
	AddMessage(ctx context.Context, message messages.Message) error
	AddMessages(ctx context.Context, batch []messages.Message) error
	Clear(ctx context.Context) error
}

// BaseChatMessageHistory provides the same fallback behavior as Python's
// BaseChatMessageHistory: single-message add can delegate to bulk add, and bulk
// add can delegate to single-message add.
type BaseChatMessageHistory struct {
	MessagesFunc    func(context.Context) ([]messages.Message, error)
	AddMessageFunc  func(context.Context, messages.Message) error
	AddMessagesFunc func(context.Context, []messages.Message) error
	ClearFunc       func(context.Context) error
}

func (h *BaseChatMessageHistory) Messages(ctx context.Context) ([]messages.Message, error) {
	if h.MessagesFunc == nil {
		return nil, fmt.Errorf("%w: messages", ErrNotImplemented)
	}
	return h.MessagesFunc(ctx)
}

func (h *BaseChatMessageHistory) AddMessage(ctx context.Context, message messages.Message) error {
	if h.AddMessageFunc != nil {
		return h.AddMessageFunc(ctx, message)
	}
	if h.AddMessagesFunc != nil {
		return h.AddMessagesFunc(ctx, []messages.Message{message})
	}
	return fmt.Errorf("%w: add message or add messages", ErrNotImplemented)
}

func (h *BaseChatMessageHistory) AddMessages(ctx context.Context, batch []messages.Message) error {
	if h.AddMessagesFunc != nil {
		return h.AddMessagesFunc(ctx, batch)
	}
	if h.AddMessageFunc == nil {
		return fmt.Errorf("%w: add message or add messages", ErrNotImplemented)
	}
	for _, message := range batch {
		if err := h.AddMessageFunc(ctx, message); err != nil {
			return err
		}
	}
	return nil
}

func (h *BaseChatMessageHistory) AddUserMessage(ctx context.Context, content string) error {
	return h.AddMessage(ctx, messages.Human(content))
}

func (h *BaseChatMessageHistory) AddAIMessage(ctx context.Context, content string) error {
	return h.AddMessage(ctx, messages.AI(content))
}

func (h *BaseChatMessageHistory) Clear(ctx context.Context) error {
	if h.ClearFunc == nil {
		return fmt.Errorf("%w: clear", ErrNotImplemented)
	}
	return h.ClearFunc(ctx)
}

func (h *BaseChatMessageHistory) String() string {
	if h.MessagesFunc == nil {
		return ""
	}
	batch, err := h.Messages(context.Background())
	if err != nil {
		return ""
	}
	return BufferString(batch)
}

type InMemoryChatMessageHistory struct {
	mu       sync.RWMutex
	messages []messages.Message
}

func NewInMemoryChatMessageHistory(initial ...messages.Message) *InMemoryChatMessageHistory {
	return &InMemoryChatMessageHistory{
		messages: cloneMessages(initial),
	}
}

func (h *InMemoryChatMessageHistory) Messages(context.Context) ([]messages.Message, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return cloneMessages(h.messages), nil
}

func (h *InMemoryChatMessageHistory) AddMessage(_ context.Context, message messages.Message) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.messages = append(h.messages, message)
	return nil
}

func (h *InMemoryChatMessageHistory) AddMessages(_ context.Context, batch []messages.Message) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.messages = append(h.messages, batch...)
	return nil
}

func (h *InMemoryChatMessageHistory) AddUserMessage(ctx context.Context, content string) error {
	return h.AddMessage(ctx, messages.Human(content))
}

func (h *InMemoryChatMessageHistory) AddAIMessage(ctx context.Context, content string) error {
	return h.AddMessage(ctx, messages.AI(content))
}

func (h *InMemoryChatMessageHistory) Clear(context.Context) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.messages = nil
	return nil
}

func (h *InMemoryChatMessageHistory) String() string {
	batch, err := h.Messages(context.Background())
	if err != nil {
		return ""
	}
	return BufferString(batch)
}

func BufferString(batch []messages.Message) string {
	lines := make([]string, 0, len(batch))
	for _, message := range batch {
		prefix := string(message.Role)
		switch message.Role {
		case messages.RoleHuman:
			prefix = "Human"
		case messages.RoleAI:
			prefix = "AI"
		case messages.RoleSystem:
			prefix = "System"
		case messages.RoleTool:
			prefix = "Tool"
		}
		lines = append(lines, prefix+": "+message.Content)
	}
	return strings.Join(lines, "\n")
}

func cloneMessages(batch []messages.Message) []messages.Message {
	if len(batch) == 0 {
		return nil
	}
	cloned := make([]messages.Message, len(batch))
	copy(cloned, batch)
	return cloned
}
