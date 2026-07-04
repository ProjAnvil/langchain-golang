package language

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
)

// personSchema is a small object schema reused across the structured-output
// tests. The "name" field is required; "age" is optional.
func personSchema() schema.Schema {
	return schema.Object(
		map[string]schema.Schema{
			"name": schema.String("full name"),
			"age":  schema.Integer("age in years"),
		},
		"name",
	)
}

func TestInvokeStructuredValidJSON(t *testing.T) {
	model := NewFakeChatModel(WithResponses(
		messages.AI(`{"name":"Ada","age":36}`),
	))

	msg, err := InvokeStructured(
		context.Background(),
		model,
		[]messages.Message{messages.Human("describe Ada")},
		personSchema(),
	)
	if err != nil {
		t.Fatalf("invoke structured: %v", err)
	}

	if want := `{"name":"Ada","age":36}`; msg.Content != want {
		t.Fatalf("content: got %q want %q", msg.Content, want)
	}
	if msg.Role != messages.RoleAI {
		t.Fatalf("role: got %q want %q", msg.Role, messages.RoleAI)
	}
}

func TestInvokeStructuredInvalidJSON(t *testing.T) {
	model := NewFakeChatModel(WithResponses(
		messages.AI("not json at all"),
	))

	_, err := InvokeStructured(
		context.Background(),
		model,
		[]messages.Message{messages.Human("describe Ada")},
		personSchema(),
	)
	if !errors.Is(err, ErrSchemaViolation) {
		t.Fatalf("error: got %v, want ErrSchemaViolation", err)
	}
}

func TestInvokeStructuredMissingRequiredKey(t *testing.T) {
	// "name" is required by personSchema but absent in the response.
	model := NewFakeChatModel(WithResponses(
		messages.AI(`{"age":36}`),
	))

	_, err := InvokeStructured(
		context.Background(),
		model,
		[]messages.Message{messages.Human("describe Ada")},
		personSchema(),
	)
	if !errors.Is(err, ErrSchemaViolation) {
		t.Fatalf("error: got %v, want ErrSchemaViolation", err)
	}
}

func TestInvokeStructuredOptionalMissingKeySucceeds(t *testing.T) {
	// Only "name" is required; omitting "age" must not fail validation.
	model := NewFakeChatModel(WithResponses(
		messages.AI(`{"name":"Ada"}`),
	))

	msg, err := InvokeStructured(
		context.Background(),
		model,
		[]messages.Message{messages.Human("describe Ada")},
		personSchema(),
	)
	if err != nil {
		t.Fatalf("invoke structured: %v", err)
	}
	if want := `{"name":"Ada"}`; msg.Content != want {
		t.Fatalf("content: got %q want %q", msg.Content, want)
	}
}

// nativeStructuredChatModel embeds *FakeChatModel to satisfy ChatModel while
// also implementing StructuredCaller. It records which call path was used so
// the test can assert the native path is preferred.
type nativeStructuredChatModel struct {
	*FakeChatModel
	nativeCalled bool
	invokeCalled bool
}

func (m *nativeStructuredChatModel) InvokeStructured(
	ctx context.Context,
	input []messages.Message,
	sch schema.Schema,
) (messages.Message, error) {
	m.nativeCalled = true
	return messages.AI(`{"name":"native"}`), nil
}

// Override Invoke to detect the fallback path. We deliberately do NOT call the
// embedded FakeChatModel.Invoke so the test fails loudly if the fallback runs.
func (m *nativeStructuredChatModel) Invoke(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (messages.Message, error) {
	m.invokeCalled = true
	return messages.AI(`{"name":"fallback"}`), nil
}

func TestInvokeStructuredPrefersNativePath(t *testing.T) {
	// Embed a FakeChatModel so the value satisfies ChatModel. The wrapper
	// shadows Invoke and adds InvokeStructured, so the native path should win.
	wrapped := &nativeStructuredChatModel{FakeChatModel: NewFakeChatModel()}

	msg, err := InvokeStructured(
		context.Background(),
		wrapped,
		[]messages.Message{messages.Human("describe Ada")},
		personSchema(),
	)
	if err != nil {
		t.Fatalf("invoke structured: %v", err)
	}

	if !wrapped.nativeCalled {
		t.Fatal("native InvokeStructured was not called")
	}
	if wrapped.invokeCalled {
		t.Fatal("fallback Invoke was called; native path must be preferred")
	}
	if want := `{"name":"native"}`; msg.Content != want {
		t.Fatalf("content: got %q want %q", msg.Content, want)
	}
}
