package middleware

import (
	"context"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

// structuredFakeChatModel is a middleware-package test fake that satisfies both
// language.ChatModel (via embedding *language.FakeChatModel) and
// language.StructuredCaller. InvokeStructured records the call and returns a
// preconfigured response (typically an AI message whose Content is JSON), so
// middleware can decode it through its schema-driven path.
//
// Modeled on language.nativeStructuredChatModel (core/language/structured_test.go)
// but lives here because the middleware tests live in package middleware and
// cannot reference the language test package's private fakes.
type structuredFakeChatModel struct {
	*language.FakeChatModel
	response        messages.Message
	structuredCalls int
	capturedInputs  [][]messages.Message
	capturedSchemas []schema.Schema
}

func newStructuredFakeChatModel(response messages.Message) *structuredFakeChatModel {
	return &structuredFakeChatModel{
		FakeChatModel: language.NewFakeChatModel(),
		response:      response,
	}
}

// InvokeStructured implements language.StructuredCaller. language.InvokeStructured
// prefers this native path over the fallback JSON-decode-and-validate path, so
// middleware routing through InvokeStructured will hit this method.
func (m *structuredFakeChatModel) InvokeStructured(
	ctx context.Context,
	input []messages.Message,
	sch schema.Schema,
) (messages.Message, error) {
	m.structuredCalls++
	m.capturedInputs = append(m.capturedInputs, append([]messages.Message(nil), input...))
	m.capturedSchemas = append(m.capturedSchemas, sch)
	return m.response, nil
}
