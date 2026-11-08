package language

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// Upstream base.py checked_parser semantics (#41582): the structured-output
// path raises ModelRefusalError on a refusal instead of a schema/parse
// error. The Go landing point is the InvokeStructured default path (partner
// StructuredCallers handle it individually; see wave5 spec D3/D4).

// refusalModel is a minimal ChatModel that does not implement
// StructuredCaller and always returns a refusal-block message, forcing the
// InvokeStructured default path.
type refusalModel struct{}

func (m *refusalModel) Invoke(_ context.Context, _ []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	out := messages.AI("")
	out.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: "no"}}
	return out, nil
}

func (m *refusalModel) Stream(_ context.Context, _ []messages.Message, _ ...runnables.Option) (runnables.Stream[messages.Message], error) {
	msg, _ := m.Invoke(context.Background(), nil)
	return runnables.NewSliceStream([]messages.Message{msg}), nil
}

func (m *refusalModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i := range inputs {
		msg, err := m.Invoke(ctx, inputs[i], opts...)
		if err != nil {
			return nil, err
		}
		out[i] = msg
	}
	return out, nil
}

func (m *refusalModel) InputSchema() schema.Schema  { return schema.Schema{"type": "array"} }
func (m *refusalModel) OutputSchema() schema.Schema { return schema.Schema{"type": "object"} }
func (m *refusalModel) ConfigSchema() schema.Schema { return schema.Schema{"type": "object"} }
func (m *refusalModel) BindTools(_ []tools.Tool) (ChatModel, error) {
	return m, nil
}
func (m *refusalModel) Capabilities() ChatModelCapabilities { return ChatModelCapabilities{} }

func TestInvokeStructuredRefused(t *testing.T) {
	_, err := InvokeStructured(context.Background(), &refusalModel{},
		[]messages.Message{messages.Human("hi")},
		schema.Schema{"type": "object"})
	var mre *lcerrors.ModelRefusalError
	if !errors.As(err, &mre) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}
}
