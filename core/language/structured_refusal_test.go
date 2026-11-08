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

// 上游 base.py 的 checked_parser 语义（#41582）：结构化输出路径遇 refusal
// 抛 ModelRefusalError 而非 schema/parse 错误。Go 落点为 InvokeStructured
// 默认路径（partner StructuredCaller 各自接，见 wave5 spec D3/D4）。

// refusalModel 是一个不实现 StructuredCaller 的最小 ChatModel，恒返回
// refusal 块消息，强制走 InvokeStructured 默认路径。
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
