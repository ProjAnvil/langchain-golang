package chatmodels

import (
	"fmt"

	"github.com/projanvil/langchain-golang/core/language"
)

// ExampleResolve registers a fake Go factory under the provider name
// "example-fake", parses a "provider:model" string, and resolves it to a
// ChatModel. In real code a partner package's init() performs the
// registration (e.g. partners/openai registers "openai"); this example
// registers inline so it is self-contained and offline.
//
// RegisterProvider normalizes the name (hyphens become underscores), and
// ParseModelString applies the same normalization, so the two sides match
// regardless of which form the caller uses.
func ExampleResolve() {
	RegisterProvider("example-fake", func(model string, opts map[string]any) (language.ChatModel, error) {
		return language.NewFakeChatModel(), nil
	})

	spec, err := ParseModelString("example-fake:gpt-x")
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	m, err := Resolve(spec)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("%T\n", m)
	// Output:
	// *language.FakeChatModel
}
