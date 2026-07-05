package chatmodels

import (
	"fmt"
	"strings"
	"sync"

	"github.com/projanvil/langchain-golang/core/language"
)

// ProviderFactory constructs a ChatModel for a registered provider.
//
// Implementations are registered via RegisterProvider (typically from a
// partner package's init()) and invoked by Resolve. The opts map is reserved
// for future expansion; callers that have no opts to pass MUST use nil.
type ProviderFactory func(model string, opts map[string]any) (language.ChatModel, error)

// providerFactories is the global registry of Go-side provider constructors.
// It is intentionally SEPARATE from BuiltinProviders, which only carries
// Python-side package/class metadata for inference/reference: a provider can
// appear in BuiltinProviders without having a registered Go ProviderFactory,
// and only the latter is resolvable by Resolve.
var (
	providerFactoriesMu sync.RWMutex
	providerFactories   = make(map[string]ProviderFactory)
)

// RegisterProvider registers a Go ProviderFactory under the given provider
// name. The name is normalized via NormalizeProvider before insert, so
// "OpenAI", "openai", and "OPENAI" intentionally collide.
//
// RegisterProvider is intended to be called from partner packages' init()
// functions (e.g. partners/openai registering "openai"). It is safe for
// concurrent use. Re-registering an existing name overwrites the previous
// factory.
func RegisterProvider(name string, f ProviderFactory) {
	normalized := NormalizeProvider(name)
	providerFactoriesMu.Lock()
	defer providerFactoriesMu.Unlock()
	providerFactories[normalized] = f
}

// lookupProviderFactory returns the factory registered for the (normalized)
// provider name, or nil if none is registered. Caller MUST hold at least a
// read lock.
func lookupProviderFactory(normalizedName string) ProviderFactory {
	return providerFactories[normalizedName]
}

// Resolve looks up a registered Go ProviderFactory for spec.Provider and
// invokes it with spec.Model. The opts argument passed to the factory is nil
// (ChatModelSpec carries no opts today).
//
// If no factory is registered for the provider, Resolve returns a
// *UnknownProviderError so callers can distinguish "no Go implementation for
// provider X" from a construction error using errors.As.
func Resolve(spec ChatModelSpec) (language.ChatModel, error) {
	normalized := NormalizeProvider(spec.Provider)

	providerFactoriesMu.RLock()
	factory := lookupProviderFactory(normalized)
	providerFactoriesMu.RUnlock()

	if factory == nil {
		return nil, &UnknownProviderError{Provider: spec.Provider}
	}
	return factory(spec.Model, nil)
}

// ParseModelString is a STRICT parser for the 'provider:model' form. It splits
// on the FIRST ':' only and returns a ChatModelSpec with the provider half
// normalized via NormalizeProvider.
//
// ParseModelString is distinct from ParseModel: it does NOT do provider
// inference and does NOT validate the provider against BuiltinProviders or the
// factory registry. A spec returned here may name a provider that has no Go
// factory; Resolve surfaces that as an *UnknownProviderError. This keeps
// parsing and resolution as separate, composable steps.
//
// It returns an error if s has no ':', or if either the provider or model
// half is empty.
func ParseModelString(s string) (ChatModelSpec, error) {
	parts := strings.SplitN(s, ":", 2)
	if len(parts) != 2 {
		return ChatModelSpec{}, fmt.Errorf(
			"chatmodels: model string %q must be in 'provider:model' form", s,
		)
	}
	providerPart := NormalizeProvider(parts[0])
	modelPart := parts[1]
	if providerPart == "" || modelPart == "" {
		return ChatModelSpec{}, fmt.Errorf(
			"chatmodels: model string %q must be in 'provider:model' form", s,
		)
	}
	return ChatModelSpec{Model: modelPart, Provider: providerPart}, nil
}

// UnknownProviderError is returned by Resolve when no Go ProviderFactory is
// registered for the requested provider. Callers can errors.As against
// *UnknownProviderError to distinguish "no Go implementation for
// provider X" from a construction error returned by the factory itself.
type UnknownProviderError struct {
	// Provider is the provider name as it appeared on the ChatModelSpec
	// (i.e. BEFORE normalization), so callers see what they passed in.
	Provider string
}

func (e *UnknownProviderError) Error() string {
	if e == nil {
		return "chatmodels: unknown provider"
	}
	return fmt.Sprintf(
		"chatmodels: no Go ProviderFactory registered for provider %q (normalized %q); "+
			"register one via RegisterProvider",
		e.Provider, NormalizeProvider(e.Provider),
	)
}
