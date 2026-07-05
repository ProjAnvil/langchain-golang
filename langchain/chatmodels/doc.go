// Package chatmodels provides provider metadata and a Go provider registry
// for resolving a chat model from a "provider:model" identifier.
//
// Two distinct concerns live in this package:
//
//   - Python provider METADATA, used for inference: BuiltinProviders,
//     BuiltinProviderNames, ProviderInfoFor, SupportsProvider, NormalizeProvider,
//     and InferModelProvider describe which Python packages and classes own
//     which provider names (e.g. "anthropic" -> langchain_anthropic.ChatAnthropic).
//     ParseModel uses this metadata to INFER a provider from a bare model name.
//
//   - A Go provider REGISTRY, used for construction: ProviderFactory is a
//     constructor func(model, opts); RegisterProvider registers one under a
//     provider name (typically from a partner package's init); Resolve
//     constructs the ChatModel for a ChatModelSpec by dispatching to the
//     registered factory, returning a typed *UnknownProviderError when no
//     factory is registered; ParseModelString is a STRICT "provider:model"
//     parser (no inference, no registry check) that pairs with Resolve.
//     This registry is independent of BuiltinProviders: a provider may have
//     Python metadata and no Go factory (adapter-only) or vice versa.
//
// partners/openai self-registers its factory under the provider name "openai"
// via an init() (see partners/openai/init.go), so
// Resolve(ChatModelSpec{Provider: "openai", Model: ...}) produces an
// openai.ChatModel once that package is imported.
package chatmodels
