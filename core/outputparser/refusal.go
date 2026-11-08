package outputparser

import (
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
)

// RaiseIfRefused returns a *lcerrors.ModelRefusalError when msg carries a
// standard refusal content block, and nil otherwise. It mirrors Python's
// langchain_core.output_parsers._refusal.raise_if_refused (#41582), shared by
// the structured output parsers and the agents' ModelRefusalMiddleware.
func RaiseIfRefused(msg messages.Message) error {
	for _, block := range msg.ContentBlocks {
		if rb, ok := block.(messages.RefusalBlock); ok {
			return lcerrors.ModelRefusalFromBlock(rb, msg)
		}
	}
	return nil
}
