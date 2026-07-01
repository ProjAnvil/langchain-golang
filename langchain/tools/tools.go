package tools

import core "github.com/projanvil/langchain-golang/core/tools"

type Result = core.Result
type Tool = core.Tool
type Simple = core.Simple
type Func = core.Func
type RetrieverToolOptions = core.RetrieverToolOptions

var (
	NewSimple                    = core.NewSimple
	NewFunc                      = core.NewFunc
	NewStructuredFunc            = core.NewStructuredFunc
	ValidateArgsSchema           = core.ValidateArgsSchema
	RenderTextDescription        = core.RenderTextDescription
	RenderTextDescriptionAndArgs = core.RenderTextDescriptionAndArgs
	ToFunctionSpec               = core.ToFunctionSpec
	ToOpenAIToolSpec             = core.ToOpenAIToolSpec
	CreateRetrieverTool          = core.CreateRetrieverTool
)
