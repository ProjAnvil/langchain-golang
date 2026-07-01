package messages

import core "github.com/projanvil/langchain-golang/core/messages"

type Role = core.Role
type ContentBlock = core.ContentBlock
type ToolCall = core.ToolCall
type UsageMetadata = core.UsageMetadata
type Message = core.Message

const (
	RoleSystem = core.RoleSystem
	RoleHuman  = core.RoleHuman
	RoleAI     = core.RoleAI
	RoleTool   = core.RoleTool
)

var (
	System              = core.System
	Human               = core.Human
	AI                  = core.AI
	Tool                = core.Tool
	MarshalJSONStable   = core.MarshalJSONStable
	UnmarshalJSONStable = core.UnmarshalJSONStable
)
