// Package tools defines the Tool abstraction used throughout the agent
// runtime: an invocable callable with a name, a description, and a
// JSON-schema argument contract that a chat model can call.
//
// The package provides three construction entry points:
//
//   - NewSimple builds a single-input tool (one string argument) from a
//     func(context.Context, string) (Result, error).
//   - NewFunc / NewStructuredFunc build a multi-input tool from an explicit
//     schema.Schema and a func(context.Context, map[string]any) (Result, error).
//   - FromFunc reflects an ordinary Go func into a Func tool: its argument
//     (a struct with `json` tags, a map[string]any, or no argument) is
//     reflected into the tool's JSON-schema, so callers get schema-driven
//     dispatch without hand-writing a schema. Rejected callables and
//     unsupported field types are reported via typed sentinels that callers
//     can classify with errors.Is.
//
// The Tool, Simple, and Func types implement the Tool interface and are the
// values passed to langchain/agents.CreateAgent and the model tool-binding
// helpers.
package tools
