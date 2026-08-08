// Package serde implements the checkpoint.Serializer contract of the Go port
// of Python's langgraph: a JSON encoding with a closed type registry,
// corresponding to Python's `langgraph.checkpoint.serde.jsonplus`.
//
// The JSON serializer (NewJSONSerializer) encodes JSON-native values (nil,
// string, float64, bool, map[string]any, []any) as plain JSON under the tag
// "json". Concrete Go types that plain JSON would degrade — numbers decode
// as float64, typed slices as []any, []byte as a base64 string, structs as
// maps — round-trip losslessly through a closed registry envelope
// {"__type__": "<name>", "data": <payload>} under the tag "json+envelope".
// The registry covers messages.Message, []messages.Message, types.Send,
// types.Interrupt, time.Time, []byte, int64, int, and []string.
//
// Registered values nest: envelopes are produced and recognized recursively
// inside maps, slices, and the any-typed payload fields of types.Send (Arg)
// and types.Interrupt (Value). messages.Message payloads encode as plain
// JSON (the message wire format), so values inside its metadata maps follow
// ordinary JSON semantics.
//
// The registry is closed by design (no import-by-name, unlike Python's
// msgpack serde): encoding an unregistered concrete type is an error, as is
// decoding an envelope with an unknown "__type__" name. Checkpointed channel
// values must therefore be JSON-native or registry members; custom structs
// belong in the registry (extend it) or the saver rejects them.
//
// The "__type__" key is reserved inside encoded maps. A user map containing
// a string "__type__" key is treated as an envelope on decode: with a known
// registry name it silently decodes as that registered type (losing the
// original map), and with an unknown name decode fails. Do not use
// "__type__" as an ordinary map key.
//
// Nil values follow JSON's inherent asymmetry: []byte(nil),
// map[string]any(nil), []string(nil), and []messages.Message(nil) all encode
// as empty/null payloads and decode to their empty (non-nil) forms. Prefer
// empty non-nil values at checkpoint boundaries.
package serde
