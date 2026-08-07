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
// belong in the registry (extend it) or the saver rejects them. The
// "__type__" key is reserved inside encoded maps.
package serde
