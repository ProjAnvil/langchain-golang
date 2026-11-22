package indexing

import (
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"golang.org/x/crypto/blake2b"

	"github.com/projanvil/langchain-golang/core/documents"
)

// namespaceUUID is the 16-byte big-endian representation of Python's
// uuid.UUID(int=1984), the namespace langchain_core.indexing.api uses for
// deterministic document UUIDs. It is built with encoding/binary so the value
// is explicit rather than a magic byte literal: 1984 == 0x07c0 occupies the
// final two bytes (00000000-0000-0000-0000-0000000007c0).
var namespaceUUID = func() [16]byte {
	var ns [16]byte
	binary.BigEndian.PutUint16(ns[14:], 1984)
	return ns
}()

// uuidV5 computes an RFC 4122 version-5 UUID for name under namespaceUUID,
// mirroring Python's uuid.uuid5(NAMESPACE_UUID, name): SHA-1 over the
// namespace bytes followed by the UTF-8 bytes of name, with the version and
// variant bits set on the first 16 bytes of the digest.
func uuidV5(name string) string {
	h := sha1.New()
	_, _ = h.Write(namespaceUUID[:])
	_, _ = h.Write([]byte(name))
	sum := h.Sum(nil)

	var u [16]byte
	copy(u[:], sum[:16])
	u[6] = (u[6] & 0x0f) | 0x50 // version 5
	u[8] = (u[8] & 0x3f) | 0x80 // RFC 4122 variant (10xx)

	return fmt.Sprintf("%x-%x-%x-%x-%x", u[0:4], u[4:6], u[6:8], u[8:10], u[10:16])
}

// sha1HexDigest returns the lowercase SHA-1 hexadecimal digest of text.
func sha1HexDigest(text string) string {
	sum := sha1.Sum([]byte(text))
	return hex.EncodeToString(sum[:])
}

// digestHex returns the lowercase hexadecimal digest of text using algorithm.
// An empty algorithm is treated as SHA-1, matching HashStringToUUID.
func digestHex(text string, algorithm KeyEncoder) (string, error) {
	data := []byte(text)
	switch algorithm {
	case "", KeyEncoderSHA1:
		sum := sha1.Sum(data)
		return hex.EncodeToString(sum[:]), nil
	case KeyEncoderSHA256:
		sum := sha256.Sum256(data)
		return hex.EncodeToString(sum[:]), nil
	case KeyEncoderSHA512:
		sum := sha512.Sum512(data)
		return hex.EncodeToString(sum[:]), nil
	case KeyEncoderBlake2b:
		sum := blake2b.Sum512(data)
		return hex.EncodeToString(sum[:]), nil
	default:
		return "", fmt.Errorf("unsupported hashing algorithm: %q", algorithm)
	}
}

// HashStringToUUID hashes input with SHA-1 and maps the digest to a
// deterministic RFC 4122 version-5 UUID under the langchain indexing
// namespace. It mirrors langchain_core.indexing.api._hash_string_to_uuid and
// always succeeds for any string input.
func HashStringToUUID(input string) string {
	return uuidV5(sha1HexDigest(input))
}

// HashString hashes input with algorithm and maps the digest to a
// deterministic version-5 UUID, mirroring
// langchain_core.indexing.api._hash_string. An empty algorithm defaults to
// SHA-1, so HashString(input, KeyEncoderSHA1) equals HashStringToUUID(input).
// It returns an error for an unsupported algorithm.
func HashString(input string, algorithm KeyEncoder) (string, error) {
	digest, err := digestHex(input, algorithm)
	if err != nil {
		return "", err
	}
	return uuidV5(digest), nil
}

// HashNestedDict deterministically serializes data with sorted keys (the Go
// encoding/json equivalent of json.dumps(sort_keys=True)) and maps the result
// to a version-5 UUID, mirroring langchain_core.indexing.api._hash_nested_dict.
// Because map keys are sorted during marshaling, the result is independent of
// map insertion order, including for nested maps.
func HashNestedDict(data map[string]any, algorithm KeyEncoder) (string, error) {
	serialized, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("failed to hash metadata: %w", err)
	}
	return HashString(string(serialized), algorithm)
}

// SourceIDAssigner returns a function that resolves a document's source id,
// mirroring langchain_core.indexing.api._get_source_id_assigner. spec may be:
//   - nil: the resolver always returns an empty source id (no source assigned);
//   - string: the source id is read from the document metadata under that key;
//   - func(documents.Document) (string, error), KeyEncoderFunc, or
//     func(documents.Document) string: used directly as the resolver.
//
// An empty returned string corresponds to Python's None. Any other spec type
// yields an error rather than panicking.
func SourceIDAssigner(spec any) (func(documents.Document) (string, error), error) {
	switch typed := spec.(type) {
	case nil:
		return func(documents.Document) (string, error) { return "", nil }, nil
	case string:
		return func(doc documents.Document) (string, error) {
			value, ok := doc.Metadata[typed]
			if !ok {
				return "", fmt.Errorf("source id key %q not found in document metadata", typed)
			}
			source, ok := value.(string)
			if !ok {
				return "", fmt.Errorf("source id key %q must be a string, got %T", typed, value)
			}
			return source, nil
		}, nil
	case func(documents.Document) (string, error):
		return typed, nil
	case KeyEncoderFunc:
		return typed, nil
	case func(documents.Document) string:
		return func(doc documents.Document) (string, error) { return typed(doc), nil }, nil
	default:
		return nil, fmt.Errorf("source_id_key should be nil, a string, or a func; got %T", spec)
	}
}

// Batch splits items into consecutive chunks of at most size elements,
// mirroring langchain_core.indexing.api._batch. size must be a positive
// integer; an empty input yields no batches. Each chunk is a copy, so callers
// may mutate the returned slices without affecting items.
func Batch[T any](size int, items []T) ([][]T, error) {
	if size <= 0 {
		return nil, fmt.Errorf("batch size must be a positive integer, got %d", size)
	}
	if len(items) == 0 {
		return nil, nil
	}
	out := make([][]T, 0, (len(items)+size-1)/size)
	for start := 0; start < len(items); start += size {
		end := min(start+size, len(items))
		chunk := make([]T, end-start)
		copy(chunk, items[start:end])
		out = append(out, chunk)
	}
	return out, nil
}
