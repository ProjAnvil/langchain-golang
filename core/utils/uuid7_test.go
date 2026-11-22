package utils_test

import (
	"encoding/hex"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/utils"
)

// uuid7UtilParse validates the canonical UUID string shape and decodes its
// 16 raw bytes.
func uuid7UtilParse(t *testing.T, id string) []byte {
	t.Helper()
	if len(id) != 36 {
		t.Fatalf("uuid %q has length %d, want 36", id, len(id))
	}
	for _, idx := range []int{8, 13, 18, 23} {
		if id[idx] != '-' {
			t.Fatalf("uuid %q is missing a hyphen at position %d", id, idx)
		}
	}
	compact := strings.ReplaceAll(id, "-", "")
	if compact != strings.ToLower(compact) {
		t.Fatalf("uuid %q is not lowercase hex", id)
	}
	raw, err := hex.DecodeString(compact)
	if err != nil {
		t.Fatalf("uuid %q is not valid hex: %v", id, err)
	}
	if len(raw) != 16 {
		t.Fatalf("uuid %q decoded to %d bytes, want 16", id, len(raw))
	}
	return raw
}

// uuid7UtilTimestamp reconstructs the 48-bit Unix millisecond timestamp
// stored in the most-significant six bytes of a UUIDv7.
func uuid7UtilTimestamp(raw []byte) int64 {
	return int64(uint64(raw[0])<<40 |
		uint64(raw[1])<<32 |
		uint64(raw[2])<<24 |
		uint64(raw[3])<<16 |
		uint64(raw[4])<<8 |
		uint64(raw[5]))
}

func TestUUID7FormatVersionVariant(t *testing.T) {
	seen := make(map[string]struct{})
	for i := 0; i < 256; i++ {
		id, err := utils.UUID7()
		if err != nil {
			t.Fatalf("uuid7 call %d: %v", i, err)
		}
		if _, dup := seen[id]; dup {
			t.Fatalf("duplicate uuid7 %q", id)
		}
		seen[id] = struct{}{}

		raw := uuid7UtilParse(t, id)
		if version := raw[6] >> 4; version != 7 {
			t.Fatalf("uuid7 %q has version nibble %d, want 7", id, version)
		}
		if variant := raw[8] >> 6; variant != 2 {
			t.Fatalf("uuid7 %q has variant bits %b, want 10", id, variant)
		}
	}
}

func TestUUID7MonotonicWithinMillisecond(t *testing.T) {
	const count = 5000
	previous := ""
	previousTimestamp := int64(0)
	sameMillisecondPairs := 0

	for i := 0; i < count; i++ {
		id, err := utils.UUID7()
		if err != nil {
			t.Fatalf("uuid7 call %d: %v", i, err)
		}
		raw := uuid7UtilParse(t, id)
		compact := strings.ReplaceAll(id, "-", "")
		timestamp := uuid7UtilTimestamp(raw)

		// Fixed-length lowercase hex strings sort lexicographically exactly
		// as their 128-bit big-endian numeric values.
		if previous != "" && compact <= previous {
			t.Fatalf("uuid7 is not strictly increasing: %q then %q", previous, compact)
		}
		if previous != "" && timestamp == previousTimestamp {
			sameMillisecondPairs++
			if compact <= previous {
				t.Fatalf("uuid7 not monotonic within one millisecond: %q then %q", previous, compact)
			}
		}
		previous = compact
		previousTimestamp = timestamp
	}

	if sameMillisecondPairs == 0 {
		t.Fatal("expected multiple UUIDs generated within the same millisecond")
	}
}

func TestUUID7TimestampExtraction(t *testing.T) {
	before := time.Now().UnixMilli()
	ids := make([]string, 100)
	for i := range ids {
		id, err := utils.UUID7()
		if err != nil {
			t.Fatalf("uuid7 call %d: %v", i, err)
		}
		ids[i] = id
	}
	after := time.Now().UnixMilli()

	// Generous tolerance absorbs scheduler pauses and minor clock adjustments
	// while still proving the field decodes to a real current-time millisecond
	// value rather than random bits.
	const tolerance = int64(1000)
	for _, id := range ids {
		raw := uuid7UtilParse(t, id)
		got := uuid7UtilTimestamp(raw)
		if got < before-tolerance || got > after+tolerance {
			t.Fatalf("uuid7 %q timestamp %d outside [%d, %d]", id, got, before-tolerance, after+tolerance)
		}
		if got < 1_600_000_000_000 {
			t.Fatalf("uuid7 %q timestamp %d is not a plausible Unix millisecond epoch", id, got)
		}
	}
}

func TestUUID7DoesNotCollideWithUUID4(t *testing.T) {
	id4, err := utils.UUID4()
	if err != nil {
		t.Fatalf("uuid4: %v", err)
	}
	id7, err := utils.UUID7()
	if err != nil {
		t.Fatalf("uuid7: %v", err)
	}
	if id4 == id7 {
		t.Fatalf("uuid4 and uuid7 unexpectedly produced the same string %q", id4)
	}
	if version := uuid7UtilParse(t, id4)[6] >> 4; version != 4 {
		t.Fatalf("uuid4 %q has version nibble %d, want 4", id4, version)
	}
	if version := uuid7UtilParse(t, id7)[6] >> 4; version != 7 {
		t.Fatalf("uuid7 %q has version nibble %d, want 7", id7, version)
	}

	// EnsureID still relies on UUID4 and must remain functional.
	ensured, err := utils.EnsureID("")
	if err != nil {
		t.Fatalf("ensure id: %v", err)
	}
	if !strings.HasPrefix(ensured, "lc_") {
		t.Fatalf("ensure id %q, want the lc_ prefix", ensured)
	}
}
