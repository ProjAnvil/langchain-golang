package utils

import (
	"crypto/rand"
	"fmt"
	"sync"
	"time"
)

// UUIDv7 bit layout (RFC 9562, 16 bytes / 128 bits):
//
//	 0                   1                   2                   3
//	 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//	|                           unix_ts_ms                          |
//	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//	|          unix_ts_ms           |  ver  |       rand_a          |
//	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//	|var|                        rand_b                             |
//	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//	|                            rand_b                             |
//	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//
// The 48-bit Unix-millisecond timestamp, the 4-bit version (7) and the 2-bit
// variant (0b10) are fixed. The remaining 74 bits (rand_a: 12 bits,
// rand_b: 62 bits) come from crypto/rand and form a monotonic counter within
// a single millisecond.

const (
	// uuid7UtilRandAMask masks the 12-bit rand_a counter field.
	uuid7UtilRandAMask = 0x0fff
	// uuid7UtilRandBMask masks the 62-bit rand_b counter field.
	uuid7UtilRandBMask = 1<<62 - 1
)

// uuid7UtilState guards the monotonic counter across concurrent UUID7 calls.
// UTC clock resolution is milliseconds, so a mutex-protected last timestamp
// and counter are kept together: within one millisecond the counter is
// incremented (Method 1 of RFC 9562, Section 6.2), and on a new millisecond it is
// reseeded with fresh crypto/rand bits. Keeping the effective timestamp at
// max(now, lastMs) also keeps ordering monotonic if the clock steps backward.
var uuid7UtilState struct {
	sync.Mutex
	haveLast bool
	lastMs   int64
	randA    uint16
	randB    uint64
}

// UUID7 returns an RFC 9562 version 7 UUID string. The first 48 bits encode
// a Unix timestamp in milliseconds; the version nibble is 7 and the variant
// bits are 0b10. The remaining 74 bits are seeded from crypto/rand and advance
// as a monotonic counter so successive calls within the same millisecond are
// strictly increasing. An error is returned only if the crypto/rand source
// fails. It is the UUIDv7 counterpart of UUID4 and deliberately shares its
// lowercase, hyphenated output format.
func UUID7() (string, error) {
	uuid7UtilState.Lock()
	defer uuid7UtilState.Unlock()

	now := time.Now().UnixMilli()

	if !uuid7UtilState.haveLast || now > uuid7UtilState.lastMs {
		// A new millisecond: reseed the 74-bit counter with fresh randomness.
		randA, randB, err := uuid7UtilRandom74()
		if err != nil {
			return "", fmt.Errorf("utils: uuid7 random source: %w", err)
		}
		uuid7UtilState.randA = randA
		uuid7UtilState.randB = randB
		uuid7UtilState.lastMs = now
		uuid7UtilState.haveLast = true
	} else {
		// Same (or an earlier, clock-stepped-back) millisecond: advance the
		// counter so the produced UUIDs stay strictly ordered.
		randA, randB, ok := uuid7UtilIncrement(uuid7UtilState.randA, uuid7UtilState.randB)
		if ok {
			uuid7UtilState.randA = randA
			uuid7UtilState.randB = randB
		} else {
			// The full 74-bit counter is exhausted within this millisecond.
			// RFC 9562, Section 6.2 recommends advancing the timestamp; reseed with
			// fresh randomness under the advanced timestamp.
			randA, randB, err := uuid7UtilRandom74()
			if err != nil {
				return "", fmt.Errorf("utils: uuid7 random source: %w", err)
			}
			uuid7UtilState.randA = randA
			uuid7UtilState.randB = randB
			uuid7UtilState.lastMs++
		}
	}

	var b [16]byte
	uuid7UtilEncode(
		b[:],
		uuid7UtilState.lastMs,
		uuid7UtilState.randA,
		uuid7UtilState.randB,
	)
	return uuid7UtilFormat(b[:]), nil
}

// uuid7UtilRandom74 draws 74 fresh random bits from crypto/rand and splits
// them into the 12-bit rand_a and 62-bit rand_b counter fields.
func uuid7UtilRandom74() (uint16, uint64, error) {
	var buf [10]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return 0, 0, err
	}
	randA := uint16(buf[0]&0x0f)<<8 | uint16(buf[1])
	randB := uint64(buf[2]&0x3f)<<56 |
		uint64(buf[3])<<48 |
		uint64(buf[4])<<40 |
		uint64(buf[5])<<32 |
		uint64(buf[6])<<24 |
		uint64(buf[7])<<16 |
		uint64(buf[8])<<8 |
		uint64(buf[9])
	return randA, randB, nil
}

// uuid7UtilIncrement advances the combined 74-bit counter by one. It returns
// ok=false when the entire counter has wrapped, in which case the caller
// advances the timestamp and reseeds.
func uuid7UtilIncrement(randA uint16, randB uint64) (uint16, uint64, bool) {
	if randB == uuid7UtilRandBMask {
		if randA == uuid7UtilRandAMask {
			return 0, 0, false
		}
		return randA + 1, 0, true
	}
	return randA, randB + 1, true
}

// uuid7UtilEncode writes a single UUIDv7 into b (len 16) from a Unix
// millisecond timestamp and the two counter fields, applying the version-7
// and RFC 4122 variant bits.
func uuid7UtilEncode(b []byte, ms int64, randA uint16, randB uint64) {
	ts := uint64(ms) & 0x0000ffffffffffff
	b[0] = byte(ts >> 40)
	b[1] = byte(ts >> 32)
	b[2] = byte(ts >> 24)
	b[3] = byte(ts >> 16)
	b[4] = byte(ts >> 8)
	b[5] = byte(ts)

	// Version 7 in the high nibble of byte 6; top 4 bits of rand_a follow.
	b[6] = 0x70 | byte((randA>>8)&0x0f)
	b[7] = byte(randA)

	// Variant 0b10 in the top two bits of byte 8; top 6 bits of rand_b follow.
	b[8] = 0x80 | byte((randB>>56)&0x3f)
	b[9] = byte(randB >> 48)
	b[10] = byte(randB >> 40)
	b[11] = byte(randB >> 32)
	b[12] = byte(randB >> 24)
	b[13] = byte(randB >> 16)
	b[14] = byte(randB >> 8)
	b[15] = byte(randB)
}

// uuid7UtilFormat renders the 16 bytes as the canonical lowercase,
// hyphenated UUID string shared with UUID4.
func uuid7UtilFormat(b []byte) string {
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4],
		b[4:6],
		b[6:8],
		b[8:10],
		b[10:16],
	)
}
