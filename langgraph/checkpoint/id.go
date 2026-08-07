package checkpoint

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"time"
)

// NewID returns a monotonic checkpoint ID: fixed-width fields
// (millisecond timestamp, clock sequence, random suffix) make lexicographic
// order match chronological order, so sorting IDs sorts checkpoints in time.
// clockSeq distinguishes IDs minted within the same millisecond; callers
// pass a per-process step counter and the random suffix guards against
// collisions across processes.
//
// This deliberately diverges from Python's uuid6 checkpoint IDs: there is no
// checkpoint-ID interop requirement between the Go and Python runtimes, and
// this format stays dependency-free.
func NewID(clockSeq int) string {
	return fmt.Sprintf("%013d-%06x-%016x", time.Now().UnixMilli(), clockSeq&0xffffff, randUint64())
}

// randUint64 reads 8 bytes from crypto/rand.
func randUint64() uint64 {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand does not fail on supported platforms; fall back to a
		// time-derived value rather than panicking.
		return uint64(time.Now().UnixNano())
	}
	return binary.BigEndian.Uint64(b[:])
}
