package redis

import (
	"strconv"
	"strings"
)

// Key layout. All keys live under two prefixes mirroring Python's
// CHECKPOINT_PREFIX / CHECKPOINT_WRITE_PREFIX
// (langgraph_redis/langgraph/checkpoint/redis/base.py:36-37), plus one
// ordering index per (thread, namespace):
//
//	checkpoint:{thread}:{ns}:{checkpoint_id}          HASH    checkpoint blob + metadata
//	checkpoint_write:{thread}:{ns}:{checkpoint_id}:{task_id}:{idx}  STRING  one pending write
//	checkpoint_zset:{thread}:{ns}                     ZSET    checkpoint IDs (all score 0) for ordering
//
// Keys are write-only constructs: reads build keys from Config components
// and enumerate writes/threads with SCAN, but no code ever parses a key back
// into components, so the escaping below exists only to keep constructed keys
// unambiguous and SCAN patterns literal.
const (
	checkpointPrefix     = "checkpoint:"
	checkpointWritePrefix = "checkpoint_write:"
	checkpointZSetPrefix = "checkpoint_zset:"
)

// keyEscaper escapes the two characters that would make a component
// ambiguous inside a colon-joined key: ":" (the separator) and "\" (the
// escape character itself). This is the Go counterpart of Python's
// storage-safe id/str helpers (util.py): Python needed its empty-string
// sentinel because RediSearch cannot index empty strings — a constraint this
// port does not have — so only structural escaping is required here.
var keyEscaper = strings.NewReplacer(`\`, `\\`, `:`, `\:`)

// globEscaper escapes Redis glob metacharacters so a key-escaped component
// can be embedded literally in a SCAN MATCH pattern.
var globEscaper = strings.NewReplacer(
	`\`, `\\`,
	`*`, `\*`,
	`?`, `\?`,
	`[`, `\[`,
	`]`, `\]`,
)

// escapeKeyComponent makes one key component safe for colon-joining.
func escapeKeyComponent(s string) string { return keyEscaper.Replace(s) }

// globComponent renders one key component literal inside a SCAN MATCH
// pattern: key-escaped first, then glob-escaped.
func globComponent(s string) string { return globEscaper.Replace(escapeKeyComponent(s)) }

// checkpointKey is the hash key for one checkpoint.
func checkpointKey(threadID, ns, checkpointID string) string {
	return checkpointPrefix + escapeKeyComponent(threadID) + ":" +
		escapeKeyComponent(ns) + ":" + escapeKeyComponent(checkpointID)
}

// writeKey is the string key for one pending write slot (task_id, idx).
func writeKey(threadID, ns, checkpointID, taskID string, idx int) string {
	return checkpointWritePrefix + escapeKeyComponent(threadID) + ":" +
		escapeKeyComponent(ns) + ":" + escapeKeyComponent(checkpointID) + ":" +
		escapeKeyComponent(taskID) + ":" + strconv.Itoa(idx)
}

// zsetKey is the sorted-set key ordering one (thread, namespace)'s
// checkpoint IDs.
func zsetKey(threadID, ns string) string {
	return checkpointZSetPrefix + escapeKeyComponent(threadID) + ":" + escapeKeyComponent(ns)
}

// checkpointScanPattern matches every checkpoint hash of one thread
// (all namespaces).
func checkpointScanPattern(threadID string) string {
	return checkpointPrefix + globComponent(threadID) + `:*`
}

// writeScanPattern matches every pending write of one checkpoint when
// checkpointID is non-empty, or every write of one thread (all namespaces
// and checkpoints) when it is empty.
func writeScanPattern(threadID, ns, checkpointID string) string {
	pattern := checkpointWritePrefix + globComponent(threadID) + ":"
	if checkpointID == "" {
		return pattern + `*`
	}
	return pattern + globComponent(ns) + ":" + globComponent(checkpointID) + `:*`
}

// zsetScanPattern matches every ordering zset of one thread (all
// namespaces).
func zsetScanPattern(threadID string) string {
	return checkpointZSetPrefix + globComponent(threadID) + `:*`
}
