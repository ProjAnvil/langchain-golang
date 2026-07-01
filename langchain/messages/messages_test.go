package messages

import "testing"

func TestV1MessagesExportCore(t *testing.T) {
	msg := Human("hello")
	if msg.Role != RoleHuman || msg.Content != "hello" {
		t.Fatalf("unexpected message: %#v", msg)
	}
}
