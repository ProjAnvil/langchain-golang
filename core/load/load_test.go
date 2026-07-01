package load

import "testing"

type fakeSerializable struct{}

func (fakeSerializable) LCNamespace() []string        { return []string{"langchain", "tests"} }
func (fakeSerializable) LCID() []string               { return []string{"langchain", "tests", "Fake"} }
func (fakeSerializable) LCAttributes() map[string]any { return map[string]any{"name": "ok"} }

func TestDumpAndLoad(t *testing.T) {
	payload, err := Dump(fakeSerializable{})
	if err != nil {
		t.Fatal(err)
	}
	registry := NewRegistry()
	if err := registry.Register(payload.ID, func(attrs map[string]any) (any, error) {
		return attrs["name"], nil
	}); err != nil {
		t.Fatal(err)
	}
	got, err := registry.Load(payload)
	if err != nil {
		t.Fatal(err)
	}
	if got != "ok" {
		t.Fatalf("Load() = %v, want ok", got)
	}
}
