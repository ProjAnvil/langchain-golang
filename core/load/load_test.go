package load

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

type fakeSerializable struct{}

func (fakeSerializable) LCNamespace() []string        { return []string{"langchain", "tests"} }
func (fakeSerializable) LCID() []string               { return []string{"langchain", "tests", "Fake"} }
func (fakeSerializable) LCAttributes() map[string]any { return map[string]any{"name": "ok"} }

type namespaceOnlySerializable struct{}

func (namespaceOnlySerializable) LCNamespace() []string        { return []string{"langchain", "tests", "NsOnly"} }
func (namespaceOnlySerializable) LCID() []string               { return nil }
func (namespaceOnlySerializable) LCAttributes() map[string]any { return nil }

type emptySerializable struct{}

func (emptySerializable) LCNamespace() []string        { return nil }
func (emptySerializable) LCID() []string               { return nil }
func (emptySerializable) LCAttributes() map[string]any { return nil }

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

func TestDump(t *testing.T) {
	t.Run("nil value", func(t *testing.T) {
		if _, err := Dump(nil); err == nil {
			t.Fatal("Dump(nil) expected error")
		}
	})

	t.Run("falls back to namespace", func(t *testing.T) {
		payload, err := Dump(namespaceOnlySerializable{})
		if err != nil {
			t.Fatal(err)
		}
		want := []string{"langchain", "tests", "NsOnly"}
		if strings.Join(payload.ID, "/") != strings.Join(want, "/") {
			t.Fatalf("Dump() ID = %v, want %v", payload.ID, want)
		}
		if payload.LC != 1 || payload.Type != "constructor" {
			t.Fatalf("Dump() = %+v, want lc=1 type=constructor", payload)
		}
	})

	t.Run("missing id", func(t *testing.T) {
		if _, err := Dump(emptySerializable{}); err == nil {
			t.Fatal("Dump() expected error for empty id and namespace")
		}
	})

	t.Run("clones id and attributes", func(t *testing.T) {
		value := fakeSerializable{}
		payload, err := Dump(value)
		if err != nil {
			t.Fatal(err)
		}
		payload.ID[0] = "mutated"
		payload.Attrs["name"] = "mutated"
		if value.LCID()[0] == "mutated" {
			t.Fatal("Dump() did not clone the id slice")
		}
		if value.LCAttributes()["name"] == "mutated" {
			t.Fatal("Dump() did not clone the attributes map")
		}
	})
}

func TestDumps(t *testing.T) {
	t.Run("round trip", func(t *testing.T) {
		data, err := Dumps(fakeSerializable{})
		if err != nil {
			t.Fatal(err)
		}
		var payload Serialized
		if err := json.Unmarshal(data, &payload); err != nil {
			t.Fatal(err)
		}
		if payload.LC != 1 || payload.Type != "constructor" {
			t.Fatalf("Dumps() decoded = %+v, want lc=1 type=constructor", payload)
		}
		if got := payload.Attrs["name"]; got != "ok" {
			t.Fatalf("Dumps() attrs[name] = %v, want ok", got)
		}
	})

	t.Run("propagates dump error", func(t *testing.T) {
		if _, err := Dumps(emptySerializable{}); err == nil {
			t.Fatal("Dumps() expected error")
		}
	})
}

func TestRegister(t *testing.T) {
	loader := func(map[string]any) (any, error) { return nil, nil }

	tests := []struct {
		name    string
		id      []string
		loader  Loader
		wantErr string
	}{
		{name: "empty id", id: nil, loader: loader, wantErr: "id is required"},
		{name: "nil loader", id: []string{"a"}, loader: nil, wantErr: "loader is required"},
		{name: "ok", id: []string{"a", "b"}, loader: loader},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := NewRegistry().Register(tt.id, tt.loader)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("Register() unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Register() error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestLoad(t *testing.T) {
	registry := NewRegistry()
	loadErr := errors.New("boom")
	if err := registry.Register([]string{"langchain", "tests", "Fake"}, func(attrs map[string]any) (any, error) {
		return attrs, nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := registry.Register([]string{"langchain", "tests", "Failing"}, func(map[string]any) (any, error) {
		return nil, loadErr
	}); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		payload Serialized
		wantErr string
	}{
		{
			name:    "unsupported lc version",
			payload: Serialized{LC: 2, Type: "constructor", ID: []string{"langchain", "tests", "Fake"}},
			wantErr: "unsupported serialized payload",
		},
		{
			name:    "unsupported type",
			payload: Serialized{LC: 1, Type: "secret", ID: []string{"langchain", "tests", "Fake"}},
			wantErr: "unsupported serialized payload",
		},
		{
			name:    "unknown id",
			payload: Serialized{LC: 1, Type: "constructor", ID: []string{"langchain", "tests", "Missing"}},
			wantErr: "unknown serialized id: langchain/tests/Missing",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := registry.Load(tt.payload)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Load() error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}

	t.Run("loader error propagates", func(t *testing.T) {
		_, err := registry.Load(Serialized{LC: 1, Type: "constructor", ID: []string{"langchain", "tests", "Failing"}})
		if !errors.Is(err, loadErr) {
			t.Fatalf("Load() error = %v, want %v", err, loadErr)
		}
	})

	t.Run("attrs cloned and nil preserved", func(t *testing.T) {
		payload := Serialized{LC: 1, Type: "constructor", ID: []string{"langchain", "tests", "Fake"}}
		got, err := registry.Load(payload)
		if err != nil {
			t.Fatal(err)
		}
		if got.(map[string]any) != nil {
			t.Fatalf("Load() attrs = %v, want nil", got)
		}

		payload.Attrs = map[string]any{"name": "ok"}
		got, err = registry.Load(payload)
		if err != nil {
			t.Fatal(err)
		}
		got.(map[string]any)["name"] = "mutated"
		if payload.Attrs["name"] != "ok" {
			t.Fatal("Load() did not clone the attributes map")
		}
	})
}
