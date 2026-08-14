package moderation

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestModerateFlagged(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/moderations" {
			t.Errorf("path = %q", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"results":[{"flagged":true,"categories":{"hate":true,"violence":false},"category_scores":{"hate":0.9,"violence":0.1}}]}`))
	}))
	defer server.Close()

	c := NewClient(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("omni-moderation-latest"))
	res, err := c.Moderate(context.Background(), "bad text")
	if err != nil {
		t.Fatalf("Moderate: %v", err)
	}
	if !res.Flagged || res.Categories["hate"] != true {
		t.Fatalf("result = %#v", res)
	}
}

func TestModerateClean(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"flagged":false,"categories":{"hate":false},"category_scores":{"hate":0.01}}]}`))
	}))
	defer server.Close()

	c := NewClient(modelconfig.WithBaseURL(server.URL))
	res, err := c.Moderate(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Moderate: %v", err)
	}
	if res.Flagged {
		t.Fatalf("expected clean result, got flagged")
	}
}

func TestMiddlewareBeforeModelFlagsHuman(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"flagged":true,"categories":{"hate":true},"category_scores":{"hate":0.9}}]}`))
	}))
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	state := map[string]any{"messages": []messages.Message{
		messages.Human("flagged human"),
	}}
	_, err := m.BeforeModel(context.Background(), state)
	var ve *ViolationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected ViolationError, got %v", err)
	}
	if len(ve.Categories) == 0 || ve.Categories[0] != "hate" {
		t.Fatalf("categories = %#v", ve.Categories)
	}
}

func TestMiddlewareBeforeModelCleanPasses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"flagged":false,"categories":{},"category_scores":{}}]}`))
	}))
	defer server.Close()

	m := NewMiddleware(NewClient(modelconfig.WithBaseURL(server.URL)))
	state := map[string]any{"messages": []messages.Message{
		messages.Human("clean"),
	}}
	out, err := m.BeforeModel(context.Background(), state)
	if err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	if out == nil {
		t.Fatal("expected state passed through")
	}
}
