package anthropic

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
)

func okServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`)
	}))
}

func TestInvokeStartCallbackError(t *testing.T) {
	server := okServer(t)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	_, err := model.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(stubHandler{failOn: callbacks.EventChatModelStart, err: errStub})),
	)
	if !errors.Is(err, errStub) {
		t.Fatalf("start callback error should propagate: %v", err)
	}
}

func TestInvokeEndCallbackError(t *testing.T) {
	server := okServer(t)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	_, err := model.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(stubHandler{failOn: callbacks.EventChatModelEnd, err: errStub})),
		runnables.WithMetadata("origin", "test"),
	)
	if !errors.Is(err, errStub) {
		t.Fatalf("end callback error should propagate: %v", err)
	}
}

func TestInvokeErrorEventEmitted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusBadRequest)
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
		modelconfig.WithMaxRetries(0),
	)
	_, err := model.Invoke(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err == nil {
		t.Fatal("invoke should fail")
	}
	errorEvents := filterEvents(recorder.Events(), callbacks.EventChatModelError)
	if len(errorEvents) != 1 || errorEvents[0].Error == "" {
		t.Fatalf("error events: %+v", errorEvents)
	}
}

func TestInvokeProviderErrorResponseBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Some gateways wrap failures in an HTTP 200 Anthropic error object.
		_, _ = fmt.Fprint(w, `{"type":"error","error":{"type":"invalid_request_error","message":"bad model"}}`)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("invoke should fail on error-typed 200 response")
	}
}

func TestInvokeEmptyResponseBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{}`)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("invoke should fail on an empty 200 response")
	}
}

func TestStreamStartCallbackError(t *testing.T) {
	server := okServer(t)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	_, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(stubHandler{failOn: callbacks.EventChatModelStart, err: errStub})),
	)
	if !errors.Is(err, errStub) {
		t.Fatalf("stream start callback error should propagate: %v", err)
	}
}
