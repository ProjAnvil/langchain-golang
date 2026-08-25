package openai

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// Mirrors libs/partners/openai/tests/unit_tests/chat_models/
// test_stream_chunk_timeout.py (pass-through, disabled, fires, structured
// attrs, env-var handling, negative rejection) adapted to the Go pull-based
// stream API. Uses the sseServer helper from stream_errors_test.go.

func TestStreamChunkTimeoutErrorAttributes(t *testing.T) {
	// test_stream_chunk_timeout_error_has_structured_attrs
	err := &StreamChunkTimeoutError{TimeoutS: 0.5, Model: "gpt-5.5", ChunksReceived: 3}
	if err.TimeoutS != 0.5 || err.Model != "gpt-5.5" || err.ChunksReceived != 3 {
		t.Fatalf("attrs = %#v", err)
	}
	text := err.Error()
	for _, want := range []string{"gpt-5.5", "chunks_received=3", "stream_chunk_timeout", "LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S"} {
		if !strings.Contains(text, want) {
			t.Fatalf("Error() = %q, want substring %q", text, want)
		}
	}
	// Python subclasses TimeoutError so existing handlers keep catching; the
	// Go analog: errors.Is(err, context.DeadlineExceeded) + Timeout() bool.
	if !errors.Is(err, context.DeadlineExceeded) || !errors.Is(err, os.ErrDeadlineExceeded) {
		t.Fatal("StreamChunkTimeoutError must match deadline-exceeded sentinels")
	}
	var timeout interface{ Timeout() bool }
	if !errors.As(err, &timeout) || !timeout.Timeout() {
		t.Fatal("StreamChunkTimeoutError must satisfy the Timeout() bool convention")
	}
}

func TestStreamChunkTimeoutPassesThrough(t *testing.T) {
	// test_astream_with_chunk_timeout_passes_through: fast source + generous
	// timeout delivers every chunk.
	server := sseServer(t,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"a"}`+"\n\n"+
			`data: {"type":"response.output_text.delta","output_index":0,"delta":"b"}`+"\n\n"+
			`data: {"type":"response.output_text.delta","output_index":0,"delta":"c"}`+"\n\n")
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithStreamChunkTimeout(5 * time.Second)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()
	var got []string
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		got = append(got, chunk.Content)
	}
	if strings.Join(got, "") != "abc" {
		t.Fatalf("chunks = %v, want [a b c]", got)
	}
}

func TestStreamChunkTimeoutDisabledPassesThrough(t *testing.T) {
	// test_astream_with_chunk_timeout_disabled_passes_through: timeout=0
	// disables the bound; a slow source still completes.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_text.delta","output_index":0,"delta":"x"}`+"\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(200 * time.Millisecond)
		_, _ = fmt.Fprint(w, `data: {"type":"response.output_text.delta","output_index":0,"delta":"y"}`+"\n\n")
	}))
	defer server.Close()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithStreamChunkTimeout(0)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()
	var got []string
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		got = append(got, chunk.Content)
	}
	if strings.Join(got, "") != "xy" {
		t.Fatalf("chunks = %v, want [x y]", got)
	}
}

// stallServer writes one SSE text delta then blocks until the client goes away.
func stallServer(t *testing.T, first string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, first)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	}))
	t.Cleanup(server.Close)
	return server
}

func TestStreamChunkTimeoutFires(t *testing.T) {
	// test_astream_with_chunk_timeout_fires + threads_model_name: slow source +
	// tight timeout raises StreamChunkTimeoutError carrying model + count.
	// Also mirrors test_astream_with_chunk_timeout_logs_on_fire: capture the
	// logger up front so the WARNING assertion below can inspect it.
	var buf bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	defer slog.SetDefault(previous)
	server := stallServer(t, sseTextDelta)
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-5.5"),
	).WithStreamChunkTimeout(50 * time.Millisecond)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok || chunk.Content != "hi" {
		t.Fatalf("first Next = (%q, %v, %v), want (\"hi\", true, nil)", chunk.Content, ok, err)
	}
	_, _, err = stream.Next(context.Background())
	var timeoutErr *StreamChunkTimeoutError
	if !errors.As(err, &timeoutErr) {
		t.Fatalf("second Next err = %v, want StreamChunkTimeoutError", err)
	}
	if timeoutErr.Model != "gpt-5.5" || timeoutErr.ChunksReceived != 1 || timeoutErr.TimeoutS != 0.05 {
		t.Fatalf("error attrs = %#v", timeoutErr)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("error must match context.DeadlineExceeded")
	}
	// test_astream_with_chunk_timeout_logs_on_fire: the fire is logged at
	// WARNING with source=stream_chunk_timeout.
	if !strings.Contains(buf.String(), "source=stream_chunk_timeout") {
		t.Fatalf("warning log = %q, want source=stream_chunk_timeout", buf.String())
	}
	// After a timeout the stream is done.
	if _, ok, err := stream.Next(context.Background()); ok || err != nil {
		t.Fatalf("post-timeout Next = (%v, %v), want (false, nil)", ok, err)
	}
}

func TestStreamChunkTimeoutFiresChatCompletions(t *testing.T) {
	// Spec 4.6: the option applies to the Chat Completions stream path too.
	server := stallServer(t, `data: {"choices":[{"delta":{"content":"hi"}}]}`+"\n\n")
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions().WithStreamChunkTimeout(50 * time.Millisecond)
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()
	if _, ok, err := stream.Next(context.Background()); !ok || err != nil {
		t.Fatalf("first Next = (%v, %v), want (true, nil)", ok, err)
	}
	var timeoutErr *StreamChunkTimeoutError
	if _, _, err := stream.Next(context.Background()); !errors.As(err, &timeoutErr) {
		t.Fatalf("second Next err = %v, want StreamChunkTimeoutError", err)
	}
}

func TestStreamChunkTimeoutEnvHandling(t *testing.T) {
	// test_invalid_stream_chunk_timeout_env_degrades_safely /
	// test_stream_chunk_timeout_env_kill_switch_zero /
	// test_negative_stream_chunk_timeout_env_rejected.
	cases := []struct {
		name string
		env  string
		set  bool
		want time.Duration
	}{
		{"unset default", "", false, 120 * time.Second},
		{"garbage degrades to default", "not-a-float", true, 120 * time.Second},
		{"zero kill switch", "0", true, 0},
		{"negative rejected", "-10", true, 120 * time.Second},
		{"valid override", "2.5", true, 2500 * time.Millisecond},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.set {
				t.Setenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", tc.env)
			} else {
				_ = os.Unsetenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S")
			}
			model := NewChatModel(modelconfig.WithModel("gpt-test"))
			if model.streamChunkTimeout != tc.want {
				t.Fatalf("streamChunkTimeout = %v, want %v", model.streamChunkTimeout, tc.want)
			}
		})
	}
}

func TestStreamChunkTimeoutEnvGarbageLogsWarning(t *testing.T) {
	// test_invalid_stream_chunk_timeout_env_emits_warning: the fallback is
	// logged at WARNING so the typo is discoverable.
	var buf bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	defer slog.SetDefault(previous)
	t.Setenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "nonsense")
	_ = NewChatModel(modelconfig.WithModel("gpt-test"))
	if !strings.Contains(buf.String(), "LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S") {
		t.Fatalf("warning log = %q, want env var named", buf.String())
	}
}

func TestWithStreamChunkTimeoutNegativeRejected(t *testing.T) {
	// test_negative_stream_chunk_timeout_kwarg_rejected: a negative kwarg falls
	// back to the (env-driven) default with a WARNING rather than silently
	// disabling the wrapper; 0 is the documented opt-out.
	model := NewChatModel(modelconfig.WithModel("gpt-test")).WithStreamChunkTimeout(-time.Second)
	if model.streamChunkTimeout != 120*time.Second {
		t.Fatalf("streamChunkTimeout = %v, want 120s fallback", model.streamChunkTimeout)
	}
	if zero := NewChatModel(modelconfig.WithModel("gpt-test")).WithStreamChunkTimeout(0); zero.streamChunkTimeout != 0 {
		t.Fatalf("streamChunkTimeout = %v, want 0 (opt-out preserved)", zero.streamChunkTimeout)
	}
}

func TestAzureChatModelStreamChunkTimeoutPassThrough(t *testing.T) {
	// AzureChatModel has no stream_chunk_timeout kwarg of its own (only the
	// env-derived default); the pass-through modifier configures the embedded
	// ChatModel that AzureChatModel.Stream delegates to.
	model := NewAzureChatModel("https://example.openai.azure.com", "dep", "2024-01-01", "key",
		modelconfig.WithModel("gpt-test"),
	).WithStreamChunkTimeout(5 * time.Second)
	if model.chat.streamChunkTimeout != 5*time.Second {
		t.Fatalf("streamChunkTimeout = %v, want 5s", model.chat.streamChunkTimeout)
	}
}
