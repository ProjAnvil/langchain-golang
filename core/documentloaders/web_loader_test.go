package documentloaders

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newArticleServer(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(readTestFixture(t, "article.html"))
	}))
	t.Cleanup(server.Close)
	return server
}

func TestWebLoaderLoadsPage(t *testing.T) {
	t.Setenv("USER_AGENT", "")
	server := newArticleServer(t)

	loader := NewWebLoader(server.Client())
	docs, err := loader.Load(t.Context(), server.URL)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[0].PageContent != goldenArticleText {
		t.Fatalf("content mismatch:\nwant: %q\ngot:  %q", goldenArticleText, docs[0].PageContent)
	}
	meta := docs[0].Metadata
	if meta["source"] != server.URL {
		t.Fatalf("source: %#v", meta)
	}
	if meta["title"] != "Sample Article" {
		t.Fatalf("title: %#v", meta)
	}
	if meta["description"] != "A sample article used by loader tests." {
		t.Fatalf("description: %#v", meta)
	}
	if meta["language"] != "en" {
		t.Fatalf("language: %#v", meta)
	}
}

func TestWebLoaderSendsDefaultHeaders(t *testing.T) {
	t.Setenv("USER_AGENT", "")
	var captured *http.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured = r.Clone(r.Context())
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte("<html><body><p>ok</p></body></html>"))
	}))
	t.Cleanup(server.Close)

	if _, err := NewWebLoader(server.Client()).Load(t.Context(), server.URL); err != nil {
		t.Fatalf("load: %v", err)
	}
	if captured == nil {
		t.Fatal("request was not captured")
	}
	if got := captured.Header.Get("User-Agent"); got != "DefaultLangchainUserAgent" {
		t.Fatalf("user agent: %q", got)
	}
	if got := captured.Header.Get("Accept"); !strings.HasPrefix(got, "text/html") {
		t.Fatalf("accept: %q", got)
	}
	if got := captured.Header.Get("Referer"); got != "https://www.google.com/" {
		t.Fatalf("referer: %q", got)
	}
}

func TestWebLoaderUserAgentEnvOverride(t *testing.T) {
	t.Setenv("USER_AGENT", "unit-agent/1.0")
	var captured *http.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured = r.Clone(r.Context())
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte("<html><body><p>ok</p></body></html>"))
	}))
	t.Cleanup(server.Close)

	if _, err := NewWebLoader(server.Client()).Load(t.Context(), server.URL); err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := captured.Header.Get("User-Agent"); got != "unit-agent/1.0" {
		t.Fatalf("user agent: %q", got)
	}
}

func TestWebLoaderNon2xxError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "missing", http.StatusNotFound)
	}))
	t.Cleanup(server.Close)

	loader := NewWebLoader(server.Client())
	_, err := loader.Load(t.Context(), server.URL)
	if err == nil {
		t.Fatal("expected status error")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Fatalf("error should mention status: %v", err)
	}
}

func TestWebLoaderBodySizeLimit(t *testing.T) {
	body := "<html><body><p>" + strings.Repeat("x", 32) + "</p></body></html>"

	t.Run("oversized body errors", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(body))
		}))
		t.Cleanup(server.Close)

		loader := NewWebLoader(server.Client(), WithMaxBodyBytes(16))
		if _, err := loader.Load(t.Context(), server.URL); err == nil {
			t.Fatal("expected body size error")
		}
	})

	t.Run("body at limit loads", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(body))
		}))
		t.Cleanup(server.Close)

		loader := NewWebLoader(server.Client(), WithMaxBodyBytes(int64(len(body))))
		docs, err := loader.Load(t.Context(), server.URL)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if len(docs) != 1 || !strings.Contains(docs[0].PageContent, "xxxx") {
			t.Fatalf("docs: %#v", docs)
		}
	})
}

func TestWebLoaderDefaultBodyLimit(t *testing.T) {
	if DefaultWebMaxBodyBytes != 10<<20 {
		t.Fatalf("default limit: %d", DefaultWebMaxBodyBytes)
	}
	loader := NewWebLoader(nil)
	if loader.maxBodyBytes != DefaultWebMaxBodyBytes {
		t.Fatalf("configured limit: %d", loader.maxBodyBytes)
	}
	if loader.client == nil {
		t.Fatal("expected default client")
	}
}

func TestWebLoaderLazyLoadRequiresURL(t *testing.T) {
	loader := NewWebLoader(nil)
	if _, err := loader.LazyLoad(t.Context()); err == nil {
		t.Fatal("expected missing URL error")
	}
}

func TestWebLoaderLazyLoadAndSplit(t *testing.T) {
	server := newArticleServer(t)

	loader := NewWebLoader(server.Client()).ForURL(server.URL)
	docs, err := LoadAndSplit(t.Context(), loader, fakeSplitter{})
	if err != nil {
		t.Fatalf("load and split: %v", err)
	}
	if len(docs) != 2 || docs[0].PageContent != "a" || docs[1].PageContent != "b" {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[0].Metadata["source"] != server.URL {
		t.Fatalf("metadata: %#v", docs[0].Metadata)
	}
}
