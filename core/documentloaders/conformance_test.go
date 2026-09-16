// Package documentloaders_test wires the concrete loaders into the
// standardtests conformance suite. It lives in the external test package so
// it can import standardtests, which itself imports documentloaders.
package documentloaders_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/projanvil/langchain-golang/core/documentloaders"
	"github.com/projanvil/langchain-golang/standardtests"
)

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return data
}

func TestHTMLLoaderConformance(t *testing.T) {
	html := readFixture(t, "article.html")
	standardtests.RunDocumentLoaderBasics(t, func(t testing.TB) documentloaders.LazyLoader {
		t.Helper()
		return documentloaders.NewHTMLLoader(bytes.NewReader(html))
	})
}

func TestWebLoaderConformance(t *testing.T) {
	t.Setenv("USER_AGENT", "")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(readFixture(t, "article.html"))
	}))
	t.Cleanup(server.Close)

	base := documentloaders.NewWebLoader(server.Client())
	standardtests.RunDocumentLoaderBasics(t, func(t testing.TB) documentloaders.LazyLoader {
		t.Helper()
		return base.ForURL(server.URL)
	})
}

func TestPDFLoaderConformance(t *testing.T) {
	pdfData := readFixture(t, "article.pdf")
	standardtests.RunDocumentLoaderBasics(t, func(t testing.TB) documentloaders.LazyLoader {
		t.Helper()
		return documentloaders.NewPDFLoader(bytes.NewReader(pdfData))
	})
}
