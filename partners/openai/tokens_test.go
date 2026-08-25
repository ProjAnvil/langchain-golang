package openai

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tiktoken-go/tokenizer"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// requireCodec loads a tiktoken codec or skips the test. The vocabularies are
// embedded in the tiktoken-go library, so a load failure indicates an
// environment problem rather than a behavioral regression; skip instead of
// hard-failing CI.
func requireCodec(t *testing.T, encoding tokenizer.Encoding) tokenizer.Codec {
	t.Helper()
	codec, err := getCodec(encoding)
	if err != nil {
		t.Skipf("tiktoken codec %s unavailable: %v", encoding, err)
	}
	return codec
}

// Known cl100k_base vector verified against Python tiktoken:
// "supercalifragilistic" → [13066 3035 278 333 4193 321 4633].
// o200k_base vector: [17789 5842 366 17764 311 6207].
func TestChatModelGetTokenIDsRealBPE(t *testing.T) {
	requireCodec(t, tokenizer.Cl100kBase)
	model := NewChatModel(modelconfig.WithModel("gpt-4")) // cl100k_base

	want := []int{13066, 3035, 278, 333, 4193, 321, 4633}
	got := model.GetTokenIDs("supercalifragilistic")
	if len(got) != len(want) {
		t.Fatalf("cl100k ids = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("cl100k ids = %v, want %v", got, want)
		}
	}
	if n := model.GetNumTokens("supercalifragilistic"); n != len(want) {
		t.Fatalf("GetNumTokens = %d, want %d", n, len(want))
	}

	// o200k_base: verified vector plus encode/decode round-trip.
	requireCodec(t, tokenizer.O200kBase)
	o200k := NewChatModel(modelconfig.WithModel("gpt-4o"))
	wantO200k := []int{17789, 5842, 366, 17764, 311, 6207}
	gotO200k := o200k.GetTokenIDs("supercalifragilistic")
	if len(gotO200k) != len(wantO200k) {
		t.Fatalf("o200k ids = %v, want %v", gotO200k, wantO200k)
	}
	for i := range wantO200k {
		if gotO200k[i] != wantO200k[i] {
			t.Fatalf("o200k ids = %v, want %v", gotO200k, wantO200k)
		}
	}
	text := "supercalifragilistic 🦜🔗"
	ids := o200k.GetTokenIDs(text)
	if len(ids) == 0 {
		t.Fatal("o200k GetTokenIDs returned no ids")
	}
	codec, _ := getCodec(tokenizer.O200kBase)
	uids := make([]uint, len(ids))
	for i, id := range ids {
		uids[i] = uint(id)
	}
	decoded, err := codec.Decode(uids)
	if err != nil || decoded != text {
		t.Fatalf("o200k round-trip = %q, err=%v; want %q", decoded, err, text)
	}

	// Invariants that held for the approximation still hold.
	if n := model.GetNumTokens(""); n != 0 {
		t.Fatalf("GetNumTokens(empty) = %d, want 0", n)
	}
	if got := language.GetNumTokens(model, "supercalifragilistic"); got != len(want) {
		t.Fatalf("language.GetNumTokens = %d, want %d", got, len(want))
	}
}

// encodingForModel mirrors Python _get_encoding_model: tiktoken's model table
// first, then o200k_base for gpt-4o/gpt-4.1/gpt-5 prefixes, else cl100k_base.
func TestEncodingForModel(t *testing.T) {
	cases := map[string]tokenizer.Encoding{
		"gpt-4o":             tokenizer.O200kBase,
		"gpt-4o-mini":        tokenizer.O200kBase,
		"gpt-4.1":            tokenizer.O200kBase,
		"gpt-4.1-nano":       tokenizer.O200kBase,
		"gpt-5":              tokenizer.O200kBase,
		"gpt-5-2025-08-07":   tokenizer.O200kBase, // only reachable via the prefix fallback
		"gpt-4":              tokenizer.Cl100kBase,
		"gpt-4-0125-preview": tokenizer.Cl100kBase,
		"gpt-3.5-turbo":      tokenizer.Cl100kBase,
		"some-future-model":  tokenizer.Cl100kBase,
	}
	for model, want := range cases {
		if got := encodingForModel(model); got != want {
			t.Errorf("encodingForModel(%q) = %s, want %s", model, got, want)
		}
	}

	// tiktoken_model_name override wins over the configured model, mirroring
	// Python _get_encoding_model.
	m := NewChatModel(
		modelconfig.WithModel("gpt-4"),
		modelconfig.WithTiktokenModelName("gpt-4o"),
	)
	resolved, _, err := m.getEncodingModel()
	if err != nil {
		t.Fatalf("getEncodingModel: %v", err)
	}
	if resolved != "gpt-4o" {
		t.Fatalf("resolved model = %q, want gpt-4o", resolved)
	}
}

// Message overhead rules with real BPE counts, mirroring
// test_base.py::test_get_num_tokens_from_messages: tokens_per_message=3,
// tokens_per_name=1, +3 reply primer (gpt-3.5-turbo-0301 uses 4/-1).
// Expectations are expressed through GetNumTokens so the test pins the
// overhead arithmetic independently of specific BPE counts.
func TestChatModelGetNumTokensFromMessagesOverhead(t *testing.T) {
	requireCodec(t, tokenizer.Cl100kBase)
	model := NewChatModel(modelconfig.WithModel("gpt-4"))

	// Empty message list: only the +3 reply primer.
	got, err := model.GetNumTokensFromMessages(nil)
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if got != 3 {
		t.Fatalf("empty message list = %d, want 3 (primer only)", got)
	}

	human := messages.Human("hi")
	got, err = model.GetNumTokensFromMessages([]messages.Message{human})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	want := 3 + model.GetNumTokens("user") + model.GetNumTokens("hi") + 3
	if got != want {
		t.Fatalf("single human message = %d, want %d", got, want)
	}

	// A name adds count(name) + tokens_per_name(1).
	named := messages.Human("hi")
	named.Name = "bob"
	got, err = model.GetNumTokensFromMessages([]messages.Message{named})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	want += model.GetNumTokens("bob") + 1
	if got != want {
		t.Fatalf("named human message = %d, want %d", got, want)
	}

	// gpt-3.5-turbo-0301: tokens_per_message=4, tokens_per_name=-1. Both
	// models use cl100k_base, so the same message differs by exactly
	// (4-1) - (3+1) = -1.
	legacy := NewChatModel(modelconfig.WithModel("gpt-3.5-turbo-0301"))
	gotLegacy, err := legacy.GetNumTokensFromMessages([]messages.Message{named})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if gotLegacy != got-1 {
		t.Fatalf("gpt-3.5-turbo-0301 named message = %d, want %d", gotLegacy, got-1)
	}

	// Tool message: tool_call_id contributes a flat 3.
	tool := messages.Tool("call_1", "ok")
	got, err = model.GetNumTokensFromMessages([]messages.Message{tool})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	want = 3 + model.GetNumTokens("tool") + model.GetNumTokens("ok") + 3 + 3
	if got != want {
		t.Fatalf("tool message = %d, want %d", got, want)
	}

	// AI message with a tool call: function name + JSON arguments.
	ai := messages.Message{
		Role:      messages.RoleAI,
		ToolCalls: []messages.ToolCall{{ID: "foo", Name: "bar", Args: map[string]any{"arg1": "arg1"}}},
	}
	got, err = model.GetNumTokensFromMessages([]messages.Message{ai})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	want = 3 + model.GetNumTokens("assistant") + model.GetNumTokens("bar") +
		model.GetNumTokens(`{"arg1":"arg1"}`) + 3
	if got != want {
		t.Fatalf("tool-call message = %d, want %d", got, want)
	}

	// System and custom roles exercise the openAIWireRole arms.
	for _, msg := range []messages.Message{
		messages.System("hi"),
		{Role: "function", Content: "hi"},
	} {
		got, err = model.GetNumTokensFromMessages([]messages.Message{msg})
		if err != nil {
			t.Fatalf("GetNumTokensFromMessages: %v", err)
		}
		want = 3 + model.GetNumTokens(openAIWireRole(msg.Role)) + model.GetNumTokens("hi") + 3
		if got != want {
			t.Fatalf("role %q message = %d, want %d", msg.Role, got, want)
		}
	}

	// TextBlock content: exercises the TextBlock arm of the ContentBlocks
	// switch.
	texty := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.TextBlock{Text: "hi"},
	})
	got, err = model.GetNumTokensFromMessages([]messages.Message{texty})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	want = 3 + model.GetNumTokens("user") + model.GetNumTokens("hi") + 3
	if got != want {
		t.Fatalf("text-block message = %d, want %d", got, want)
	}
}

// pngBase64 encodes a width x height PNG and returns it as base64.
func pngBase64(t *testing.T, width, height int) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.RGBA{R: 0x7f, A: 0xff})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png.Encode: %v", err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// Image token counting: detail "low" is a flat 85; otherwise the image is
// sized (base64 or URL) and counted with OpenAI's formula; images that cannot
// be sized are ignored, like Python without PIL/httpx.
func TestChatModelGetNumTokensFromMessagesImages(t *testing.T) {
	requireCodec(t, tokenizer.Cl100kBase)
	model := NewChatModel(modelconfig.WithModel("gpt-4"))
	roleTokens := model.GetNumTokens("user")

	low := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{URL: "https://example.com/x.png", Extras: map[string]any{"detail": "low"}},
	})
	got, err := model.GetNumTokensFromMessages([]messages.Message{low})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if want := 3 + roleTokens + 85 + 3; got != want {
		t.Fatalf("low-detail image message = %d, want %d", got, want)
	}

	// Base64 PNG, 100x200: no resize, ceil(100/512)=ceil(200/512)=1 → 255.
	b64 := pngBase64(t, 100, 200)
	sized := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{Base64: b64},
	})
	got, err = model.GetNumTokensFromMessages([]messages.Message{sized})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if want := 3 + roleTokens + 255 + 3; got != want {
		t.Fatalf("base64 image message = %d, want %d", got, want)
	}

	// Same image as a data: URL in the URL field.
	dataURL := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{URL: "data:image/png;base64," + b64},
	})
	got, err = model.GetNumTokensFromMessages([]messages.Message{dataURL})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if want := 3 + roleTokens + 255 + 3; got != want {
		t.Fatalf("data-URL image message = %d, want %d", got, want)
	}

	// Image fetched over HTTP: 1000x500 → resize smaller side:
	// 1000 > 768 and 500 ≤ 768? Only the >2048 caps apply → no resize
	// (long side 1000 ≤ 2048, and 500 ≤ 768) → ceil(500/512)=1,
	// ceil(1000/512)=2 → 170*2+85 = 425.
	pngBytes, err := base64.StdEncoding.DecodeString(pngBase64(t, 1000, 500))
	if err != nil {
		t.Fatalf("base64 decode: %v", err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(pngBytes)
	}))
	defer server.Close()
	fetched := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{URL: server.URL + "/img.png"},
	})
	got, err = model.GetNumTokensFromMessages([]messages.Message{fetched})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if want := 3 + roleTokens + 425 + 3; got != want {
		t.Fatalf("fetched image message = %d, want %d", got, want)
	}

	// Unsizable images (unreachable URL, bad base64) are ignored.
	unsized := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{URL: "http://127.0.0.1:1/x.png"},
		messages.ImageBlock{Base64: "!!!not-base64!!!"},
	})
	got, err = model.GetNumTokensFromMessages([]messages.Message{unsized})
	if err != nil {
		t.Fatalf("GetNumTokensFromMessages: %v", err)
	}
	if want := 3 + roleTokens + 3; got != want {
		t.Fatalf("unsizable image message = %d, want %d", got, want)
	}
}

// resizeImage/countImageTokens against hand-computed applications of the
// Python formulas (_resize and _count_image_tokens, chat_models/base.py:4012).
func TestImageResizeAndTokenFormula(t *testing.T) {
	resizeCases := []struct {
		w, h         int
		wantW, wantH int
	}{
		{100, 100, 100, 100},    // small: untouched
		{512, 512, 512, 512},    // exactly one tile
		{768, 768, 768, 768},    // exactly at the smaller-side cap
		{1024, 1024, 768, 768},  // square: shrunk to 768x768
		{2048, 2048, 768, 768},  // 2048 cap, then 768 cap
		{4096, 1024, 2048, 512}, // long side capped, short side already ≤768
		{1024, 4096, 512, 2048}, // tall variant
		{3000, 500, 2048, 341},  // integer-division aspect ratio: 500*2048/3000
		{2048, 100, 2048, 100},  // long side exactly 2048: untouched
	}
	for _, tc := range resizeCases {
		gotW, gotH := resizeImage(tc.w, tc.h)
		if gotW != tc.wantW || gotH != tc.wantH {
			t.Errorf("resizeImage(%d, %d) = (%d, %d), want (%d, %d)",
				tc.w, tc.h, gotW, gotH, tc.wantW, tc.wantH)
		}
	}

	tokenCases := []struct {
		w, h int
		want int
	}{
		{100, 100, 255},   // 1x1 tiles: 170*1*1 + 85
		{512, 512, 255},   // 1x1 tiles
		{513, 512, 425},   // ceil(513/512)=2 → 170*2*1 + 85
		{1024, 1024, 765}, // resized to 768x768 → 170*2*2 + 85
		{4096, 1024, 765}, // resized to 2048x512 → ceil = 4x1 → 170*4*1 + 85
		{3000, 500, 765},  // resized to 2048x341 → ceil = 4x1 → 765
	}
	for _, tc := range tokenCases {
		if got := countImageTokens(tc.w, tc.h); got != tc.want {
			t.Errorf("countImageTokens(%d, %d) = %d, want %d", tc.w, tc.h, got, tc.want)
		}
	}
}

// Python raises NotImplementedError for models outside the
// gpt-3.5-turbo/gpt-4/gpt-5 families (chat_models/base.py:2142-2149).
func TestChatModelGetNumTokensFromMessagesUnsupportedModel(t *testing.T) {
	model := NewChatModel(modelconfig.WithModel("o3"))
	_, err := model.GetNumTokensFromMessages([]messages.Message{messages.Human("hi")})
	if err == nil {
		t.Fatal("expected error for o3")
	}
	if !strings.Contains(err.Error(), "not presently implemented") || !strings.Contains(err.Error(), "o3") {
		t.Fatalf("error = %v, want 'not presently implemented' mentioning o3", err)
	}

	// The tiktoken_model_name override also steers the family check, like
	// Python: a gpt-4 model overridden to a non-chat name errors.
	overridden := NewChatModel(
		modelconfig.WithModel("gpt-4"),
		modelconfig.WithTiktokenModelName("text-embedding-ada-002"),
	)
	_, err = overridden.GetNumTokensFromMessages([]messages.Message{messages.Human("hi")})
	if err == nil || !strings.Contains(err.Error(), "text-embedding-ada-002") {
		t.Fatalf("error = %v, want 'not presently implemented' mentioning the override", err)
	}
}
