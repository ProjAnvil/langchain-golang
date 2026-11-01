package modelprofiles

import (
	"slices"
	"testing"
)

// Expected mid_conversation coverage, mirrored from the upstream partner
// augmentation data that declares it (langchain-anthropic 1.7.6 and
// langchain-openai 1.7.1, Python #41180). The flags come solely from
// profile_augmentations.toml — models.dev does not carry them — so the carrier
// sets below must match the augmentation files exactly.
var (
	wantAnthropicSystemMessages = []string{
		"claude-fable-5",
		"claude-fable-5-1",
		"claude-haiku-5-5",
		"claude-mythos-5",
		"claude-mythos-5-1",
		"claude-opus-4-8",
		"claude-opus-5",
		"claude-opus-5-5",
		"claude-sonnet-5",
		"claude-sonnet-5-5",
	}
	wantAnthropicToolsTrue = []string{
		"claude-fable-5",
		"claude-fable-5-1",
		"claude-haiku-5-5",
		"claude-mythos-5",
		"claude-mythos-5-1",
		"claude-opus-4-8",
		"claude-opus-5",
		"claude-opus-5-5",
		"claude-sonnet-5-5",
	}
	wantOpenAIMidConversation = []string{
		"gpt-4.1",
		"gpt-4.1-mini",
		"gpt-4.1-nano",
		"gpt-5",
		"gpt-5-chat-latest",
		"gpt-5-codex",
		"gpt-5-mini",
		"gpt-5-nano",
		"gpt-5-pro",
		"gpt-5.1",
		"gpt-5.1-chat-latest",
		"gpt-5.1-codex",
		"gpt-5.1-codex-max",
		"gpt-5.1-codex-mini",
		"gpt-5.2",
		"gpt-5.2-chat-latest",
		"gpt-5.2-codex",
		"gpt-5.2-pro",
		"gpt-5.3-chat-latest",
		"gpt-5.3-codex",
		"gpt-5.3-codex-spark",
		"gpt-5.4",
		"gpt-5.4-mini",
		"gpt-5.4-nano",
		"gpt-5.4-pro",
		"gpt-5.5",
		"gpt-5.5-pro",
		"gpt-5.6",
		"gpt-5.6-luna",
		"gpt-5.6-sol",
		"gpt-5.6-terra",
		"gpt-6-astra",
		"gpt-6-luna",
		"gpt-6-sol",
		"gpt-6.1-sol",
		"o3",
		"o3-mini",
		"o3-pro",
	}
)

// carriers returns the sorted model IDs in provider's registry whose profile
// carries key explicitly set to the given boolean value. Absent keys are not
// carriers, so value=false means "present and false".
func carriers(t *testing.T, provider, key string, value bool) []string {
	t.Helper()
	reg, ok := materialized()[provider]
	if !ok {
		t.Fatalf("provider %q missing from materialized profiles", provider)
	}
	var ids []string
	for modelID, profile := range reg {
		if raw, present := profile[key]; present {
			if b, isBool := raw.(bool); isBool && b == value {
				ids = append(ids, modelID)
			}
		}
	}
	slices.Sort(ids)
	return ids
}

// carries reports whether the profile of provider/model carries key at all.
func carries(t *testing.T, provider, model, key string) bool {
	t.Helper()
	profile, ok := Lookup(provider, model)
	if !ok {
		t.Fatalf("Lookup(%q, %q) missed", provider, model)
	}
	_, present := profile[key]
	return present
}

func TestLookupAnthropicMidConversationFlags(t *testing.T) {
	profile, ok := Lookup("anthropic", "claude-opus-4-8")
	if !ok {
		t.Fatal("Lookup(claude-opus-4-8) missed, want hit")
	}
	if !profile.Supports(FieldMidConversationSystemMessages) {
		t.Errorf("claude-opus-4-8: mid_conversation_system_messages = %v, want true",
			profile[FieldMidConversationSystemMessages])
	}
	if !profile.Supports(FieldMidConversationTools) {
		t.Errorf("claude-opus-4-8: mid_conversation_tools = %v, want true",
			profile[FieldMidConversationTools])
	}
}

func TestLookupOpenAIMidConversationFlags(t *testing.T) {
	profile, ok := Lookup("openai", "gpt-5")
	if !ok {
		t.Fatal("Lookup(gpt-5) missed, want hit")
	}
	if !profile.Supports(FieldMidConversationSystemMessages) {
		t.Errorf("gpt-5: mid_conversation_system_messages = %v, want true",
			profile[FieldMidConversationSystemMessages])
	}
	if !profile.Supports(FieldMidConversationTools) {
		t.Errorf("gpt-5: mid_conversation_tools = %v, want true",
			profile[FieldMidConversationTools])
	}
}

func TestLookupMidConversationToolsFalseForClaudeSonnet5(t *testing.T) {
	// Upstream declares sonnet-5 as system-messages-capable but NOT
	// mid-conversation-tools-capable: the key must be present and false, not
	// absent.
	profile, ok := Lookup("anthropic", "claude-sonnet-5")
	if !ok {
		t.Fatal("Lookup(claude-sonnet-5) missed, want hit")
	}
	if !profile.Supports(FieldMidConversationSystemMessages) {
		t.Errorf("claude-sonnet-5: mid_conversation_system_messages = %v, want true",
			profile[FieldMidConversationSystemMessages])
	}
	if got, present := profile[FieldMidConversationTools]; !present {
		t.Error("claude-sonnet-5: mid_conversation_tools absent, want present-and-false")
	} else if got != false {
		t.Errorf("claude-sonnet-5: mid_conversation_tools = %v, want false", got)
	}
	if profile.Supports(FieldMidConversationTools) {
		t.Error("claude-sonnet-5: Supports(mid_conversation_tools) = true, want false")
	}
}

func TestLookupOlderModelsDoNotCarryMidConversationKeys(t *testing.T) {
	for _, tc := range []struct{ provider, model string }{
		{"anthropic", "claude-sonnet-4-5"},
		{"anthropic", "claude-opus-4-5"},
		{"openai", "gpt-3.5-turbo"},
	} {
		if _, ok := Lookup(tc.provider, tc.model); !ok {
			t.Errorf("Lookup(%q, %q) missed, want hit", tc.provider, tc.model)
			continue
		}
		for _, key := range []string{FieldMidConversationSystemMessages, FieldMidConversationTools} {
			if carries(t, tc.provider, tc.model, key) {
				t.Errorf("%s/%s: %s present, want absent (upstream does not declare it)",
					tc.provider, tc.model, key)
			}
		}
	}
}

func TestLookupAnthropicCoverageMirrorsUpstream(t *testing.T) {
	if got := carriers(t, "anthropic", FieldMidConversationSystemMessages, true); !slices.Equal(got, wantAnthropicSystemMessages) {
		t.Errorf("anthropic mid_conversation_system_messages=true carriers = %v, want %v",
			got, wantAnthropicSystemMessages)
	}
	if got := carriers(t, "anthropic", FieldMidConversationTools, true); !slices.Equal(got, wantAnthropicToolsTrue) {
		t.Errorf("anthropic mid_conversation_tools=true carriers = %v, want %v",
			got, wantAnthropicToolsTrue)
	}
	if got := carriers(t, "anthropic", FieldMidConversationTools, false); !slices.Equal(got, []string{"claude-sonnet-5"}) {
		t.Errorf("anthropic mid_conversation_tools=false carriers = %v, want [claude-sonnet-5]", got)
	}
}

func TestLookupOpenAICoverageMirrorsUpstream(t *testing.T) {
	if got := carriers(t, "openai", FieldMidConversationSystemMessages, true); !slices.Equal(got, wantOpenAIMidConversation) {
		t.Errorf("openai mid_conversation_system_messages=true carriers = %v, want %v",
			got, wantOpenAIMidConversation)
	}
	if got := carriers(t, "openai", FieldMidConversationTools, true); !slices.Equal(got, wantOpenAIMidConversation) {
		t.Errorf("openai mid_conversation_tools=true carriers = %v, want %v",
			got, wantOpenAIMidConversation)
	}
}

func TestLookupMidConversationToolsImpliesSystemMessages(t *testing.T) {
	// Python documents mid_conversation_tools as implying
	// mid_conversation_system_messages; enforce it across every provider.
	for provider, reg := range materialized() {
		for modelID, profile := range reg {
			if profile.Supports(FieldMidConversationTools) && !profile.Supports(FieldMidConversationSystemMessages) {
				t.Errorf("%s/%s: mid_conversation_tools=true but mid_conversation_system_messages is not true",
					provider, modelID)
			}
		}
	}
}

func TestLookupMiss(t *testing.T) {
	if profile, ok := Lookup("anthropic", "claude-does-not-exist"); ok {
		t.Errorf("Lookup(unknown model) = (%v, true), want miss", profile)
	}
	if profile, ok := Lookup("not-a-provider", "gpt-5"); ok {
		t.Errorf("Lookup(unknown provider) = (%v, true), want miss", profile)
	}
	if profile, ok := Lookup("", ""); ok {
		t.Errorf("Lookup(empty args) = (%v, true), want miss", profile)
	}
}
