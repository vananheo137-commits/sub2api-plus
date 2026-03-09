package service

import (
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var openAICompatibilityModelAliases = map[string]string{
	"g5":             "gpt-5.1",
	"g5mini":         "gpt-5.1",
	"g5nano":         "gpt-5.1",
	"g5codex":        "gpt-5.1-codex",
	"g5codexmini":    "gpt-5.1-codex-mini",
	"g51":            "gpt-5.1",
	"g51codex":       "gpt-5.1-codex",
	"g51codexmax":    "gpt-5.1-codex-max",
	"g51codexmini":   "gpt-5.1-codex-mini",
	"g52":            "gpt-5.2",
	"g52codex":       "gpt-5.2-codex",
	"g53":            "gpt-5.3-codex",
	"g53codex":       "gpt-5.3-codex",
	"g54":            "gpt-5.4",
	"gpt5":           "gpt-5.1",
	"gpt5mini":       "gpt-5.1",
	"gpt5nano":       "gpt-5.1",
	"gpt5codex":      "gpt-5.1-codex",
	"gpt5codexmini":  "gpt-5.1-codex-mini",
	"gpt51":          "gpt-5.1",
	"gpt51codex":     "gpt-5.1-codex",
	"gpt51codexmax":  "gpt-5.1-codex-max",
	"gpt51codexmini": "gpt-5.1-codex-mini",
	"gpt52":          "gpt-5.2",
	"gpt52codex":     "gpt-5.2-codex",
	"gpt53":          "gpt-5.3-codex",
	"gpt53codex":     "gpt-5.3-codex",
	"gpt54":          "gpt-5.4",
	"codexmini":      "gpt-5.1-codex-mini",
}

// NormalizeOpenAICompatibilityRequestBody rewrites known CLI/client-facing
// model aliases into the canonical OpenAI model names already understood by
// the Responses gateway path. Unknown models are left untouched.
func NormalizeOpenAICompatibilityRequestBody(body []byte) ([]byte, error) {
	if len(body) == 0 {
		return body, nil
	}
	if !gjson.ValidBytes(body) {
		return nil, fmt.Errorf("invalid json body")
	}

	modelValue := gjson.GetBytes(body, "model")
	if !modelValue.Exists() || modelValue.Type != gjson.String {
		return body, nil
	}

	model := strings.TrimSpace(modelValue.String())
	normalized := normalizeOpenAICompatibilityRequestModel(model)
	if normalized == "" || normalized == model {
		return body, nil
	}

	next, err := sjson.SetBytes(body, "model", normalized)
	if err != nil {
		return nil, fmt.Errorf("rewrite compatibility model alias: %w", err)
	}
	return next, nil
}

func normalizeOpenAICompatibilityRequestModel(model string) string {
	baseModel := strings.TrimSpace(model)
	if baseModel == "" {
		return ""
	}
	if strings.Contains(baseModel, "/") {
		parts := strings.Split(baseModel, "/")
		baseModel = strings.TrimSpace(parts[len(parts)-1])
	}

	aliasKey := normalizeOpenAICompatibilityAliasKey(baseModel)
	if aliasKey != "" {
		if mapped, ok := openAICompatibilityModelAliases[aliasKey]; ok {
			return mapped
		}
	}

	if mapped := getNormalizedCodexModel(baseModel); mapped != "" {
		return mapped
	}

	if looksLikeOpenAICompatibilityCodexFamily(baseModel) {
		return normalizeCodexModel(baseModel)
	}

	return ""
}

func normalizeOpenAICompatibilityAliasKey(model string) string {
	lowered := strings.ToLower(strings.TrimSpace(model))
	if lowered == "" {
		return ""
	}

	replacer := strings.NewReplacer(
		"-", "",
		"_", "",
		".", "",
		"/", "",
		" ", "",
	)
	return replacer.Replace(lowered)
}

func looksLikeOpenAICompatibilityCodexFamily(model string) bool {
	lowered := strings.ToLower(strings.TrimSpace(model))
	if lowered == "" {
		return false
	}

	return strings.HasPrefix(lowered, "gpt-5") ||
		strings.HasPrefix(lowered, "gpt 5") ||
		strings.HasPrefix(lowered, "g5") ||
		strings.HasPrefix(lowered, "codex")
}
