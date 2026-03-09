package service

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestNormalizeOpenAICompatibilityRequestBody_NormalizesAlias(t *testing.T) {
	body := []byte(`{"model":"g5-codex","stream":true,"prompt":"hello"}`)

	normalized, err := NormalizeOpenAICompatibilityRequestBody(body)
	require.NoError(t, err)
	require.Equal(t, "gpt-5.1-codex", gjson.GetBytes(normalized, "model").String())
	require.Equal(t, "hello", gjson.GetBytes(normalized, "prompt").String())
}

func TestNormalizeOpenAICompatibilityRequestBody_LeavesUnknownModelUntouched(t *testing.T) {
	body := []byte(`{"model":"custom-non-openai-model","stream":true}`)

	normalized, err := NormalizeOpenAICompatibilityRequestBody(body)
	require.NoError(t, err)
	require.JSONEq(t, string(body), string(normalized))
}
