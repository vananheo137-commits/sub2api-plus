package service

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNormalizeOpenAICompatibilityRequestBody_StrictModeLeavesAliasUntouched(t *testing.T) {
	body := []byte(`{"model":"g5-codex","stream":true,"prompt":"hello"}`)

	normalized, err := NormalizeOpenAICompatibilityRequestBody(body)
	require.NoError(t, err)
	require.JSONEq(t, string(body), string(normalized))
}

func TestNormalizeOpenAICompatibilityRequestBody_StrictModeLeavesGPT54ProUntouched(t *testing.T) {
	body := []byte(`{"model":"gpt-5.4-pro","stream":false,"input":"hello"}`)

	normalized, err := NormalizeOpenAICompatibilityRequestBody(body)
	require.NoError(t, err)
	require.JSONEq(t, string(body), string(normalized))
}

func TestNormalizeOpenAICompatibilityRequestBody_StrictModeLeavesVersionedGPT54Untouched(t *testing.T) {
	body := []byte(`{"model":"gpt-5.4-2026-03-05","stream":false}`)

	normalized, err := NormalizeOpenAICompatibilityRequestBody(body)
	require.NoError(t, err)
	require.JSONEq(t, string(body), string(normalized))
}

func TestNormalizeOpenAICompatibilityRequestBody_LeavesCanonicalModelUntouched(t *testing.T) {
	body := []byte(`{"model":"gpt-5.4","stream":false}`)

	normalized, err := NormalizeOpenAICompatibilityRequestBody(body)
	require.NoError(t, err)
	require.JSONEq(t, string(body), string(normalized))
}
