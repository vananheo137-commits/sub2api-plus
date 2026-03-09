package apicompat

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOpenAICompletionsToResponses_Basic(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.1",
		"stream":true,
		"max_tokens":256,
		"prompt":"hello from completions"
	}`)

	converted, err := OpenAICompletionsToResponses(body)
	require.NoError(t, err)

	var parsed map[string]any
	require.NoError(t, json.Unmarshal(converted, &parsed))
	require.Equal(t, "gpt-5.1", parsed["model"])
	require.Equal(t, true, parsed["stream"])
	require.Equal(t, false, parsed["store"])
	require.EqualValues(t, 256, parsed["max_output_tokens"])
	require.Equal(t, "hello from completions", parsed["input"])
}

func TestResponsesToOpenAICompletion_Basic(t *testing.T) {
	body := []byte(`{
		"id":"resp_456",
		"created_at":1700000000,
		"model":"gpt-5.1",
		"status":"completed",
		"output":[
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"completion text"}]}
		],
		"usage":{"input_tokens":9,"output_tokens":4,"total_tokens":13}
	}`)

	converted, err := ResponsesToOpenAICompletion(body)
	require.NoError(t, err)

	var parsed map[string]any
	require.NoError(t, json.Unmarshal(converted, &parsed))
	require.Equal(t, "text_completion", parsed["object"])
	require.Equal(t, "resp_456", parsed["id"])
	require.Equal(t, "gpt-5.1", parsed["model"])

	choices, ok := parsed["choices"].([]any)
	require.True(t, ok)
	require.Len(t, choices, 1)

	choice, ok := choices[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "completion text", choice["text"])
	require.Equal(t, "stop", choice["finish_reason"])

	usage, ok := parsed["usage"].(map[string]any)
	require.True(t, ok)
	require.EqualValues(t, 9, usage["prompt_tokens"])
	require.EqualValues(t, 4, usage["completion_tokens"])
	require.EqualValues(t, 13, usage["total_tokens"])
}

func TestResponsesEventToOpenAICompletions_DoneFallback(t *testing.T) {
	state := NewOpenAICompletionsStreamState("gpt-5.1")

	payloads, done, err := ResponsesEventToOpenAICompletions([]byte(`{
		"type":"response.output_text.done",
		"output_index":0,
		"content_index":0,
		"text":"hello from done"
	}`), state)
	require.NoError(t, err)
	require.False(t, done)
	require.Len(t, payloads, 1)
	require.Contains(t, string(payloads[0]), `"text":"hello from done"`)
}
