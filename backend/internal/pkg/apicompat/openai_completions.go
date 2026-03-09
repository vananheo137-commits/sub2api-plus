package apicompat

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// OpenAICompletionsStreamState tracks state while converting Responses SSE
// events into legacy Completions chunks.
type OpenAICompletionsStreamState struct {
	ResponseID          string
	CreatedAt           int64
	Model               string
	OutputTextDeltaSeen map[string]bool
}

// NewOpenAICompletionsStreamState returns an initialized stream state.
func NewOpenAICompletionsStreamState(model string) *OpenAICompletionsStreamState {
	return &OpenAICompletionsStreamState{
		Model:               strings.TrimSpace(model),
		OutputTextDeltaSeen: make(map[string]bool),
	}
}

// OpenAICompletionsToResponses converts a legacy Completions request into a
// Responses API request body.
func OpenAICompletionsToResponses(body []byte) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("parse completions request: %w", err)
	}

	out := map[string]any{
		"store": false,
	}

	if model := strings.TrimSpace(stringFromAny(raw["model"])); model != "" {
		out["model"] = model
	}
	if stream, ok := raw["stream"].(bool); ok {
		out["stream"] = stream
	}
	if temperature, ok := float64FromAny(raw["temperature"]); ok {
		out["temperature"] = temperature
	}
	if topP, ok := float64FromAny(raw["top_p"]); ok {
		out["top_p"] = topP
	}
	if maxTokens, ok := intFromAny(raw["max_tokens"]); ok {
		out["max_output_tokens"] = maxTokens
	}

	if prompt, ok := buildOpenAICompletionsInput(raw["prompt"]); ok {
		out["input"] = prompt
	}

	return json.Marshal(out)
}

// ResponsesToOpenAICompletion converts a final Responses JSON body into a
// legacy Completions JSON body.
func ResponsesToOpenAICompletion(body []byte) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("parse responses body: %w", err)
	}

	choice := map[string]any{
		"text":     extractResponsesCompletionText(raw["output"]),
		"index":    0,
		"logprobs": nil,
		"finish_reason": openAICompletionFinishReason(
			stringFromAny(raw["status"]),
			extractIncompleteReason(raw["incomplete_details"]),
		),
	}

	out := map[string]any{
		"id":      stringFromAny(raw["id"]),
		"object":  "text_completion",
		"created": createdAtFromMap(raw),
		"model":   stringFromAny(raw["model"]),
		"choices": []any{choice},
	}
	if usage := buildOpenAIChatUsage(raw["usage"]); len(usage) > 0 {
		out["usage"] = usage
	}

	return json.Marshal(out)
}

// ResponsesEventToOpenAICompletions converts one Responses SSE payload into
// zero or more legacy Completions chunk payloads. The returned boolean
// indicates whether the stream reached a terminal event and should emit [DONE].
func ResponsesEventToOpenAICompletions(
	data []byte,
	state *OpenAICompletionsStreamState,
) ([][]byte, bool, error) {
	if len(data) == 0 {
		return nil, false, nil
	}

	var evt ResponsesStreamEvent
	if err := json.Unmarshal(data, &evt); err != nil {
		return nil, false, fmt.Errorf("parse responses stream event: %w", err)
	}

	if state == nil {
		state = NewOpenAICompletionsStreamState("")
	}
	if state.CreatedAt == 0 {
		state.CreatedAt = time.Now().Unix()
	}

	switch evt.Type {
	case "response.created":
		if evt.Response != nil {
			if state.ResponseID == "" {
				state.ResponseID = evt.Response.ID
			}
			if state.Model == "" {
				state.Model = evt.Response.Model
			}
		}
		return nil, false, nil
	case "response.output_text.delta":
		if evt.Delta == "" {
			return nil, false, nil
		}
		state.OutputTextDeltaSeen[openAICompletionsOutputTextKey(evt)] = true
		return marshalOpenAICompletionsChunks([]map[string]any{
			buildOpenAICompletionsChunk(state, evt.Delta, nil, nil),
		}, false)
	case "response.output_text.done":
		if evt.Text == "" {
			return nil, false, nil
		}
		if state.OutputTextDeltaSeen[openAICompletionsOutputTextKey(evt)] {
			return nil, false, nil
		}
		return marshalOpenAICompletionsChunks([]map[string]any{
			buildOpenAICompletionsChunk(state, evt.Text, nil, nil),
		}, false)
	case "response.completed", "response.done", "response.incomplete", "response.failed":
		finishReason := openAICompletionFinishReason(
			responseStatusFromEvent(evt),
			incompleteReasonFromEvent(evt.Response),
		)
		return marshalOpenAICompletionsChunks([]map[string]any{
			buildOpenAICompletionsChunk(state, "", &finishReason, buildOpenAIChatUsageFromResponse(evt.Response)),
		}, true)
	default:
		return nil, false, nil
	}
}

func buildOpenAICompletionsInput(raw any) (string, bool) {
	switch value := raw.(type) {
	case string:
		return value, true
	case []any:
		parts := make([]string, 0, len(value))
		for _, item := range value {
			switch v := item.(type) {
			case string:
				parts = append(parts, v)
			default:
				encoded, err := json.Marshal(v)
				if err != nil {
					continue
				}
				parts = append(parts, string(encoded))
			}
		}
		if len(parts) == 0 {
			return "", false
		}
		return strings.Join(parts, "\n\n"), true
	case nil:
		return "", false
	default:
		encoded, err := json.Marshal(value)
		if err != nil {
			return "", false
		}
		return string(encoded), true
	}
}

func extractResponsesCompletionText(raw any) string {
	outputs, ok := raw.([]any)
	if !ok || len(outputs) == 0 {
		return ""
	}

	var builder strings.Builder
	for _, item := range outputs {
		output, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(stringFromAny(output["type"])) != "message" {
			continue
		}
		builder.WriteString(extractResponsesMessageText(output["content"]))
	}
	return builder.String()
}

func buildOpenAICompletionsChunk(
	state *OpenAICompletionsStreamState,
	text string,
	finishReason *string,
	usage map[string]any,
) map[string]any {
	choice := map[string]any{
		"text":          text,
		"index":         0,
		"logprobs":      nil,
		"finish_reason": nil,
	}
	if finishReason != nil {
		choice["finish_reason"] = *finishReason
	}

	chunk := map[string]any{
		"id":      openAICompletionsResponseID(state),
		"object":  "text_completion",
		"created": openAICompletionsCreatedAt(state),
		"model":   openAICompletionsModel(state),
		"choices": []any{choice},
	}
	if len(usage) > 0 {
		chunk["usage"] = usage
	}
	return chunk
}

func marshalOpenAICompletionsChunks(chunks []map[string]any, done bool) ([][]byte, bool, error) {
	if len(chunks) == 0 {
		return nil, done, nil
	}

	out := make([][]byte, 0, len(chunks))
	for _, chunk := range chunks {
		payload, err := json.Marshal(chunk)
		if err != nil {
			return nil, false, err
		}
		out = append(out, payload)
	}
	return out, done, nil
}

func openAICompletionFinishReason(status, incompleteReason string) string {
	switch strings.TrimSpace(status) {
	case "incomplete":
		switch strings.TrimSpace(incompleteReason) {
		case "max_output_tokens":
			return "length"
		case "content_filter":
			return "content_filter"
		default:
			return "stop"
		}
	default:
		return "stop"
	}
}

func openAICompletionsOutputTextKey(evt ResponsesStreamEvent) string {
	return fmt.Sprintf("%d:%d", evt.OutputIndex, evt.ContentIndex)
}

func openAICompletionsResponseID(state *OpenAICompletionsStreamState) string {
	if state == nil {
		return ""
	}
	return state.ResponseID
}

func openAICompletionsCreatedAt(state *OpenAICompletionsStreamState) int64 {
	if state != nil && state.CreatedAt > 0 {
		return state.CreatedAt
	}
	return time.Now().Unix()
}

func openAICompletionsModel(state *OpenAICompletionsStreamState) string {
	if state == nil {
		return ""
	}
	return state.Model
}
