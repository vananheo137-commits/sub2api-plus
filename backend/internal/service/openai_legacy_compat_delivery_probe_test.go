package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

type openAICompatibilityProbeUpstream struct {
	mu       sync.Mutex
	lastBody []byte
}

func (u *openAICompatibilityProbeUpstream) Do(req *http.Request, proxyURL string, accountID int64, accountConcurrency int) (*http.Response, error) {
	u.mu.Lock()
	defer u.mu.Unlock()

	if req != nil && req.Body != nil {
		body, _ := io.ReadAll(req.Body)
		u.lastBody = append(u.lastBody[:0], body...)
		_ = req.Body.Close()
		req.Body = io.NopCloser(bytes.NewReader(body))
	}

	model := gjson.GetBytes(u.lastBody, "model").String()
	if model == "" {
		model = "gpt-5.1"
	}
	stream := gjson.GetBytes(u.lastBody, "stream").Bool()
	if stream {
		sse := fmt.Sprintf(
			"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n"+
				"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_%s\",\"model\":%q,\"usage\":{\"input_tokens\":3,\"output_tokens\":4,\"total_tokens\":7}}}\n\n",
			probeModelID(model),
			model,
		)
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(sse)),
		}, nil
	}

	payload := map[string]any{
		"id":         "resp_" + probeModelID(model),
		"created_at": 1700000000,
		"model":      model,
		"status":     "completed",
		"output": []any{
			map[string]any{
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type": "output_text",
						"text": probeResponseText(model),
					},
				},
			},
		},
		"usage": map[string]any{
			"input_tokens":  5,
			"output_tokens": 7,
			"total_tokens":  12,
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewReader(body)),
	}, nil
}

func (u *openAICompatibilityProbeUpstream) DoWithTLS(req *http.Request, proxyURL string, accountID int64, accountConcurrency int, enableTLSFingerprint bool) (*http.Response, error) {
	return u.Do(req, proxyURL, accountID, accountConcurrency)
}

func (u *openAICompatibilityProbeUpstream) LastModel() string {
	u.mu.Lock()
	defer u.mu.Unlock()
	return gjson.GetBytes(u.lastBody, "model").String()
}

func TestOpenAIForward_StrictCompatibilityDoesNotRewriteModelWithoutExplicitMapping(t *testing.T) {
	gin.SetMode(gin.TestMode)
	logSink, restore := captureStructuredLog(t)
	defer restore()

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{"model":"gpt-5.4-pro","stream":false,"input":"hello"}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{
		resp: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"id":"resp_no_mapping","model":"gpt-5.4-pro","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"strict no mapping"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`,
			)),
		},
	}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:          1,
		Name:        "strict-openai",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Credentials: map[string]any{"api_key": "sk-test"},
		Status:      StatusActive,
		Schedulable: true,
		Concurrency: 1,
	}

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "gpt-5.4-pro", result.Model)
	require.Equal(t, "gpt-5.4-pro", gjson.GetBytes(upstream.lastBody, "model").String())
	require.Contains(t, rec.Body.String(), `"model":"gpt-5.4-pro"`)
	require.True(t, logSink.ContainsFieldValue("requested_model", "gpt-5.4-pro"))
	require.True(t, logSink.ContainsFieldValue("upstream_model", "gpt-5.4-pro"))
}

func TestOpenAIForward_StrictCompatibilityAppliesExplicitModelMapping(t *testing.T) {
	gin.SetMode(gin.TestMode)
	logSink, restore := captureStructuredLog(t)
	defer restore()

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{"model":"gpt-5.4-pro","stream":false,"input":"hello"}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{
		resp: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"id":"resp_mapped","model":"gpt-5.4","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"explicit mapping"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`,
			)),
		},
	}
	svc := &OpenAIGatewayService{
		cfg:          &config.Config{},
		httpUpstream: upstream,
	}
	account := &Account{
		ID:       1,
		Name:     "mapped-openai",
		Platform: PlatformOpenAI,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_key": "sk-test",
			"model_mapping": map[string]any{
				"gpt-5.4-pro": "gpt-5.4",
			},
		},
		Status:      StatusActive,
		Schedulable: true,
		Concurrency: 1,
	}

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "gpt-5.4", gjson.GetBytes(upstream.lastBody, "model").String())
	require.Equal(t, http.StatusOK, rec.Code)
	require.True(t, logSink.ContainsFieldValue("requested_model", "gpt-5.4-pro"))
	require.True(t, logSink.ContainsFieldValue("upstream_model", "gpt-5.4"))
}

func TestOpenAICompatibilityDeliveryProbe_LegacyProtocolOnly(t *testing.T) {
	gin.SetMode(gin.TestMode)

	plainAccount := &Account{
		ID:          1,
		Name:        "plain-openai",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Credentials: map[string]any{"api_key": "sk-test"},
		Status:      StatusActive,
		Schedulable: true,
		Concurrency: 1,
	}
	mappedAccount := &Account{
		ID:       2,
		Name:     "mapped-openai",
		Platform: PlatformOpenAI,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_key": "sk-test",
			"model_mapping": map[string]any{
				"gpt-5.4-pro": "gpt-5.4",
			},
		},
		Status:      StatusActive,
		Schedulable: true,
		Concurrency: 1,
	}

	plainUpstream := &openAICompatibilityProbeUpstream{}
	plainServer := newOpenAICompatibilityProbeServer(t, plainAccount, plainUpstream)
	defer plainServer.Close()

	mappedUpstream := &openAICompatibilityProbeUpstream{}
	mappedServer := newOpenAICompatibilityProbeServer(t, mappedAccount, mappedUpstream)
	defer mappedServer.Close()

	chatNonStream := runProbeCurl(t, "chat_nonstream", []string{
		"-sS", "-i", "-X", "POST", plainServer.URL + "/v1/chat/completions",
		"-H", "Content-Type: application/json",
		"--data", `{"model":"gpt-5.1","stream":false,"messages":[{"role":"user","content":"hi"}]}`,
	})
	require.Contains(t, chatNonStream, "HTTP/1.1 200 OK")
	require.Contains(t, chatNonStream, `"object":"chat.completion"`)
	require.Contains(t, chatNonStream, `"content":"hello from compat"`)

	chatStream := runProbeCurl(t, "chat_stream", []string{
		"-sS", "-N", "-i", "-X", "POST", plainServer.URL + "/v1/chat/completions",
		"-H", "Content-Type: application/json",
		"--data", `{"model":"gpt-5.1","stream":true,"messages":[{"role":"user","content":"hi"}]}`,
	})
	require.Contains(t, chatStream, "HTTP/1.1 200 OK")
	require.Contains(t, chatStream, `"object":"chat.completion.chunk"`)
	require.Contains(t, chatStream, `data: [DONE]`)

	responsesStream := runProbeCurl(t, "responses_stream", []string{
		"-sS", "-N", "-i", "-X", "POST", plainServer.URL + "/v1/responses",
		"-H", "Content-Type: application/json",
		"--data", `{"model":"gpt-5.1","stream":true,"input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}]}`,
	})
	require.Contains(t, responsesStream, "HTTP/1.1 200 OK")
	require.Contains(t, responsesStream, `"type":"response.output_text.delta"`)
	require.NotContains(t, responsesStream, `"object":"chat.completion.chunk"`)

	mappedResponses := runProbeCurl(t, "responses_explicit_mapping", []string{
		"-sS", "-i", "-X", "POST", mappedServer.URL + "/v1/responses",
		"-H", "Content-Type: application/json",
		"--data", `{"model":"gpt-5.4-pro","stream":false,"input":"hi"}`,
	})
	require.Contains(t, mappedResponses, "HTTP/1.1 200 OK")
	require.Equal(t, "gpt-5.4", mappedUpstream.LastModel())
	t.Logf("[responses_explicit_mapping] upstream_model=%s", mappedUpstream.LastModel())
}

func newOpenAICompatibilityProbeServer(t *testing.T, account *Account, upstream *openAICompatibilityProbeUpstream) *httptest.Server {
	t.Helper()

	svc := &OpenAIGatewayService{
		cfg: &config.Config{
			Gateway: config.GatewayConfig{
				StreamDataIntervalTimeout: 0,
				StreamKeepaliveInterval:   0,
				MaxLineSize:               defaultMaxLineSize,
			},
		},
		httpUpstream: upstream,
	}

	router := gin.New()
	router.POST("/v1/responses", func(c *gin.Context) {
		probeForwardRequest(c, svc, account, nil)
	})
	router.POST("/v1/chat/completions", func(c *gin.Context) {
		probeForwardRequest(c, svc, account, func(body []byte) ([]byte, error) {
			SetOpenAICompatibilityMode(c, OpenAICompatibilityModeChatCompletions)
			return apicompat.OpenAIChatCompletionsToResponses(body)
		})
	})
	router.POST("/v1/completions", func(c *gin.Context) {
		probeForwardRequest(c, svc, account, func(body []byte) ([]byte, error) {
			SetOpenAICompatibilityMode(c, OpenAICompatibilityModeCompletions)
			return apicompat.OpenAICompletionsToResponses(body)
		})
	})
	return httptest.NewServer(router)
}

func probeForwardRequest(c *gin.Context, svc *OpenAIGatewayService, account *Account, convert func([]byte) ([]byte, error)) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": "failed to read request body"}})
		return
	}
	if convert == nil {
		if body, err = NormalizeOpenAICompatibilityRequestBody(body); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": err.Error()}})
			return
		}
	} else {
		if body, err = NormalizeOpenAICompatibilityRequestBody(body); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": err.Error()}})
			return
		}
		if body, err = convert(body); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": err.Error()}})
			return
		}
	}
	if _, err = svc.Forward(c.Request.Context(), c, account, body); err != nil && !c.Writer.Written() {
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": err.Error()}})
	}
}

func runProbeCurl(t *testing.T, label string, args []string) string {
	t.Helper()

	curlPath, err := exec.LookPath("curl.exe")
	if err != nil {
		curlPath, err = exec.LookPath("curl")
	}
	require.NoError(t, err)

	t.Logf("[%s] curl %s", label, renderCurlArgs(args))
	cmd := exec.Command(curlPath, args...)
	output, err := cmd.CombinedOutput()
	require.NoError(t, err, "[%s] curl failed: %s", label, string(output))
	result := string(output)
	t.Logf("[%s] response:\n%s", label, result)
	return result
}

func renderCurlArgs(args []string) string {
	quoted := make([]string, 0, len(args))
	for _, arg := range args {
		quoted = append(quoted, strconv.Quote(arg))
	}
	return strings.Join(quoted, " ")
}

func probeModelID(model string) string {
	replacer := strings.NewReplacer("/", "_", "-", "_", ".", "_", ":", "_", " ", "_")
	return replacer.Replace(model)
}

func probeResponseText(model string) string {
	if model == "gpt-5.4" {
		return "explicit mapping"
	}
	return "hello from compat"
}
