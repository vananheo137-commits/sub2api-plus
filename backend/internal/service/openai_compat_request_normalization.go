package service

import (
	"fmt"

	"github.com/tidwall/gjson"
)

// NormalizeOpenAICompatibilityRequestBody validates compatibility request JSON
// without performing any implicit model rewriting. Strict model mode only
// allows explicit account-level mappings later in the routing pipeline.
func NormalizeOpenAICompatibilityRequestBody(body []byte) ([]byte, error) {
	if len(body) == 0 {
		return body, nil
	}
	if !gjson.ValidBytes(body) {
		return nil, fmt.Errorf("invalid json body")
	}
	return body, nil
}
