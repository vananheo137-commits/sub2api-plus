package handler

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type strictModelsAccountRepo struct {
	service.AccountRepository
	accounts []service.Account
}

func (r *strictModelsAccountRepo) ListSchedulable(context.Context) ([]service.Account, error) {
	return r.accounts, nil
}

func (r *strictModelsAccountRepo) ListSchedulableByGroupID(context.Context, int64) ([]service.Account, error) {
	return r.accounts, nil
}

func TestGatewayModels_StrictModeDoesNotFallbackToDefaultModels(t *testing.T) {
	gin.SetMode(gin.TestMode)

	accountRepo := &strictModelsAccountRepo{
		accounts: []service.Account{
			{
				ID:          1,
				Name:        "openai-no-mapping",
				Platform:    service.PlatformOpenAI,
				Type:        service.AccountTypeAPIKey,
				Status:      service.StatusActive,
				Schedulable: true,
				Credentials: map[string]any{"api_key": "sk-test"},
			},
		},
	}

	h := &GatewayHandler{
		gatewayService: service.NewGatewayService(
			accountRepo, nil, nil, nil, nil, nil, nil, nil,
			nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil,
		),
	}

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodGet, "/v1/models", nil)

	groupID := int64(7)
	c.Set(string(middleware.ContextKeyAPIKey), &service.APIKey{
		ID:      101,
		GroupID: &groupID,
		Group: &service.Group{
			ID:       groupID,
			Platform: service.PlatformOpenAI,
		},
	})

	h.Models(c)

	require.Equal(t, http.StatusOK, rec.Code)

	var payload struct {
		Object string           `json:"object"`
		Data   []map[string]any `json:"data"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &payload))
	require.Equal(t, "list", payload.Object)
	require.Len(t, payload.Data, 0)
}
