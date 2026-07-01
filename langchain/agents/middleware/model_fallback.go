package middleware

import "context"

type ModelFallbackMiddleware struct {
	Models []any
}

func NewModelFallbackMiddleware(firstModel any, additionalModels ...any) *ModelFallbackMiddleware {
	models := make([]any, 0, 1+len(additionalModels))
	models = append(models, firstModel)
	models = append(models, additionalModels...)
	return &ModelFallbackMiddleware{Models: models}
}

func (m *ModelFallbackMiddleware) WrapModelCall(ctx context.Context, request ModelRequest, handler ModelHandler) (ModelResponse, error) {
	response, err := handler(ctx, request)
	if err == nil {
		return response, nil
	}
	lastErr := err
	for _, fallbackModel := range m.Models {
		next, overrideErr := request.Override(WithModel(fallbackModel))
		if overrideErr != nil {
			return ModelResponse{}, overrideErr
		}
		response, err = handler(ctx, next)
		if err == nil {
			return response, nil
		}
		lastErr = err
	}
	return ModelResponse{}, lastErr
}
