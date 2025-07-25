package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/nexxia-ai/aigentic/ai"
)

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// OpenAI-specific request/response types
type OpenAIChatRequest struct {
	Model            string          `json:"model"`
	Messages         []OpenAIMessage `json:"messages"`
	Tools            []OpenAITool    `json:"tools,omitempty"`
	Temperature      float64         `json:"temperature,omitempty"`
	MaxTokens        int             `json:"max_tokens,omitempty"`
	TopP             float64         `json:"top_p,omitempty"`
	FrequencyPenalty float64         `json:"frequency_penalty,omitempty"`
	PresencePenalty  float64         `json:"presence_penalty,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
}

// OpenAIContentPart represents a single content part in a message
type OpenAIContentPart struct {
	Type string      `json:"type"`
	Text string      `json:"text,omitempty"`
	File *OpenAIFile `json:"file,omitempty"`
}

// OpenAIFile represents a file reference in a message
type OpenAIFile struct {
	FileID string `json:"file_id"`
}

type OpenAIMessage struct {
	Role       string           `json:"role"`
	Content    interface{}      `json:"content"` // Can be string or []OpenAIContentPart
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type OpenAIToolCall struct {
	ID           string `json:"id"`
	Type         string `json:"type"`
	FunctionCall struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type OpenAITool struct {
	Type     string             `json:"type"`
	Function OpenAIToolFunction `json:"function"`
}

type OpenAIToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type OpenAIChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role        string           `json:"role"`
			Content     string           `json:"content"`
			Refusal     interface{}      `json:"refusal"`
			Annotations []interface{}    `json:"annotations"`
			ToolCalls   []OpenAIToolCall `json:"tool_calls,omitempty"`
		} `json:"message"`
		Logprobs     interface{} `json:"logprobs"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
			AudioTokens  int `json:"audio_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails struct {
			ReasoningTokens          int `json:"reasoning_tokens"`
			AudioTokens              int `json:"audio_tokens"`
			AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
			RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
	ServiceTier string `json:"service_tier"`
}

// Retry configuration
const (
	maxRetries   = 5
	baseDelay    = 1 * time.Second
	maxDelay     = 30 * time.Second
	jitterFactor = 0.1
)

// NewOpenAIModel creates a new OpenAI model using the model struct
func NewOpenAIModel(modelName string, apiKey string) *ai.Model {
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	// Initialize random seed for jitter calculation
	rand.Seed(time.Now().UnixNano())

	model := &ai.Model{
		ModelName: modelName,
		APIKey:    apiKey,
		BaseURL:   "https://api.openai.com/v1",
	}
	model.SetGenerateFunc(openaiGenerate)
	return model
}

// isRetryableError checks if an error should trigger a retry
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}

	errStr := err.Error()

	// Check for specific HTTP status codes that are retryable
	if strings.Contains(errStr, "status: 502") ||
		strings.Contains(errStr, "status: 503") ||
		strings.Contains(errStr, "status: 504") ||
		strings.Contains(errStr, "status: 429") {
		return true
	}

	// Check for network-related errors
	if strings.Contains(errStr, "connection refused") ||
		strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "network") ||
		strings.Contains(errStr, "temporary") {
		return true
	}

	return false
}

// calculateBackoffDelay calculates the delay for the next retry with exponential backoff and jitter
func calculateBackoffDelay(attempt int) time.Duration {
	// Exponential backoff: baseDelay * 2^attempt
	delay := float64(baseDelay) * math.Pow(2, float64(attempt))

	// Cap the delay at maxDelay
	if delay > float64(maxDelay) {
		delay = float64(maxDelay)
	}

	// Add jitter to prevent thundering herd
	jitter := delay * jitterFactor * rand.Float64()
	delay += jitter

	return time.Duration(delay)
}

// openaiGenerateWithRetry is the generate function for OpenAI models with retry logic
func openaiGenerateWithRetry(ctx context.Context, model *ai.Model, messages []ai.Message, tools []ai.Tool) (ai.AIMessage, error) {
	var lastErr error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Convert our message format to OpenAI's format
		openaiMessages := openAIConvertMessages(messages)
		openaiTools := openAIConvertTools(tools)

		// Make a single LLM call
		response, err := openaiREST(ctx, model, openaiMessages, openaiTools)
		if err == nil {
			// Success - return the response
			return response, nil
		}

		lastErr = err

		// Check if this is a retryable error
		if !isRetryableError(err) {
			// Non-retryable error - return immediately
			return ai.AIMessage{}, fmt.Errorf("failed to generate (non-retryable): %w", err)
		}

		// If this is the last attempt, don't retry
		if attempt == maxRetries {
			break
		}

		// Calculate delay for next retry
		delay := calculateBackoffDelay(attempt)

		slog.Warn("OpenAI API request failed, retrying",
			"attempt", attempt+1,
			"max_attempts", maxRetries+1,
			"delay", delay,
			"error", err.Error())

		// Wait before retrying
		select {
		case <-ctx.Done():
			return ai.AIMessage{}, fmt.Errorf("context cancelled during retry: %w", ctx.Err())
		case <-time.After(delay):
			// Continue to next attempt
		}
	}

	// All retries exhausted
	return ai.AIMessage{}, fmt.Errorf("failed to generate after %d attempts. Last error: %w", maxRetries+1, lastErr)
}

// openaiGenerate is the generate function for OpenAI models
func openaiGenerate(ctx context.Context, model *ai.Model, messages []ai.Message, tools []ai.Tool) (ai.AIMessage, error) {
	return openaiGenerateWithRetry(ctx, model, messages, tools)
}

// openAIConvertMessages converts our message format to OpenAI's format
func openAIConvertMessages(messages []ai.Message) []OpenAIMessage {
	openaiMessages := make([]OpenAIMessage, len(messages))
	for i, msg := range messages {
		role, _ := msg.Value()
		openaiMessages[i] = OpenAIMessage{
			Role: string(role),
		}

		switch r := msg.(type) {
		case ai.UserMessage:
			openaiMessages[i].Content = r.Content
		case ai.AIMessage:
			openaiMessages[i].Content = r.Content
			for _, toolCall := range r.ToolCalls {
				openaiMessages[i].ToolCalls = append(openaiMessages[i].ToolCalls, OpenAIToolCall{
					ID:   toolCall.ID,
					Type: toolCall.Type,
					FunctionCall: struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					}{
						Name:      toolCall.Name,
						Arguments: toolCall.Args,
					},
				})
			}
		case ai.ToolMessage:
			openaiMessages[i].Content = r.Content
			openaiMessages[i].ToolCallID = r.ToolCallID
		case ai.SystemMessage:
			openaiMessages[i].Content = r.Content
		case ai.ResourceMessage:
			switch r.Type {
			// Handle file IDs first (OpenAI Files API)
			case "file":
				// Create content array with file reference
				contentParts := []OpenAIContentPart{
					{
						Type: "file",
						File: &OpenAIFile{
							FileID: r.Name,
						},
					},
				}

				// Add text content if there's a description or name
				if r.Description != "" {
					contentParts = append(contentParts, OpenAIContentPart{Type: "text", Text: r.Description})
				}
				openaiMessages[i].Content = contentParts
			case "image":
				contentParts := []OpenAIContentPart{
					{
						Type: "image_url",
						Text: base64.StdEncoding.EncodeToString(r.Body.([]byte)),
					},
				}
				openaiMessages[i].Content = contentParts
			default:
				// Handle text content by converting bytes to string to avoid base64 encoding
				if bodyBytes, ok := r.Body.([]byte); ok {
					openaiMessages[i].Content = string(bodyBytes)
				} else if bodyStr, ok := r.Body.(string); ok {
					openaiMessages[i].Content = bodyStr
				} else {
					openaiMessages[i].Content = r.Body
				}
			}
		default:
			panic(fmt.Sprintf("unsupported message type: %T - check that message is not a pointer", r))
		}
	}
	return openaiMessages
}

// openAIConvertTools converts our tool format to OpenAI's format
func openAIConvertTools(tools []ai.Tool) []OpenAITool {
	openaiTools := make([]OpenAITool, len(tools))
	for i, tool := range tools {
		openaiTools[i] = OpenAITool{
			Type: "function",
			Function: OpenAIToolFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			},
		}
	}
	return openaiTools
}

// openaiREST makes a single call to the OpenAI API
func openaiREST(ctx context.Context, model *ai.Model, messages []OpenAIMessage, tools []OpenAITool) (ai.AIMessage, error) {
	req := &OpenAIChatRequest{
		Model:    model.ModelName,
		Messages: messages,
		Tools:    tools,
	}

	// Apply configuration values from model pointer fields
	// Only set values that were explicitly set (non-nil pointers)
	if model.Temperature != nil {
		req.Temperature = *model.Temperature
	}
	if model.MaxTokens != nil {
		req.MaxTokens = *model.MaxTokens
	}
	if model.TopP != nil {
		req.TopP = *model.TopP
	}
	if model.FrequencyPenalty != nil {
		req.FrequencyPenalty = *model.FrequencyPenalty
	}
	if model.PresencePenalty != nil {
		req.PresencePenalty = *model.PresencePenalty
	}
	if model.StopSequences != nil {
		req.Stop = *model.StopSequences
	}
	if model.Stream != nil {
		req.Stream = *model.Stream
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return ai.AIMessage{}, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", model.BaseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return ai.AIMessage{}, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+model.APIKey)

	client := http.Client{Timeout: 10 * time.Minute} // Add timeout to prevent hanging requests ,
	resp, err := client.Do(httpReq)
	if err != nil {
		return ai.AIMessage{}, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return ai.AIMessage{}, &ai.StatusError{
			StatusCode:   resp.StatusCode,
			Status:       resp.Status,
			ErrorMessage: string(respBody),
		}
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return ai.AIMessage{}, fmt.Errorf("failed to read response body: %w", err)
	}

	var openaiResp OpenAIChatResponse
	if err := json.Unmarshal(respBody, &openaiResp); err != nil {
		return ai.AIMessage{}, fmt.Errorf("failed to decode response body: %w", err)
	}

	if len(openaiResp.Choices) == 0 {
		return ai.AIMessage{}, fmt.Errorf("no choices in response")
	}

	choice := openaiResp.Choices[0]
	content, thinkPart := ai.ExtractThinkTags(choice.Message.Content)

	msg := ai.AIMessage{
		Role:    ai.MessageRole(choice.Message.Role),
		Content: content,
		Think:   thinkPart,
	}

	// Convert tool calls
	for _, toolCall := range choice.Message.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, ai.ToolCall{
			ID:     toolCall.ID,
			Type:   toolCall.Type,
			Name:   toolCall.FunctionCall.Name,
			Args:   toolCall.FunctionCall.Arguments,
			Result: "",
		})
	}

	// Set response metadata
	msg.Response = ai.Response{
		ID:      openaiResp.ID,
		Object:  openaiResp.Object,
		Created: openaiResp.Created,
		Model:   openaiResp.Model,
		Usage: ai.Usage{
			PromptTokens:     openaiResp.Usage.PromptTokens,
			CompletionTokens: openaiResp.Usage.CompletionTokens,
			TotalTokens:      openaiResp.Usage.TotalTokens,
		},
		ServiceTier: openaiResp.ServiceTier,
	}

	return msg, nil
}
