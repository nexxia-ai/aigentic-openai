package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/nexxia-ai/aigentic/ai"
)

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

// OpenAIChatStreamResponse represents a streaming chunk from OpenAI
type OpenAIChatStreamResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role      string           `json:"role,omitempty"`
			Content   string           `json:"content,omitempty"`
			ToolCalls []OpenAIToolCall `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
}

// OpenAIContentPart represents a single content part in a message
type OpenAIContentPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	File     *OpenAIFile     `json:"file,omitempty"`
	ImageURL *OpenAIImageURL `json:"image_url,omitempty"`
}

// OpenAIImageURL represents an image URL in a message
type OpenAIImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
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
	Index        int    `json:"index,omitempty"`
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

// NewModel creates a new OpenAI model using the model struct
func NewModel(modelName string, apiKey string) *ai.Model {
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	model := &ai.Model{
		ModelName: modelName,
		APIKey:    apiKey,
		BaseURL:   "https://api.openai.com/v1",
	}
	model.SetGenerateFunc(openaiGenerate)
	model.SetStreamingFunc(openaiStream)
	return model
}

// isRetryableError checks if an error should trigger a retry
func isRetryableError(err error) error {
	if err == nil {
		return nil
	}

	errStr := err.Error()

	// Check for specific HTTP status codes that are retryable
	if strings.Contains(errStr, "status: 502") ||
		strings.Contains(errStr, "status: 503") ||
		strings.Contains(errStr, "status: 504") ||
		strings.Contains(errStr, "status: 429") {
		return fmt.Errorf("%w: %v", ai.ErrTemporary, err)
	}

	// Check for network-related errors
	if strings.Contains(errStr, "connection refused") ||
		strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "network") ||
		strings.Contains(errStr, "temporary") {
		return fmt.Errorf("%w: %v", ai.ErrTemporary, err)
	}

	return err
}

// openaiGenerate is the generate function for OpenAI models
func openaiGenerate(ctx context.Context, model *ai.Model, messages []ai.Message, tools []ai.Tool) (ai.AIMessage, error) {
	openaiMessages := openAIConvertMessages(messages)
	openaiTools := openAIConvertTools(tools)
	return openaiREST(ctx, model, openaiMessages, openaiTools)
}

// openaiStream is the streaming function for OpenAI models
func openaiStream(ctx context.Context, model *ai.Model, messages []ai.Message, tools []ai.Tool, chunkFunction func(ai.AIMessage) error) (ai.AIMessage, error) {
	openaiMessages := openAIConvertMessages(messages)
	openaiTools := openAIConvertTools(tools)
	return openaiStreamREST(ctx, model, openaiMessages, openaiTools, chunkFunction)
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
			// Handle file IDs first (OpenAI Files API)
			if strings.HasPrefix(r.URI, "file://") {
				// Extract file ID from URI
				fileID := strings.TrimPrefix(r.URI, "file://")
				contentParts := []OpenAIContentPart{
					{
						Type: "file",
						File: &OpenAIFile{
							FileID: fileID,
						},
					},
				}

				// Add text content if there's a description, otherwise use file name
				if r.Description != "" {
					contentParts = append(contentParts, OpenAIContentPart{Type: "text", Text: r.Description})
				} else if r.Name != "" {
					contentParts = append(contentParts, OpenAIContentPart{Type: "text", Text: "File: " + r.Name})
				}
				openaiMessages[i].Content = contentParts
			} else if r.MIMEType != "" && strings.HasPrefix(r.MIMEType, "image/") {
				// Handle image content with proper content parts structure
				if bodyBytes, ok := r.Body.([]byte); ok {
					base64Data := base64.StdEncoding.EncodeToString(bodyBytes)
					dataURL := fmt.Sprintf("data:%s;base64,%s", r.MIMEType, base64Data)

					contentParts := []OpenAIContentPart{
						{
							Type: "image_url",
							ImageURL: &OpenAIImageURL{
								URL:    dataURL,
								Detail: "auto", // Let OpenAI decide the level of detail
							},
						},
					}

					// Add text content if there's a description
					if r.Description != "" {
						contentParts = append(contentParts, OpenAIContentPart{Type: "text", Text: r.Description})
					} else if r.Name != "" {
						contentParts = append(contentParts, OpenAIContentPart{Type: "text", Text: "Image: " + r.Name})
					}

					openaiMessages[i].Content = contentParts
				} else {
					// Fallback for non-byte body
					openaiMessages[i].Content = r.Body
				}
			} else {
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

	reqBody, err := json.Marshal(req)
	if err != nil {
		return ai.AIMessage{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", model.BaseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return ai.AIMessage{}, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+model.APIKey)

	client := http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(httpReq)
	if err != nil {
		return ai.AIMessage{}, isRetryableError(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		errStatus := &ai.StatusError{
			StatusCode:   resp.StatusCode,
			Status:       resp.Status,
			ErrorMessage: string(respBody),
		}
		return ai.AIMessage{}, isRetryableError(errStatus)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return ai.AIMessage{}, isRetryableError(err)
	}

	var openaiResp OpenAIChatResponse
	if err := json.Unmarshal(respBody, &openaiResp); err != nil {
		return ai.AIMessage{}, isRetryableError(err)
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

// openaiStreamREST makes a streaming call to the OpenAI API
func openaiStreamREST(ctx context.Context, model *ai.Model, messages []OpenAIMessage, tools []OpenAITool, chunkFunction func(ai.AIMessage) error) (ai.AIMessage, error) {
	req := &OpenAIChatRequest{
		Model:    model.ModelName,
		Messages: messages,
		Tools:    tools,
		Stream:   true, // Enable streaming
	}

	// Apply configuration values from model pointer fields
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

	reqBody, err := json.Marshal(req)
	if err != nil {
		return ai.AIMessage{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", model.BaseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return ai.AIMessage{}, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+model.APIKey)

	client := http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(httpReq)
	if err != nil {
		return ai.AIMessage{}, isRetryableError(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		errStatus := &ai.StatusError{
			StatusCode:   resp.StatusCode,
			Status:       resp.Status,
			ErrorMessage: string(respBody),
		}
		return ai.AIMessage{}, isRetryableError(errStatus)
	}

	// Parse SSE response
	return parseSSEResponse(resp, chunkFunction)
}

// parseSSEResponse parses Server-Sent Events from OpenAI streaming API
func parseSSEResponse(resp *http.Response, chunkFunction func(ai.AIMessage) error) (ai.AIMessage, error) {
	scanner := bufio.NewScanner(resp.Body)
	var finalMessage ai.AIMessage
	var accumulatedContent strings.Builder
	var accumulatedThink strings.Builder
	var toolCallsMap = make(map[int]*ai.ToolCall)
	var responseID string
	var responseCreated int64
	var responseModel string

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines
		if line == "" {
			continue
		}

		// Check for [DONE] message
		if line == "data: [DONE]" {
			break
		}

		// Parse data lines (start with "data: ")
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		// Extract JSON data
		jsonData := strings.TrimPrefix(line, "data: ")

		// Parse the JSON chunk
		var chunk OpenAIChatStreamResponse
		if err := json.Unmarshal([]byte(jsonData), &chunk); err != nil {
			// Log the error but continue processing other chunks
			slog.Warn("Failed to parse SSE chunk", "error", err, "data", jsonData)
			continue
		}

		// Store response metadata from first chunk
		if responseID == "" {
			responseID = chunk.ID
			responseCreated = chunk.Created
			responseModel = chunk.Model
		}

		// Process the chunk
		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]

			// Handle new content from this chunk
			// var newContent string
			var contentForChunk string
			var thinkForChunk string

			if choice.Delta.Content != "" {
				contentForChunk, thinkForChunk = ai.ExtractThinkTags(choice.Delta.Content)
				accumulatedContent.WriteString(contentForChunk)
				accumulatedThink.WriteString(thinkForChunk)
			}

			// Handle tool calls
			if len(choice.Delta.ToolCalls) > 0 {
				for _, deltaToolCall := range choice.Delta.ToolCalls {
					index := deltaToolCall.Index

					if toolCallsMap[index] == nil {
						toolCallsMap[index] = &ai.ToolCall{
							ID:   deltaToolCall.ID,
							Type: deltaToolCall.Type,
							Name: deltaToolCall.FunctionCall.Name,
						}
					}

					// If we have a valid tool call, accumulate arguments
					// long arguments are sent in multiple chunks
					if toolCallsMap[index] != nil && deltaToolCall.FunctionCall.Arguments != "" {
						toolCallsMap[index].Args += deltaToolCall.FunctionCall.Arguments
					}
				}
			}

			// Set role if provided
			if choice.Delta.Role != "" && finalMessage.Role == "" {
				finalMessage.Role = ai.MessageRole(choice.Delta.Role)
			}

			// Only send chunks when there's actually new content
			if contentForChunk != "" || thinkForChunk != "" {
				// Create partial message for chunk function (only new content, no accumulated data)
				partialMessage := ai.AIMessage{
					Role:    finalMessage.Role,
					Content: contentForChunk,
					Think:   thinkForChunk,
					// Don't send ToolCalls in streaming chunks - they'll be in the final message
				}

				// Call chunk function with partial message
				if err := chunkFunction(partialMessage); err != nil {
					return ai.AIMessage{}, err
				}
			}

			// Check if streaming is complete
			if choice.FinishReason != "" {
				break
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return ai.AIMessage{}, fmt.Errorf("error reading SSE stream: %w", err)
	}

	// Set final accumulated content (without think tags) and think content
	finalMessage.Content = accumulatedContent.String()
	finalMessage.Think = accumulatedThink.String()

	var finalToolCalls []ai.ToolCall
	for i := 0; i < len(toolCallsMap); i++ {
		if toolCall, exists := toolCallsMap[i]; exists {
			finalToolCalls = append(finalToolCalls, *toolCall)
		}
	}
	finalMessage.ToolCalls = finalToolCalls

	// Set response metadata
	finalMessage.Response = ai.Response{
		ID:      responseID,
		Object:  "chat.completion",
		Created: responseCreated,
		Model:   responseModel,
	}

	return finalMessage, nil
}
