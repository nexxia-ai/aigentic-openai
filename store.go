package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/nexxia-ai/aigentic/document"
)

// OpenAIStore manages temporary files for OpenAI chat sessions
type OpenAIStore struct {
	apiKey  string
	baseURL string
	client  *http.Client
	docs    map[string]*document.Document // Track uploaded documents
	mu      sync.RWMutex
}

var _ document.DocumentStore = &OpenAIStore{}

// NewOpenAIFileManager creates a new OpenAI file manager
func NewOpenAIFileManager(apiKey string) *OpenAIStore {
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	return &OpenAIStore{
		apiKey:  apiKey,
		baseURL: "https://api.openai.com/v1",
		client:  &http.Client{Timeout: 60 * time.Second},
		docs:    make(map[string]*document.Document),
	}
}

// Open implements the DocumentStore interface - retrieves a file from OpenAI by ID
func (fm *OpenAIStore) Open(ctx context.Context, fileID string) (*document.Document, error) {
	// Check if we already have this document in memory
	fm.mu.RLock()
	if doc, exists := fm.docs[fileID]; exists {
		fm.mu.RUnlock()
		return doc, nil
	}
	fm.mu.RUnlock()

	// Retrieve file info from OpenAI
	fileInfo, err := fm.getFileInfoFromOpenAI(ctx, fileID)
	if err != nil {
		return nil, fmt.Errorf("failed to get file info from OpenAI: %w", err)
	}

	// Create Document
	doc := document.NewInMemoryDocument(fileID, fileInfo.Filename, []byte{}, nil)

	// Store in memory
	fm.mu.Lock()
	fm.docs[fileID] = doc
	fm.mu.Unlock()

	return doc, nil
}

// AddDocument uploads a document to OpenAI and returns the document
func (fm *OpenAIStore) AddDocument(ctx context.Context, doc *document.Document) (*document.Document, error) {
	// Get document content using Bytes()
	content, err := doc.Bytes()
	if err != nil {
		return nil, fmt.Errorf("failed to get document content: %w", err)
	}

	// Upload to OpenAI
	fileID, err := fm.uploadBytesToOpenAI(ctx, doc)
	if err != nil {
		return nil, err
	}

	// Create Document
	uploadedDoc := document.NewInMemoryDocument(fileID, doc.Filename, content, nil)

	// Store in memory
	fm.mu.Lock()
	fm.docs[fileID] = uploadedDoc
	fm.mu.Unlock()

	return uploadedDoc, nil
}

// DeleteDocument deletes a document from OpenAI
func (fm *OpenAIStore) DeleteDocument(ctx context.Context, docID string) error {
	// Delete from OpenAI
	err := fm.deleteFromOpenAI(ctx, docID)
	if err != nil {
		return err
	}

	// Remove from memory
	fm.mu.Lock()
	delete(fm.docs, docID)
	fm.mu.Unlock()

	return nil
}

// ListDocuments retrieves documents created by this instance
func (fm *OpenAIStore) ListDocuments() []*document.Document {
	fm.mu.RLock()
	defer fm.mu.RUnlock()

	docs := make([]*document.Document, 0, len(fm.docs))
	for _, doc := range fm.docs {
		docs = append(docs, doc)
	}

	return docs
}

// FileInfo represents file information from OpenAI API
type FileInfo struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Bytes     int64  `json:"bytes"`
	CreatedAt int64  `json:"created_at"`
	Filename  string `json:"filename"`
	Purpose   string `json:"purpose"`
}

// NativeListDocuments retrieves file information from OpenAI API with retry logic
// TODO: this is a temporary function to get the file info from OpenAI API (Aug 2025
//
//	it should be replaced with ListAllDocuments when the Document type includes a creation date.
func (fm *OpenAIStore) NativeListDocuments(ctx context.Context) ([]FileInfo, error) {
	maxRetries := 3
	for attempt := 1; attempt <= maxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, "GET", fm.baseURL+"/files", nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("Authorization", "Bearer "+fm.apiKey)

		resp, err := fm.client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to list files: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			var listResp struct {
				Data []FileInfo `json:"data"`
			}

			if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
				return nil, fmt.Errorf("failed to decode response: %w", err)
			}

			return listResp.Data, nil
		}

		body, _ := io.ReadAll(resp.Body)

		if resp.StatusCode >= 500 && resp.StatusCode < 600 && attempt < maxRetries {
			backoff := time.Duration(attempt) * time.Second
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
				continue
			}
		}

		return nil, fmt.Errorf("list files failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil, fmt.Errorf("list files failed after %d attempts", maxRetries)
}

// ListAllDocuments retrieves all documents from OpenAI and returns them
func (fm *OpenAIStore) ListAllDocuments(ctx context.Context) ([]*document.Document, error) {
	files, err := fm.NativeListDocuments(ctx)
	if err != nil {
		return nil, err
	}

	var docs []*document.Document
	for _, file := range files {
		doc := document.NewInMemoryDocument(file.ID, file.Filename, []byte{}, nil)
		docs = append(docs, doc)
	}

	return docs, nil
}

// DeleteOldDocuments deletes documents from OpenAI that are older than the specified duration
func (fm *OpenAIStore) DeleteOldDocuments(ctx context.Context, maxAge time.Duration) error {
	files, err := fm.NativeListDocuments(ctx)
	if err != nil {
		return fmt.Errorf("failed to list documents: %w", err)
	}

	cutoffTime := time.Now().Add(-maxAge).Unix()
	var deletionErrors []error

	for _, file := range files {
		if file.CreatedAt < cutoffTime {
			if err := fm.DeleteDocument(ctx, file.ID); err != nil {
				deletionErrors = append(deletionErrors, fmt.Errorf("failed to delete document %s (%s): %w", file.ID, file.Filename, err))
			}
		}
	}

	if len(deletionErrors) > 0 {
		var errMsg string
		for i, err := range deletionErrors {
			if i > 0 {
				errMsg += "; "
			}
			errMsg += err.Error()
		}
		return fmt.Errorf("failed to delete some documents: %s", errMsg)
	}

	return nil
}

// Close deletes all documents and cleans up
func (fm *OpenAIStore) Close(ctx context.Context) error {
	fm.mu.RLock()
	docIDs := make([]string, 0, len(fm.docs))
	for docID := range fm.docs {
		docIDs = append(docIDs, docID)
	}
	fm.mu.RUnlock()

	for _, docID := range docIDs {
		if err := fm.DeleteDocument(ctx, docID); err != nil {
			// Log error but continue with cleanup
			fmt.Printf("Failed to remove document %s: %v\n", docID, err)
		}
	}

	return nil
}

// uploadBytesToOpenAI uploads a document to OpenAI's file API
func (fm *OpenAIStore) uploadBytesToOpenAI(ctx context.Context, doc *document.Document) (string, error) {
	// Retry logic for server errors
	maxRetries := 3
	for attempt := 1; attempt <= maxRetries; attempt++ {
		// Create multipart form
		var buf bytes.Buffer
		writer := multipart.NewWriter(&buf)

		// Get document content
		content, err := doc.Bytes()
		if err != nil {
			return "", fmt.Errorf("failed to get document content: %w", err)
		}

		// Add file field
		part, err := writer.CreateFormFile("file", doc.Filename)
		if err != nil {
			return "", fmt.Errorf("failed to create form file: %w", err)
		}

		// Copy content
		_, err = io.Copy(part, bytes.NewReader(content))
		if err != nil {
			return "", fmt.Errorf("failed to copy content: %w", err)
		}

		// Add purpose field
		// err = writer.WriteField("purpose", "assistants")
		err = writer.WriteField("purpose", "user_data")
		if err != nil {
			return "", fmt.Errorf("failed to add purpose field: %w", err)
		}

		writer.Close()

		// Create request
		req, err := http.NewRequestWithContext(ctx, "POST", fm.baseURL+"/files", &buf)
		if err != nil {
			return "", fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("Authorization", "Bearer "+fm.apiKey)
		req.Header.Set("Content-Type", writer.FormDataContentType())

		// Make request
		resp, err := fm.client.Do(req)
		if err != nil {
			return "", fmt.Errorf("failed to upload file: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			// Parse response
			var uploadResp struct {
				ID string `json:"id"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&uploadResp); err != nil {
				return "", fmt.Errorf("failed to decode response: %w", err)
			}
			return uploadResp.ID, nil
		}

		body, _ := io.ReadAll(resp.Body)

		// If it's a server error (5xx), retry with exponential backoff
		if resp.StatusCode >= 500 && resp.StatusCode < 600 && attempt < maxRetries {
			// Wait before retrying (exponential backoff)
			backoff := time.Duration(attempt) * time.Second
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(backoff):
				continue
			}
		}

		// For non-retryable errors or final attempt, return the error
		return "", fmt.Errorf("upload failed with status %d: %s", resp.StatusCode, string(body))
	}

	return "", fmt.Errorf("upload failed after %d attempts", maxRetries)
}

// deleteFromOpenAI deletes a file from OpenAI's file API
func (fm *OpenAIStore) deleteFromOpenAI(ctx context.Context, fileID string) error {
	// Retry logic for server errors
	maxRetries := 3
	for attempt := 1; attempt <= maxRetries; attempt++ {
		// Create request
		req, err := http.NewRequestWithContext(ctx, "DELETE", fmt.Sprintf("%s/files/%s", fm.baseURL, fileID), nil)
		if err != nil {
			return fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("Authorization", "Bearer "+fm.apiKey)

		// Make request
		resp, err := fm.client.Do(req)
		if err != nil {
			return fmt.Errorf("failed to delete file: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			return nil
		}

		body, _ := io.ReadAll(resp.Body)

		// If it's a server error (5xx), retry with exponential backoff
		if resp.StatusCode >= 500 && resp.StatusCode < 600 && attempt < maxRetries {
			// Wait before retrying (exponential backoff)
			backoff := time.Duration(attempt) * time.Second
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
				continue
			}
		}

		// For non-retryable errors or final attempt, return the error
		return fmt.Errorf("delete failed with status %d: %s", resp.StatusCode, string(body))
	}

	return fmt.Errorf("delete failed after %d attempts", maxRetries)
}

// getFileInfoFromOpenAI retrieves file information from OpenAI by file ID
func (fm *OpenAIStore) getFileInfoFromOpenAI(ctx context.Context, fileID string) (*struct {
	ID       string `json:"id"`
	Object   string `json:"object"`
	Bytes    int64  `json:"bytes"`
	Filename string `json:"filename"`
	Purpose  string `json:"purpose"`
}, error) {
	// Retry logic for server errors
	maxRetries := 3
	for attempt := 1; attempt <= maxRetries; attempt++ {
		// Create request
		req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/files/%s", fm.baseURL, fileID), nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("Authorization", "Bearer "+fm.apiKey)

		// Make request
		resp, err := fm.client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to get file info: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			// Parse response
			var fileInfo struct {
				ID       string `json:"id"`
				Object   string `json:"object"`
				Bytes    int64  `json:"bytes"`
				Filename string `json:"filename"`
				Purpose  string `json:"purpose"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&fileInfo); err != nil {
				return nil, fmt.Errorf("failed to decode response: %w", err)
			}
			return &fileInfo, nil
		}

		body, _ := io.ReadAll(resp.Body)

		// If it's a server error (5xx), retry with exponential backoff
		if resp.StatusCode >= 500 && resp.StatusCode < 600 && attempt < maxRetries {
			// Wait before retrying (exponential backoff)
			backoff := time.Duration(attempt) * time.Second
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
				continue
			}
		}

		// For non-retryable errors or final attempt, return the error
		return nil, fmt.Errorf("get file info failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil, fmt.Errorf("get file info failed after %d attempts", maxRetries)
}
