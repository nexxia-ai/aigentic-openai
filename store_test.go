package openai

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/nexxia-ai/aigentic"
	"github.com/nexxia-ai/aigentic/utils"
)

func init() {
	utils.LoadEnvFile("../../.env")
}

// TestSimpleAddAndDelete tests basic file upload and delete functionality
func TestSimpleAddAndDelete(t *testing.T) {
	// Require OPENAI_API_KEY environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	// Create file manager
	fileManager := NewOpenAIFileManager(apiKey)
	defer fileManager.Close(context.Background())

	// Create a simple test file
	tempFile, err := os.CreateTemp("", "simple-test-*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tempFile.Name())

	// Write simple content
	testContent := "Simple test file for OpenAI file API."
	_, err = tempFile.WriteString(testContent)
	if err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tempFile.Close()

	t.Logf("Created test file: %s (%d bytes)", tempFile.Name(), len(testContent))

	// Create Document
	inputDoc := aigentic.NewInMemoryDocument(filepath.Base(tempFile.Name()), filepath.Base(tempFile.Name()), []byte(testContent), nil)
	inputDoc.FilePath = tempFile.Name()

	// Upload file
	doc, err := fileManager.AddDocument(context.Background(), inputDoc)
	if err != nil {
		t.Fatalf("Failed to add file: %v", err)
	}

	t.Logf("✅ Successfully uploaded file:")
	t.Logf("   ID: %s", doc.ID())
	t.Logf("   Filename: %s", doc.Filename)
	t.Logf("   Size: %d bytes", doc.FileSize)
	t.Logf("   MIME Type: %s", doc.MimeType)

	// Verify Document
	if doc.ID() == "" {
		t.Error("Document ID should not be empty")
	}
	// Extract just the filename from the full path for comparison
	expectedFilename := filepath.Base(tempFile.Name())
	if doc.Filename != expectedFilename {
		t.Errorf("Expected filename '%s', got '%s'", expectedFilename, doc.Filename)
	}
	if doc.FileSize != int64(len(testContent)) {
		t.Errorf("Expected size %d, got %d", len(testContent), doc.FileSize)
	}
	if doc.MimeType != "text/plain; charset=utf-8" {
		t.Errorf("Expected MIME type 'text/plain; charset=utf-8', got '%s'", doc.MimeType)
	}
	// Note: Files with purpose "assistants" cannot be downloaded via /files/{id}/content endpoint
	// This is a limitation of OpenAI's API. The content verification is skipped for this test.
	t.Logf("✅ File uploaded successfully. Content verification skipped due to OpenAI API limitations.")

	// Delete the file
	err = fileManager.DeleteDocument(context.Background(), doc.ID())
	if err != nil {
		t.Fatalf("Failed to remove file: %v", err)
	}

	t.Logf("✅ Successfully deleted file: %s", doc.ID())

	// Verify file was deleted by trying to remove it again (should fail)
	err = fileManager.DeleteDocument(context.Background(), doc.ID())
	if err == nil {
		t.Error("Expected error when trying to delete already deleted file")
	} else {
		t.Logf("✅ Expected error when deleting already deleted file: %v", err)
	}
}

// TestOpenAIFileManagerRealIntegration tests real OpenAI file API operations
func TestOpenAIFileManagerRealIntegration(t *testing.T) {
	// Require OPENAI_API_KEY environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	// Create file manager
	fileManager := NewOpenAIFileManager(apiKey)
	defer fileManager.Close(context.Background())

	// Create a test file with more complex content
	tempFile, err := os.CreateTemp("", "integration-test-*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tempFile.Name())

	// Write more complex content
	testContent := `This is a more complex test file for OpenAI file API integration testing.
It contains multiple lines and various characters.
Testing: !@#$%^&*()_+-=[]{}|;':",./<>?
Numbers: 1234567890
Special chars: éñüßåøæ
End of file.`
	_, err = tempFile.WriteString(testContent)
	if err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tempFile.Close()

	t.Logf("Created test file: %s (%d bytes)", tempFile.Name(), len(testContent))

	// Create Document
	inputDoc := aigentic.NewInMemoryDocument(filepath.Base(tempFile.Name()), filepath.Base(tempFile.Name()), []byte(testContent), nil)
	inputDoc.FilePath = tempFile.Name()

	// Upload file
	doc, err := fileManager.AddDocument(context.Background(), inputDoc)
	if err != nil {
		t.Fatalf("Failed to add file: %v", err)
	}

	// Verify Document
	if doc.ID() == "" {
		t.Error("Document ID should not be empty")
	}
	// Extract just the filename from the full path for comparison
	expectedFilename := filepath.Base(tempFile.Name())
	if doc.Filename != expectedFilename {
		t.Errorf("Expected filename '%s', got '%s'", expectedFilename, doc.Filename)
	}
	if doc.FileSize != int64(len(testContent)) {
		t.Errorf("Expected size %d, got %d", len(testContent), doc.FileSize)
	}
	if doc.MimeType != "text/plain; charset=utf-8" {
		t.Errorf("Expected MIME type 'text/plain; charset=utf-8', got '%s'", doc.MimeType)
	}

	t.Logf("✅ Successfully uploaded file: ID=%s, Filename=%s, Size=%d, MIME=%s",
		doc.ID(), doc.Filename, doc.FileSize, doc.MimeType)

	// Test listing files
	b, err := doc.Bytes()
	if err != nil {
		t.Fatalf("Failed to get file bytes: %v", err)
	}
	t.Logf("File bytes: %s", string(b))

	t.Run("ListAllFiles", func(t *testing.T) {
		allFiles, err := fileManager.ListAllDocuments(context.Background())
		if err != nil {
			t.Fatalf("Failed to list all files: %v", err)
		}

		t.Logf("Found %d total files on OpenAI", len(allFiles))

		// Look for our uploaded file
		found := false
		for _, file := range allFiles {
			if file.ID() == doc.ID() {
				found = true
				break
			}
		}

		if !found {
			t.Error("Uploaded file not found in list of all files")
		} else {
			t.Logf("✅ Found uploaded file in list of all files")
		}
	})

	// Clean up
	err = fileManager.DeleteDocument(context.Background(), doc.ID())
	if err != nil {
		t.Fatalf("Failed to remove file: %v", err)
	}

	t.Logf("✅ Successfully cleaned up test file: %s", doc.ID())
}

// TestMultipleFiles tests uploading and managing multiple files
func TestMultipleFiles(t *testing.T) {
	// Require OPENAI_API_KEY environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	// Create file manager
	fileManager := NewOpenAIFileManager(apiKey)
	defer fileManager.Close(context.Background())

	// Upload multiple files
	var docs []*aigentic.Document
	for i := 1; i <= 3; i++ {
		// Create a test file
		tempFile, err := os.CreateTemp("", fmt.Sprintf("multi-test-%d-*.txt", i))
		if err != nil {
			t.Fatalf("Failed to create temp file %d: %v", i, err)
		}
		defer os.Remove(tempFile.Name())

		// Write content
		testContent := fmt.Sprintf("Test file %d content", i)
		_, err = tempFile.WriteString(testContent)
		if err != nil {
			t.Fatalf("Failed to write to temp file %d: %v", i, err)
		}
		tempFile.Close()

		// Create Document
		inputDoc := aigentic.NewInMemoryDocument(filepath.Base(tempFile.Name()), filepath.Base(tempFile.Name()), []byte(testContent), nil)
		inputDoc.FilePath = tempFile.Name()

		// Upload file
		doc, err := fileManager.AddDocument(context.Background(), inputDoc)
		if err != nil {
			t.Fatalf("Failed to add file %d: %v", i, err)
		}

		docs = append(docs, doc)

		t.Logf("Uploaded file %d: ID=%s, Filename=%s", i, doc.ID(), doc.Filename)
	}

	// Verify we have the expected number of files
	if len(docs) != 3 {
		t.Errorf("Expected 3 files, got %d", len(docs))
	}

	// Verify all files have unique IDs
	ids := make(map[string]bool)
	for _, doc := range docs {
		if ids[doc.ID()] {
			t.Errorf("Duplicate file ID: %s", doc.ID())
		}
		ids[doc.ID()] = true
	}

	// Test listing files
	t.Run("ListFiles", func(t *testing.T) {
		files := fileManager.ListDocuments()
		t.Logf("Found %d files in local list", len(files))

		// Verify all uploaded files are in the list
		for _, doc := range docs {
			found := false
			for _, file := range files {
				if file.ID() == doc.ID() {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("Uploaded file %s not found in local list", doc.ID())
			}
		}

		t.Logf("✅ All uploaded files found in local list")
	})

	// Clean up all files
	for _, doc := range docs {
		err := fileManager.DeleteDocument(context.Background(), doc.ID())
		if err != nil {
			t.Fatalf("Failed to remove file %s: %v", doc.ID(), err)
		}
		t.Logf("Successfully removed file: ID=%s", doc.ID())
	}

	t.Logf("✅ Successfully cleaned up all test files")
}

// TestFileContentRetrieval tests retrieving file content
func TestFileContentRetrieval(t *testing.T) {
	// Require OPENAI_API_KEY environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	// Create file manager
	fileManager := NewOpenAIFileManager(apiKey)
	defer fileManager.Close(context.Background())

	// Create a test file
	tempFile, err := os.CreateTemp("", "content-test-*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tempFile.Name())

	// Write test content
	testContent := "This is test content for retrieval testing."
	_, err = tempFile.WriteString(testContent)
	if err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tempFile.Close()

	// Create Document
	inputDoc := aigentic.NewInMemoryDocument(filepath.Base(tempFile.Name()), filepath.Base(tempFile.Name()), []byte(testContent), nil)
	inputDoc.FilePath = tempFile.Name()

	// Upload file
	doc, err := fileManager.AddDocument(context.Background(), inputDoc)
	if err != nil {
		t.Fatalf("Failed to add file: %v", err)
	}

	t.Logf("✅ Successfully uploaded file: ID=%s, Filename=%s, Size=%d",
		doc.ID(), doc.Filename, doc.FileSize)

	// Test content retrieval
	t.Run("RetrieveContent", func(t *testing.T) {
		content, err := doc.Bytes()
		if err != nil {
			// This is expected behavior for files with purpose "assistants"
			// OpenAI API doesn't allow downloading files with this purpose
			if strings.Contains(err.Error(), "Not allowed to download files of purpose: assistants") {
				t.Logf("✅ Expected behavior: Files with purpose 'assistants' cannot be downloaded via content API")
				t.Logf("   This is a limitation of the OpenAI API for security reasons")
				return
			}
			t.Fatalf("Failed to retrieve file content: %v", err)
		}

		t.Logf("File ID: %s", doc.ID())
		t.Logf("File Name: %s", doc.Filename)
		t.Logf("File Size: %d bytes", doc.FileSize)
		t.Logf("Retrieved Content: %s", string(content))

		// Verify content matches
		if string(content) != testContent {
			t.Errorf("Content mismatch. Expected: '%s', Got: '%s'", testContent, string(content))
		} else {
			t.Logf("✅ Content matches expected value")
		}
	})

	// Clean up
	err = fileManager.DeleteDocument(context.Background(), doc.ID())
	if err != nil {
		t.Fatalf("Failed to remove file: %v", err)
	}

	t.Logf("✅ Successfully cleaned up test file: ID=%s", doc.ID())
}

// TestCleanupOnClose tests that Close() properly cleans up all files
func TestCleanupOnClose(t *testing.T) {
	// Require OPENAI_API_KEY environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	// Create a separate file manager for this test
	testFileManager := NewOpenAIFileManager(apiKey)

	// Create and upload a test file
	tempFile, err := os.CreateTemp("", "cleanup-test-*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tempFile.Name())

	// Write test content
	testContent := "Test content for cleanup testing."
	_, err = tempFile.WriteString(testContent)
	if err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tempFile.Close()

	// Create Document
	inputDoc := aigentic.NewInMemoryDocument(filepath.Base(tempFile.Name()), filepath.Base(tempFile.Name()), []byte(testContent), nil)
	inputDoc.FilePath = tempFile.Name()

	// Upload file
	doc, err := testFileManager.AddDocument(context.Background(), inputDoc)
	if err != nil {
		t.Fatalf("Failed to add file: %v", err)
	}

	t.Logf("Uploaded file for cleanup: ID=%s", doc.ID())

	// Close the file manager (should clean up all files)
	err = testFileManager.Close(context.Background())
	if err != nil {
		t.Fatalf("Failed to close file manager: %v", err)
	}

	t.Logf("✅ Successfully closed file manager and cleaned up files")

	// Verify file was deleted by trying to remove it again (should fail)
	err = testFileManager.DeleteDocument(context.Background(), doc.ID())
	if err == nil {
		t.Error("Expected error when trying to delete already deleted file")
	} else {
		t.Logf("✅ Expected error when deleting already deleted file: %v", err)
	}
}

// TestOpenByFileID tests retrieving a file from OpenAI by file ID
func TestOpenByFileID(t *testing.T) {
	// Require OPENAI_API_KEY environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	// Create file manager
	fileManager := NewOpenAIFileManager(apiKey)
	defer fileManager.Close(context.Background())

	// First, upload a file to get a file ID
	tempFile, err := os.CreateTemp("", "open-test-*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tempFile.Name())

	// Write test content
	testContent := "Test content for Open by ID testing."
	_, err = tempFile.WriteString(testContent)
	if err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tempFile.Close()

	// Create Document
	inputDoc := aigentic.NewInMemoryDocument(filepath.Base(tempFile.Name()), filepath.Base(tempFile.Name()), []byte(testContent), nil)
	inputDoc.FilePath = tempFile.Name()

	// Upload file to get a file ID
	uploadedDoc, err := fileManager.AddDocument(context.Background(), inputDoc)
	if err != nil {
		t.Fatalf("Failed to add file: %v", err)
	}

	fileID := uploadedDoc.ID()
	t.Logf("✅ Successfully uploaded file with ID: %s", fileID)

	// Test retrieving the file by ID using Open method
	t.Run("OpenByFileID", func(t *testing.T) {
		// Retrieve the file using Open method
		retrievedDoc, err := fileManager.Open(context.Background(), fileID)
		if err != nil {
			t.Fatalf("Failed to open file by ID: %v", err)
		}

		t.Logf("✅ Successfully retrieved file by ID:")
		t.Logf("   ID: %s", retrievedDoc.ID())
		t.Logf("   Filename: %s", retrievedDoc.Filename)
		t.Logf("   Size: %d bytes", retrievedDoc.FileSize)
		t.Logf("   MIME Type: %s", retrievedDoc.MimeType)

		// Verify the retrieved document matches the uploaded one
		if retrievedDoc.ID() != uploadedDoc.ID() {
			t.Errorf("Document ID mismatch. Expected: %s, Got: %s", uploadedDoc.ID(), retrievedDoc.ID())
		}
		if retrievedDoc.Filename != uploadedDoc.Filename {
			t.Errorf("Document filename mismatch. Expected: %s, Got: %s", uploadedDoc.Filename, retrievedDoc.Filename)
		}
		if retrievedDoc.FileSize != uploadedDoc.FileSize {
			t.Errorf("Document size mismatch. Expected: %d, Got: %d", uploadedDoc.FileSize, retrievedDoc.FileSize)
		}
		if retrievedDoc.MimeType != uploadedDoc.MimeType {
			t.Errorf("Document MIME type mismatch. Expected: %s, Got: %s", uploadedDoc.MimeType, retrievedDoc.MimeType)
		}

		t.Logf("✅ Retrieved document matches uploaded document")
	})

	// Clean up
	err = fileManager.DeleteDocument(context.Background(), fileID)
	if err != nil {
		t.Fatalf("Failed to remove file: %v", err)
	}

	t.Logf("✅ Successfully cleaned up test file: %s", fileID)
}

// TestErrorHandling tests various error conditions
func TestErrorHandling(t *testing.T) {
	// Require OPENAI_API_KEY environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	// Create file manager
	fileManager := NewOpenAIFileManager(apiKey)
	defer fileManager.Close(context.Background())

	// Test uploading non-existent file
	t.Run("NonExistentFile", func(t *testing.T) {
		// Create Document with non-existent file path
		nonExistentDoc := aigentic.NewInMemoryDocument("non-existent-file.txt", "non-existent-file.txt", []byte{}, nil)
		nonExistentDoc.FilePath = "non-existent-file.txt"

		_, err := fileManager.AddDocument(context.Background(), nonExistentDoc)
		if err == nil {
			t.Error("Expected error when uploading non-existent file")
		} else {
			t.Logf("✅ Expected error when uploading non-existent file: %v", err)
		}
	})

	// Test deleting non-existent file
	t.Run("NonExistentFileDelete", func(t *testing.T) {
		err := fileManager.DeleteDocument(context.Background(), "non-existent-file-id")
		if err == nil {
			t.Error("Expected error when deleting non-existent file")
		} else {
			t.Logf("✅ Expected error when deleting non-existent file: %v", err)
		}
	})
}

// TestNativeListDocuments tests the NativeListDocuments method and prints old documents
func TestNativeListDocuments(t *testing.T) {
	// Require OPENAI_API_KEY environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	// Create file manager
	fileManager := NewOpenAIFileManager(apiKey)
	defer fileManager.Close(context.Background())

	// Test listing all documents using NativeListDocuments
	t.Run("ListAllNativeDocuments", func(t *testing.T) {
		files, err := fileManager.NativeListDocuments(context.Background())
		if err != nil {
			t.Fatalf("Failed to list native documents: %v", err)
		}

		t.Logf("Found %d total files using NativeListDocuments", len(files))

		// Print basic info about all files
		for i, file := range files {
			createdTime := time.Unix(file.CreatedAt, 0)
			t.Logf("File %d: ID=%s, Name=%s, Size=%d bytes, Created=%s",
				i+1, file.ID, file.Filename, file.Bytes, createdTime.Format(time.RFC3339))
		}
	})

	// Test finding and printing documents older than 1 hour
	t.Run("PrintOldDocuments", func(t *testing.T) {
		files, err := fileManager.NativeListDocuments(context.Background())
		if err != nil {
			t.Fatalf("Failed to list native documents: %v", err)
		}

		// Calculate cutoff time (1 hour ago)
		oneHourAgo := time.Now().Add(-1 * time.Hour)
		cutoffTime := oneHourAgo.Unix()

		t.Logf("Looking for documents older than: %s", oneHourAgo.Format(time.RFC3339))
		t.Logf("Cutoff timestamp: %d", cutoffTime)

		// Find and print documents older than 1 hour
		var oldDocuments []FileInfo
		for _, file := range files {
			if file.CreatedAt < cutoffTime {
				oldDocuments = append(oldDocuments, file)
			}
		}

		t.Logf("Found %d documents older than 1 hour:", len(oldDocuments))

		if len(oldDocuments) == 0 {
			t.Logf("✅ No documents found older than 1 hour")
		} else {
			for i, file := range oldDocuments {
				createdTime := time.Unix(file.CreatedAt, 0)
				age := time.Since(createdTime)
				t.Logf("Old Document %d:", i+1)
				t.Logf("  ID: %s", file.ID)
				t.Logf("  Filename: %s", file.Filename)
				t.Logf("  Size: %d bytes", file.Bytes)
				t.Logf("  Purpose: %s", file.Purpose)
				t.Logf("  Created: %s", createdTime.Format(time.RFC3339))
				t.Logf("  Age: %s", age.String())
				t.Logf("  ---")
			}
		}
	})

	// Upload a test file and verify it appears in the list (and is NOT old)
	t.Run("UploadAndVerifyNew", func(t *testing.T) {
		// Create a test file
		tempFile, err := os.CreateTemp("", "native-list-test-*.txt")
		if err != nil {
			t.Fatalf("Failed to create temp file: %v", err)
		}
		defer os.Remove(tempFile.Name())

		// Write content
		testContent := "Test file for NativeListDocuments testing"
		_, err = tempFile.WriteString(testContent)
		if err != nil {
			t.Fatalf("Failed to write to temp file: %v", err)
		}
		tempFile.Close()

		// Create Document
		inputDoc := aigentic.NewInMemoryDocument(filepath.Base(tempFile.Name()), filepath.Base(tempFile.Name()), []byte(testContent), nil)
		inputDoc.FilePath = tempFile.Name()

		// Upload file
		doc, err := fileManager.AddDocument(context.Background(), inputDoc)
		if err != nil {
			t.Fatalf("Failed to add file: %v", err)
		}

		t.Logf("Uploaded test file: ID=%s, Filename=%s", doc.ID(), doc.Filename)

		// List documents again and find our uploaded file
		files, err := fileManager.NativeListDocuments(context.Background())
		if err != nil {
			t.Fatalf("Failed to list native documents after upload: %v", err)
		}

		// Find our uploaded file
		var uploadedFile *FileInfo
		for _, file := range files {
			if file.ID == doc.ID() {
				uploadedFile = &file
				break
			}
		}

		if uploadedFile == nil {
			t.Errorf("Uploaded file not found in NativeListDocuments results")
		} else {
			createdTime := time.Unix(uploadedFile.CreatedAt, 0)
			age := time.Since(createdTime)

			t.Logf("✅ Found uploaded file in NativeListDocuments:")
			t.Logf("  ID: %s", uploadedFile.ID)
			t.Logf("  Filename: %s", uploadedFile.Filename)
			t.Logf("  Size: %d bytes", uploadedFile.Bytes)
			t.Logf("  Purpose: %s", uploadedFile.Purpose)
			t.Logf("  Created: %s", createdTime.Format(time.RFC3339))
			t.Logf("  Age: %s", age.String())

			// Verify this file is NOT older than 1 hour
			oneHourAgo := time.Now().Add(-1 * time.Hour)
			if uploadedFile.CreatedAt < oneHourAgo.Unix() {
				t.Errorf("Newly uploaded file appears to be older than 1 hour - this shouldn't happen")
			} else {
				t.Logf("✅ Newly uploaded file is correctly identified as recent (not older than 1 hour)")
			}
		}

		// Clean up the test file
		err = fileManager.DeleteDocument(context.Background(), doc.ID())
		if err != nil {
			t.Logf("Failed to clean up test file: %v", err)
		} else {
			t.Logf("✅ Successfully cleaned up test file")
		}
	})

	t.Logf("✅ NativeListDocuments test completed successfully")
}
