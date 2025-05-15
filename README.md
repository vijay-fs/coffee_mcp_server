# Coffee MCP Server

A powerful document extraction and processing API server built with FastAPI, designed to extract text, tables, and generate embeddings from documents.

## 📋 Overview

Coffee MCP Server provides a robust API for processing documents (PDFs, images, etc.) and extracting their content with advanced OCR and text processing capabilities. The server handles documents asynchronously, making it suitable for processing large files without blocking the client.

Key features include:

- Asynchronous document processing with responsive API during long-running tasks
- Page-by-page PDF processing for real-time status updates
- Text extraction using OCR (Optical Character Recognition)
- Table detection and extraction
- Generation of text embeddings (using OpenAI or other providers)
- MongoDB storage for persistent job tracking and results
- RESTful API with comprehensive endpoints
- Background thread processing to maintain API responsiveness

## 🚀 Setup Guide

### Prerequisites

- Python 3.8+
- MongoDB installed and running
- OCR dependencies:
  - Tesseract OCR engine
  - Poppler (for PDF processing)
- API keys for embedding providers (OpenAI, Anthropic)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd coffee_mcp_server
   ```

2. **Install dependencies**

   The project requires several Python packages and system dependencies:

   ```bash
   # Install system dependencies (macOS example)
   brew install tesseract poppler

   # Install Python dependencies
   pip install -r requirements.txt
   ```

   Key dependencies include:

   - FastAPI and Uvicorn for the API server
   - PyMongo for MongoDB interaction
   - OpenAI and Anthropic for embeddings generation
   - PyTesseract and PDF2Image for document processing
   - OpenCV for image processing
   - Various utilities for handling different file formats

3. **Environment Configuration**

   Create a `.env` file in the project root with the following variables:

   ```
   # MongoDB Connection
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DATABASE=coffee_mcp

   # API Keys
   OPENAI_API_KEY=your_openai_api_key

   # Optional: Other embedding providers
   # ANTHROPIC_API_KEY=your_anthropic_api_key

   # Server Configuration
   PORT=8000
   HOST=localhost
   ```

   > **Important**: Never commit your `.env` file to the repository. Make sure it's included in `.gitignore`.
   > Use `.env.example` as a template without actual secrets.

4. **Start MongoDB**

   Ensure your MongoDB instance is running:

   ```bash
   mongod --dbpath /path/to/data/directory
   ```

5. **Run the Server**
   ```bash
   uvicorn app:app --host localhost --port 8000 --reload
   ```

## 🔌 API Endpoints

### Document Extraction API

#### `POST /v1/extract_data`

Submit a document for extraction.

**Request:**

- Content-Type: `multipart/form-data`
- Body:
  - `file`: The document file to process (Required)
  - `embedding_provider`: The provider to use for generating embeddings (Default: "openai")

**Response:**

```json
{
  "job_id": "65a1f2d3b4c5d6e7f8g9h0i1",
  "status": "pending",
  "message": "Job created and queued for processing."
}
```

**Processing Flow:**

1. The document is uploaded and a job is created
2. Document processing happens asynchronously in the background
3. The client receives a job ID that can be used to check the status
4. Text extraction, table detection, and embedding generation occur sequentially
5. Results are stored in MongoDB for later retrieval

#### `GET /v1/extract_data_job`

Get the status of a document extraction job.

**Request:**

- Query Parameters:
  - `job_id`: The ID of the job to check (Required)

**Response:**

```json
{
  "job_id": "65a1f2d3b4c5d6e7f8g9h0i1",
  "status": "processing",
  "progress": 65.0,
  "message": "Processing page 13 of 20...",
  "estimated_completion_time": "2025-05-09T12:45:21.000Z",
  "next_poll_time": 5
}
```

**Possible Status Values:**

- `pending`: Job is queued but not yet started
- `processing`: Job is actively being processed
- `completed`: Job has completed successfully
- `failed`: Job has failed (includes error message)

#### `GET /v1/extract_data_result`

Get the result of a completed document extraction job.

**Request:**

- Query Parameters:
  - `job_id`: The ID of the job to get results for (Required)
  - `page`: Page number for pagination (Optional)
  - `page_size`: Number of pages to return per request (Optional, default: 10)

**Response:**

```json
{
  "job_id": "65a1f2d3b4c5d6e7f8g9h0i1",
  "status": "completed",
  "filename": "example.pdf",
  "fileFormat": "application/pdf",
  "totalPages": 20,
  "created_at": "2025-05-09T10:15:30.000Z",
  "completed_at": "2025-05-09T10:17:45.000Z",
  "processing_time_seconds": 135,
  "pages": [
    {
      "pageNumber": 1,
      "text": "Full extracted text content...",
      "textChunks": [
        {
          "text": "Text chunk content...",
          "bbox": [100, 200, 300, 400],
          "embeddings": [0.12, 0.34, 0.56, ...],
          "embedding_model": "text-embedding-3-large"
        }
      ],
      "hasTable": true,
      "tables": [
        {
          "table_id": "table_1",
          "title": "Table Title",
          "rows": 5,
          "columns": 3,
          "data": [["Header1", "Header2", "Header3"], ["Row1Col1", "Row1Col2", "Row1Col3"], ...]
        }
      ]
    }
  ],
  "metadata": {
    "title": "Document Title",
    "author": "Document Author",
    "creation_date": "2025-01-01"
  }
}
```

When using pagination, only the requested pages are returned, along with metadata about the pagination:

```json
{
  "job_id": "65a1f2d3b4c5d6e7f8g9h0i1",
  "status": "completed",
  "filename": "example.pdf",
  "fileFormat": "application/pdf",
  "totalPages": 20,
  "pageCount": 10,
  "currentPage": 1,
  "pageSize": 10,
  "pages": [
    // Only includes the first 10 pages
  ]
}
```

## 🛠️ Architecture

### Component Overview

The server is built around several key components:

1. **FastAPI Application (`app.py`)**: Main entry point and API definition
2. **API Routes (`routes/ragnor_routes.py`)**: API endpoint implementation with three primary endpoints:
   - `POST /v1/extract_data`: Submit documents for processing
   - `GET /v1/extract_data_job`: Check job status
   - `GET /v1/extract_data_result`: Retrieve processing results
3. **Document Processor (`utils/ragnor_processor.py`)**: Core processing logic for document extraction
4. **Text Extractor (`utils/ragnor_text_extractor.py`)**: OCR and text extraction using Tesseract
5. **Format Handlers (`utils/ragnor_format_handlers.py`)**: Specialized handlers for different file types (PDF, images)
6. **Embedding Generator (`utils/ragnor_embedding_generator.py`)**: Generates text embeddings using multiple providers (OpenAI, Anthropic)
7. **Database Models (`db/ragnor_db_models.py`)**: Data models for MongoDB storage
8. **Database Connection (`db/db.py`)**: MongoDB connection management

### Document Processing Flow

```
┌─────────────┐     ┌───────────────┐     ┌────────────────────┐
│ Client      │     │ FastAPI Router │     │ Document Processor │
│ Application │────▶│ (ragnor_routes)│────▶│  (Background Task) │
└─────────────┘     └───────────────┘     └────────────────────┘
                                                      │
                                                      ▼
┌─────────────┐     ┌───────────────┐     ┌────────────────────┐
│ Result      │     │ MongoDB       │     │ Text Extraction    │
│ Retrieval   │◀────│ Storage       │◀────│ & Embedding        │
└─────────────┘     └───────────────┘     └────────────────────┘
```

1. **Document Upload**:

   - Client uploads document to `/v1/extract_data`
   - Server creates a job entry in MongoDB
   - Background task is triggered for processing

2. **Optimized Background Processing**:

   - Document is analyzed and format detected
   - For large PDFs (>10 pages), processing happens page-by-page in real-time
   - Each page is converted to an image and immediately processed with OCR
   - MongoDB is updated after each page is processed, enabling real-time progress tracking
   - Background thread processing ensures API remains responsive during intensive OCR tasks
   - Progress percentage is continuously updated in MongoDB
   - For smaller documents, batch processing is used for efficiency

3. **Result Retrieval**:
   - Client polls `/v1/extract_data_job` for status
   - When completed, client retrieves results from `/v1/extract_data_result`
   - Results can be paginated for large documents

## 🔐 Security Considerations

- **API Keys Management**:
  - Store all API keys (OpenAI, Anthropic) in the `.env` file
  - **IMPORTANT**: Never commit the `.env` file to your Git repository
  - Use `.env.example` as a template without actual secrets
  - Add `.env` to your `.gitignore` file to prevent accidental commits
  - If secrets are accidentally committed, follow these steps:
    1. Remove sensitive data from Git history using `git filter-branch`
    2. Update all compromised API keys immediately
    3. Ensure `.gitignore` is properly configured
- **Input Validation**: All inputs are validated to prevent injection attacks
- **Error Handling**: Robust error handling prevents exposing sensitive details
- **CORS Configuration**: API is configured with appropriate CORS settings
- **Secure Development Workflow**:
  - Use feature branches for development
  - Review code for security issues before merging
  - Regularly update dependencies to patch security vulnerabilities

## 🔧 Performance Optimizations

### Responsive PDF Processing

The server implements several optimizations to ensure responsiveness when processing large PDF documents:

1. **Page-by-Page Processing**:

   - Large PDFs are processed one page at a time
   - Each page is immediately processed after conversion instead of waiting for all pages to be converted
   - MongoDB is updated after each page completes, providing real-time status updates

2. **Background Thread Processing**:

   - Document processing runs in a background thread
   - Main API thread remains responsive for status queries and other requests
   - Non-blocking architecture allows for concurrent processing of multiple documents

3. **Optimized MongoDB Interaction**:

   - Support for both string IDs and ObjectIds in queries
   - Efficient update operations with minimal database overhead
   - Atomic updates for page data to prevent race conditions
   - Robust error handling with automatic recovery mechanisms

4. **Flexible Debug Image Handling**:
   - Debug images can be saved to a configurable path using `RAGNOR_DEBUG_IMAGES_PATH` environment variable
   - If the variable is not set, no debug images are created, improving performance

### Memory and Performance Considerations

- Processing large documents (500+ pages) requires sufficient memory for OCR operations
- For very large documents, consider increasing server memory allocation
- API remains responsive even during intensive processing tasks
- Progress updates allow clients to accurately track status of long-running jobs

## 🧩 Development and Extensions

### Adding New Features

To add a new feature to the Coffee MCP Server:

1. Implement the core functionality in the `utils/` directory
2. Add any new models to `db/ragnor_db_models.py`
3. Create appropriate routes in `routes/`
4. Update this documentation to reflect the new features

### Testing

Run tests with:

```bash
pytest
```

## Suggested Folder Structure and Naming Conventions for Coffee MCP Server

Link to the folder structure and naming conventions document:
[docs/folder_structure_and_naming_conventions.md](docs/folder_structure_and_naming_conventions.md)

!NOTE: The folder structure and naming conventions document is a work in progress and is subject to change.

## 📄 License

[Include your license information here]

## 👥 Contributors

[List of contributors]

## Author

Vijay
