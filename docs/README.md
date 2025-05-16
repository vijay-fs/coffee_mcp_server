coffee_mcp_server/
│
├── app/ # Main application code
│ ├── **init**.py # Makes app a package
│ ├── main.py # Application entry point
│ ├── config.py # Configuration management
│ ├── api/ # API endpoints
│ │ ├── **init**.py
│ │ ├── routes/ # Route definitions
│ │ │ ├── **init**.py
│ │ │ ├── document.py # Document processing routes (was ragnor_routes.py)
│ │ │ └── health.py # Health check routes
│ │ └── dependencies.py # Shared API dependencies
│ │
│ ├── core/ # Core business logic
│ │ ├── **init**.py
│ │ ├── processors/ # Document processing logic
│ │ │ ├── **init**.py
│ │ │ ├── document.py # Document processor (was ragnor_processor.py)
│ │ │ └── ocr.py # OCR-specific processing
│ │ ├── handlers/ # Format handlers
│ │ │ ├── **init**.py
│ │ │ ├── pdf.py # PDF handling
│ │ │ └── image.py # Image handling
│ │ └── embeddings/ # Embedding generation
│ │ ├── **init**.py
│ │ └── openai.py # OpenAI embedding generator
│ │
│ ├── db/ # Database connections and models
│ │ ├── **init**.py
│ │ ├── mongodb.py # MongoDB connection
│ │ └── models/ # Data models
│ │ ├── **init**.py
│ │ ├── document.py # Document models
│ │ └── job.py # Job models
│ │
│ └── utils/ # Utility functions
│ ├── **init**.py
│ ├── logging.py # Logging utilities
│ └── helpers.py # General helpers
│
├── tests/ # Test suite
│ ├── **init**.py
│ ├── conftest.py # Test configuration
│ ├── api/ # API tests
│ └── core/ # Core logic tests
│
├── scripts/ # Utility scripts
│ ├── seed_data.py # Data seeding
│ └── benchmark.py # Performance benchmarks
│
├── docs/ # Documentation
│ ├── api.md # API documentation
│ └── development.md # Development guide
│
├── .env.example # Example environment variables
├── .gitignore # Git ignore file
├── Dockerfile # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── requirements.txt # Production dependencies
├── requirements-dev.txt # Development dependencies
└── README.md # Project readme

## Changelog

### May 16, 2025 - Hybrid PDF Extraction System Implementation

The hybrid PDF extraction system has been implemented to optimize text extraction from PDFs. This system intelligently handles both text-based and scanned PDF pages, using the most appropriate extraction method for each page type.

Key changes include:

1. Created a new module `ragnor_pdf_analyzer.py` that:

   - Analyzes PDFs to determine which pages are scanned vs. text-based
   - Provides functions to extract text directly from text-based PDF pages using pdfplumber
   - Contains heuristics to detect scanned pages based on text content analysis

2. Enhanced the `PDFHandler` class with new methods:

   - `is_page_scanned()` to determine if a page is scanned or text-based
   - `extract_text_from_page()` to directly extract text from text-based PDF pages

3. Added a new `_process_text_page()` method to the `RagnorDocumentProcessor` class to:

   - Process text-based PDF pages with direct text extraction
   - Generate embeddings for text content if requested
   - Store the extracted text with the extraction method marked as "pdfplumber"

4. Updated the document processing workflow to:

   - Check each PDF page to determine if it's scanned or text-based
   - Use pdfplumber for text-based pages and existing OCR for scanned pages
   - Apply this approach to both large PDFs (>10 pages) and smaller PDFs

5. Added an "extractionMethod" field to the output structure to track whether text was extracted using:
   - "pdfplumber" for text-based PDF pages
   - "tesseract_ocr" for scanned pages

> How the ragnor_pdf_analyzer.py is analysing whether its a scanned or text page ?

⏺ The ragnor_pdf_analyzer.py module determines if a PDF page is scanned or text-based primarily by analyzing the character
objects extracted by pdfplumber. Here's how it works:

1. It uses pdfplumber to extract all the "chars" (character objects) from the page
2. It considers a page to be text-based if it has a substantial number of character objects (more than 10)
3. Pages with few or no character objects are identified as scanned pages

The key detection logic in is_scanned_page() looks at:

- Whether any character objects exist at all (has_text)
- The number of character objects (text_count)

This approach works because:

- Text-based PDFs have embedded text characters that pdfplumber can directly extract
- Scanned PDFs contain images of text with no actual character data, resulting in few or no characters being extracted
  by pdfplumber

The system uses this detection to intelligently choose between direct text extraction (faster for text-based PDFs) and
OCR (required for scanned PDFs).

> /cost
> ⎿ Total cost: $1.35

     Total duration (API):  5m 38.4s
     Total duration (wall): 19m 18.8s
     Total code changes:    504 lines added, 42 lines removed
