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
