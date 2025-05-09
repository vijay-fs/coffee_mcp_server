"""
Pydantic models for Ragnor document extraction MongoDB documents.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Enum for job status tracking."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RagnorDocJob(BaseModel):
    """Model for tracking the status of document extraction jobs."""
    id: Any = Field(None, alias="_id")  # MongoDB ObjectId
    filename: str
    originalFilename: str
    fileSize: int
    mimeType: str
    fileFormat: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0  # 0 to 100 percent
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    completedAt: Optional[datetime] = None
    errorMessage: Optional[str] = None
    retryCount: int = 0
    version: str = "1.0.0"  # For future compatibility


class TextChunk(BaseModel):
    """Model for storing text chunks with position information."""
    text: str
    confidence: Optional[float] = None
    position: Optional[Dict[str, int]] = None  # {x, y, width, height}


class PageContent(BaseModel):
    """Model for storing page-level content extraction."""
    pageNumber: int
    text: str
    textChunks: List[TextChunk] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)
    embeddings: Optional[List[float]] = None
    embeddingModel: Optional[str] = None


class RagnorDocExtraction(BaseModel):
    """Model for storing document extraction results."""
    id: Any = Field(None, alias="_id")  # MongoDB ObjectId
    jobId: Any  # Reference to RagnorDocJob
    filename: str
    fileFormat: str
    totalPages: int
    processedPages: List[int] = Field(default_factory=list)
    pageContents: List[PageContent] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"  # For future compatibility
