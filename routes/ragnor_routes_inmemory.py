"""
In-memory version of API routes for Ragnor document extraction (no MongoDB dependency).
This version uses in-memory dictionaries to store job data for testing purposes.
"""
import os
import io
import time
import uuid
import asyncio
import tempfile
import random
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from starlette.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

# Initialize router
router = APIRouter()

# In-memory storage for jobs and results
JOBS = {}
RESULTS = {}

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

class ExtractDataResponse(BaseModel):
    """Response model for the extract_data endpoint."""
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    """Response model for the extract_data_job endpoint."""
    job_id: str
    status: str
    progress: float
    message: str
    estimated_completion_time: Optional[datetime] = None
    next_poll_time: Optional[int] = None  # seconds to wait before next poll

# Mock document processor functions
async def process_document_inmemory(job_id: str, file_content: bytes, filename: str, mime_type: str, retry_count=0):
    """Mock document processing function using in-memory storage with retry mechanism."""
    MAX_RETRIES = 3
    
    try:
        # Update job status to processing
        JOBS[job_id]["status"] = JobStatus.PROCESSING
        JOBS[job_id]["progress"] = 0.0
        JOBS[job_id]["updated_at"] = datetime.utcnow()
        
        # Simulate processing steps with delays
        processing_steps = [
            {"name": "Extracting metadata", "progress": 10},
            {"name": "Converting to images", "progress": 30},
            {"name": "Performing OCR", "progress": 50},
            {"name": "Detecting tables", "progress": 70},
            {"name": "Generating embeddings", "progress": 90}
        ]
        
        print(f"Processing document in-memory for job {job_id}, file size: {len(file_content)} bytes")
        
        for step in processing_steps:
            # Add longer delays to simulate real processing (3-5 seconds per step)
            print(f"Step: {step['name']} - Progress: {step['progress']}%")
            step_time = 3 + (random.random() * 2)  # 3-5s per step
            await asyncio.sleep(step_time)
            JOBS[job_id]["progress"] = float(step["progress"])
            JOBS[job_id]["status_detail"] = step["name"]
            JOBS[job_id]["updated_at"] = datetime.utcnow()
            print(f"Completed step: {step['name']} after {step_time:.1f} seconds")
        
        # Create mock extraction result
        page_count = max(1, min(5, len(file_content) // 100000))  # Simulate page count based on file size
        result = {
            "job_id": job_id,
            "status": "completed",
            "filename": filename,
            "fileFormat": mime_type,
            "totalPages": page_count,
            "processedPages": list(range(1, page_count + 1)),
            "metadata": {
                "format": mime_type.split("/")[-1] if "/" in mime_type else mime_type,
                "pages": page_count,
                "title": filename,
                "fileSize": len(file_content)
            },
            "created_at": JOBS[job_id]["created_at"].isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "pages": []
        }
        
        # Calculate processing time
        created_at = JOBS[job_id]["created_at"]
        completed_at = datetime.utcnow()
        processing_time = (completed_at - created_at).total_seconds()
        result["processing_time_seconds"] = processing_time
        
        # Generate mock page content
        for page_num in range(1, page_count + 1):
            # Create realistic looking content based on file type
            if "pdf" in mime_type.lower():
                page_text = f"This is sample PDF text from page {page_num}. The content includes paragraphs, tables and figures."
            elif "docx" in mime_type.lower():
                page_text = f"DOCX document - Page {page_num}. This document contains formatted text with headings and lists."
            elif "image" in mime_type.lower():
                page_text = f"Extracted OCR text from image. The image contains text content that has been processed using OCR technology."
            else:
                page_text = f"Generic extracted content from page {page_num} of {filename}."
            
            # Generate chunks with realistic positions
            chunks = []
            y_position = 100
            for i in range(1, 4):  # 3 chunks per page
                chunk_text = f"Text chunk {i} on page {page_num}: {page_text[:30]}..."
                chunks.append({
                    "text": chunk_text,
                    "confidence": random.uniform(0.85, 0.98),  # Realistic OCR confidence
                    "position": {
                        "x": 50,
                        "y": y_position,
                        "width": 500,
                        "height": 20
                    }
                })
                y_position += 150  # Space between text chunks
            
            # Create page object
            page = {
                "pageNumber": page_num,
                "text": page_text,
                "textChunks": chunks,
                "hasEmbeddings": True,
                "embeddingModel": "OpenAIEmbeddingGenerator"
            }
            
            result["pages"].append(page)
        
        # Store the result
        RESULTS[job_id] = result
        
        # Update job status to completed
        JOBS[job_id]["status"] = JobStatus.COMPLETED
        JOBS[job_id]["progress"] = 100.0
        JOBS[job_id]["updated_at"] = datetime.utcnow()
        JOBS[job_id]["completed_at"] = datetime.utcnow()
        
    except Exception as e:
        # Log the error
        error_message = f"Error processing document: {str(e)}"
        print(error_message)
        
        # Check if we can retry
        if retry_count < MAX_RETRIES:
            # Increment retry count
            retry_count += 1
            
            # Update job status for retry
            JOBS[job_id]["status"] = JobStatus.PROCESSING
            JOBS[job_id]["progress"] = 0.0
            JOBS[job_id]["error_message"] = f"Retrying after error: {str(e)} (Attempt {retry_count} of {MAX_RETRIES})"
            JOBS[job_id]["updated_at"] = datetime.utcnow()
            JOBS[job_id]["retryCount"] = retry_count
            
            # Exponential backoff for retries (2^retry_count seconds)
            backoff_time = 2 ** retry_count
            print(f"Retrying job {job_id} in {backoff_time} seconds... (Attempt {retry_count} of {MAX_RETRIES})")
            await asyncio.sleep(backoff_time)
            
            # Retry processing
            await process_document_inmemory(job_id, file_content, filename, mime_type, retry_count)
        else:
            # Max retries reached, mark job as failed
            JOBS[job_id]["status"] = JobStatus.FAILED
            JOBS[job_id]["error_message"] = f"{error_message} (after {MAX_RETRIES} retries)"
            JOBS[job_id]["updated_at"] = datetime.utcnow()

@router.post("/v4/extract_data", response_model=ExtractDataResponse)
async def extract_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    embedding_provider: str = Form("openai"),
):
    """
    Submit a document for extraction.
    
    This endpoint accepts a file upload and creates a job for extracting text and generating embeddings.
    The job will be processed asynchronously, and the client can poll for status updates.
    
    Parameters:
    - file: The document file to process
    - embedding_provider: The provider to use for generating embeddings (openai or modal)
    
    Returns:
    - job_id: The ID of the created job
    - status: The initial status of the job (pending)
    - message: A message describing the job status
    """
    try:
        # For large file handling, use a temporary file to stream the content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Generate a unique job ID
            job_id = str(uuid.uuid4())
            
            # Create a new job record in memory
            JOBS[job_id] = {
                "filename": file.filename,
                "file_size": 0,  # Will update after reading
                "mime_type": file.content_type,
                "status": JobStatus.PROCESSING,
                "progress": 0.0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "error_message": None
            }
            
            # Stream file to disk in chunks
            CHUNK_SIZE = 1024 * 1024  # 1MB chunks
            total_size = 0
            
            try:
                while chunk := await file.read(CHUNK_SIZE):
                    temp_file.write(chunk)
                    total_size += len(chunk)
                    
                    # Update progress during upload
                    if file.size:
                        upload_progress = min(5.0, (total_size / file.size) * 5.0)
                        JOBS[job_id]["progress"] = upload_progress
                        JOBS[job_id]["updated_at"] = datetime.utcnow()
            except Exception as upload_error:
                # Handle upload error
                JOBS[job_id]["status"] = JobStatus.FAILED
                JOBS[job_id]["error_message"] = f"File upload failed: {str(upload_error)}"
                JOBS[job_id]["updated_at"] = datetime.utcnow()
                raise upload_error
            
            # Update file size now that we know it
            JOBS[job_id]["file_size"] = total_size
        
        # Read the complete file
        with open(temp_path, 'rb') as f:
            file_content = f.read()
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Update job status to pending (waiting for processing)
        JOBS[job_id]["status"] = JobStatus.PENDING
        JOBS[job_id]["updated_at"] = datetime.utcnow()
        
        # Start processing in background
        background_tasks.add_task(
            process_document_inmemory, 
            job_id, 
            file_content, 
            file.filename, 
            file.content_type
        )
        
        # Return job ID and initial status
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Document extraction job created successfully. Poll for status updates using the /v4/extract_data_job endpoint."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating document extraction job: {str(e)}"
        )

@router.get("/v4/extract_data_job", response_model=JobStatusResponse)
async def get_extraction_job_status(job_id: str = Query(...)):
    """
    Get the status of a document extraction job.
    
    Parameters:
    - job_id: The ID of the job to check
    
    Returns:
    - job_id: The ID of the job
    - status: The current status of the job (pending, processing, completed, failed)
    - progress: The progress percentage (0-100)
    - message: A message describing the job status
    - next_poll_time: Recommended seconds to wait before polling again
    """
    try:
        # Get job from in-memory storage
        job = JOBS.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        # Format response based on status
        response = {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "message": "",
            "next_poll_time": None
        }
        
        # Add status-specific details
        if job["status"] == JobStatus.PENDING:
            response["message"] = "Job is pending processing."
            response["next_poll_time"] = 5
            
        elif job["status"] == JobStatus.PROCESSING:
            response["message"] = f"Job is processing. Progress: {job['progress']:.1f}%"
            
            # Adjust polling frequency based on progress
            if job["progress"] < 20:
                response["next_poll_time"] = 5  # More frequent at start
            elif job["progress"] < 80:
                response["next_poll_time"] = 10  # Less frequent during main processing
            else:
                response["next_poll_time"] = 3  # More frequent near end
                
        elif job["status"] == JobStatus.COMPLETED:
            response["message"] = "Job has completed successfully."
            
        elif job["status"] == JobStatus.FAILED:
            response["message"] = f"Job failed: {job.get('error_message', 'Unknown error')}"
        
        return response
        
    except ValueError as e:
        # Job not found
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {str(e)}"
        )
    except Exception as e:
        # Other errors
        raise HTTPException(
            status_code=500,
            detail=f"Error getting job status: {str(e)}"
        )

@router.get("/v4/extract_data_result")
async def get_extraction_result(
    job_id: str = Query(...),
    page: int = Query(None),
    page_size: int = Query(10)
):
    """
    Get the result of a completed document extraction job.
    
    Parameters:
    - job_id: The ID of the job to get results for
    - page: Page number for pagination (optional)
    - page_size: Number of pages to return per request (optional, default 10)
    
    Returns:
    - The full extraction result with metadata and page-by-page content
    - If page parameter is provided, returns only that subset of pages
    """
    try:
        # Check job status
        job = JOBS.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        # If job is not completed, return a proper error
        if job["status"] != JobStatus.COMPLETED:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Job not completed",
                    "job_id": job_id,
                    "status": job["status"],
                    "progress": job["progress"]
                }
            )
        
        # Get result from in-memory storage
        result = RESULTS.get(job_id)
        if not result:
            raise ValueError(f"Result not found for job: {job_id}")
        
        # Apply pagination if requested
        if page is not None:
            total_pages = len(result.get("pages", []))
            
            # Validate page number
            if page < 1:
                raise ValueError("Page number must be greater than 0")
            
            # Calculate pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            # Get paginated results
            paginated_pages = result.get("pages", [])[start_idx:end_idx]
            
            # Create paginated response
            paginated_result = {
                "job_id": result.get("job_id"),
                "status": result.get("status"),
                "filename": result.get("filename"),
                "fileFormat": result.get("fileFormat"),
                "totalPages": total_pages,
                "pageCount": len(paginated_pages),
                "currentPage": page,
                "pageSize": page_size,
                "totalPages": total_pages,
                "pages": paginated_pages,
                "metadata": result.get("metadata", {})
            }
            
            return paginated_result
            
        # Return full extraction result if no pagination
        return result
        
    except ValueError as e:
        # Job or result not found
        raise HTTPException(
            status_code=404,
            detail=f"Result not found: {str(e)}"
        )
    except Exception as e:
        # Other errors
        raise HTTPException(
            status_code=500,
            detail=f"Error getting extraction result: {str(e)}"
        )
