"""
API routes for Ragnor document extraction.
"""
import os
import asyncio
import tempfile
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from starlette.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime

from utils.ragnor_processor import RagnorDocumentProcessor
from db.ragnor_db_models import JobStatus

# Initialize router
router = APIRouter()

# Initialize document processor
document_processor = RagnorDocumentProcessor()

# Background task for processing documents


async def process_document_task(job_id: str, file_content: bytes):
    """Background task for processing a document."""
    print(
        f"Starting background processing for job {job_id}, file size: {len(file_content)} bytes")
    try:
        # Delay processing a bit to ensure job creation has completed
        await asyncio.sleep(1)

        # Perform the actual document processing
        await document_processor.process_document(job_id, file_content)

        print(f"Document processing completed successfully for job {job_id}")
    except Exception as e:
        print(f"ERROR in background processing for job {job_id}: {str(e)}")
        # Ensure job is marked as failed even if document_processor.process_document doesn't handle it
        try:
            await document_processor.update_job_status(
                job_id,
                JobStatus.FAILED,
                error_message=f"Processing failed: {str(e)}"
            )
        except Exception as status_err:
            print(
                f"ERROR updating job status for failed job {job_id}: {str(status_err)}")


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


@router.post("/v1/extract_data", response_model=ExtractDataResponse)
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
        # This prevents loading the entire file into memory at once
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

            # Stream the file in chunks to avoid memory issues with large files
            CHUNK_SIZE = 1024 * 1024  # 1MB chunks

            # Create a new extraction job before reading the full file
            # This allows us to start processing immediately
            job_id = await document_processor.create_extraction_job(
                file_content=b'',  # Empty content initially
                filename=file.filename,
                mime_type=file.content_type
            )

            # Update job status to indicate file is being received
            await document_processor.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                progress=0.0,
                error_message=None
            )

            # Stream file to disk
            total_size = 0
            try:
                while chunk := await file.read(CHUNK_SIZE):
                    temp_file.write(chunk)
                    total_size += len(chunk)

                    # Update progress based on file size if Content-Length is available
                    if file.size:
                        upload_progress = min(
                            5.0, (total_size / file.size) * 5.0)
                        await document_processor.update_job_status(
                            job_id,
                            JobStatus.PROCESSING,
                            progress=upload_progress
                        )
            except Exception as upload_error:
                # If file upload fails, update job status and raise error
                await document_processor.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message=f"File upload failed: {str(upload_error)}"
                )
                raise upload_error

        # Read the file from disk now that it's fully uploaded
        with open(temp_path, 'rb') as f:
            file_content = f.read()

        # Clean up the temporary file
        os.unlink(temp_path)

        # Add document processing to background tasks
        background_tasks.add_task(process_document_task, job_id, file_content)

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


@router.get("/v1/extract_data_job", response_model=JobStatusResponse)
async def get_extraction_job_status(job_id: str = Query(...)):
    """
    Get the status of a document extraction job.

    This endpoint allows clients to poll for status updates on an extraction job.

    Parameters:
    - job_id: The ID of the job to check

    Returns:
    - job_id: The ID of the job
    - status: The current status of the job (pending, processing, completed, failed)
    - progress: The progress percentage (0-100)
    - message: A message describing the job status
    - estimated_completion_time: Estimated time when the job will complete (if available)
    - next_poll_time: Recommended seconds to wait before polling again
    """
    try:
        # Get job status
        job_status = await document_processor.get_job_status(job_id)

        # Format response based on status
        response = {
            "job_id": job_id,
            "status": job_status.get("status"),
            "progress": job_status.get("progress", 0.0),
            "message": "",
            "next_poll_time": None
        }

        # Add status-specific details
        if job_status.get("status") == JobStatus.PENDING:
            response["message"] = "Job is pending processing."
            response["next_poll_time"] = 5  # Check again in 5 seconds

        elif job_status.get("status") == JobStatus.PROCESSING:
            response["message"] = f"Job is processing. Progress: {job_status.get('progress', 0.0):.1f}%"

            # Adjust polling frequency based on progress
            if job_status.get("progress", 0) < 20:
                response["next_poll_time"] = 5  # More frequent at start
            elif job_status.get("progress", 0) < 80:
                # Less frequent during main processing
                response["next_poll_time"] = 10
            else:
                response["next_poll_time"] = 3  # More frequent near end

        elif job_status.get("status") == JobStatus.COMPLETED:
            response["message"] = "Job has completed successfully."

        elif job_status.get("status") == JobStatus.FAILED:
            response["message"] = f"Job failed: {job_status.get('error_message', 'Unknown error')}"

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


@router.get("/v1/extract_data_result")
async def get_extraction_result(
    job_id: str = Query(...),
    page: int = Query(None),
    page_size: int = Query(10)
):
    """
    Get the result of a completed document extraction job.

    This endpoint returns the full extraction results for a completed job.

    Parameters:
    - job_id: The ID of the job to get results for
    - page: Page number for pagination (optional)
    - page_size: Number of pages to return per request (optional, default 10)

    Returns:
    - The full extraction result with metadata and page-by-page content
    - If page parameter is provided, returns only that subset of pages
    """
    try:
        print(f"Retrieving extraction result for job ID: {job_id}")

        # First check job status directly to give better error messages
        job_status = await document_processor.get_job_status(job_id)
        print(f"Current job status: {job_status}")

        # If job is not completed, return a proper error
        if job_status.get("status") != JobStatus.COMPLETED:
            print(
                f"Job {job_id} is not completed yet. Status: {job_status.get('status')}, Progress: {job_status.get('progress', 0.0)}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Job not completed",
                    "job_id": job_id,
                    "status": job_status.get("status"),
                    "progress": job_status.get("progress", 0.0),
                    "message": f"Job processing is {job_status.get('progress', 0.0):.1f}% complete. Please wait until processing completes."
                }
            )

        # Get extraction result
        print(f"Job is completed, retrieving extraction results...")
        result = await document_processor.get_extraction_result(job_id)

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
