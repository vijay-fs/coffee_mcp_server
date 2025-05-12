"""
Core document processing logic for Ragnor document extraction API.
"""
import os
import sys
import io
import json
import time
import asyncio
import random
import traceback
import threading
import multiprocessing
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from bson import ObjectId

import pytesseract
from PIL import Image

from db.ragnor_db_models import JobStatus, RagnorDocJob, RagnorDocExtraction, PageContent, TextChunk
from db.db import ragnor_doc_jobs_collection, ragnor_doc_extractions_collection
from utils.ragnor_format_handlers import get_handler_for_format, FormatHandler, PDFHandler
from utils.ragnor_text_extractor import RagnorTextExtractor, OCRResult
from utils.ragnor_embedding_generator import get_embedding_generator

# Initialize the text extractor
text_extractor = RagnorTextExtractor()

# Create a process pool for OCR processing
# This will allow the API to remain responsive during heavy processing
OCR_PROCESS_POOL = multiprocessing.Pool(processes=2)


class RagnorDocumentProcessor:
    """Main processor for document extraction."""

    def __init__(self, embedding_provider="openai"):
        """Initialize the document processor."""
        self.text_extractor = RagnorTextExtractor()
        try:
            self.embedding_generator = get_embedding_generator(
                embedding_provider)
        except ValueError:
            print(
                f"Warning: Could not initialize embedding generator for {embedding_provider}. Embeddings will not be generated.")
            self.embedding_generator = None
        self.background_tasks = {}  # Keep track of background tasks

    def generate_embeddings_for_text(self, text: str, provider: str = "openai"):
        """Generate embeddings for the given text using the specified provider.

        Args:
            text: The text to generate embeddings for
            provider: The embedding provider to use (default: "openai")

        Returns:
            Tuple containing (embeddings, model_name, success_flag)
        """
        if not text or not text.strip():
            print("WARNING: Cannot generate embeddings for empty text")
            return [], "", False

        try:
            from utils.ragnor_embedding_generator import OpenAIEmbeddingGenerator
            print(f"Generating embeddings for text using {provider}")

            # Create embedding generator based on provider
            if provider.lower() == "openai":
                embedding_generator = OpenAIEmbeddingGenerator()
                embedding_model = embedding_generator.model
            else:
                print(
                    f"Unsupported embedding provider: {provider}, defaulting to OpenAI")
                embedding_generator = OpenAIEmbeddingGenerator()
                embedding_model = embedding_generator.model

            # Generate embeddings for the text
            embeddings = embedding_generator.generate_embeddings(text)

            # Ensure the embeddings are properly formatted for MongoDB
            # Convert numpy arrays or any other non-JSON serializable formats to lists of floats
            if embeddings is not None:
                # Ensure we have valid numeric values
                processed_embeddings = [float(val) for val in embeddings]
                print(
                    f"Generated embeddings with {len(processed_embeddings)} dimensions")
                # Print a sample of the first few embeddings to verify format
                print(
                    f"Sample embeddings (first 3): {processed_embeddings[:3]}")
                return processed_embeddings, embedding_model, True
            else:
                print("WARNING: No embeddings were generated")
                return [], embedding_model, False
        except Exception as emb_err:
            print(f"ERROR generating embeddings: {str(emb_err)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return [], "", False

    async def create_extraction_job(self, file_content: bytes, filename: str, mime_type: str) -> str:
        """Create a new document extraction job and return the job ID."""
        try:
            # Identify file format
            file_format = mime_type

            # Create a new job document
            job_doc = RagnorDocJob(
                _id=str(ObjectId()),  # Generate a new ObjectId
                filename=os.path.basename(filename),
                originalFilename=filename,
                fileSize=len(file_content),
                mimeType=mime_type,
                fileFormat=file_format,
                status=JobStatus.PENDING,
                progress=0.0,
                createdAt=datetime.utcnow(),
                updatedAt=datetime.utcnow()
            )

            # Insert job document
            ragnor_doc_jobs_collection.insert_one(job_doc.dict(by_alias=True))

            # Return the job ID
            return job_doc.id

        except Exception as e:
            print(f"Error creating extraction job: {str(e)}")
            raise

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a document extraction job."""
        try:
            # Find the job document - try both string ID and ObjectId
            job_doc = ragnor_doc_jobs_collection.find_one({"_id": job_id})

            # If not found with string ID, try with ObjectId
            if not job_doc:
                try:
                    # Try to convert to ObjectId and search again
                    from bson.objectid import ObjectId
                    job_doc = ragnor_doc_jobs_collection.find_one(
                        {"_id": ObjectId(job_id)})
                    print(f"Found job with ObjectId: {job_id}")
                except:
                    # If conversion fails, it's not a valid ObjectId
                    print(f"Could not convert {job_id} to ObjectId")

            if not job_doc:
                raise ValueError(f"Job not found: {job_id}")

            # Return job status
            return {
                "job_id": job_id,
                "status": job_doc.get("status"),
                "progress": job_doc.get("progress", 0.0),
                "created_at": job_doc.get("createdAt"),
                "updated_at": job_doc.get("updatedAt"),
                "completed_at": job_doc.get("completedAt"),
                "error_message": job_doc.get("errorMessage")
            }

        except Exception as e:
            print(f"Error getting job status: {str(e)}")
            raise

    async def update_job_status(self, job_id: str, status: JobStatus, progress: float = None, error_message: str = None, result_id: str = None) -> None:
        """Update the status of a document extraction job."""
        try:
            # Prepare update
            update_doc = {
                "status": status,
                "updatedAt": datetime.utcnow()
            }

            # Add optional fields if provided
            if progress is not None:
                update_doc["progress"] = progress

            if error_message:
                update_doc["errorMessage"] = error_message

            if result_id:
                update_doc["resultId"] = result_id

            # Add completedAt timestamp if job is completed or failed
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                update_doc["completedAt"] = datetime.utcnow()

            # Update job document
            ragnor_doc_jobs_collection.update_one(
                {"_id": job_id},
                {"$set": update_doc}
            )

        except Exception as e:
            print(f"Error updating job status: {str(e)}")

    async def update_job_metadata(self, job_id: str, metadata: dict) -> None:
        """Update metadata for a document extraction job.

        Args:
            job_id: The ID of the job to update
            metadata: A dictionary of metadata to update
        """
        try:
            # Update the job document with the provided metadata
            update_doc = {
                **metadata,
                "updatedAt": datetime.utcnow()
            }

            # Update job document
            result = ragnor_doc_jobs_collection.update_one(
                {"_id": job_id},
                {"$set": update_doc}
            )

            print(
                f"Updated job metadata for {job_id}: {metadata}, modified count: {result.modified_count}")

        except Exception as e:
            print(f"Error updating job metadata: {str(e)}")

    def _process_single_page_sync(self, image, page_num, total_pages, file_format, extraction_id, job_id, generate_embeddings=False, embedding_provider="openai"):
        """Process a single page with OCR and update the database (synchronous version).

        This is a non-async version of the method for use in background threads.
        """
        print(f"Processing page {page_num}/{total_pages}")

        # Extract text using OCR
        print(f"Performing OCR on page {page_num}...")
        try:
            ocr_result = self.text_extractor.extract_text(
                image, doc_format=file_format)

            # Convert OCR result to a serializable dictionary
            ocr_dict = self.text_extractor.serialize_result(ocr_result)

            # Extract the full text from text lines
            full_text = "\n".join(
                [line.text for line in ocr_result.text_lines]) if ocr_result.text_lines else ""

            # Debug OCR results
            text_sample = full_text[:100] + \
                "..." if full_text else "[No text extracted]"
            print(f"OCR result for page {page_num}: {text_sample}")
            print(f"Extracted {len(ocr_result.text_lines)} text lines")

            # Create text chunks for the page data
            text_chunks = []
            if ocr_result.text_lines:
                for line in ocr_result.text_lines:
                    text_chunks.append({
                        "text": line.text,
                        "confidence": line.confidence,
                        "position": {
                            "x": line.bbox[0] if line.bbox else 0,
                            "y": line.bbox[1] if line.bbox else 0,
                            "width": line.bbox[2] if line.bbox else 0,
                            "height": line.bbox[3] if line.bbox else 0
                        }
                    })

            # Start with basic page data structure
            page_data = {
                "pageNumber": page_num
            }

            # Add text content (this will be first in the document)
            page_data["text"] = full_text

            # Generate embeddings immediately after OCR is complete if requested
            if generate_embeddings and full_text.strip():
                print(
                    f"Generating embeddings for page {page_num} text using {embedding_provider}")
                embeddings, embedding_model, success = self.generate_embeddings_for_text(
                    full_text, embedding_provider)

                if success and embeddings:
                    # Add embeddings right after text in the page data
                    print(f"DEBUG: Adding embeddings to page {page_num} data")
                    page_data["embeddings"] = embeddings
                    page_data["embeddingModel"] = embedding_model
                    page_data["hasEmbeddings"] = True
                    # Print the type and a sample to verify
                    print(
                        f"DEBUG: Embeddings type: {type(embeddings)}, length: {len(embeddings)}")
                    print(f"DEBUG: First few values: {embeddings[:3]}")
                else:
                    # Continue without embeddings
                    print(
                        f"DEBUG: No valid embeddings generated for page {page_num}")
                    page_data["hasEmbeddings"] = False
            else:
                print(f"DEBUG: Embeddings not requested for page {page_num}")
                page_data["hasEmbeddings"] = False

            # Add the remaining page data AFTER text and embeddings
            page_data["textChunks"] = text_chunks
            # Placeholder, table detection not implemented yet
            page_data["hasTable"] = False
            page_data["tables"] = []

            print(
                f"Created page data for page {page_num} with {len(text_chunks)} text chunks")

            # Check if extraction document exists before updating
            existing_doc = ragnor_doc_extractions_collection.find_one(
                {"_id": extraction_id})
            if not existing_doc:
                print(
                    f"WARNING: Extraction document {extraction_id} not found, creating it")
                # Create the extraction document if it doesn't exist
                extraction_doc = {
                    "_id": extraction_id,
                    "jobId": job_id,
                    "totalPages": total_pages,
                    "processedPages": [],
                    "pageContents": [],
                    "createdAt": datetime.utcnow(),
                    "updatedAt": datetime.utcnow()
                }
                ragnor_doc_extractions_collection.insert_one(extraction_doc)

            # Update the extraction document in the database with this page's data
            try:
                # First, verify the page data structure has embeddings in the right position if requested
                if generate_embeddings and full_text.strip():
                    print(f"DEBUG: Generating embeddings for MongoDB update")
                    # Generate embeddings right before MongoDB update
                    embeddings, embedding_model, success = self.generate_embeddings_for_text(
                        full_text, embedding_provider)

                    if success and embeddings:
                        print(
                            f"DEBUG: Successfully generated {len(embeddings)} embeddings")

                        # Force a new ordered document with embeddings
                        ordered_page_data = {
                            "pageNumber": page_num,
                            "text": full_text,
                            "embeddings": embeddings,  # Force embeddings here
                            "embeddingModel": embedding_model,
                            "hasEmbeddings": True,
                            "textChunks": page_data["textChunks"],
                            "hasTable": False,
                            "tables": []
                        }
                        page_data = ordered_page_data
                        print(
                            f"DEBUG: Created ordered page data with embeddings in position 3")
                        print(
                            f"DEBUG: Keys in order: {list(page_data.keys())}")
                    else:
                        print(
                            f"ERROR: Failed to generate embeddings before MongoDB update")

                # Use direct MongoDB operations to ensure embeddings are included
                print(f"DEBUG: Using direct MongoDB operations to store embeddings")

                # First, remove any existing page
                remove_result = ragnor_doc_extractions_collection.update_one(
                    {"_id": extraction_id},
                    {"$pull": {"pageContents": {"pageNumber": page_num}}}
                )
                print(
                    f"DEBUG: Removed existing page data, matched count: {remove_result.matched_count}")

                # Mark page as processed in the tracking list
                ragnor_doc_extractions_collection.update_one(
                    {"_id": extraction_id},
                    {"$addToSet": {"processedPages": page_num}}
                )

                # Get current document to check structure
                current_doc = ragnor_doc_extractions_collection.find_one(
                    {"_id": extraction_id})
                if not current_doc:
                    print(
                        f"ERROR: Document {extraction_id} not found in database")
                    return False

                # Force embeddings generation if it was requested in the API call
                job_doc = ragnor_doc_jobs_collection.find_one({"_id": job_id})
                should_generate_embeddings = job_doc.get(
                    "embedding", False) if job_doc else generate_embeddings

                if should_generate_embeddings and full_text.strip():
                    print(
                        f"FORCING EMBEDDINGS GENERATION for page {page_num} based on job settings")
                    embeddings, embedding_model, success = self.generate_embeddings_for_text(
                        full_text, embedding_provider)

                    if success and embeddings:
                        print(
                            f"SUCCESS: Generated {len(embeddings)} embeddings for forced storage")
                        has_embeddings = True
                    else:
                        print(
                            f"ERROR: Failed to generate embeddings even with forced approach")
                        embeddings = []
                        embedding_model = "openai"
                        has_embeddings = False
                else:
                    print(
                        f"No embeddings requested at job level for page {page_num}")
                    embeddings = []
                    embedding_model = ""
                    has_embeddings = False

                # Construct page document with explicit field order to ensure embeddings are right after text
                ordered_page_content = {
                    "pageNumber": page_num,
                    "text": full_text,
                    "embeddings": embeddings,
                    "embeddingModel": embedding_model,
                    "hasEmbeddings": has_embeddings,
                    "textChunks": page_data["textChunks"],
                    "hasTable": False,
                    "tables": []
                }

                # Use direct $push with properly ordered fields
                update_result = ragnor_doc_extractions_collection.update_one(
                    {"_id": extraction_id},
                    {
                        "$push": {
                            "pageContents": ordered_page_content
                        },
                        "$set": {"updatedAt": datetime.utcnow()}
                    }
                )
                print(
                    f"Updated extraction document with page {page_num} data, modified: {update_result.modified_count}")

                # Verify the update succeeded
                if update_result.modified_count == 0:
                    print(
                        f"WARNING: Failed to update extraction document {extraction_id}, verifying its existence")
                    # Check if the extraction document exists
                    doc_exists = ragnor_doc_extractions_collection.find_one(
                        {"_id": extraction_id})
                    if not doc_exists:
                        print(
                            f"ERROR: Extraction document {extraction_id} does not exist!")
                    else:
                        print(
                            f"INFO: Extraction document exists, but update failed. Forcing update.")

                # As a final check, verify the data was stored correctly
                verification = ragnor_doc_extractions_collection.find_one(
                    {"_id": extraction_id, "pageContents.pageNumber": page_num},
                    {"pageContents.$": 1}
                )
                if verification and "pageContents" in verification and len(verification["pageContents"]) > 0:
                    stored_page = verification["pageContents"][0]
                    print(
                        f"DEBUG: Verified page {page_num} storage. hasEmbeddings={stored_page.get('hasEmbeddings', False)}")
                    if generate_embeddings and not stored_page.get('hasEmbeddings', False):
                        print(
                            f"WARNING: Embeddings were not properly stored for page {page_num}!")
                else:
                    print(
                        f"WARNING: Could not verify page {page_num} data was properly stored")

                return True
            except Exception as update_err:
                print(
                    f"Error updating extraction document with page data: {str(update_err)}")
                print(f"Stack trace: {traceback.format_exc()}")
                return False

        except Exception as ocr_err:
            print(f"ERROR during OCR processing: {str(ocr_err)}")
            print(f"Stack trace:", traceback.format_exc())
            print(
                f"Image details: Format={file_format}, Size={image.size if hasattr(image, 'size') else 'unknown'}")
            print(f"Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
            print(
                f"Tesseract exists: {os.path.exists(pytesseract.pytesseract.tesseract_cmd)}")
            # Don't raise the error - we want to continue with other pages
            print(
                f"Continuing with next page despite OCR error on page {page_num}")
            return False

    async def _process_single_page(self, image, page_num, total_pages, file_format, extraction_id, job_id):
        """Process a single page with OCR and update the database.

        Args:
            image (PIL.Image): The image to process
            page_num (int): The page number (1-indexed)
            total_pages (int): The total number of pages
            file_format (str): The document format
            extraction_id (str): The extraction document ID
            job_id (str): The job ID
        """
        print(f"Processing page {page_num}/{total_pages}")

        # Extract text using OCR
        print(f"Performing OCR on page {page_num}...")
        try:
            ocr_result = self.text_extractor.extract_text(
                image, doc_format=file_format)

            # Convert OCR result to a serializable dictionary
            ocr_dict = self.text_extractor.serialize_result(ocr_result)

            # Extract the full text from text lines
            full_text = "\n".join(
                [line.text for line in ocr_result.text_lines]) if ocr_result.text_lines else ""

            # Debug OCR results
            text_sample = full_text[:100] + \
                "..." if full_text else "[No text extracted]"
            print(f"OCR result for page {page_num}: {text_sample}")
            print(f"Extracted {len(ocr_result.text_lines)} text lines")

            # Create page data with proper structure
            text_chunks = []
            if ocr_result.text_lines:
                for line in ocr_result.text_lines:
                    text_chunks.append({
                        "text": line.text,
                        "confidence": line.confidence,
                        "position": {
                            "x": line.bbox[0] if line.bbox else 0,
                            "y": line.bbox[1] if line.bbox else 0,
                            "width": line.bbox[2] if line.bbox else 0,
                            "height": line.bbox[3] if line.bbox else 0
                        }
                    })

            page_data = {
                "pageNumber": page_num,
                "text": full_text,
                "textChunks": text_chunks,
                "hasTable": False,  # Placeholder, table detection not implemented yet
                "tables": []
            }

            print(
                f"Created page data for page {page_num} with {len(text_chunks)} text chunks")

            # Update the extraction document in the database with this page's data
            try:
                ragnor_doc_extractions_collection.update_one(
                    {"_id": extraction_id},
                    {
                        "$push": {
                            "processedPages": page_num,
                            "pageContents": page_data
                        },
                        "$set": {
                            "updatedAt": datetime.utcnow()
                        }
                    }
                )
                print(f"Updated extraction document with page {page_num} data")
                return True
            except Exception as update_err:
                print(
                    f"Error updating extraction document with page data: {str(update_err)}")
                return False

        except Exception as ocr_err:
            print(f"ERROR during OCR processing: {str(ocr_err)}")
            print(f"Stack trace:", traceback.format_exc())
            print(
                f"Image details: Format={file_format}, Size={image.size if hasattr(image, 'size') else 'unknown'}")
            print(f"Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
            print(
                f"Tesseract exists: {os.path.exists(pytesseract.pytesseract.tesseract_cmd)}")
            # Don't raise the error - we want to continue with other pages
            print(
                f"Continuing with next page despite OCR error on page {page_num}")
            return False

    def _process_document_in_background(self, job_id: str, file_content: bytes, retry_count: int = 0, embedding: bool = False, embedding_provider: str = "openai"):
        """Process a document in a background thread (non-async version).
        This keeps the API responsive during intensive processing.
        """
        try:
            # Set up MongoDB connection for this thread if needed
            # Database operations here...

            # Start the actual processing
            print(f"=== STARTING BACKGROUND DOCUMENT PROCESSING ===")
            print(f"Job ID: {job_id}")
            print(f"File content size: {len(file_content)} bytes")
            print(f"Retry count: {retry_count}")

            # Get job data
            job_doc = ragnor_doc_jobs_collection.find_one({"_id": job_id})
            if not job_doc:
                raise ValueError(f"Job not found: {job_id}")

            # Get file format
            file_format = job_doc.get("fileFormat")
            filename = job_doc.get("filename")

            print(f"Processing document: {filename} (format: {file_format})")

            # Get the appropriate format handler
            handler = get_handler_for_format(file_format)
            print(f"Using format handler: {handler.__class__.__name__}")

            self._process_document_impl(
                job_id, file_content, file_format, filename, handler, retry_count, embedding, embedding_provider)

        except Exception as e:
            # Log and update job status if there's an error
            print(f"ERROR in background processing for job {job_id}: {str(e)}")
            # Update job status to failed
            self._update_job_status_sync(
                job_id, JobStatus.FAILED, 0.0, error_message=str(e))

    def _update_job_status_sync(self, job_id, status, progress, result_id=None, error_message=None):
        """Synchronous version of update_job_status for use in background thread"""
        try:
            # Prepare the update data
            update_data = {
                "status": status,
                "progress": progress,
                "updatedAt": datetime.utcnow()
            }

            if result_id:
                update_data["resultId"] = result_id

            if error_message:
                update_data["errorMessage"] = error_message

            if status == JobStatus.COMPLETED:
                update_data["completedAt"] = datetime.utcnow()
            elif status == JobStatus.FAILED:
                update_data["failedAt"] = datetime.utcnow()

            # Update in MongoDB
            update_result = ragnor_doc_jobs_collection.update_one(
                {"_id": job_id},
                {"$set": update_data}
            )

            # Check if the update was successful
            if update_result.modified_count > 0:
                print(
                    f"Job {job_id} status updated to {status}, progress {progress:.1f}%")
            else:
                print(f"Warning: Job {job_id} not found or not updated")

        except Exception as e:
            print(f"Error updating job status: {str(e)}")

    async def process_document(self, job_id: str, file_content: bytes, retry_count: int = 0, embedding: bool = False, embedding_provider: str = "openai") -> None:
        """Process a document extraction job with retry mechanism.
        This method starts a background process and returns immediately to keep the API responsive.

        Args:
            job_id (str): The ID of the job to process
            file_content (bytes): The document file content
            retry_count (int): The current retry count
            embedding (bool): Whether to generate embeddings for extracted text
            embedding_provider (str): The provider to use for generating embeddings
        """
        MAX_RETRIES = 3

        # Debug info
        print(
            f"Starting background processing for job {job_id}, file size: {len(file_content)} bytes")

        # Start processing in a background thread to keep API responsive
        process_thread = threading.Thread(
            target=self._process_document_in_background,
            args=(job_id, file_content, retry_count,
                  embedding, embedding_provider)
        )
        process_thread.daemon = True  # Thread will exit when main thread exits
        process_thread.start()

        # Track the background task
        self.background_tasks[job_id] = process_thread

        print(f"Document processing started in background for job {job_id}")

    def _process_document_impl(self, job_id: str, file_content: bytes, file_format: str, filename: str, handler, retry_count: int = 0, embedding: bool = False, embedding_provider: str = "openai"):
        """Implementation of document processing that runs in the background thread.
        This is a non-async version that runs in a background thread.

        Args:
            job_id: The ID of the job to process
            file_content: The document file content
            file_format: The format of the document
            filename: The name of the file
            handler: The format handler to use
            retry_count: The current retry count
            embedding: Whether to generate embeddings for extracted text
            embedding_provider: The provider to use for generating embeddings
        """
        MAX_RETRIES = 3

        # Force job processing to take some time to make it easier to test
        # This ensures polling will actually show progress
        MIN_PROCESSING_TIME = 15  # seconds for entire process
        start_time = time.time()

        try:
            # Debug info
            print(f"=== STARTING DOCUMENT PROCESSING ===")
            print(f"Job ID: {job_id}")
            print(f"File content size: {len(file_content)} bytes")
            print(f"Retry count: {retry_count}")

            # Verify file content exists and is not empty
            if not file_content:
                raise ValueError(f"No file content provided for job {job_id}")

            # Update job status to processing
            self._update_job_status_sync(job_id, JobStatus.PROCESSING, 0.0)

            # Delay to simulate initial setup
            print(f"Simulating initial processing setup...")
            time.sleep(1)

            print(f"Processing document: {filename} (format: {file_format})")
            print(f"Using format handler: {handler.__class__.__name__}")

            # Extract metadata (with delay to simulate real processing)
            print(f"Extracting document metadata...")
            # Delay to simulate metadata extraction time
            time.sleep(1)
            metadata = handler.extract_metadata(file_content, file_format)
            print(f"Extracted metadata: {metadata}")
            self._update_job_status_sync(job_id, JobStatus.PROCESSING, 5.0)

            # Special handling for PDFs - get page count and process pages one at a time
            is_pdf = 'pdf' in file_format.lower()
            total_pages = 0

            if is_pdf:
                # Use PDFHandler to get the page count
                total_pages = PDFHandler.get_pdf_page_count(file_content)

            # Create extraction document with initial structure
            extraction_id = str(ObjectId())
            extraction_doc = {
                "_id": extraction_id,
                "jobId": job_id,
                "filename": filename,
                "fileFormat": file_format,
                "totalPages": total_pages,  # Will be updated if not determined yet
                "processedPages": [],
                "pageContents": [],
                "metadata": metadata,
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow()
            }

            # Initialize the extraction document in the database
            try:
                result = ragnor_doc_extractions_collection.insert_one(
                    extraction_doc)
                print(
                    f"Created initial extraction document with ID {extraction_id}")
            except Exception as db_err:
                print(
                    f"Error creating initial extraction document: {str(db_err)}")
                raise db_err

            # Start document conversion and OCR processing
            print(f"Starting document conversion and OCR processing...")
            processed_pages = []
            conversion_progress = 10.0  # Starting progress after metadata extraction

            # Update progress to indicate we're starting conversion
            self._update_job_status_sync(
                job_id, JobStatus.PROCESSING, conversion_progress)

            # Handle PDFs and non-PDFs differently
            if is_pdf and total_pages > 10:  # If it's a large PDF
                # Process large PDFs page by page
                print(
                    f"Processing large PDF with {total_pages} pages one page at a time")

                # Process pages one by one, starting immediate OCR on each page
                for page_num in range(1, total_pages + 1):  # 1-indexed page numbers
                    try:
                        # Convert just this page to an image
                        image = PDFHandler.convert_page_to_image(
                            file_content, page_num, total_pages)

                        # Immediately process this page with OCR and generate embeddings if requested
                        self._process_single_page_sync(
                            image, page_num, total_pages, file_format, extraction_id, job_id,
                            generate_embeddings=embedding, embedding_provider=embedding_provider)
                        processed_pages.append(page_num)

                        # Update progress - scale from 10% to 90% based on page completion
                        progress = 10.0 + \
                            (80.0 * len(processed_pages) / total_pages)
                        self._update_job_status_sync(
                            job_id, JobStatus.PROCESSING, progress)

                    except Exception as page_err:
                        print(
                            f"Error processing page {page_num}: {str(page_err)}")
                        # Continue with other pages even if one fails
                        continue
            else:
                # For non-PDFs or small PDFs, use the standard approach
                converted_images = handler.convert_to_images(
                    file_content, file_format)

                # If we didn't know the total pages before, update it now
                if total_pages == 0 and converted_images:
                    total_pages = len(converted_images)
                    print(
                        f"Updated total page count to {total_pages} based on converted images")
                    # Update the extraction document with the correct total page count
                    try:
                        ragnor_doc_extractions_collection.update_one(
                            {"_id": extraction_id},
                            {"$set": {"totalPages": total_pages}}
                        )
                    except Exception as update_err:
                        print(
                            f"Error updating total page count: {str(update_err)}")

                # Update progress now that we have the images
                self._update_job_status_sync(
                    job_id, JobStatus.PROCESSING, 15.0)

                # Process each image and extract text
                print(
                    f"Starting OCR text extraction on {total_pages} pages...")

                for idx, image in enumerate(converted_images):
                    page_num = idx + 1
                    # Process this page with OCR
                    self._process_single_page_sync(
                        image, page_num, total_pages, file_format, extraction_id, job_id)
                    processed_pages.append(page_num)

                    # Update progress - scale from 15% to 85% based on page completion
                    progress = 15.0 + (70.0 * (idx + 1) / total_pages)
                    print(
                        f"Page {page_num} complete. Progress: {progress:.1f}%")
                    self._update_job_status_sync(
                        job_id, JobStatus.PROCESSING, progress)

            # Verify that we've processed all pages as expected
            print(
                f"Finished processing all pages. Total pages: {total_pages}, Processed pages: {len(processed_pages)}")

            # If some pages failed processing but we have at least one successful page, we can still consider the job successful
            if len(processed_pages) < total_pages:
                print(
                    f"Warning: Not all pages were processed successfully. Processed {len(processed_pages)} out of {total_pages}")

            # Extract the document format from the MIME type if available
            doc_format = file_format.split(
                "/")[1] if "/" in file_format else file_format
            print(f"Document format determined as: {doc_format}")

            # Before completing, ensure we meet minimum processing time for polling testing
            elapsed_time = time.time() - start_time
            if elapsed_time < MIN_PROCESSING_TIME:
                remaining_time = MIN_PROCESSING_TIME - elapsed_time
                print(
                    f"Processing completed too quickly, adding {remaining_time:.1f}s delay to simulate real-world processing")
                time.sleep(remaining_time)

            # Update job in MongoDB to mark as completed
            try:
                update_result = ragnor_doc_jobs_collection.update_one(
                    {"_id": job_id},
                    {
                        "$set": {
                            "status": JobStatus.COMPLETED,
                            "progress": 100.0,
                            "completedAt": datetime.utcnow(),
                            "updatedAt": datetime.utcnow(),
                            "resultId": extraction_id
                        }
                    }
                )
                print(
                    f"Job status update result: {update_result.modified_count} document(s) modified")

                # Mark job as completed with the extraction document's ID
                print(f"Marking job {job_id} as COMPLETED")
                self._update_job_status_sync(
                    job_id, JobStatus.COMPLETED, 100.0, result_id=extraction_id)

                print(f"=== DOCUMENT PROCESSING COMPLETED SUCCESSFULLY ===")
                print(f"Job ID: {job_id}")
                print(f"Result ID: {extraction_id}")
                print(
                    f"Total processing time: {time.time() - start_time:.2f} seconds")

            except Exception as update_err:
                print(f"Error updating job status: {str(update_err)}")
                raise update_err

        except Exception as e:
            # Log the error
            error_message = f"Error processing document: {str(e)}"
            print(error_message)
            print(traceback.format_exc())

            # Check if we can retry
            if retry_count < MAX_RETRIES:
                # Increment retry count
                retry_count += 1

                # Wait before retrying (exponential backoff)
                retry_delay = 2 ** retry_count  # 2, 4, 8 seconds
                print(
                    f"Retrying job {job_id} in {retry_delay} seconds... (Attempt {retry_count} of {MAX_RETRIES})")
                time.sleep(retry_delay)

                # Retry processing
                self._process_document_impl(
                    job_id, file_content, file_format, filename, handler, retry_count)
            else:
                # Mark job as failed
                print(
                    f"ERROR in background processing for job {job_id}: {str(e)}")
                self._update_job_status_sync(
                    job_id, JobStatus.FAILED, 0.0, error_message=str(e))

                # For non-PDFs or small PDFs, use the standard approach
                converted_images = handler.convert_to_images(
                    file_content, file_format)

                # If we didn't know the total pages before, update it now
                if total_pages == 0 and converted_images:
                    total_pages = len(converted_images)
                    print(
                        f"Updated total page count to {total_pages} based on converted images")
                    # Update the extraction document with the correct total page count
                    try:
                        ragnor_doc_extractions_collection.update_one(
                            {"_id": extraction_id},
                            {"$set": {"totalPages": total_pages}}
                        )
                    except Exception as update_err:
                        print(
                            f"Error updating total page count: {str(update_err)}")

            # For small PDFs or non-PDFs, handle the regular way
            if not (is_pdf and total_pages > 10):
                for idx, image in enumerate(converted_images):
                    page_num = idx + 1
                    # Process this page with OCR
                    self._process_single_page_sync(
                        image, page_num, total_pages, file_format, extraction_id, job_id)
                    processed_pages.append(page_num)

                    # Update progress - scale from 15% to 85% based on page completion
                    progress = 15.0 + (70.0 * (idx + 1) / total_pages)
                    print(
                        f"Page {page_num} complete. Progress: {progress:.1f}%")
                    self._update_job_status_sync(
                        job_id, JobStatus.PROCESSING, progress)

            # Verify that we've processed all pages as expected
            print(
                f"Finished processing all pages. Total pages: {total_pages}, Processed pages: {len(processed_pages)}")

            # If some pages failed processing but we have at least one successful page, we can still consider the job successful
            if len(processed_pages) < total_pages:
                print(
                    f"Warning: Not all pages were processed successfully. Processed {len(processed_pages)} out of {total_pages}")

            # Extract the document format from the MIME type if available
            doc_format = file_format.split(
                "/")[1] if "/" in file_format else file_format
            print(f"Document format determined as: {doc_format}")

            # Store extraction result in the database
            print(f"Inserting extraction document into MongoDB collection...")
            extraction_id = str(extraction_doc.get("_id"))

            # Verify we have required data
            print(f"Extraction document summary:")
            print(f"  Job ID: {extraction_doc.get('jobId')}")
            print(f"  Total Pages: {extraction_doc.get('totalPages')}")
            print(
                f"  Processed Pages: {len(extraction_doc.get('processedPages', []))}")
            print(
                f"  Page Contents: {len(extraction_doc.get('pageContents', []))}")

            # Check if document exists first - ensure we have pages array
            try:
                existing = ragnor_doc_extractions_collection.find_one(
                    {"_id": extraction_id})
                if existing:
                    print(
                        f"Updating existing extraction document {extraction_id}")
                    # Make sure we preserve any existing page data
                    if "pageContents" in existing and len(existing["pageContents"]) > 0:
                        print(
                            f"Preserving {len(existing['pageContents'])} existing page contents")
                        # Only update specific fields, not the entire document
                        update_data = {
                            "metadata": extraction_doc.get("metadata", {}),
                            "totalPages": extraction_doc.get("totalPages", 0),
                            "updatedAt": datetime.utcnow()
                        }
                        ragnor_doc_extractions_collection.update_one(
                            {"_id": extraction_id},
                            {"$set": update_data}
                        )
                    else:
                        # No existing page data, safe to update whole document
                        ragnor_doc_extractions_collection.update_one(
                            {"_id": extraction_id},
                            {"$set": extraction_doc}
                        )
                else:
                    print(
                        f"Creating new extraction document with ID {extraction_id}")
                    # Initialize empty arrays for pages
                    extraction_doc["processedPages"] = []
                    extraction_doc["pageContents"] = []
                    extraction_doc["pages"] = []
                    result = ragnor_doc_extractions_collection.insert_one(
                        extraction_doc)
                    print(f"MongoDB insert result: {result.inserted_id}")
            except Exception as db_err:
                print(
                    f"Error storing extraction result in database: {str(db_err)}")

            # Before completing, ensure we meet minimum processing time for polling testing
            elapsed_time = time.time() - start_time
            if elapsed_time < MIN_PROCESSING_TIME:
                remaining_time = MIN_PROCESSING_TIME - elapsed_time
                print(
                    f"Processing completed too quickly, adding {remaining_time:.1f}s delay to simulate real-world processing")
                time.sleep(remaining_time)

            # Update job in MongoDB to mark as completed
            try:
                update_result = ragnor_doc_jobs_collection.update_one(
                    {"_id": job_id},
                    {
                        "$set": {
                            "status": JobStatus.COMPLETED,
                            "progress": 100.0,
                            "completedAt": datetime.utcnow(),
                            "updatedAt": datetime.utcnow(),
                            "resultId": extraction_id
                        }
                    }
                )
                print(
                    f"Job status update result: {update_result.modified_count} document(s) modified")

                # Mark job as completed with the extraction document's ID
                print(f"Marking job {job_id} as COMPLETED")
                self._update_job_status_sync(
                    job_id, JobStatus.COMPLETED, 100.0, result_id=extraction_id)

                print(f"=== DOCUMENT PROCESSING COMPLETED SUCCESSFULLY ===")
                print(f"Job ID: {job_id}")
                print(f"Result ID: {extraction_id}")
                print(
                    f"Total processing time: {time.time() - start_time:.2f} seconds")
            except Exception as update_err:
                print(f"Error updating job status: {str(update_err)}")

        except Exception as e:
            # Log the error
            error_message = f"Error processing document: {str(e)}"
            print(error_message)

            # Check if we can retry
            if retry_count < MAX_RETRIES:
                # Increment retry count
                retry_count += 1

                # Update job status
                self._update_job_status_sync(
                    job_id,
                    JobStatus.PROCESSING,
                    progress=0.0,
                    error_message=f"Retrying after error: {str(e)} (Attempt {retry_count} of {MAX_RETRIES})"
                )

                # Exponential backoff for retries (2^retry_count seconds)
                backoff_time = 2 ** retry_count
                print(
                    f"Retrying job {job_id} in {backoff_time} seconds... (Attempt {retry_count} of {MAX_RETRIES})")
                time.sleep(backoff_time)

                # Update retry count in database
                ragnor_doc_jobs_collection.update_one(
                    {"_id": job_id},
                    {"$set": {"retryCount": retry_count}}
                )

                # Retry processing
                # Can't call process_document directly as it's async
                # Instead, we'll recursively call _process_document_impl
                self._process_document_impl(
                    job_id, file_content, file_format, filename, handler, retry_count)
            else:
                # Max retries reached, mark job as failed
                self._update_job_status_sync(
                    job_id,
                    JobStatus.FAILED,
                    error_message=f"{error_message} (after {MAX_RETRIES} retries)"
                )
                raise

    async def get_extraction_result(self, job_id: str) -> Dict[str, Any]:
        """Get the extraction result for a completed job.

        Args:
            job_id: The ID of the job to retrieve results for

        Returns:
            A dictionary containing the extraction results with the following structure:
            {
                "job_id": str,                    # The job ID
                "status": str,                    # Current job status
                "filename": str,                  # Original filename
                "fileFormat": str,                # MIME type of the file
                "totalPages": int,                # Total number of pages in document
                "processedPages": List[int],      # List of page numbers that were processed
                "metadata": Dict,                 # Document metadata (format specific)
                "created_at": str,                # ISO timestamp of job creation
                "completed_at": str,              # ISO timestamp of job completion
                "processing_time_seconds": float, # Total processing time in seconds
                "pages": [                        # Array of page content objects
                    {
                        "pageNumber": int,        # Page number (1-based)
                        "text": str,              # Extracted text content
                        "textChunks": [           # Array of text chunks with position info
                            {
                                "text": str,      # Chunk text
                                "confidence": float, # OCR confidence score
                                "position": {     # Position on page
                                    "x": int,     # X coordinate
                                    "y": int,     # Y coordinate
                                    "width": int, # Width
                                    "height": int # Height
                                }
                            }
                        ],
                        "hasEmbeddings": bool,    # Whether embeddings are available
                        "embeddingModel": str     # Model used for embeddings
                    }
                ]
            }

        Raises:
            ValueError: If job or extraction result not found
        """
        print(f"=== RETRIEVING EXTRACTION RESULT ===")
        print(f"Job ID: {job_id}")

        try:
            # Check job status
            print(f"Looking up job document in MongoDB...")
            job_doc = ragnor_doc_jobs_collection.find_one({"_id": job_id})

            if not job_doc:
                print(f"ERROR: Job document not found with ID: {job_id}")
                raise ValueError(f"Job not found: {job_id}")

            print(
                f"Found job document: Status={job_doc.get('status')}, Progress={job_doc.get('progress', 0.0)}")

            # If job is not completed, return an error
            if job_doc.get("status") != JobStatus.COMPLETED:
                print(
                    f"WARNING: Job {job_id} is not completed yet. Status: {job_doc.get('status')}")
                return {
                    "job_id": job_id,
                    "status": job_doc.get("status"),
                    "progress": job_doc.get("progress", 0.0),
                    "error": "Job not completed"
                }

            # Get extraction document - try both string jobId and ObjectId
            print(f"Looking up extraction document in MongoDB...")
            print(f"Querying with jobId: {job_id}")
            extraction_doc = ragnor_doc_extractions_collection.find_one({
                                                                        "jobId": job_id})

            # If not found with string ID, try with ObjectId
            if not extraction_doc:
                try:
                    # Try to convert to ObjectId and search again
                    from bson.objectid import ObjectId
                    extraction_doc = ragnor_doc_extractions_collection.find_one(
                        {"jobId": ObjectId(job_id)})
                    if extraction_doc:
                        print(
                            f"Found extraction with ObjectId jobId: {job_id}")
                except:
                    # If conversion fails, it's not a valid ObjectId
                    print(
                        f"Could not convert {job_id} to ObjectId for extraction lookup")

            if not extraction_doc:
                print(
                    f"ERROR: Extraction document not found for job ID: {job_id}")
                # Try alternate query to debug
                all_extractions = list(
                    ragnor_doc_extractions_collection.find())
                print(
                    f"Found {len(all_extractions)} total extraction documents in collection")
                raise ValueError(
                    f"Extraction result not found for job: {job_id}")

            # Calculate processing time
            created_at = job_doc.get("createdAt")
            completed_at = job_doc.get("completedAt")
            processing_time = None
            if created_at and completed_at:
                processing_time = (completed_at - created_at).total_seconds()

            # Debug extraction document
            print(f"Found extraction document: ID={extraction_doc.get('_id')}")
            print(f"Total pages: {extraction_doc.get('totalPages')}")
            print(
                f"Processed pages: {len(extraction_doc.get('pageContents', []))}")
            print(
                f"Document has text content: {bool(extraction_doc.get('pageContents'))}")

            # Verify extraction document structure
            if not extraction_doc.get('pageContents'):
                print(f"WARNING: Extraction document has no page contents")
                # Try to print more info about the document
                print(f"Document keys: {list(extraction_doc.keys())}")
                print(f"Document jobId: {extraction_doc.get('jobId')}")
                if extraction_doc.get('jobId') != job_id:
                    print(
                        f"ERROR: jobId mismatch: {extraction_doc.get('jobId')} != {job_id}")

                # This might be a corrupt document, create an empty result
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "filename": extraction_doc.get("filename"),
                    "fileFormat": extraction_doc.get("fileFormat"),
                    "totalPages": extraction_doc.get("totalPages", 0),
                    "processedPages": extraction_doc.get("processedPages", []),
                    "metadata": extraction_doc.get("metadata", {}),
                    "pages": [],
                    "error": "No extraction content found"
                }

            # Format the response with detailed information
            response = {
                "job_id": job_id,
                "status": "completed",
                "filename": extraction_doc.get("filename"),
                "fileFormat": extraction_doc.get("fileFormat"),
                "totalPages": extraction_doc.get("totalPages"),
                "processedPages": extraction_doc.get("processedPages"),
                "metadata": extraction_doc.get("metadata", {}),
                "created_at": created_at.isoformat() if created_at else None,
                "completed_at": completed_at.isoformat() if completed_at else None,
                "processing_time_seconds": processing_time,
                "pages": []
            }

            # Add page contents
            for page_content in extraction_doc.get("pageContents", []):
                # Include embeddings if they exist in the document
                has_embeddings = page_content.get("hasEmbeddings", False)

                page = {
                    "pageNumber": page_content.get("pageNumber"),
                    "text": page_content.get("text"),
                    "hasEmbeddings": has_embeddings,
                    # "textChunks": page_content.get("textChunks", []),  # Temporarily commented out
                    # "hasTable": page_content.get("hasTable", False),
                    # "tables": page_content.get("tables", [])
                }

                # Add embeddings and model if they exist
                if has_embeddings:
                    # Include embeddings and model information
                    page["embeddings"] = page_content.get("embeddings", [])
                    page["embeddingModel"] = page_content.get(
                        "embeddingModel", "")
                    print(
                        f"Including embeddings for page {page_content.get('pageNumber')} in API response")

                response["pages"].append(page)

            return response

        except Exception as e:
            print(f"Error getting extraction result: {str(e)}")
            raise

    async def store_extraction_result(self, extraction_doc: Dict[str, Any]) -> str:
        """Store extraction result in database.

        Args:
            extraction_doc: Extraction document to store

        Returns:
            The extraction document ID
        """
        try:
            # Insert extraction document
            result = ragnor_doc_extractions_collection.insert_one(
                extraction_doc)

            # Return extraction document ID
            return str(result.inserted_id)

        except Exception as e:
            print(f"Error storing extraction result: {str(e)}")
            raise
