"""
Core document processing logic for Ragnor document extraction API.
"""
import os
import time
import random
import asyncio
from typing import Dict, Any
from bson import ObjectId
from datetime import datetime


from db.ragnor_db_models import JobStatus, RagnorDocJob, RagnorDocExtraction, PageContent, TextChunk
from db.db import ragnor_doc_jobs_collection, ragnor_doc_extractions_collection
from utils.ragnor_format_handlers import get_handler_for_format, FormatHandler
from utils.ragnor_text_extractor import RagnorTextExtractor, OCRResult
from utils.ragnor_embedding_generator import get_embedding_generator

# Initialize the text extractor
text_extractor = RagnorTextExtractor()


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
            # Find the job document
            job_doc = ragnor_doc_jobs_collection.find_one({"_id": job_id})

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

    async def process_document(self, job_id: str, file_content: bytes, retry_count: int = 0) -> None:
        """Process a document extraction job with retry mechanism.

        Args:
            job_id (str): The ID of the job to process
            file_content (bytes): The document file content
            retry_count (int): The current retry count
        """
        MAX_RETRIES = 3

        # Debug info
        print(f"=== STARTING DOCUMENT PROCESSING ===")
        print(f"Job ID: {job_id}")
        print(f"File content size: {len(file_content)} bytes")
        print(f"Retry count: {retry_count}")

        # Force job processing to take some time to make it easier to test
        # This ensures polling will actually show progress
        MIN_PROCESSING_TIME = 15  # seconds for entire process

        start_time = time.time()
        try:
            # Verify file content exists and is not empty
            if not file_content:
                raise ValueError(f"No file content provided for job {job_id}")

            # Update job status to processing
            await self.update_job_status(job_id, JobStatus.PROCESSING, 0.0)

            # Delay to simulate initial setup
            print(f"Simulating initial processing setup...")
            await asyncio.sleep(1)

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

            # Extract metadata (with delay to simulate real processing)
            print(f"Extracting document metadata...")
            # Delay to simulate metadata extraction time
            await asyncio.sleep(2)
            metadata = handler.extract_metadata(file_content, file_format)
            print(f"Extracted metadata: {metadata}")
            await self.update_job_status(job_id, JobStatus.PROCESSING, 5.0)

            # Convert document to images (this is time-consuming)
            print(f"Converting document to images for OCR processing...")
            # Simulate the time it takes to convert document to images
            await asyncio.sleep(3)
            images = handler.convert_to_images(file_content, file_format)
            print(f"Successfully converted document to {len(images)} images")

            # Save some debug information about the images
            if images:
                print(
                    f"First image details: Size={images[0].size}, Mode={images[0].mode}")

                # Save a sample image for debugging
                debug_dir = os.path.join(
                    os.path.dirname(__file__), "debug_images")
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(
                    debug_dir, f"job_{job_id}_sample.png")
                images[0].save(debug_path)
                print(f"Saved sample image to {debug_path}")
            else:
                print("WARNING: No images were extracted from the document!")

            # Update progress
            await self.update_job_status(job_id, JobStatus.PROCESSING, 15.0)

            # Create extraction document with proper structure
            extraction_doc = {
                "_id": str(ObjectId()),
                "jobId": job_id,
                "filename": filename,
                "fileFormat": file_format,
                "totalPages": len(images),
                "processedPages": [],
                "pageContents": [],
                "metadata": metadata,
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow()
            }

            # Process each image and extract text
            print(f"Starting OCR text extraction on {len(images)} pages...")
            total_pages = len(images)
            processed_pages = []

            for idx, image in enumerate(images):
                page_num = idx + 1
                print(f"Processing page {page_num}/{total_pages}")

                # OCR processing can take a long time, here we'll add a delay to simulate
                # the actual processing time and ensure our polling works correctly
                # 2-4 seconds per page
                await asyncio.sleep(2 + random.random() * 2)

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
                except Exception as ocr_err:
                    print(f"Error during OCR processing: {str(ocr_err)}")
                    # Create empty OCR result to continue processing
                    ocr_result = OCRResult(
                        text_lines=[], languages=[], image_bbox=None)
                    ocr_dict = {"text_lines": [],
                                "languages": [], "image_bbox": None}
                    full_text = ""
                    print(f"Using empty OCR result due to error")

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

                # Track successfully processed pages
                processed_pages.append(page_num)
                extraction_doc["processedPages"].append(page_num)

                # Add the page to the page contents
                extraction_doc["pageContents"].append(page_data)

                # Update progress - adjust from 15% to 85% progress based on page completion
                progress = 15.0 + (70.0 * (idx + 1) / len(images))
                print(f"Page {page_num} complete. Progress: {progress:.1f}%")
                await self.update_job_status(job_id, JobStatus.PROCESSING, progress)

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

            # Check if document exists first
            try:
                existing = ragnor_doc_extractions_collection.find_one(
                    {"_id": extraction_id})
                if existing:
                    print(
                        f"Updating existing extraction document {extraction_id}")
                    ragnor_doc_extractions_collection.update_one(
                        {"_id": extraction_id},
                        {"$set": extraction_doc}
                    )
                else:
                    print(
                        f"Creating new extraction document with ID {extraction_id}")
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
                await asyncio.sleep(remaining_time)

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
                await self.update_job_status(job_id, JobStatus.COMPLETED, 100.0, result_id=extraction_id)

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
                await self.update_job_status(
                    job_id,
                    JobStatus.PROCESSING,
                    progress=0.0,
                    error_message=f"Retrying after error: {str(e)} (Attempt {retry_count} of {MAX_RETRIES})"
                )

                # Exponential backoff for retries (2^retry_count seconds)
                backoff_time = 2 ** retry_count
                print(
                    f"Retrying job {job_id} in {backoff_time} seconds... (Attempt {retry_count} of {MAX_RETRIES})")
                await asyncio.sleep(backoff_time)

                # Update retry count in database
                ragnor_doc_jobs_collection.update_one(
                    {"_id": job_id},
                    {"$set": {"retryCount": retry_count}}
                )

                # Retry processing
                await self.process_document(job_id, file_content, retry_count)
            else:
                # Max retries reached, mark job as failed
                await self.update_job_status(
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

            # Get extraction document
            print(f"Looking up extraction document in MongoDB...")
            print(f"Querying with jobId: {job_id}")
            extraction_doc = ragnor_doc_extractions_collection.find_one({
                                                                        "jobId": job_id})

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
                page = {
                    "pageNumber": page_content.get("pageNumber"),
                    "text": page_content.get("text"),
                    "textChunks": page_content.get("textChunks", []),
                    "hasTable": page_content.get("hasTable", False),
                    "tables": page_content.get("tables", [])
                }
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
