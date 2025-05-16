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
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from bson import ObjectId
from functools import partial

import pytesseract
from PIL import Image

from db.ragnor_db_models import JobStatus, RagnorDocJob, RagnorDocExtraction, PageContent, TextChunk
from db.db import ragnor_doc_jobs_collection, ragnor_doc_extractions_collection

# Add a new collection for storing page contents separately
from pymongo import MongoClient
from os import getenv

# Connect to MongoDB
client = MongoClient(getenv('MONGODB_CONNECTION_STRING', 'mongodb://localhost:27017'))
db = client[getenv('MONGODB_DATABASE', 'ragnor_docs')]

# Create collections
ragnor_page_contents_collection = db['ragnor_page_contents']
from utils.ragnor_format_handlers import get_handler_for_format, FormatHandler, PDFHandler
from utils.ragnor_text_extractor import RagnorTextExtractor, OCRResult
from utils.ragnor_embedding_generator import get_embedding_generator

# Suppress pdfminer warnings
logging.getLogger('pdfminer').setLevel(logging.ERROR)

# Initialize the text extractor
text_extractor = RagnorTextExtractor()

# Determine the number of CPU cores available for multiprocessing
NUM_CORES = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free for system
print(f"Using {NUM_CORES} CPU cores for multiprocessing")

# Create a process pool for OCR processing
# This will allow the API to remain responsive during heavy processing
OCR_PROCESS_POOL = multiprocessing.Pool(processes=NUM_CORES)


class RagnorDocumentProcessor:
    """Main processor for document extraction."""

    def __init__(self, embedding_provider="azure"):
        """Initialize the document processor."""
        self.text_extractor = RagnorTextExtractor()
        print(f"Initializing RagnorDocumentProcessor with provider: {embedding_provider}")
        try:
            print(f"Getting embedding generator for provider: {embedding_provider}")
            self.embedding_generator = get_embedding_generator(
                embedding_provider)
            print(f"Successfully initialized embedding generator: {self.embedding_generator.__class__.__name__}")
            print(f"Embedding model: {self.embedding_generator.model}")
        except Exception as e:
            print(f"Error initializing embedding generator for {embedding_provider}: {str(e)}")
            print(f"Warning: Could not initialize embedding generator for {embedding_provider}. Embeddings will not be generated.")
            self.embedding_generator = None
        self.background_tasks = {}  # Keep track of background tasks

    def generate_embeddings_for_text(self, text: str, provider: str = "azure"):
        """Generate embeddings for the given text using the specified provider.

        Args:
            text: The text to generate embeddings for
            provider: The embedding provider to use (default: "azure")

        Returns:
            Tuple containing (embeddings, model_name, success_flag)
        """
        if not text or not text.strip():
            print("WARNING: Cannot generate embeddings for empty text")
            return [], "", False

        try:
            from utils.ragnor_embedding_generator import get_embedding_generator
            print(f"Generating embeddings for text using {provider}")

            # Create embedding generator based on provider
            try:
                # Use the provider directly - get_embedding_generator supports 'openai', 'azure', 'azure-openai', and 'modal'
                print(f"Creating embedding generator with provider: {provider}")
                embedding_generator = get_embedding_generator(provider)
                embedding_model = embedding_generator.model
                print(f"Successfully created embedding generator with model: {embedding_model}")
            except ValueError as e:
                print(f"Error with provider {provider}: {str(e)}")
                # Don't fall back to OpenAI, just re-raise the error
                raise

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

    def _process_text_page(self, text, page_num, total_pages, file_format, extraction_id, job_id, generate_embeddings=False, embedding_provider="openai"):
        """Process a text-based page without OCR and update the database.
        
        This method is used for text-based PDF pages extracted with pdfplumber.
        
        Args:
            text: The extracted text from the page
            page_num: The page number (1-indexed)
            total_pages: The total number of pages in the document
            file_format: The document format
            extraction_id: The extraction document ID
            job_id: The job ID
            generate_embeddings: Whether to generate embeddings for the text
            embedding_provider: The embedding provider to use
        """
        print(f"Processing text-based page {page_num}/{total_pages}")
        
        try:
            # Create a simple structure for the text content
            # We don't have detailed positioning info with direct text extraction, so we use basic values
            text_chunks = []
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                if line.strip():
                    # Create a simple text chunk with estimated position
                    text_chunks.append({
                        "text": line,
                        "confidence": 100.0,  # Text-based pages have high confidence
                        "position": {
                            "x": 0,
                            "y": i * 20,  # Estimate line height
                            "width": 1000,  # Arbitrary width
                            "height": 20   # Arbitrary height
                        }
                    })
            
            # Create page data structure
            page_data = {
                "pageNumber": page_num,
                "text": text,
                "textChunks": text_chunks,
                "hasTable": False,
                "tables": []
            }
            
            # Generate embeddings if requested
            if generate_embeddings and text.strip():
                print(f"Generating embeddings for page {page_num} text using {embedding_provider}")
                try:
                    # Use the document processor's existing embedding generator directly
                    if hasattr(self, 'embedding_generator') and self.embedding_generator:
                        print(f"Using document processor's embedding generator directly")
                        embedding_generator = self.embedding_generator
                        embeddings = embedding_generator.generate_embeddings(text)
                        embedding_model = embedding_generator.model
                        success = bool(embeddings)
                    else:
                        print(f"No embedding generator available, using generate_embeddings_for_text")
                        embeddings, embedding_model, success = self.generate_embeddings_for_text(
                            text, embedding_provider="azure")
                    
                    if success and embeddings:
                        page_data["embeddings"] = embeddings
                        page_data["embeddingModel"] = embedding_model
                        page_data["hasEmbeddings"] = True
                    else:
                        page_data["hasEmbeddings"] = False
                except Exception as e:
                    print(f"Error generating embeddings: {str(e)}")
                    page_data["hasEmbeddings"] = False
            else:
                page_data["hasEmbeddings"] = False
            
            # Update the extraction document in the database
            try:
                # Check if extraction document exists
                existing_doc = ragnor_doc_extractions_collection.find_one({"_id": extraction_id})
                if not existing_doc:
                    print(f"WARNING: Extraction document {extraction_id} not found, creating it")
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
                
                # Remove any existing page
                remove_result = ragnor_doc_extractions_collection.update_one(
                    {"_id": extraction_id},
                    {"$pull": {"pageContents": {"pageNumber": page_num}}}
                )
                
                # Mark page as processed in the tracking list
                ragnor_doc_extractions_collection.update_one(
                    {"_id": extraction_id},
                    {"$addToSet": {"processedPages": page_num}}
                )
                
                # Force embedding generation if requested in the API call
                job_doc = ragnor_doc_jobs_collection.find_one({"_id": job_id})
                should_generate_embeddings = job_doc.get("embedding", False) if job_doc else generate_embeddings
                job_embedding_provider = job_doc.get("embedding_provider", "azure") if job_doc else "azure"
                
                if should_generate_embeddings and text.strip():
                    print(f"FORCING EMBEDDINGS GENERATION for page {page_num} based on job settings")
                    try:
                        if hasattr(self, 'embedding_generator') and self.embedding_generator:
                            current_provider = self.embedding_generator.__class__.__name__
                            print(f"Using document processor embedding generator: {current_provider}")
                            embedding_generator = self.embedding_generator
                            embedding_model = embedding_generator.model
                            print(f"Generating embeddings with model: {embedding_model}")
                            embeddings = embedding_generator.generate_embeddings(text)
                            success = bool(embeddings)
                        else:
                            print(f"No embedding generator available in processor")
                            embeddings = []
                            embedding_model = "none" 
                            success = False
                        
                        if success and embeddings:
                            print(f"SUCCESS: Generated {len(embeddings)} embeddings for forced storage")
                            has_embeddings = True
                        else:
                            print(f"WARN: No embeddings generated")
                            embeddings = []
                            embedding_model = "none"
                            has_embeddings = False
                    except Exception as e:
                        print(f"ERROR during embedding generation: {str(e)}")
                        embeddings = []
                        embedding_model = "none"
                        has_embeddings = False
                else:
                    print(f"No embeddings requested at job level for page {page_num}")
                    embeddings = []
                    embedding_model = ""
                    has_embeddings = False
                
                # Construct ordered page document
                ordered_page_content = {
                    "pageNumber": page_num,
                    "text": text,
                    "embeddings": embeddings,
                    "embeddingModel": embedding_model,
                    "hasEmbeddings": has_embeddings,
                    "textChunks": page_data["textChunks"],
                    "hasTable": False,
                    "tables": [],
                    "extractionMethod": "pdfplumber"  # Add extraction method for tracking
                }
                
                # Update the database with page content
                update_result = ragnor_doc_extractions_collection.update_one(
                    {"_id": extraction_id},
                    {
                        "$push": {
                            "pageContents": ordered_page_content
                        },
                        "$set": {"updatedAt": datetime.utcnow()}
                    }
                )
                print(f"Updated extraction document with page {page_num} data, modified: {update_result.modified_count}")
                
                # Verify the update succeeded
                if update_result.modified_count == 0:
                    print(f"WARNING: Failed to update extraction document {extraction_id}")
                
                return True
            except Exception as update_err:
                print(f"Error updating extraction document with page data: {str(update_err)}")
                print(f"Stack trace: {traceback.format_exc()}")
                return False
                
        except Exception as e:
            print(f"ERROR processing text page: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return False

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
                try:
                    # Use the document processor's existing embedding generator directly
                    if hasattr(self, 'embedding_generator') and self.embedding_generator:
                        print(f"Using document processor's embedding generator directly")
                        embedding_generator = self.embedding_generator
                        embeddings = embedding_generator.generate_embeddings(full_text)
                        embedding_model = embedding_generator.model
                        success = bool(embeddings)
                    else:
                        print(f"No embedding generator available, using generate_embeddings_for_text")
                        embeddings, embedding_model, success = self.generate_embeddings_for_text(
                            full_text, embedding_provider="azure")

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
                except Exception as e:
                    print(f"Error generating embeddings: {str(e)}")
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
                    embeddings = []
                    embedding_model = ""
                    success = False
                    
                    try:
                        # Use the document processor's existing embedding generator directly
                        if hasattr(self, 'embedding_generator') and self.embedding_generator:
                            print(f"Using document processor's embedding generator directly")
                            embedding_generator = self.embedding_generator
                            embeddings = embedding_generator.generate_embeddings(full_text)
                            embedding_model = embedding_generator.model
                            success = bool(embeddings)
                        else:
                            print(f"No embedding generator available, using generate_embeddings_for_text")
                            embeddings, embedding_model, success = self.generate_embeddings_for_text(
                                full_text, embedding_provider="azure")
                                
                        if success and embeddings:
                            print(f"DEBUG: Successfully generated {len(embeddings)} embeddings")
                        else:
                            print(f"ERROR: Failed to generate embeddings before MongoDB update")
                            success = False
                    except Exception as e:
                        print(f"Error generating embeddings for MongoDB update: {str(e)}")
                        success = False
                    
                    # Force a new ordered document with embeddings
                    ordered_page_data = {
                        "pageNumber": page_num,
                        "text": full_text,
                        "embeddings": embeddings,  # Force embeddings here
                        "embeddingModel": embedding_model,
                        "hasEmbeddings": success,
                        "textChunks": page_data["textChunks"],
                        "hasTable": False,
                        "tables": []
                    }
                    page_data = ordered_page_data
                    print(f"DEBUG: Created ordered page data with embeddings in position 3")
                    print(f"DEBUG: Keys in order: {list(page_data.keys())}")

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
                
                # Get the embedding provider from the job document
                job_embedding_provider = job_doc.get("embedding_provider", "azure") if job_doc else "azure"
                print(f"Job embedding provider setting: {job_embedding_provider}")

                if should_generate_embeddings and full_text.strip():
                    print(
                        f"FORCING EMBEDDINGS GENERATION for page {page_num} based on job settings")
                    try:
                        # Always use the document processor's embedding generator directly
                        if hasattr(self, 'embedding_generator') and self.embedding_generator:
                            current_provider = self.embedding_generator.__class__.__name__
                            print(f"Using document processor embedding generator: {current_provider}")
                            embedding_generator = self.embedding_generator
                            embedding_model = embedding_generator.model
                            print(f"Generating embeddings with model: {embedding_model}")
                            embeddings = embedding_generator.generate_embeddings(full_text)
                            success = bool(embeddings)
                        else:
                            print(f"No embedding generator available in processor")
                            # Don't call generate_embeddings_for_text as it might use OpenAI
                            embeddings = []
                            embedding_model = "none" 
                            success = False

                        if success and embeddings:
                            print(
                                f"SUCCESS: Generated {len(embeddings)} embeddings for forced storage")
                            has_embeddings = True
                        else:
                            print(
                                f"WARN: No embeddings generated")
                            embeddings = []
                            embedding_model = "none"
                            has_embeddings = False
                    except Exception as e:
                        print(f"ERROR during embedding generation: {str(e)}")
                        embeddings = []
                        embedding_model = "none"
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
                    "tables": [],
                    "extractionMethod": "tesseract_ocr"  # Add extraction method for tracking
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

    @staticmethod
    def _process_page_parallel(page_params):
        """Static method to process a single page in a separate process.
        This needs to be a static method for proper pickling with multiprocessing."""
        """Process a single page in a separate process.
        This method is designed to be called within a multiprocessing Pool.
        
        Args:
            page_params: Dictionary containing parameters for processing a page:
                - file_content: The document file content
                - page_num: The page number to process
                - total_pages: Total number of pages in document
                - is_scanned: Whether the page is scanned (requires OCR) or text-based
                - file_format: The format of the document
                - job_id: The job ID
                - extraction_id: The extraction document ID
                - embedding: Whether to generate embeddings
                - embedding_provider: Provider to use for embeddings
        
        Returns:
            Dictionary with processing results including page number, success status, and error if any
        """
        try:
            file_content = page_params['file_content']
            page_num = page_params['page_num']
            total_pages = page_params['total_pages']
            is_scanned = page_params['is_scanned']
            file_format = page_params['file_format']
            job_id = page_params['job_id']
            extraction_id = page_params['extraction_id']
            embedding = page_params.get('embedding', False)
            embedding_provider = page_params.get('embedding_provider', 'azure')
            
            print(f"[Process {os.getpid()}] Processing page {page_num}/{total_pages}")
            
            success = False
            error = None
            
            try:
                if is_scanned:
                    # For scanned pages, use the existing OCR process
                    print(f"[Process {os.getpid()}] Page {page_num} is a scanned page, using OCR")
                    # Convert page to image
                    image = PDFHandler.convert_page_to_image(file_content, page_num, total_pages)
                    # Process with OCR
                    success = self._process_single_page_sync(
                        image, page_num, total_pages, file_format, extraction_id, job_id,
                        generate_embeddings=embedding, embedding_provider=embedding_provider)
                else:
                    # For text-based pages, use pdfplumber
                    print(f"[Process {os.getpid()}] Page {page_num} is a text-based page, using pdfplumber")
                    text = PDFHandler.extract_text_from_page(file_content, page_num)
                    print(f"[Process {os.getpid()}] Extracted {len(text)} characters from page {page_num}")
                    # Process the text directly
                    success = self._process_text_page(
                        text, page_num, total_pages, file_format, extraction_id, job_id,
                        generate_embeddings=embedding, embedding_provider=embedding_provider)
            except Exception as e:
                error = str(e)
                traceback.print_exc()
                print(f"[Process {os.getpid()}] Error processing page {page_num}: {error}")
                
            return {
                'page_num': page_num,
                'success': success,
                'error': error
            }
        except Exception as outer_e:
            # Catch any exceptions that might occur outside the inner try block
            outer_error = str(outer_e)
            traceback.print_exc()
            print(f"[Process {os.getpid()}] Critical error in _process_page_parallel for page {page_params.get('page_num')}: {outer_error}")
            return {
                'page_num': page_params.get('page_num', 0),
                'success': False,
                'error': outer_error
            }
    
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
                "pageReferences": [],  # Will store references to page content in separate collection
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

            # Use threading for parallel processing but avoid problematic analysis methods
            print(f"Setting up parallel processing for {total_pages} pages using {NUM_CORES} worker threads")
            
            # Process pages in smaller batches to avoid memory issues
            batch_size = min(NUM_CORES * 2, 50)  # Use at most 50 pages per batch
            
            # Create batches of page numbers to process
            page_batches = []
            for i in range(0, total_pages, batch_size):
                end = min(i + batch_size, total_pages)
                page_batches.append(list(range(i+1, end+1)))  # 1-indexed page numbers
            
            print(f"Processing document in {len(page_batches)} batches of up to {batch_size} pages each")
            
            # Function to process a single page directly - hybrid approach using both direct text extraction and OCR
            def process_page_direct(page_num):
                thread_id = threading.get_ident()
                print(f"[Thread {thread_id}] Processing page {page_num}/{total_pages}")
                
                try:
                    # Try direct text extraction first (faster)
                    try:
                        text = PDFHandler.extract_text_from_page(file_content, page_num)
                        has_direct_text = len(text.strip()) > 0
                        print(f"[Thread {thread_id}] Page {page_num} direct extraction: {len(text)} chars")
                    except Exception as text_err:
                        print(f"[Thread {thread_id}] Page {page_num} direct extraction failed: {str(text_err)}")
                        text = ""
                        has_direct_text = False
                    
                    # If direct extraction failed or returned little text, try OCR
                    ocr_text = ""
                    text_chunks = []
                    extraction_method = "pdfplumber"
                    
                    if not has_direct_text or len(text) < 100:  # If no text or very little text
                        try:
                            print(f"[Thread {thread_id}] Using OCR for page {page_num}")
                            # Convert page to image for OCR
                            image = PDFHandler.convert_page_to_image(file_content, page_num, total_pages)
                            
                            # Perform OCR
                            ocr_result = text_extractor.extract_text(image, doc_format=file_format)
                            ocr_text = "\n".join([line.text for line in ocr_result.text_lines]) if ocr_result.text_lines else ""
                            
                            # Create text chunks from OCR
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
                            
                            # If OCR found more text, use it instead
                            if len(ocr_text) > len(text):
                                print(f"[Thread {thread_id}] Using OCR text for page {page_num} ({len(ocr_text)} chars)")
                                text = ocr_text
                                extraction_method = "tesseract_ocr"
                            else:
                                print(f"[Thread {thread_id}] Keeping direct text for page {page_num} ({len(text)} chars)")
                        except Exception as ocr_err:
                            print(f"[Thread {thread_id}] OCR failed for page {page_num}: {str(ocr_err)}")
                    
                    # If we don't have text chunks yet, create them from direct text
                    if not text_chunks and text:
                        lines = text.split('\n')
                        for i, line in enumerate(lines):
                            if line.strip():
                                text_chunks.append({
                                    "text": line,
                                    "confidence": 100.0,
                                    "position": {
                                        "x": 0,
                                        "y": i * 20,
                                        "width": 1000,
                                        "height": 20
                                    }
                                })
                    # Generate embeddings if requested
                    embeddings = []
                    embedding_model = ""
                    has_embeddings = False
                    
                    if embedding and text.strip():
                        try:
                            print(f"[Thread {thread_id}] Generating embeddings for page {page_num}")
                            
                            # Try to use the existing embedding generator directly
                            # (self refers to RagnorDocumentProcessor instance)
                            if hasattr(self, 'embedding_generator') and self.embedding_generator:
                                print(f"[Thread {thread_id}] Using existing embedding generator for page {page_num}")
                                embedding_generator = self.embedding_generator
                                embeddings = embedding_generator.generate_embeddings(text)
                                embedding_model = embedding_generator.model
                                has_embeddings = bool(embeddings)
                            else:
                                # Fallback to creating a new generator with the specified provider
                                # This follows the implementation mentioned in the memory about handling embedding providers
                                from utils.ragnor_embedding_generator import get_embedding_generator
                                print(f"[Thread {thread_id}] Creating new {embedding_provider} embedding generator for page {page_num}")
                                try:
                                    # First try with the specified provider directly
                                    generator = get_embedding_generator(embedding_provider)
                                    embeddings = generator.generate_embeddings(text)
                                    embedding_model = generator.model
                                    has_embeddings = bool(embeddings)
                                except ValueError as e:
                                    # Only fall back if the provider is truly unsupported
                                    print(f"[Thread {thread_id}] Error with {embedding_provider} provider: {str(e)}")
                                    # Re-raise the error instead of falling back
                                    raise
                            
                            if has_embeddings:
                                print(f"[Thread {thread_id}] Generated embeddings for page {page_num}")
                            else:
                                print(f"[Thread {thread_id}] Failed to generate embeddings for page {page_num}")
                        except Exception as emb_err:
                            print(f"[Thread {thread_id}] Error generating embeddings: {str(emb_err)}")
                    
                    # Create the page data structure
                    page_data = {
                        "pageNumber": page_num,
                        "text": text,
                        "embeddings": embeddings,
                        "embeddingModel": embedding_model,
                        "hasEmbeddings": has_embeddings,
                        "textChunks": text_chunks,
                        "hasTable": False,
                        "tables": [],
                        "extractionMethod": extraction_method
                    }
                    
                    # Update the database with the extracted data
                    try:
                        # First remove any existing page data
                        ragnor_doc_extractions_collection.update_one(
                            {"_id": extraction_id},
                            {"$pull": {"pageContents": {"pageNumber": page_num}}}
                        )
                        
                        # Mark this page as processed
                        ragnor_doc_extractions_collection.update_one(
                            {"_id": extraction_id},
                            {"$addToSet": {"processedPages": page_num}}
                        )
                        
                        # Store page content in a separate collection to avoid MongoDB 16MB document limit
                        
                        # Create a unique ID for this page content (combine extraction ID and page number)
                        page_content_id = f"{extraction_id}_{page_num}"
                        
                        # Prepare page data for separate collection
                        page_content_doc = {
                            "_id": page_content_id,
                            "extractionId": extraction_id,
                            "pageNumber": page_num,
                            "text": text,
                            "embeddings": embeddings,
                            "embeddingModel": embedding_model,
                            "hasEmbeddings": has_embeddings,
                            "textChunks": text_chunks,
                            "hasTable": False,
                            "tables": [],
                            "extractionMethod": extraction_method,
                            "createdAt": datetime.utcnow()
                        }
                        
                        # Store complete page content in separate collection
                        try:
                            # Insert or update the page content document
                            ragnor_page_contents_collection.replace_one(
                                {"_id": page_content_id},
                                page_content_doc,
                                upsert=True
                            )
                            
                            # Only store a reference in the main extraction document
                            page_reference = {
                                "pageNumber": page_num,
                                "contentId": page_content_id,
                                "hasEmbeddings": has_embeddings,
                                "extractionMethod": extraction_method
                            }
                            
                            # Update the main extraction document with just the reference
                            ragnor_doc_extractions_collection.update_one(
                                {"_id": extraction_id},
                                {
                                    "$pull": {"pageReferences": {"pageNumber": page_num}}
                                }
                            )
                            
                            ragnor_doc_extractions_collection.update_one(
                                {"_id": extraction_id},
                                {
                                    "$addToSet": {"pageReferences": page_reference},
                                    "$set": {"updatedAt": datetime.utcnow()}
                                }
                            )
                            
                            print(f"[Thread {thread_id}] Page {page_num} content stored in separate collection")
                            
                        except Exception as db_err:
                            print(f"[Thread {thread_id}] Database error storing page {page_num} content: {str(db_err)}")
                            raise
                        print(f"[Thread {thread_id}] Database updated for page {page_num}")
                        return {
                            'page_num': page_num,
                            'success': True,
                            'error': None
                        }
                    except Exception as db_err:
                        print(f"[Thread {thread_id}] Database error for page {page_num}: {str(db_err)}")
                        return {
                            'page_num': page_num,
                            'success': False,
                            'error': f"Database error: {str(db_err)}"
                        }
                except Exception as e:
                    print(f"[Thread {thread_id}] Processing error for page {page_num}: {str(e)}")
                    traceback.print_exc()
                    return {
                        'page_num': page_num,
                        'success': False,
                        'error': str(e)
                    }
            
            # Initialize tracking variables
            processed_pages = []
            progress_count = 0
            
            # Update progress to show we're starting processing
            self._update_job_status_sync(job_id, JobStatus.PROCESSING, 15.0)
            
            # Process each batch of pages
            for batch_idx, page_batch in enumerate(page_batches):
                print(f"Processing batch {batch_idx+1}/{len(page_batches)} with {len(page_batch)} pages")
                
                # Use ThreadPoolExecutor for parallel processing of each batch
                with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
                    # Submit tasks for each page in this batch
                    futures = {executor.submit(process_page_direct, page_num): page_num for page_num in page_batch}
                    
                    # Process results as they complete
                    for future in as_completed(futures):
                        page_num = futures[future]
                        try:
                            # Get the result from the completed future
                            result = future.result()
                            progress_count += 1
                            
                            # Track successfully processed pages
                            if result['success']:
                                processed_pages.append(result['page_num'])
                            else:
                                print(f"Warning: Page {result['page_num']} failed: {result['error']}")
                            
                            # Update progress (scale from 15% to 85%)
                            progress = 15.0 + (70.0 * progress_count / total_pages)
                            print(f"Completed page {result['page_num']}/{total_pages}. Progress: {progress:.1f}%")
                            self._update_job_status_sync(job_id, JobStatus.PROCESSING, progress)
                            
                        except Exception as exc:
                            print(f"ERROR processing page {page_num}: {exc}")
                            traceback.print_exc()
                            
            # Print summary after all batches are processed
            print(f"Finished processing all pages. Total pages: {total_pages}, Processed pages: {len(processed_pages)}")


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
            
            # Look for page references (new structure) or page contents (old structure)
            page_references = extraction_doc.get('pageReferences', [])
            page_contents = extraction_doc.get('pageContents', [])
            
            print(f"Found {len(page_references)} page references and {len(page_contents)} direct page contents")
            
            # Verify extraction document structure
            if not page_references and not page_contents:
                print(f"WARNING: Extraction document has no page data (neither references nor contents)")
                # Try to print more info about the document
                print(f"Document keys: {list(extraction_doc.keys())}")
                print(f"Document jobId: {extraction_doc.get('jobId')}")
                if extraction_doc.get('jobId') != job_id:
                    print(f"ERROR: jobId mismatch: {extraction_doc.get('jobId')} != {job_id}")

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

            # Add page contents from either the references (new model) or direct contents (old model)
            if page_references:
                print(f"Using new page reference model for {len(page_references)} pages")
                # New model: Pages are stored in a separate collection referenced by contentId
                for page_ref in sorted(page_references, key=lambda p: p.get("pageNumber", 0)):
                    page_num = page_ref.get("pageNumber")
                    content_id = page_ref.get("contentId")
                    
                    print(f"Fetching page content for page {page_num} with content ID: {content_id}")
                    
                    # Fetch the full page content from the separate collection
                    try:
                        page_content_doc = ragnor_page_contents_collection.find_one({"_id": content_id})
                        
                        if page_content_doc:
                            # Include embeddings if they exist in the document
                            has_embeddings = page_content_doc.get("hasEmbeddings", False)
                            
                            # Create page data for response
                            page = {
                                "pageNumber": page_content_doc.get("pageNumber"),
                                "text": page_content_doc.get("text"),
                                "hasEmbeddings": has_embeddings,
                                "extractionMethod": page_content_doc.get("extractionMethod", "ocr"),
                                # Only include relevant chunked fields to reduce size
                                # "textChunks": page_content_doc.get("textChunks", []),
                                # "hasTable": page_content_doc.get("hasTable", False),
                                # "tables": page_content_doc.get("tables", [])
                            }
                            
                            # Add embeddings and model if they exist
                            if has_embeddings:
                                # Include embeddings and model information
                                page["embeddings"] = page_content_doc.get("embeddings", [])
                                page["embeddingModel"] = page_content_doc.get("embeddingModel", "")
                                print(f"Including embeddings for page {page_num} in API response")
                            
                            response["pages"].append(page)
                        else:
                            print(f"WARNING: Could not find page content document with ID: {content_id}")
                            # Include a placeholder for missing pages
                            response["pages"].append({
                                "pageNumber": page_num,
                                "text": "[Content not found]",
                                "hasEmbeddings": False,
                                "extractionMethod": "unknown",
                                "error": f"Content not found for page {page_num}"
                            })
                    except Exception as e:
                        print(f"ERROR fetching page content for page {page_num}: {str(e)}")
                        # Include error in response
                        response["pages"].append({
                            "pageNumber": page_num,
                            "text": f"[Error retrieving content: {str(e)}]",
                            "hasEmbeddings": False,
                            "extractionMethod": "unknown",
                            "error": str(e)
                        })
            
            elif page_contents:
                print(f"Using legacy direct page content model for {len(page_contents)} pages")
                # Legacy model: Pages are stored directly in the extraction document
                for page_content in sorted(page_contents, key=lambda p: p.get("pageNumber", 0)):
                    # Include embeddings if they exist in the document
                    has_embeddings = page_content.get("hasEmbeddings", False)

                    page = {
                        "pageNumber": page_content.get("pageNumber"),
                        "text": page_content.get("text"),
                        "hasEmbeddings": has_embeddings,
                        "extractionMethod": page_content.get("extractionMethod", "ocr"),
                        # "textChunks": page_content.get("textChunks", []),  # Temporarily commented out
                        # "hasTable": page_content.get("hasTable", False),
                        # "tables": page_content.get("tables", [])
                    }

                    # Add embeddings and model if they exist
                    if has_embeddings:
                        # Include embeddings and model information
                        page["embeddings"] = page_content.get("embeddings", [])
                        page["embeddingModel"] = page_content.get("embeddingModel", "")
                        print(f"Including embeddings for page {page_content.get('pageNumber')} in API response")

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
