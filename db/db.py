"""
Central MongoDB connection and collection accessors for all databases.
Handles both main application database and Ragnor document extraction database.
"""
import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Main database configuration
MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGODB_DATABASE", "ragnor_docs")

# MongoDB client connection setup
try:
    # Create client and select database with appropriate parameters
    logger.info(f"Connecting to MongoDB at {MONGO_URI}")

    # Configure client options based on connection string
    # Only use TLS for remote connections, not for localhost
    if "localhost" in MONGO_URI or "127.0.0.1" in MONGO_URI:
        # Local connection - no TLS needed
        mongo_options = {
            "connectTimeoutMS": 30000,     # 30 seconds
            "socketTimeoutMS": 30000,      # 30 seconds
            "retryWrites": True
        }
        logger.info("Using local MongoDB connection without TLS")
    else:
        # Remote connection - use TLS
        mongo_options = {
            "tls": True,
            "tlsAllowInvalidCertificates": True,
            "connectTimeoutMS": 30000,     # 30 seconds
            "socketTimeoutMS": 30000,      # 30 seconds
            "retryWrites": True,
            "w": "majority"
        }
        logger.info("Using remote MongoDB connection with TLS")

    # Connect with appropriate options
    _client = MongoClient(MONGO_URI, **mongo_options)

    # Test connection
    _client.admin.command('ping')
    logger.info("MongoDB connection successful")

    # Set up main database
    _db = _client[MONGO_DB]

    # Test if we can write to the Ragnor database
    test_result = _db.command("ping")
    if test_result.get("ok") == 1:
        logger.info(f"Successfully connected to Ragnor database: {MONGO_DB}")
    else:
        logger.error("Failed to ping Ragnor database")
        raise Exception("Failed to connect to Ragnor database")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    raise

# Create indexes for performance and uniqueness


def _safe_create_index(collection, keys, **kwargs):
    """Create an index safely, handling errors"""
    try:
        collection.create_index(keys, **kwargs)
    except Exception as e:
        logger.warning(
            f"Could not create index {keys} on {collection.name}: {e}")


def get_collection(name):
    """Get a collection from the specified database"""
    return _db.get_collection(name)


# Collections with collection names in the ragnor_docs database
ragnor_doc_jobs_collection = get_collection("jobs")
ragnor_doc_extractions_collection = get_collection("extractions")

# Create indexes for Ragnor collections
# Job collection indexes
_safe_create_index(ragnor_doc_jobs_collection, [("createdAt", -1)])
_safe_create_index(ragnor_doc_jobs_collection, [("status", 1)])
_safe_create_index(ragnor_doc_jobs_collection, [("filename", 1)])

# Extraction collection indexes
_safe_create_index(ragnor_doc_extractions_collection,
                   [("jobId", 1)], unique=True)
_safe_create_index(ragnor_doc_extractions_collection, [("filename", 1)])
_safe_create_index(ragnor_doc_extractions_collection, [("createdAt", -1)])
