"""
Embedding generation for Ragnor document extraction API.
"""
import os
from typing import List, Optional
from openai import OpenAI
import requests
from dotenv import load_dotenv

load_dotenv()

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODAL_API_KEY = os.getenv("MODAL_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-large")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Default embedding models
DEFAULT_OPENAI_MODEL = "text-embedding-3-large"
DEFAULT_MODAL_MODEL = "e5-large-v2"
DEFAULT_AZURE_MODEL = "text-embedding-3-large"  # Default model for Azure


class EmbeddingGenerator:
    """Base class for embedding generators."""

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for a text string."""
        raise NotImplementedError("Subclasses must implement this method")

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text strings."""
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """Generate embeddings using OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize OpenAI embedding generator."""
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model or DEFAULT_OPENAI_MODEL

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for a text string using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embeddings: {str(e)}")
            # Return empty embedding on error
            return []

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text strings using OpenAI."""
        try:
            # OpenAI supports batch embedding generation
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            # Extract embeddings from response in the same order
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)

            return embeddings
        except Exception as e:
            print(f"Error generating batch OpenAI embeddings: {str(e)}")
            # Return empty embeddings on error
            return [[] for _ in texts]


class AzureOpenAIEmbeddingGenerator(EmbeddingGenerator):
    """Generate embeddings using Azure OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None,
                 deployment: Optional[str] = None, api_version: Optional[str] = None,
                 model: Optional[str] = None):
        """Initialize Azure OpenAI embedding generator."""
        self.api_key = api_key or AZURE_OPENAI_API_KEY
        self.endpoint = endpoint or AZURE_OPENAI_ENDPOINT
        self.deployment = deployment or AZURE_OPENAI_DEPLOYMENT
        self.api_version = api_version or AZURE_OPENAI_API_VERSION

        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required")
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint is required")

        # Remove quotes if present in the endpoint URL
        if self.endpoint and (self.endpoint.startswith('"') or self.endpoint.startswith("'")):
            self.endpoint = self.endpoint.strip('"\'')
            print(f"Removed quotes from endpoint: {self.endpoint}")

        self.model = model or DEFAULT_AZURE_MODEL

        # Print configuration for debugging
        print(f"Azure OpenAI Configuration:")
        print(f"  Endpoint: {self.endpoint}")
        print(f"  Deployment: {self.deployment}")
        print(f"  API Version: {self.api_version}")
        print(f"  Model: {self.model}")

        base_url = f"{self.endpoint}/openai/deployments/{self.deployment}"
        print(f"  Base URL: {base_url}")

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                default_headers={
                    "api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                default_query={"api-version": self.api_version}
            )
            print(f"Successfully created Azure OpenAI client")
        except Exception as e:
            print(f"Error creating Azure OpenAI client: {str(e)}")

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for a text string using Azure OpenAI."""
        try:
            print(
                f"Generating Azure OpenAI embeddings for text: {text[:50]}...")
            response = self.client.embeddings.create(
                model=self.model,  # Model is ignored as it's specified in the deployment
                input=text
            )
            print(f"Successfully generated embeddings with Azure OpenAI")
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating Azure OpenAI embeddings: {str(e)}")
            print(
                f"Client configuration: endpoint={self.endpoint}, deployment={self.deployment}")
            # Return empty embedding on error
            return []

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text strings using Azure OpenAI."""
        try:
            # Azure OpenAI supports batch embedding generation
            response = self.client.embeddings.create(
                model=self.model,  # Model is ignored as it's specified in the deployment
                input=texts
            )

            # Extract embeddings from response in the same order
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)

            return embeddings
        except Exception as e:
            print(f"Error generating batch Azure OpenAI embeddings: {str(e)}")
            # Return empty embeddings on error
            return [[] for _ in texts]


class ModalEmbeddingGenerator(EmbeddingGenerator):
    """Generate embeddings using Modal API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize Modal embedding generator."""
        self.api_key = api_key or MODAL_API_KEY
        if not self.api_key:
            raise ValueError("Modal API key is required")

        self.model = model or DEFAULT_MODAL_MODEL
        self.endpoint = "https://api.modal.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for a text string using Modal."""
        try:
            payload = {
                "model": self.model,
                "input": text
            }

            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload
            )

            if response.status_code != 200:
                print(
                    f"Modal API error: {response.status_code} - {response.text}")
                return []

            response_json = response.json()
            return response_json.get("data", [{}])[0].get("embedding", [])
        except Exception as e:
            print(f"Error generating Modal embeddings: {str(e)}")
            # Return empty embedding on error
            return []

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text strings using Modal."""
        try:
            payload = {
                "model": self.model,
                "input": texts
            }

            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload
            )

            if response.status_code != 200:
                print(
                    f"Modal API error: {response.status_code} - {response.text}")
                return [[] for _ in texts]

            response_json = response.json()

            # Extract embeddings from response
            embeddings = []
            for item in response_json.get("data", []):
                embeddings.append(item.get("embedding", []))

            return embeddings
        except Exception as e:
            print(f"Error generating batch Modal embeddings: {str(e)}")
            # Return empty embeddings on error
            return [[] for _ in texts]


def get_embedding_generator(provider: str = "azure") -> EmbeddingGenerator:
    """Get an embedding generator based on provider."""
    if provider.lower() == "openai":
        return OpenAIEmbeddingGenerator()
    elif provider.lower() == "azure":
        return AzureOpenAIEmbeddingGenerator()
    elif provider.lower() == "azure-openai":
        return AzureOpenAIEmbeddingGenerator()
    elif provider.lower() == "modal":
        return ModalEmbeddingGenerator()
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
