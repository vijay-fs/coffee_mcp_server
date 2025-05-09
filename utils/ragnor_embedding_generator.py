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

# Default embedding models
DEFAULT_OPENAI_MODEL = "text-embedding-ada-002"
DEFAULT_MODAL_MODEL = "e5-large-v2"


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


def get_embedding_generator(provider: str = "openai") -> EmbeddingGenerator:
    """Get an embedding generator based on provider."""
    if provider.lower() == "openai":
        return OpenAIEmbeddingGenerator()
    elif provider.lower() == "modal":
        return ModalEmbeddingGenerator()
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
