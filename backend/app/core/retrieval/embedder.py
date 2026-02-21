from openai import OpenAI
from typing import List
import asyncio

from backend.app.config import get_settings

settings = get_settings()

class Embedder: 
    """ Wrapper for OpenAI Embeddings API.

    """
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    def embed_documents(self, text: str) -> List[float]:
        """
        Embed 1 text
        Args: 
            text: Text to embed
        Returns: 
            List of floats (1536 dims)
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts
        Args: 
            texts: List of texts to embed
        Returns: 
            List of vectors
        Note: 
            OpenAI limits batch size to 2048 
            If texts > 2048, we will split it into multiple batches
        """
        batch_size = 2048
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                model=self.model, 
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        
        return all_embeddings