from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
from functools import lru_cache

from config import Config

class EmbeddingService:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2") # For semantic similarity

    @lru_cache(maxsize=Config.EMBEDDING_CACHE_SIZE)
    def embed_content(self, content: str) -> List[float]:
        """Generates an embedding for the given content."""
        return self.embedder.encode(content).tolist()

    def calculate_semantic_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculates cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        return cosine_similarity(np.array(embedding1).reshape(1, -1), np.array(embedding2).reshape(1, -1))[0][0]
