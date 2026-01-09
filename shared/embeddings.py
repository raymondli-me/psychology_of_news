"""Embedding generation using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import torch

# Use consistent model across all sources for comparability
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wrapper for sentence embedding model."""

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("Embedding model ready.")

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL) -> "EmbeddingModel":
        """Get singleton instance."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text."""
        return self.model.encode([text], convert_to_numpy=True)[0]


def generate_embeddings(texts: List[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    model = EmbeddingModel.get_instance(model_name)
    return model.encode(texts)


def semantic_search(
    query: str,
    embeddings: np.ndarray,
    texts: List[str],
    top_k: int = 10
) -> List[dict]:
    """Search for similar texts using cosine similarity."""
    model = EmbeddingModel.get_instance()
    query_vec = model.encode_single(query)

    # Cosine similarity
    similarities = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {"index": int(idx), "text": texts[idx], "score": float(similarities[idx])}
        for idx in top_indices
    ]
