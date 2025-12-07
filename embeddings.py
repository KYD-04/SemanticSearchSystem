"""
Модуль для работы с эмбеддингами.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Менеджер эмбеддингов."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def fragment_text(self, text: str, fragment_type: str = "sentences") -> list:
        """Разбить текст на фрагменты."""
        if fragment_type == "words":
            return [w for w in text.split() if w.strip()]
        elif fragment_type == "paragraphs":
            return [p.strip() for p in text.split('\n\n') if p.strip()]
        else:  # sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def get_embeddings(self, texts: list) -> np.ndarray:
        """Получить эмбеддинги для списка текстов."""
        if not texts:
            return np.array([])
        return self.model.encode(texts, convert_to_numpy=True)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Получить эмбеддинг для одного текста."""
        return self.model.encode([text], convert_to_numpy=True)[0]
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Вычислить косинусную меру."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
