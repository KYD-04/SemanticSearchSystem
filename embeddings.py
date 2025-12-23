"""
Модуль для работы с эмбеддингами.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Менеджер эмбеддингов."""
    
    def __init__(self, model_name: str = "deepvk/USER-base", device: str = None):
        self.model_name = model_name

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)

    
    def fragment_text(self, text: str, fragment_type: str = "sentences") -> list:
        """Разбить текст на фрагменты."""
        if fragment_type == "words":
            return [w for w in text.split() if w.strip()]
        elif fragment_type == "paragraphs":
            return self._split_into_paragraphs(text)
        else:  # sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def _split_into_paragraphs(self, text: str) -> list:
        """
        Умное разделение текста на параграфы.

        PDF-файлы часто не имеют чётких разделителей \n\n между параграфами.
        Этот метод использует несколько стратегий для корректного разделения.
        """
        # Нормализуем переносы строк
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Стратегия 1: Попытка разделить по двойным переносам
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # Если получили слишком мало параграфов, используем более агрессивное разделение
        if len(paragraphs) < 5:
            # Стратегия 2: Разделяем по одинарным переносам, но объединяем короткие строки
            lines = text.split('\n')

            # Объединяем короткие строки (менее 50 символов) со следующей строкой
            merged_lines = []
            current_paragraph = ""

            for line in lines:
                line = line.strip()
                if not line:
                    # Пустая строка - это конец параграфа
                    if current_paragraph.strip():
                        merged_lines.append(current_paragraph.strip())
                        current_paragraph = ""
                    continue

                # Если строка короткая (< 50 символов) и не заканчивается на знак препинания,
                # это скорее всего часть того же параграфа
                if len(line) < 50 and not line.endswith(('.', '!', '?', ':', '"', "'")):
                    current_paragraph += " " + line if current_paragraph else line
                else:
                    # Длинная строка или строка заканчивается на знак препинания
                    if current_paragraph:
                        merged_lines.append(current_paragraph)
                    current_paragraph = line

            # Добавляем последний параграф
            if current_paragraph.strip():
                merged_lines.append(current_paragraph.strip())

            paragraphs = [p for p in merged_lines if p.strip()]

        # Если всё ещё мало параграфов, пробуем ещё более агрессивное разделение
        # на основе максимального размера параграфа (около 2000 символов)
        if len(paragraphs) < 10:
            # Разбиваем большие параграфы на части
            final_paragraphs = []
            for para in paragraphs:
                if len(para) > 3000:
                    # Разбиваем по предложениям
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < 2500:
                            current_chunk += (" " + sentence if current_chunk else sentence)
                        else:
                            if current_chunk.strip():
                                final_paragraphs.append(current_chunk.strip())
                            current_chunk = sentence
                    if current_chunk.strip():
                        final_paragraphs.append(current_chunk.strip())
                else:
                    final_paragraphs.append(para)
            paragraphs = final_paragraphs

        return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]
    
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
