"""
Класс RelevanceMetric для оценки релевантности фрагментов.
Пользователь может добавлять новые метрики metric_<name>(query, fragment, embed=True) -> float
"""

import numpy as np
import re


class RelevanceMetric:
    """Класс метрик релевантности."""
    
    @staticmethod
    def metric_cosine(query, fragment, embed: bool = True) -> float:
        """Косинусное расстояние (работает с эмбеддингами)."""
        if not embed:
            raise ValueError("Косинусная метрика требует эмбеддинги")
        
        norm_q = np.linalg.norm(query)
        norm_f = np.linalg.norm(fragment)
        if norm_q == 0 or norm_f == 0:
            return 0.0
        return float(np.dot(query, fragment) / (norm_q * norm_f))
    
    @staticmethod
    def metric_word_match(query, fragment, embed: bool = False) -> float:
        """Процент совпадения слов запроса (работает с текстом)."""
        if embed:
            raise ValueError("Метрика совпадения слов требует текст")
        
        # Нормализация: lowercase, только буквы и цифры
        def normalize(text):
            return set(re.findall(r'\w+', text.lower()))
        
        query_words = normalize(query)
        fragment_words = normalize(fragment)
        
        if not query_words:
            return 0.0
        
        matched = query_words & fragment_words
        return len(matched) / len(query_words)
    
    @staticmethod
    def metric_euclidean(query, fragment, embed: bool = True) -> float:
        """Евклидово расстояние (инвертированное, работает с эмбеддингами)."""
        if not embed:
            raise ValueError("Евклидова метрика требует эмбеддинги")
        
        distance = np.linalg.norm(query - fragment)
        # Инвертируем: меньше расстояние = выше score
        return 1.0 / (1.0 + distance)
    
    @classmethod
    def get_method(cls, method_name: str):
        """Получить метод по имени."""
        return getattr(cls, method_name, None)
    
    @classmethod
    def get_all_metrics(cls) -> list:
        """Получить список всех метрик с их параметрами."""
        metrics = []
        for name in dir(cls):
            if name.startswith('metric_'):
                method = getattr(cls, name)
                # Получить значение embed по умолчанию
                import inspect
                sig = inspect.signature(method)
                embed_default = sig.parameters.get('embed')
                embed_value = embed_default.default if embed_default else True
                
                metrics.append({
                    "name": name,
                    "display_name": name.replace('metric_', '').replace('_', ' ').title(),
                    "embed": embed_value
                })
        return metrics
