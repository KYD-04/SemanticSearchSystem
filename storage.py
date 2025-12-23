"""
Модуль для хранения эмбеддингов и метаданных.
"""

import os
import json
import hashlib
import numpy as np


class Storage:
    """Хранилище эмбеддингов и метаданных документов."""
    
    def __init__(self, hash_folder: str):
        self.hash_folder = hash_folder
        os.makedirs(hash_folder, exist_ok=True)
        self.index_file = os.path.join(hash_folder, "index.json")
        self.index = self._load_index()
    
    def _load_index(self) -> dict:
        """Загрузить индекс документов."""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Сохранить индекс документов."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """Получить MD5 хеш файла."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    @staticmethod
    def _make_safe_filename(name: str) -> str:
        """Заменить небезопасные символы в имени файла."""
        return name.replace('/', '_').replace('\\', '_')
    
    def is_indexed(self, filepath: str, model_name: str, fragment_type: str = None) -> bool:
        """Проверить, проиндексирован ли файл для данной модели и типа фрагментации."""
        file_hash = self.get_file_hash(filepath)
        # Нормализуем имя модели, чтобы убрать / из пути файла
        safe_model = self._make_safe_filename(model_name)
        if fragment_type:
            key = f"{file_hash}_{safe_model}_{fragment_type}"
        else:
            key = f"{file_hash}_{safe_model}"
        return key in self.index
    
    def save_document(self, filepath: str, model_name: str, fragment_type: str,
                      fragments: list, embeddings: np.ndarray, full_text: str):
        """Сохранить документ с эмбеддингами."""
        file_hash = self.get_file_hash(filepath)
        # Нормализуем имя модели, чтобы убрать / из пути файла
        safe_model = self._make_safe_filename(model_name)
        key = f"{file_hash}_{safe_model}_{fragment_type}"
        
        # Сохранить эмбеддинги
        emb_file = os.path.join(self.hash_folder, f"{key}.npy")
        # Нормализовать путь для Windows (убрать проблемные разделители)
        emb_file = os.path.normpath(emb_file)
        np.save(emb_file, embeddings)
        
        # Обновить индекс
        self.index[key] = {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "model": model_name,
            "fragment_type": fragment_type,
            "file_hash": file_hash,
            "fragments": fragments,
            "full_text": full_text,
            "embeddings_file": emb_file
        }
        self._save_index()
    
    def get_all_documents(self, model_name: str, fragment_type: str = None) -> list:
        """Получить все документы для заданной модели и типа фрагментации."""
        results = []
        for key, data in self.index.items():
            # Проверяем соответствие модели и типа фрагментации
            if data["model"] == model_name:
                if fragment_type is None or data.get("fragment_type") == fragment_type:
                    emb_file = data["embeddings_file"]
                    if os.path.exists(emb_file):
                        embeddings = np.load(emb_file)
                        results.append({
                            **data,
                            "embeddings": embeddings
                        })
        return results
    
    def clear_missing_files(self):
        """Удалить записи для несуществующих файлов."""
        to_remove = []
        for key, data in self.index.items():
            if not os.path.exists(data["filepath"]):
                to_remove.append(key)
                emb_file = data.get("embeddings_file")
                if emb_file and os.path.exists(emb_file):
                    os.remove(emb_file)
        for key in to_remove:
            del self.index[key]
        if to_remove:
            self._save_index()
    
    def clear_all(self):
        """Очистить весь индекс (полезно при смене типа фрагментации)."""
        # Удалить все файлы эмбеддингов
        for key, data in self.index.items():
            emb_file = data.get("embeddings_file")
            if emb_file and os.path.exists(emb_file):
                os.remove(emb_file)
        # Очистить индекс
        self.index = {}
        self._save_index()
