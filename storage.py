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
    
    def is_indexed(self, filepath: str, model_name: str) -> bool:
        """Проверить, проиндексирован ли файл для данной модели."""
        file_hash = self.get_file_hash(filepath)
        key = f"{file_hash}_{model_name}"
        return key in self.index
    
    def save_document(self, filepath: str, model_name: str, fragments: list, 
                      embeddings: np.ndarray, full_text: str):
        """Сохранить документ с эмбеддингами."""
        file_hash = self.get_file_hash(filepath)
        key = f"{file_hash}_{model_name}"
        
        # Сохранить эмбеддинги
        emb_file = os.path.join(self.hash_folder, f"{key}.npy")
        np.save(emb_file, embeddings)
        
        # Обновить индекс
        self.index[key] = {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "model": model_name,
            "file_hash": file_hash,
            "fragments": fragments,
            "full_text": full_text,
            "embeddings_file": emb_file
        }
        self._save_index()
    
    def get_all_documents(self, model_name: str) -> list:
        """Получить все документы для заданной модели."""
        results = []
        for key, data in self.index.items():
            if data["model"] == model_name:
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
