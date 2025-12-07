"""
Класс OpenFile для открытия различных форматов файлов.
Пользователь может добавлять новые методы open_<format>(path) -> str
"""

import os
from bs4 import BeautifulSoup


class OpenFile:
    """Класс для открытия файлов различных форматов."""
    
    @staticmethod
    def open_txt(path: str) -> str:
        """Открыть текстовый файл."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    @staticmethod
    def open_md(path: str) -> str:
        """Открыть Markdown файл."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    @staticmethod
    def open_html(path: str) -> str:
        """Открыть HTML файл и извлечь текст."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text(separator='\n', strip=True)
    
    @staticmethod
    def open_pdf(path: str) -> str:
        """Открыть PDF файл."""
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or '')
            return '\n'.join(text)
        except ImportError:
            raise ImportError("pypdf не установлен. Установите: pip install pypdf")
    
    @staticmethod
    def open_docx(path: str) -> str:
        """Открыть DOCX файл."""
        try:
            from docx import Document
            doc = Document(path)
            return '\n'.join([para.text for para in doc.paragraphs])
        except ImportError:
            raise ImportError("python-docx не установлен. Установите: pip install python-docx")
    
    @classmethod
    def get_method(cls, method_name: str):
        """Получить метод по имени."""
        return getattr(cls, method_name, None)
    
    @classmethod
    def get_supported_formats(cls) -> list:
        """Получить список поддерживаемых форматов."""
        return [m.replace('open_', '') for m in dir(cls) if m.startswith('open_')]
