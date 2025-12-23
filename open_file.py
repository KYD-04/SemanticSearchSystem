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
        """Открыть PDF файл и вернуть текст, разделенный на абзацы."""
        try:
            import pdfplumber

            all_paragraphs = []

            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    # Извлекаем текст с use_text_flow для объединения строк в блоки
                    page_text = page.extract_text(
                        use_text_flow=True,
                        x_tolerance=3,
                        y_tolerance=3
                    )

                    if page_text:
                        # Разделяем текст на блоки по двойному переносу (абзацы)
                        raw_blocks = page_text.split('\n\n')

                        for block in raw_blocks:
                            # Очищаем блок: заменяем внутренние переносы на пробелы
                            # Это превращает "разорванный" PDF-текст в цельные абзацы
                            clean_block = " ".join(block.splitlines())

                            if clean_block.strip():
                                all_paragraphs.append(clean_block.strip())

            # Склеиваем все абзацы через двойной перенос
            return '\n\n'.join(all_paragraphs)

        except ImportError:
            raise ImportError("pdfplumber не установлен. Установите: pip install pdfplumber")
        except Exception as e:
            return f"Ошибка при чтении файла: {e}"
    
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
