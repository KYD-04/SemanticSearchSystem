"""
Главное приложение системы семантического поиска.
"""

import os
import yaml
from flask import Flask, render_template, request, jsonify

from open_file import OpenFile
from embeddings import EmbeddingManager
from storage import Storage

app = Flask(__name__)

# Загрузка конфигурации
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()
storage = Storage(config['hash_folder'])
embedding_manager = EmbeddingManager(config['embedding_model'])


def get_file_status(filepath: str) -> dict:
    """Получить статус файла."""
    ext = os.path.splitext(filepath)[1].lower().lstrip('.')
    handlers = config.get('file_handlers', {})
    
    if ext not in handlers:
        return {"status": "unsupported", "icon": "red", "message": "Формат не поддерживается"}
    
    method_name = handlers[ext]
    method = OpenFile.get_method(method_name)
    
    if method is None:
        return {"status": "no_method", "icon": "red", "message": f"Метод {method_name} не найден"}
    
    if storage.is_indexed(filepath, config['embedding_model']):
        return {"status": "indexed", "icon": "green", "message": "Проиндексирован"}
    
    return {"status": "pending", "icon": "yellow", "message": "Ожидает индексации"}


def process_file(filepath: str) -> dict:
    """Обработать один файл."""
    ext = os.path.splitext(filepath)[1].lower().lstrip('.')
    handlers = config.get('file_handlers', {})
    
    if ext not in handlers:
        return {"success": False, "error": "Формат не поддерживается", "icon": "red"}
    
    method_name = handlers[ext]
    method = OpenFile.get_method(method_name)
    
    if method is None:
        return {"success": False, "error": f"Метод {method_name} не найден", "icon": "red"}
    
    try:
        # Извлечь текст
        text = method(filepath)
        
        # Разбить на фрагменты
        fragments = embedding_manager.fragment_text(text, config['fragment_type'])
        
        if not fragments:
            return {"success": False, "error": "Нет текста для индексации", "icon": "yellow"}
        
        # Получить эмбеддинги
        embeddings = embedding_manager.get_embeddings(fragments)
        
        # Сохранить
        storage.save_document(
            filepath=filepath,
            model_name=config['embedding_model'],
            fragments=fragments,
            embeddings=embeddings,
            full_text=text
        )
        
        return {"success": True, "icon": "green", "fragments_count": len(fragments)}
    
    except Exception as e:
        return {"success": False, "error": str(e), "icon": "yellow"}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/files')
def get_files():
    """Получить список файлов с их статусами."""
    files_folder = config['files_folder']
    os.makedirs(files_folder, exist_ok=True)
    
    files = []
    for filename in os.listdir(files_folder):
        filepath = os.path.join(files_folder, filename)
        if os.path.isfile(filepath):
            status = get_file_status(filepath)
            files.append({
                "filename": filename,
                "filepath": filepath,
                **status
            })
    
    return jsonify(files)


@app.route('/api/refresh', methods=['POST'])
def refresh():
    """Обновить индекс - обработать все файлы."""
    files_folder = config['files_folder']
    os.makedirs(files_folder, exist_ok=True)
    
    # Очистить записи для удалённых файлов
    storage.clear_missing_files()
    
    results = []
    for filename in os.listdir(files_folder):
        filepath = os.path.join(files_folder, filename)
        if os.path.isfile(filepath):
            # Проверить, нужно ли индексировать
            if not storage.is_indexed(filepath, config['embedding_model']):
                result = process_file(filepath)
                results.append({"filename": filename, **result})
            else:
                results.append({"filename": filename, "success": True, "icon": "green", "skipped": True})
    
    return jsonify(results)


@app.route('/api/search', methods=['POST'])
def search():
    """Выполнить поиск."""
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"error": "Пустой запрос"}), 400
    
    # Получить эмбеддинг запроса
    query_embedding = embedding_manager.get_embedding(query)
    
    # Получить все документы
    documents = storage.get_all_documents(config['embedding_model'])
    
    results = []
    for doc in documents:
        fragments = doc['fragments']
        embeddings = doc['embeddings']
        
        for i, (fragment, emb) in enumerate(zip(fragments, embeddings)):
            similarity = embedding_manager.cosine_similarity(query_embedding, emb)
            results.append({
                "filename": doc['filename'],
                "filepath": doc['filepath'],
                "fragment": fragment,
                "fragment_index": i,
                "similarity": round(similarity, 4),
                "full_text": doc['full_text']
            })
    
    # Сортировка по убыванию сходства
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Вернуть топ-50
    return jsonify(results[:50])


@app.route('/api/document/<path:filepath>')
def get_document(filepath):
    """Получить документ для предпросмотра."""
    documents = storage.get_all_documents(config['embedding_model'])
    
    for doc in documents:
        if doc['filepath'] == filepath:
            return jsonify({
                "filename": doc['filename'],
                "full_text": doc['full_text'],
                "fragments": doc['fragments']
            })
    
    return jsonify({"error": "Документ не найден"}), 404


if __name__ == '__main__':
    print(f"Запуск сервера на http://localhost:{config['port']}")
    app.run(host='0.0.0.0', port=config['port'], debug=True)
