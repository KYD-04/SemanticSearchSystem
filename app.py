"""
Главное приложение системы семантического поиска.
"""

import os
import json
import time
import yaml
from flask import Flask, render_template, request, jsonify, Response

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


def get_file_size(filepath: str) -> int:
    """Получить размер файла."""
    try:
        return os.path.getsize(filepath)
    except:
        return 0


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
                "size": get_file_size(filepath),
                **status
            })
    
    return jsonify(files)


@app.route('/api/refresh', methods=['GET'])
def refresh_stream():
    """Обновить индекс с потоковой передачей прогресса (SSE)."""
    
    def generate():
        files_folder = config['files_folder']
        os.makedirs(files_folder, exist_ok=True)
        storage.clear_missing_files()
        
        # Собрать все файлы
        all_files = []
        for filename in os.listdir(files_folder):
            filepath = os.path.join(files_folder, filename)
            if os.path.isfile(filepath):
                status = get_file_status(filepath)
                all_files.append({
                    "filename": filename,
                    "filepath": filepath,
                    "size": get_file_size(filepath),
                    **status
                })
        
        # Отправить начальный список файлов
        yield f"data: {json.dumps({'type': 'init', 'files': all_files})}\n\n"
        
        # Найти файлы для обработки
        to_process = [f for f in all_files if f['status'] == 'pending']
        total = len(to_process)
        
        if total == 0:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        
        # Оценка времени на основе размера файлов
        total_size = sum(f['size'] for f in to_process)
        processed_size = 0
        start_time = time.time()
        
        for i, file_info in enumerate(to_process):
            filename = file_info['filename']
            filepath = file_info['filepath']
            file_size = file_info['size']
            
            # Отправить начало обработки
            yield f"data: {json.dumps({'type': 'start', 'filename': filename, 'index': i, 'total': total})}\n\n"
            
            ext = os.path.splitext(filepath)[1].lower().lstrip('.')
            handlers = config.get('file_handlers', {})
            method_name = handlers.get(ext)
            method = OpenFile.get_method(method_name) if method_name else None
            
            result = {"filename": filename}
            
            if method is None:
                result.update({"success": False, "icon": "red"})
            else:
                try:
                    # Извлечь текст
                    text = method(filepath)
                    fragments = embedding_manager.fragment_text(text, config['fragment_type'])
                    
                    if not fragments:
                        result.update({"success": False, "icon": "yellow"})
                    else:
                        # Обработка фрагментов с прогрессом
                        total_frags = len(fragments)
                        batch_size = max(1, total_frags // 10)
                        
                        all_embeddings = []
                        for j in range(0, total_frags, batch_size):
                            batch = fragments[j:j+batch_size]
                            embs = embedding_manager.get_embeddings(batch)
                            all_embeddings.append(embs)
                            
                            # Прогресс внутри файла
                            file_progress = min(100, int((j + len(batch)) / total_frags * 100))
                            
                            # Общий прогресс
                            elapsed = time.time() - start_time
                            current_processed = processed_size + (file_size * file_progress / 100)
                            if current_processed > 0:
                                speed = current_processed / elapsed
                                remaining_size = total_size - current_processed
                                eta_total = remaining_size / speed if speed > 0 else 0
                                eta_file = (file_size * (100 - file_progress) / 100) / speed if speed > 0 else 0
                            else:
                                eta_total = 0
                                eta_file = 0
                            
                            yield f"data: {json.dumps({'type': 'progress', 'filename': filename, 'progress': file_progress, 'eta_file': round(eta_file, 1), 'eta_total': round(eta_total, 1), 'file_index': i, 'total_files': total})}\n\n"
                        
                        import numpy as np
                        embeddings = np.vstack(all_embeddings)
                        
                        storage.save_document(
                            filepath=filepath,
                            model_name=config['embedding_model'],
                            fragments=fragments,
                            embeddings=embeddings,
                            full_text=text
                        )
                        result.update({"success": True, "icon": "green"})
                
                except Exception as e:
                    result.update({"success": False, "icon": "yellow", "error": str(e)})
            
            processed_size += file_size
            yield f"data: {json.dumps({'type': 'complete', **result})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    })


@app.route('/api/search', methods=['POST'])
def search():
    """Выполнить поиск."""
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"error": "Пустой запрос"}), 400
    
    query_embedding = embedding_manager.get_embedding(query)
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
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
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
    app.run(host='0.0.0.0', port=config['port'], debug=True, threaded=True)
