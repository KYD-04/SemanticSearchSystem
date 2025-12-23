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
from metrics import RelevanceMetric

app = Flask(__name__)

# Загрузка конфигурации
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()
storage = Storage(config['hash_folder'])
embedding_manager = EmbeddingManager(config['embedding_model'], device="cuda")


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
    
    if storage.is_indexed(filepath, config['embedding_model'], config.get('fragment_type')):
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


@app.route('/api/metrics')
def get_metrics():
    """Получить список доступных метрик."""
    return jsonify(RelevanceMetric.get_all_metrics())


@app.route('/api/clear_index', methods=['POST'])
def clear_index():
    """Очистить весь индекс и переиндексировать документы."""
    storage.clear_all()
    return jsonify({"success": True, "message": "Индекс очищен. Переиндексируйте документы."})


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
        
        yield f"data: {json.dumps({'type': 'init', 'files': all_files})}\n\n"
        
        to_process = [f for f in all_files if f['status'] == 'pending']
        total = len(to_process)
        
        if total == 0:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        
        total_size = sum(f['size'] for f in to_process)
        processed_size = 0
        start_time = time.time()
        
        for i, file_info in enumerate(to_process):
            filename = file_info['filename']
            filepath = file_info['filepath']
            file_size = file_info['size']
            
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
                    text = method(filepath)
                    fragment_type = config.get('fragment_type')
                    fragments = embedding_manager.fragment_text(text, fragment_type)
                    
                    if not fragments:
                        result.update({"success": False, "icon": "yellow"})
                    else:
                        total_frags = len(fragments)
                        batch_size = max(1, total_frags // 10)

                        all_embeddings = []
                        for j in range(0, total_frags, batch_size):
                            batch = fragments[j:j+batch_size]
                            embs = embedding_manager.get_embeddings(batch)
                            all_embeddings.append(embs)

                            file_progress = min(100, int((j + len(batch)) / total_frags * 100))
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
                            fragment_type=fragment_type,
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
    """Выполнить поиск с выбранными метриками."""
    data = request.json
    query = data.get('query', '').strip()
    selected_metrics = data.get('metrics', ['metric_cosine'])
    sort_by = data.get('sort_by', 'metric_cosine')
    
    if not query:
        return jsonify({"error": "Пустой запрос"}), 400
    
    # Получить эмбеддинг запроса для метрик с embed=True
    query_embedding = embedding_manager.get_embedding(query)
    documents = storage.get_all_documents(config['embedding_model'], config.get('fragment_type'))
    
    # Получить информацию о метриках
    all_metrics_info = {m['name']: m for m in RelevanceMetric.get_all_metrics()}
    
    results = []
    for doc in documents:
        fragments = doc['fragments']
        embeddings = doc['embeddings']
        
        for i, (fragment, emb) in enumerate(zip(fragments, embeddings)):
            # Вычислить все выбранные метрики
            scores = {}
            for metric_name in selected_metrics:
                metric_method = RelevanceMetric.get_method(metric_name)
                if metric_method:
                    metric_info = all_metrics_info.get(metric_name, {})
                    uses_embed = metric_info.get('embed', True)
                    
                    try:
                        if uses_embed:
                            score = metric_method(query_embedding, emb, embed=True)
                        else:
                            score = metric_method(query, fragment, embed=False)
                        scores[metric_name] = round(float(score), 4)
                    except:
                        scores[metric_name] = 0.0
            
            results.append({
                "filename": doc['filename'],
                "filepath": doc['filepath'],
                "fragment": fragment,
                "fragment_index": i,
                "scores": scores,
                "full_text": doc['full_text']
            })
    
    # Сортировка по выбранной метрике
    results.sort(key=lambda x: x['scores'].get(sort_by, 0), reverse=True)
    return jsonify(results[:50])


@app.route('/api/refine', methods=['POST'])
def refine_search():
    """Уточнить поиск на основе обратной связи."""
    import numpy as np
    
    data = request.json
    query = data.get('query', '').strip()
    positive_fragments = data.get('positive', [])  # [{filepath, fragment_index}]
    negative_fragments = data.get('negative', [])  # [{filepath, fragment_index}]
    selected_metrics = data.get('metrics', ['metric_cosine'])
    refine_metrics = data.get('refine_metrics', ['metric_cosine'])  # метрики для уточнения
    sort_by = data.get('sort_by', 'metric_cosine')
    
    if not query:
        return jsonify({"error": "Пустой запрос"}), 400
    
    # Получить исходный эмбеддинг запроса
    query_embedding = embedding_manager.get_embedding(query)
    documents = storage.get_all_documents(config['embedding_model'], config.get('fragment_type'))
    
    # Построить индекс документов для быстрого доступа
    doc_index = {doc['filepath']: doc for doc in documents}
    
    # Rocchio algorithm: q' = α*q + β*avg(pos) - γ*avg(neg)
    alpha, beta, gamma = 1.0, 0.75, 0.25
    
    # Собрать положительные и отрицательные эмбеддинги
    pos_embeddings = []
    neg_embeddings = []
    
    for item in positive_fragments:
        doc = doc_index.get(item['filepath'])
        if doc and item['fragment_index'] < len(doc['embeddings']):
            pos_embeddings.append(doc['embeddings'][item['fragment_index']])
    
    for item in negative_fragments:
        doc = doc_index.get(item['filepath'])
        if doc and item['fragment_index'] < len(doc['embeddings']):
            neg_embeddings.append(doc['embeddings'][item['fragment_index']])
    
    # Модифицировать вектор запроса
    modified_query = alpha * query_embedding
    if pos_embeddings:
        modified_query = modified_query + beta * np.mean(pos_embeddings, axis=0)
    if neg_embeddings:
        modified_query = modified_query - gamma * np.mean(neg_embeddings, axis=0)
    
    # Нормализовать
    modified_query = modified_query / (np.linalg.norm(modified_query) + 1e-8)
    
    # Получить информацию о метриках
    all_metrics_info = {m['name']: m for m in RelevanceMetric.get_all_metrics()}
    
    results = []
    for doc in documents:
        fragments = doc['fragments']
        embeddings = doc['embeddings']
        
        for i, (fragment, emb) in enumerate(zip(fragments, embeddings)):
            scores = {}
            for metric_name in selected_metrics:
                metric_method = RelevanceMetric.get_method(metric_name)
                if metric_method:
                    metric_info = all_metrics_info.get(metric_name, {})
                    uses_embed = metric_info.get('embed', True)
                    
                    try:
                        if uses_embed:
                            # Использовать модифицированный вектор для метрик уточнения
                            q_vec = modified_query if metric_name in refine_metrics else query_embedding
                            score = metric_method(q_vec, emb, embed=True)
                        else:
                            score = metric_method(query, fragment, embed=False)
                        scores[metric_name] = round(float(score), 4)
                    except:
                        scores[metric_name] = 0.0
            
            results.append({
                "filename": doc['filename'],
                "filepath": doc['filepath'],
                "fragment": fragment,
                "fragment_index": i,
                "scores": scores,
                "full_text": doc['full_text']
            })
    
    results.sort(key=lambda x: x['scores'].get(sort_by, 0), reverse=True)
    return jsonify(results[:50])


@app.route('/api/document/<path:filepath>')
def get_document(filepath):
    """Получить документ для предпросмотра."""
    documents = storage.get_all_documents(config['embedding_model'], config.get('fragment_type'))
    
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
