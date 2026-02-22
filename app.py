import sys
import os
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# ── Serve frontend ──
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

# ── API: Health check ──
@app.route('/api/status')
def status():
    return jsonify({"status": "ok", "message": "FedFortress API is running"})

# ── API: Run Baseline Training (SSE) ──
@app.route('/api/train/baseline', methods=['POST'])
def run_baseline_training():
    data = request.json or {}
    epochs = min(int(data.get('epochs', 3)), 5)

    def generate():
        try:
            from src.baseline import train_baseline
            for result in train_baseline(epochs=epochs):
                yield f"data: {json.dumps(result)}\n\n"
                time.sleep(0.05)
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'error': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no',
                             'Connection': 'keep-alive'})

# ── API: Run Federated Training (SSE) ──
@app.route('/api/train/federated', methods=['POST'])
def run_federated_training_api():
    data = request.json or {}

    max_samples = int(data.get('max_samples', 1000))
    # Automatically use quick_mode for smaller sample counts (fast demo)
    quick_mode = max_samples <= 2000

    config = {
        'aggregation':       data.get('aggregation', 'FedAvg'),
        'num_clients':       int(data.get('num_clients', 3)),
        'malicious_clients': int(data.get('malicious_clients', 1)),
        'rounds':            min(int(data.get('rounds', 3)), 5),
        'quick_mode':        quick_mode,
        'max_samples':       max_samples,
        'local_epochs':      int(data.get('local_epochs', 1)),
        'dp_enabled':        bool(data.get('dp_enabled', False)),
        'dp_epsilon':        float(data.get('dp_epsilon', 1.0)),
        'attack_type':       data.get('attack_type', 'noise_injection'),
    }

    def generate():
        try:
            from src.main import run_federated_training, save_training_results
            all_results = []
            for update in run_federated_training(**config):
                all_results.append(update)
                yield f"data: {json.dumps(update)}\n\n"
                time.sleep(0.05)

            # Save results (non-fatal if it fails)
            try:
                saved_path = save_training_results(
                    results=all_results,
                    aggregation=config['aggregation'],
                    num_clients=config['num_clients'],
                    malicious_clients=config['malicious_clients'],
                    rounds=config['rounds'],
                    quick_mode=config['quick_mode'],
                    max_samples=config['max_samples'],
                    dp_enabled=config['dp_enabled'],
                    dp_epsilon=config['dp_epsilon'],
                )
                yield f"data: {json.dumps({'done': True, 'saved_path': str(saved_path), 'total_rounds': len(all_results)})}\n\n"
            except Exception:
                yield f"data: {json.dumps({'done': True, 'total_rounds': len(all_results)})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'error': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no',
                             'Connection': 'keep-alive'})


if __name__ == '__main__':
    print("\n🛡️  FedFortress API Server")
    print("   Frontend: http://localhost:5000")
    print("   API:      http://localhost:5000/api/status\n")
    app.run(debug=True, port=5000, threaded=True)
