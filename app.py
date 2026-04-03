import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# ── ensure pipeline is on path ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import load_artifacts, predict_transaction, generate_dataset, preprocess, train_models, evaluate_models, get_feature_importance, save_artifacts, MODEL_DIR

# ── load (or train) artifacts ────────────────────────────────
MODEL_PKL = os.path.join(MODEL_DIR, 'models.pkl')
if not os.path.exists(MODEL_PKL):
    print("[!] No saved models found — training now …")
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = generate_dataset(10_000)
    df.to_csv(os.path.join(MODEL_DIR, 'dataset.csv'), index=False)
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = preprocess(df)
    trained = train_models(X_train, y_train, X_train_s)
    results = evaluate_models(trained, X_test, y_test, X_test_s)
    fi = get_feature_importance(trained)
    save_artifacts(trained, scaler, results, fi)

MODELS, SCALER, META = load_artifacts()
print(f"[✓] Models loaded: {list(MODELS.keys())}")

# ── read the HTML template ───────────────────────────────────
HTML_PATH = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
with open(HTML_PATH, 'r', encoding='utf-8') as f:
    HTML_TEMPLATE = f.read()

# ── inject model metadata into page ─────────────────────────
RESULTS_JS = json.dumps(META['model_results'])
FI_JS = json.dumps(META['feature_importance'])

HTML_PAGE = (HTML_TEMPLATE
             .replace('__MODEL_RESULTS__', RESULTS_JS)
             .replace('__FEATURE_IMPORTANCE__', FI_JS))

# ════════════════════════════════════════════════════════════════
class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ('/', '/index.html'):
            self._send(200, 'text/html', HTML_PAGE.encode())
        else:
            self._send(404, 'text/plain', b'Not found')

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == '/predict':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)

            try:
                data = json.loads(body)

                txn = {
                    'amount': float(data.get('amount', 1000)),
                    'hour': int(data.get('hour', 12)),
                    'freq_1h': int(data.get('freq_1h', 1)),
                    'freq_24h': int(data.get('freq_24h', 4)),
                    'avg_spend': float(data.get('avg_spend', 3000)),
                    'new_device': int(data.get('new_device', 0)),
                    'new_merchant': int(data.get('new_merchant', 0)),
                    'weekend': int(data.get('weekend', 0)),
                }

                model_name = data.get('model', 'Random Forest')

                prob, risk, factors = predict_transaction(
                    MODELS, SCALER, txn, model_name
                )

                resp = {
                    'prob': round(prob * 100, 2),
                    'risk': risk,
                    'factors': factors,
                    'model_used': model_name,
                }

                self._send(200, 'application/json', json.dumps(resp).encode())

            except Exception as e:
                self._send(500, 'application/json',
                           json.dumps({'error': str(e)}).encode())
        else:
            self._send(404, 'text/plain', b'Not found')

    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    PORT = 8000
    server = HTTPServer(('', PORT), Handler)

    print(f"\n{'='*50}")
    print(f"  FraudShield AI — UPI Fraud Detection System")
    print(f"{'='*50}")
    print(f"  Server running at http://localhost:{PORT}")
    print(f"  Models loaded: {len(MODELS)}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[✓] Server stopped.")
