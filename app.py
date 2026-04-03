import json
import os
import sys
from flask import Flask, request, jsonify

# ── ensure pipeline is on path ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import (
    load_artifacts,
    predict_transaction,
    generate_dataset,
    preprocess,
    train_models,
    evaluate_models,
    get_feature_importance,
    save_artifacts,
    MODEL_DIR
)

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

# ── load trained artifacts ──────────────────────────────────
MODELS, SCALER, META = load_artifacts()
print(f"[✓] Models loaded: {list(MODELS.keys())}")

# ── read HTML template ──────────────────────────────────────
HTML_PATH = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')

with open(HTML_PATH, 'r', encoding='utf-8') as f:
    HTML_TEMPLATE = f.read()

# ── inject model metadata into HTML ─────────────────────────
RESULTS_JS = json.dumps(META['model_results'])
FI_JS = json.dumps(META['feature_importance'])

HTML_PAGE = (
    HTML_TEMPLATE
    .replace('__MODEL_RESULTS__', RESULTS_JS)
    .replace('__FEATURE_IMPORTANCE__', FI_JS)
)

# ── Flask App ───────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def home():
    return HTML_PAGE


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

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

        return jsonify({
            'prob': round(prob * 100, 2),
            'risk': risk,
            'factors': factors,
            'model_used': model_name,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Run Server ──────────────────────────────────────────────
if __name__ == '__main__':
    PORT = int(os.environ.get("PORT", 5000))

    print(f"\n{'='*50}")
    print(f"  FraudShield AI — UPI Fraud Detection System")
    print(f"{'='*50}")
    print(f"  Server running on port {PORT}")
    print(f"  Models loaded: {len(MODELS)}")
    print(f"{'='*50}\n")

    app.run(host="0.0.0.0", port=PORT)
