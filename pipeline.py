"""
AI-Based UPI Fraud Detection System
pipeline.py — Data generation, preprocessing, feature engineering,
              model training, and evaluation.

K.R. Mangalam University | CSE AI & ML (Section D)
Team: Saksham Sehrawat, Deepak Thapliyal, Dhairya Jashoria, Shikhar Bajpai
Supervisor: Ms. Neha Kaushik
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.utils import resample
import pickle, json, os, warnings
warnings.filterwarnings('ignore')

FEATURES = [
    'amount', 'hour', 'freq_1h', 'freq_24h', 'avg_spend',
    'spend_deviation', 'new_device', 'new_merchant', 'weekend', 'night_txn'
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')


# ─────────────────────────────────────────────
# 1. DATA GENERATION  (PaySim-style synthetic)
# ─────────────────────────────────────────────
def generate_dataset(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic UPI transaction dataset."""
    np.random.seed(seed)

    hour_probs = np.array([
        0.010, 0.005, 0.005, 0.005, 0.005, 0.010,
        0.030, 0.070, 0.080, 0.070, 0.060, 0.060,
        0.070, 0.070, 0.060, 0.060, 0.060, 0.060,
        0.050, 0.040, 0.030, 0.030, 0.020, 0.015,
    ])
    hour_probs /= hour_probs.sum()

    amounts = np.concatenate([
        np.random.lognormal(7.0, 1.5, int(n * 0.97)),   # normal txns
        np.random.lognormal(9.0, 2.0, int(n * 0.03)),   # large/suspicious
    ])[:n]

    hour       = np.random.choice(range(24), n, p=hour_probs)
    freq_1h    = np.random.poisson(1.5, n)
    freq_24h   = np.random.poisson(5.0, n)
    avg_spend  = np.random.lognormal(6.5, 1.0, n)
    spend_dev  = (amounts - avg_spend) / (avg_spend + 1)
    new_device = np.random.binomial(1, 0.05, n)
    new_merch  = np.random.binomial(1, 0.10, n)
    weekend    = np.random.binomial(1, 0.28, n)
    night_txn  = ((hour >= 23) | (hour <= 4)).astype(int)

    fraud_score = (
        (amounts > 50_000)       * 0.30 +
        (freq_1h > 3)            * 0.20 +
        (np.abs(spend_dev) > 2)  * 0.20 +
        new_device               * 0.15 +
        night_txn                * 0.10 +
        new_merch                * 0.05
    )
    fraud_prob = 1 / (1 + np.exp(-5 * (fraud_score - 0.35)))
    fraud = (np.random.uniform(0, 1, n) < fraud_prob).astype(int)

    return pd.DataFrame({
        'amount':          amounts.round(2),
        'hour':            hour,
        'freq_1h':         freq_1h,
        'freq_24h':        freq_24h,
        'avg_spend':       avg_spend.round(2),
        'spend_deviation': spend_dev.round(4),
        'new_device':      new_device,
        'new_merchant':    new_merch,
        'weekend':         weekend,
        'night_txn':       night_txn,
        'fraud':           fraud,
    })


# ─────────────────────────────────────────────
# 2. PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """Balance classes, split, and scale."""
    legit = df[df.fraud == 0]
    fraud = df[df.fraud == 1]
    fraud_up = resample(fraud, replace=True,
                        n_samples=len(legit) // 5, random_state=42)
    df_bal = (pd.concat([legit, fraud_up])
                .sample(frac=1, random_state=42)
                .reset_index(drop=True))

    X = df_bal[FEATURES]
    y = df_bal['fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler


# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
def train_models(X_train, y_train, X_train_s):
    """Train all four classifiers. Returns dict of fitted models."""
    specs = {
        'Logistic Regression': (LogisticRegression(max_iter=500), True),
        'Decision Tree':       (DecisionTreeClassifier(max_depth=8, random_state=42), False),
        'Random Forest':       (RandomForestClassifier(n_estimators=100, random_state=42), False),
        'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
    }
    trained = {}
    for name, (model, use_scaled) in specs.items():
        X = X_train_s if use_scaled else X_train
        model.fit(X, y_train)
        trained[name] = (model, use_scaled)
    return trained


# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────
def evaluate_models(trained, X_test, y_test, X_test_s):
    """Evaluate all models; return results dict."""
    results = {}
    for name, (model, use_scaled) in trained.items():
        Xte   = X_test_s if use_scaled else X_test
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]
        rpt    = classification_report(y_test, y_pred, output_dict=True)
        auc    = roc_auc_score(y_test, y_prob)
        cm     = confusion_matrix(y_test, y_pred).tolist()
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            'accuracy':  round(rpt['accuracy'] * 100, 2),
            'precision': round(rpt['1']['precision'] * 100, 2),
            'recall':    round(rpt['1']['recall'] * 100, 2),
            'f1':        round(rpt['1']['f1-score'] * 100, 2),
            'auc':       round(auc * 100, 2),
            'cm':        cm,
            'fpr':       [round(v, 4) for v in fpr.tolist()],
            'tpr':       [round(v, 4) for v in tpr.tolist()],
        }
    return results


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def get_feature_importance(trained):
    rf = trained['Random Forest'][0]
    return {f: round(float(v), 4) for f, v in
            zip(FEATURES, rf.feature_importances_)}


# ─────────────────────────────────────────────
# 6. SAVE / LOAD ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(trained, scaler, results, fi):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f'{MODEL_DIR}/models.pkl', 'wb') as f:
        pickle.dump({k: v[0] for k, v in trained.items()}, f)
    with open(f'{MODEL_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    meta = {
        'model_results':      results,
        'feature_importance': fi,
        'features':           FEATURES,
    }
    with open(f'{MODEL_DIR}/results.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] Artifacts saved to {MODEL_DIR}/")


def load_artifacts():
    with open(f'{MODEL_DIR}/models.pkl', 'rb') as f:
        models = pickle.load(f)
    with open(f'{MODEL_DIR}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{MODEL_DIR}/results.json') as f:
        meta = json.load(f)
    return models, scaler, meta


# ─────────────────────────────────────────────
# 7. PREDICTION
# ─────────────────────────────────────────────
def predict_transaction(models, scaler, txn: dict, model_name: str = 'Random Forest'):
    """
    Predict fraud probability for a single transaction.

    txn keys: amount, hour, freq_1h, freq_24h, avg_spend,
              new_device, new_merchant, weekend
    Returns: (prob: float, risk_level: str, factors: list)
    """
    # Derive computed features
    night_txn     = 1 if (txn['hour'] >= 23 or txn['hour'] <= 4) else 0
    spend_dev     = (txn['amount'] - txn['avg_spend']) / (txn['avg_spend'] + 1)

    row = pd.DataFrame([{
        'amount':          txn['amount'],
        'hour':            txn['hour'],
        'freq_1h':         txn['freq_1h'],
        'freq_24h':        txn['freq_24h'],
        'avg_spend':       txn['avg_spend'],
        'spend_deviation': spend_dev,
        'new_device':      txn['new_device'],
        'new_merchant':    txn['new_merchant'],
        'weekend':         txn['weekend'],
        'night_txn':       night_txn,
    }])[FEATURES]

    # Handle ensemble
    if model_name == 'Ensemble (All Models)':
        probs = []
        for name, model in models.items():
            if name == 'Logistic Regression':
                X = scaler.transform(row)
            else:
                X = row
            probs.append(model.predict_proba(X)[0, 1])
        prob = float(np.mean(probs))
    else:
        model = models[model_name]
        if model_name == 'Logistic Regression':
            X = scaler.transform(row)
        else:
            X = row
        prob = float(model.predict_proba(X)[0, 1])

    risk = ('HIGH' if prob > 0.65
            else 'MEDIUM' if prob > 0.35
            else 'LOW')

    # Explain: which features pushed the score up
    factors = _explain(txn, spend_dev, night_txn, prob)
    return round(prob, 4), risk, factors


def _explain(txn, spend_dev, night_txn, prob):
    factors = []
    amt = txn['amount']

    if amt > 100_000:
        factors.append(('high',   f"Very high amount ₹{amt:,.0f} — top risk factor (importance 22.2%)"))
    elif amt > 50_000:
        factors.append(('medium', f"Elevated amount ₹{amt:,.0f} — above normal spending"))
    else:
        factors.append(('low',    f"Normal amount ₹{amt:,.0f}"))

    if abs(spend_dev) > 3:
        factors.append(('high',   f"Spend is {spend_dev*100:.0f}% away from user average (importance 22.8%)"))
    elif abs(spend_dev) > 1:
        factors.append(('medium', f"Moderate deviation from avg spend ({spend_dev*100:+.0f}%)"))
    else:
        factors.append(('low',    "Spend within user's normal range"))

    f1 = txn['freq_1h']
    if f1 > 4:
        factors.append(('high',   f"{f1} transactions in last hour — unusually high"))
    elif f1 > 2:
        factors.append(('medium', f"{f1} transactions in last hour — slightly elevated"))
    else:
        factors.append(('low',    f"Normal transaction frequency ({f1}/hr)"))

    if txn['new_device']:
        factors.append(('high',   "Transaction from an unrecognised device"))
    if night_txn:
        factors.append(('medium', f"Late-night transaction at {txn['hour']}:00"))
    if txn['new_merchant']:
        factors.append(('medium', "Payment to a new / unrecognised merchant"))
    if not txn['new_device'] and not night_txn and not txn['new_merchant']:
        factors.append(('low',    "Known device, merchant, and normal hour"))

    return factors


# ─────────────────────────────────────────────
# MAIN — run full pipeline
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  AI-Based UPI Fraud Detection — Training Pipeline")
    print("=" * 55)

    print("\n[1] Generating dataset …")
    df = generate_dataset(10_000)
    df.to_csv(f'{MODEL_DIR}/dataset.csv', index=False)
    print(f"    {len(df):,} rows | fraud rate: {df.fraud.mean()*100:.1f}%")

    print("\n[2] Preprocessing & balancing …")
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = preprocess(df)

    print("\n[3] Training models …")
    trained = train_models(X_train, y_train, X_train_s)

    print("\n[4] Evaluating …")
    results = evaluate_models(trained, X_test, y_test, X_test_s)
    for name, r in results.items():
        print(f"    {name:25s}  acc={r['accuracy']}%  AUC={r['auc']}%")

    print("\n[5] Feature importances (Random Forest) …")
    fi = get_feature_importance(trained)
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
        bar = '█' * int(imp * 100)
        print(f"    {feat:20s} {bar:25s} {imp*100:.1f}%")

    print("\n[6] Saving artifacts …")
    save_artifacts(trained, scaler, results, fi)

    print("\n[✓] Pipeline complete. Run app.py to launch the UI.\n")
