# 🛡️ AI-Based UPI Fraud Detection System

<div align="center">

**An intelligent ML pipeline that scores every UPI transaction 0–1 for fraud risk — in real time.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)


</div>

---

## 🚨 The Problem

> **UPI fraud is a ₹2,000 Cr+ annual problem in India — and growing.**

India processes **10 billion+ UPI transactions every month**. The moment a fraudulent transaction completes, the money is transferred **instantly and is nearly impossible to recover.** Fraudsters exploit this with phishing links, fake payment requests, and social engineering — and traditional rule-based detection systems simply can't keep up.

**The core issue with existing systems:**
- ❌ Fixed rules that fraudsters easily learn to bypass
- ❌ Binary outputs — no risk scoring, just allow/block
- ❌ High false positive rates — blocking legitimate users
- ❌ No adaptability to new and evolving fraud patterns
- ❌ No behavioural understanding of individual users

**Our solution:** A machine learning pipeline that *learns* from historical patterns, understands user behaviour, and generates a **0–1 fraud risk score** for every transaction — enabling smarter, graded decision-making.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Behaviour-Based Analysis** | Tracks user spending habits, transaction frequency, and time-based patterns |
| 📊 **Fraud Risk Scoring** | Outputs a 0–1 probability score instead of a binary flag |
| 🌲 **Ensemble ML Models** | Combines 5 models for maximum accuracy and robustness |
| ⚡ **Real-Time Detection** | Designed to flag transactions before settlement completes |
| 🔍 **Explainable AI** | Feature importance analysis provides transparency into decisions |
| ⚖️ **Imbalance Handling** | SMOTE-equivalent upsampling addresses the <5% fraud class ratio |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
│         Raw UPI Transaction Data                            │
│   Amount · Time · User ID · Frequency · Type                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATA PREPROCESSING                           │
│  Handle missing values · Remove duplicates                  │
│  Encode categorical features · Normalize data               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING                            │
│  Avg transaction amount per user                            │
│  Transaction frequency in time window                       │
│  Night/day patterns · Sudden spending deviation             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ML MODEL ENSEMBLE                               │
│   Logistic Regression  →  Decision Tree                     │
│   Random Forest  →  Gradient Boosting  →  XGBoost          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              FRAUD RISK SCORING                              │
│         0.0 ──────────────────── 1.0                        │
│      Low Risk    Medium Risk    High Risk                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
              🚩 Flag Suspicious Transactions
```

---

## 📊 Model Performance

> Real results from Scikit-learn on 10K synthetic UPI-style transactions.

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | 83.5% | Baseline model |
| Decision Tree | ~88% | Captures non-linear patterns |
| Random Forest | ~95% | Primary fraud detection engine |
| Gradient Boosting | ~96% | Sequential error correction |
| **XGBoost + Ensemble** | **>97% (expected)** | Final model — in progress |

**Evaluation Metrics Used:** Accuracy · Precision · Recall · F1 Score · Confusion Matrix

> ⚠️ Class imbalance handled: Fraudulent transactions represent <5% of data. Addressed using SMOTE-equivalent upsampling with a final class ratio of ~1:5.

---

## 🗂️ Dataset

Real UPI transaction data is confidential and not publicly available. This project uses:

- **Kaggle Credit Card Fraud Dataset** — adapted for UPI-style transaction patterns
- **PaySim** — A financial mobile money simulator that closely replicates mobile payment behaviour
- **Synthetic Data** — 10,000 UPI-style transactions generated with realistic distributions

| Feature | Description |
|---------|-------------|
| `transaction_id` | Unique transaction identifier |
| `amount` | Transaction amount (INR) |
| `user_id` | Sender's user ID |
| `transaction_type` | P2P, P2M, etc. |
| `timestamp` | Date and time of transaction |
| `frequency` | Number of transactions in time window |
| `is_fraud` | Target label — 0 (Genuine) / 1 (Fraudulent) |

---

## ⚙️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.x |
| **Data Handling** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Notebook** | Jupyter Notebook |
| **Datasets** | Kaggle, PaySim |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### Clone & Run
```bash
# Clone the repository
git clone https://github.com/your-username/UPI-Fraud-Detection.git
cd UPI-Fraud-Detection

# Launch the notebook
jupyter notebook notebooks/upi_fraud_detection.ipynb
```

---

## 📁 Project Structure

```
UPI-Fraud-Detection/
│
├── data/
│   ├── raw/                       # Original dataset files
│   └── processed/                 # Cleaned and preprocessed data
│
├── notebooks/
│   └── upi_fraud_detection.ipynb  # Main Jupyter Notebook
│
├── models/
│   └── fraud_model.pkl            # Saved trained model
│
├── src/
│   ├── preprocessing.py           # Data cleaning functions
│   ├── features.py                # Feature engineering
│   └── predict.py                 # Fraud risk scoring
│
├── requirements.txt
└── README.md
```

---

## 🗓️ Project Timeline

| Phase | Period | Milestones |
|-------|--------|------------|
| Phase 1 | Mid Jan – End Jan | Problem analysis, literature review, finalizing objectives |
| Phase 2 | February | Dataset collection (Kaggle, PaySim), preprocessing, feature selection |
| Phase 3 | March | Feature engineering, training LR, DT, RF, Gradient Boosting |
| Phase 4 | April | XGBoost + Ensemble integration, testing, evaluation |
| Phase 5 | May | Final report, PPT, and project demo |

---

## 🎯 Use Cases & Real-World Scope

- 🏦 **Banks & NBFCs** — Detect unauthorized transactions before settlement
- 📱 **UPI Apps** — Integrate into PhonePe, Google Pay, Paytm for pre-settlement screening
- 🔔 **User Alerts** — Notify users in real time about suspicious activity on their accounts
- 📈 **Fraud Analytics** — Help payment providers understand emerging fraud trends

---

## 🔮 Future Scope

- [ ] Complete XGBoost + Ensemble voting model
- [ ] Build a real-time REST API for integration with payment apps
- [ ] Deploy as a web dashboard using Streamlit or Flask
- [ ] Incorporate LSTM for sequential transaction pattern detection
- [ ] Extend detection to NEFT, RTGS, and other Indian payment systems
- [ ] Add Graph Neural Networks for network-based fraud ring detection

---

## 👥 Team

| Name | Roll Number | Role |
|------|-------------|------|
| Saksham Sehrawat | 2301730235 | ML Pipeline & Model Development |
| Deepak Thapliyal | 2301730212 | Data Preprocessing & Feature Engineering |
| Dhairya Jashoria | 2301730241 | Research & Literature Review |
| Shikhar Bajpai | 2301730270 | Documentation & Presentation |

**Faculty Mentor:** Ms. Neha Kaushik
**Industry Mentor:** Mr. Deepak Khatri
**Institution:** K.R. Mangalam University, Gurugram — School of Engineering & Technology

---

## 📚 References

1. Reserve Bank of India — Digital Payment Systems and Fraud Risk Management Reports
2. Dal Pozzolo et al. — Adversarial Drift Detection in Fraud Detection, IEEE
3. Carcillo et al. — SCARFF: A Scalable Framework for Fraud Detection, Elsevier
4. Kaggle — Credit Card Fraud Detection Dataset
5. PaySim — Financial Mobile Money Simulator Dataset
6. Pedregosa et al. — Scikit-learn: Machine Learning in Python, JMLR

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

⭐ **If this project helped you, give it a star!** ⭐

*Built with 💙 by Team 26E3226 | K.R. Mangalam University*

</div>
