import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="UPI Fraud Detection", layout="wide")

# Custom CSS (Paytm style)
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    .card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
    }
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #00baf2;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='title'>💳 Paytm UPI Fraud Detection Demo</div>", unsafe_allow_html=True)
st.markdown("---")


df = pd.DataFrame(np.random.rand(1000, 30), columns=[f"V{i}" for i in range(30)])
df["Class"] = np.random.randint(0, 2, 1000)
X = df.drop('Class', axis=1)
y = df['Class']


model = RandomForestClassifier(n_estimators=80)
model.fit(X, y)


col1, col2 = st.columns([2, 1])

# LEFT SIDE (UPI UI)
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📲 Send Money")

    name = st.text_input("Recipient Name", "Rahul Sharma")
    upi = st.text_input("UPI ID", "rahul@paytm")
    amount = st.number_input("Amount (₹)", 1, 100000, 500)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Transaction Behaviour")

    txn_count = st.slider("Transactions in last hour", 0, 20, 2)
    new_device = st.checkbox("New Device Login")
    night = st.checkbox("Late Night Transaction")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Pay Now"):

        # Model input
        features = list(np.zeros(28))
        input_data = [50000] + features + [amount]
        input_array = np.array(input_data).reshape(1, -1)

        prob = model.predict_proba(input_array)[0][1]

        # Risk logic
        risk_score = prob
        if txn_count > 10:
            risk_score += 0.2
        if new_device:
            risk_score += 0.2
        if night:
            risk_score += 0.1
        if amount > 50000:
            risk_score += 0.2

        risk_score = min(risk_score, 1)

        # Classification
        if risk_score < 0.3:
            level = "🟢 LOW"
        elif risk_score < 0.7:
            level = "🟡 MEDIUM"
        else:
            level = "🔴 HIGH"

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("🔍 Transaction Result")
        st.write(f"Fraud Probability: **{prob:.2f}**")
        st.write(f"Risk Score: **{risk_score:.2f}**")
        st.write(f"Risk Level: {level}")

        if risk_score > 0.7:
            st.error("🚨 Payment Blocked: High Risk")
        elif risk_score > 0.3:
            st.warning("⚠️ Payment Pending Verification")
        else:
            st.success("✅ Payment Successful")

        st.markdown("</div>", unsafe_allow_html=True)

# RIGHT SIDE (COOL GRAPH ONLY)
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Fraud Insights")

    # Simple graph
    fraud_counts = df['Class'].value_counts()

    fig, ax = plt.subplots()
    ax.bar(["Normal", "Fraud"], fraud_counts.values)
    ax.set_title("Transaction Distribution")

    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("💡 AI + Behavioural Analysis")
    st.write("• Detects unusual patterns")
    st.write("• Prevents fraud in real-time")
    st.write("• Learns from past transactions")
    st.markdown("</div>", unsafe_allow_html=True)
