# app/streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
from predict import predict_transaction


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection Engine",
    layout="wide"
)

st.title("💳 Real-Time Fraud Detection Engine")


# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.header("⚙ Model Settings")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.4
)

fraud_loss = st.sidebar.number_input(
    "Loss per Fraud (₹)",
    value=5000
)

false_positive_cost = st.sidebar.number_input(
    "Cost per False Alarm (₹)",
    value=200
)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Transaction Analysis", "Model Performance"])


# ==================================================
# TAB 1 — Transaction Prediction
# ==================================================
with tab1:

    st.subheader("Enter Transaction Details")

    amount = st.number_input("Transaction Amount (₹)", min_value=0.0)
    hour = st.slider("Transaction Hour", 0, 23)
    is_international = st.selectbox("International Transaction?", [0, 1])
    transaction_gap = st.number_input("Minutes Since Last Transaction", min_value=0.0)
    location_risk = st.slider("Location Risk Score", 0.0, 1.0)
    device_risk = st.slider("Device Risk Score", 0.0, 1.0)
    merchant_risk = st.slider("Merchant Risk Score", 0.0, 1.0)

    if st.button("Analyze Transaction"):

        features = [
            amount,
            hour,
            is_international,
            transaction_gap,
            location_risk,
            device_risk,
            merchant_risk
        ]

        prediction, probability = predict_transaction(features, threshold)

        st.subheader("Fraud Risk Result")

        st.progress(float(probability))
        st.write(f"Fraud Probability: {probability:.2%}")

        if prediction == 1:
            st.error("⚠ High Fraud Risk - Recommend BLOCK")
            expected_loss = fraud_loss * probability
        else:
            st.success("✅ Low Fraud Risk - Approve")
            expected_loss = false_positive_cost * probability

        st.write(f"Estimated Business Risk: ₹{expected_loss:,.2f}")


        # -----------------------------
        # Feature Importance (Global)
        # -----------------------------
        st.subheader("Model Feature Importance")

        model = joblib.load("models/xgb_model.pkl")

        feature_names = [
            "amount",
            "hour",
            "is_international",
            "transaction_gap",
            "location_risk",
            "device_risk",
            "merchant_risk"
        ]

        importance = model.feature_importances_

        fig, ax = plt.subplots()
        ax.barh(feature_names, importance)
        ax.set_title("Feature Importance")
        st.pyplot(fig)


# ==================================================
# TAB 2 — Model Performance
# ==================================================
with tab2:

    st.subheader("Model Performance Dashboard")

    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    df = pd.read_csv("data/realistic_fraud_dataset_200k.csv")

    X = df.drop(["fraud", "transaction_id"], axis=1)
    y = df["fraud"]

    X_scaled = scaler.transform(X)

    y_proba = model.predict_proba(X_scaled)[:, 1]

    auc = roc_auc_score(y, y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    st.pyplot(fig_cm)
    plt.close(fig_cm)

    st.write(f"ROC AUC Score: {auc:.3f}")

    st.write(f"Fraud Ratio: {y.mean():.3f}")

    # -----------------------------
    # ROC Curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(y, y_proba)

    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr)
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    st.pyplot(fig1)

    # -----------------------------
    # Precision-Recall Curve
    # -----------------------------
    precision, recall, _ = precision_recall_curve(y, y_proba)

    fig2, ax2 = plt.subplots()
    ax2.plot(recall, precision)
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    st.pyplot(fig2)
