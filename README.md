## 💳 Real-Time Fraud Detection Engine

An end-to-end Machine Learning system built using **XGBoost** to detect fraudulent financial transactions in real time.

This project implements the complete ML lifecycle:

- Data preprocessing
- Model training
- Performance evaluation
- Model explainability (SHAP)
- Deployment using Streamlit

---

## 📌 Project Overview

Fraud detection is a highly imbalanced **binary classification problem** where the goal is to predict whether a transaction is:

- `1` → Fraudulent  
- `0` → Legitimate  

This system provides:

- Fraud probability score
- Adjustable decision threshold
- Business risk estimation
- Model performance dashboard
- Feature importance visualization

---

## 🏗️ Project Structure

```
ML PROJECT - FRAUD DET/
│
├── app/
│   ├── predict.py
│   └── streamlit_app.py
│
|──  generate_data.py
|
├── data/
│   └── realistic_fraud_dataset_200k.csv
│
├── models/
│   ├── scaler.pkl
│   └── xgb_model.pkl
│
├── src/
│   ├── train.py
│   ├── preprocess.py
│   ├── evaluate.py
│   └── explain.py
│
├── requirements.txt
└── README.md
```


## 🛠️ Installation & Setup

### Step 1: Clone Repository

```bash
git clone <your-repository-link>
cd ML PROJECT - FRAUD DET
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model

```bash
python src/train.py
```

### Step 4: Run the Application

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Model Performance Metrics

The system evaluates performance using:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC

Visual performance monitoring includes:

- ROC Curve
- Precision-Recall Curve
- Feature Importance Plot

---

## 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV
- SMOTE for class imbalance handling
- FastAPI REST endpoint for production inference
- Docker containerization
- Model monitoring and drift detection

---

## 📦 Tech Stack

- Python
- XGBoost
- Scikit-Learn
- Pandas
- NumPy
- SHAP
- Streamlit
- Matplotlib

---

## 📄 License

MIT License
