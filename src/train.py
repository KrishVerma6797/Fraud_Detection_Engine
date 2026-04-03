# src/train.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

def train_model():

    df = pd.read_csv("data/realistic_fraud_dataset_200k.csv")

    X = df.drop(["fraud", "transaction_id"], axis=1)
    y = df["fraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (len(y) - sum(y)) / sum(y)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    joblib.dump(model, "models/xgb_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model Saved Successfully!")

if __name__ == "__main__":
    train_model()
