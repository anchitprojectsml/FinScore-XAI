import os
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from data_loader import load_credit_data, preprocess_data


def train_credit_model(models_dir: str = "models"):
    """
    Trains an XGBoost model on processed credit risk data and saves model artifacts.
    """
    print("[INFO] Initiating Model Training Pipeline...")

    # 1. Load and Preprocess Data
    raw_data = load_credit_data()
    X_train, X_test, y_train, y_test, encoders = preprocess_data(raw_data)

    # 2. Calculate Scale Position Weight for Class Imbalance
    # Ratio of negative class (Good Credit) to positive class (Default Risk)
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0

    print(f"[INFO] Applied Class Imbalance Ratio (scale_pos_weight): {scale_weight:.2f}")

    # 3. Initialize & Train XGBoost Model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    print("[SUCCESS] XGBoost Model Training Completed.")

    # 4. Evaluate Model Performance
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 50)
    print("           MODEL PERFORMANCE METRICS           ")
    print("=" * 50)
    print(f"Accuracy Score : {accuracy:.4f}")
    print(f"ROC-AUC Score  : {roc_auc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Good Credit (0)", "Default Risk (1)"]))
    print("=" * 50 + "\n")

    # 5. Save Artifacts
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "xgboost_model.joblib")
    encoders_path = os.path.join(models_dir, "encoders.joblib")

    joblib.dump(model, model_path)
    joblib.dump(encoders, encoders_path)

    print(f"[SUCCESS] Model artifact saved at: {model_path}")
    print(f"[SUCCESS] Encoders artifact saved at: {encoders_path}")

    return model, X_test, y_test


if __name__ == "__main__":
    train_credit_model()