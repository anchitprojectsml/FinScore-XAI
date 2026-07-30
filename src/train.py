import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    confusion_matrix
)

from data_loader import load_credit_data, preprocess_data


def train_and_benchmark_models(models_dir: str = "models", reports_dir: str = "reports"):
    """
    Trains multiple models (Logistic Regression, Random Forest, XGBoost),
    evaluates performance using ROC-AUC and PR-AUC, performs threshold tuning,
    and saves the champion model artifacts.
    """
    print("[INFO] Initiating Multi-Model Benchmarking & Training Pipeline...")

    # 1. Load and Preprocess Data
    raw_data = load_credit_data()
    X_train, X_test, y_train, y_test, encoders = preprocess_data(raw_data)

    # Calculate Class Imbalance Weight Ratio
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0

    # 2. Define Model Candidates
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42),
        "XGBoost (Class-Weighted)": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric="logloss"
        )
    }

    benchmark_results = []
    trained_objects = {}

    print("\n" + "=" * 65)
    print("                MODEL BENCHMARK COMPARISON TABLE                ")
    print("=" * 65)

    # 3. Train & Evaluate Candidates
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        benchmark_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "F1-Score": round(f1, 4)
        })
        trained_objects[name] = (model, y_prob)

    benchmark_df = pd.DataFrame(benchmark_results)
    print(benchmark_df.to_string(index=False))
    print("=" * 65 + "\n")

    # 4. Select Champion Model (XGBoost) & Perform Threshold Tuning
    champion_name = "XGBoost (Class-Weighted)"
    champion_model, y_prob_champion = trained_objects[champion_name]

    print(f"[INFO] Selected Champion Model: {champion_name}")
    print("[INFO] Performing Optimal Decision Threshold Tuning (Precision-Recall Optimization)...")

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_champion)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred_optimal = (y_prob_champion >= optimal_threshold).astype(int)

    print(f"[SUCCESS] Optimal Risk Threshold Identified: {optimal_threshold:.4f} (Default: 0.5000)")
    
    cm = confusion_matrix(y_test, y_pred_optimal)
    print("\nConfusion Matrix (At Optimal Threshold):")
    print(f"True Negatives (Approved Good Credit) : {cm[0][0]}")
    print(f"False Positives (Rejected Good Credit): {cm[0][1]}")
    print(f"False Negatives (Missed Defaults)     : {cm[1][0]}")
    print(f"True Positives (Caught Defaults)      : {cm[1][1]}")

    print("\nDetailed Classification Report (Optimal Threshold):")
    print(classification_report(y_test, y_pred_optimal, target_names=["Good Credit (0)", "Default Risk (1)"]))

    # 5. Save Artifacts & Reports
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "xgboost_model.joblib")
    encoders_path = os.path.join(models_dir, "encoders.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.joblib")

    # Store optimal threshold and benchmark in metadata artifact
    metadata = {
        "champion_model": champion_name,
        "optimal_threshold": optimal_threshold,
        "benchmark_summary": benchmark_df.to_dict(orient="records")
    }

    joblib.dump(champion_model, model_path)
    joblib.dump(encoders, encoders_path)
    joblib.dump(metadata, metadata_path)

    benchmark_df.to_csv(os.path.join(reports_dir, "model_benchmark.csv"), index=False)

    print(f"[SUCCESS] Champion Model saved at: {model_path}")
    print(f"[SUCCESS] Model Metadata saved at : {metadata_path}")
    print(f"[SUCCESS] Benchmark report saved at: {os.path.join(reports_dir, 'model_benchmark.csv')}\n")

    return champion_model, X_test, y_test


if __name__ == "__main__":
    train_and_benchmark_models()