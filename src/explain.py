import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from data_loader import load_credit_data, preprocess_data


def generate_shap_explanations(
    model_path: str = "models/xgboost_model.joblib",
    output_dir: str = "reports"
):
    """
    Generates SHAP feature importance plots and audit trail explanations for compliance.
    """
    print("[INFO] Initiating SHAP Explainability Engine...")

    # 1. Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Trained model not found at {model_path}. Please run src/train.py first.")

    model = joblib.load(model_path)
    
    # 2. Load data for SHAP evaluation
    raw_data = load_credit_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(raw_data)

    # 3. Create SHAP TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    os.makedirs(output_dir, exist_ok=True)

    # 4. Save Global Summary Plot (Summary of all feature contributions)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    summary_plot_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Global SHAP Summary Plot saved at: {summary_plot_path}")

    # 5. Local Explanation Function for Single Applicant Decision Audit
    sample_idx = 0  # First applicant in test set
    applicant_data = X_test.iloc[[sample_idx]]
    applicant_shap = shap_values[sample_idx]

    print("\n" + "=" * 50)
    print(f"       INDIVIDUAL APPLICANT AUDIT TRAIN (Index {sample_idx})       ")
    print("=" * 50)
    
    # Extract top 3 features pushing risk HIGHER and LOWER
    feature_names = X_test.columns
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": applicant_data.values[0],
        "SHAP_Impact": applicant_shap.values
    }).sort_values(by="SHAP_Impact", key=abs, ascending=False)

    print("Top Risk Factors for this decision:")
    for _, row in shap_df.head(5).iterrows():
        impact_type = "INCREASED RISK" if row["SHAP_Impact"] > 0 else "DECREASED RISK"
        print(f" - {row['Feature']} = {row['Value']} -> {impact_type} (Impact: {row['SHAP_Impact']:.4f})")
    print("=" * 50 + "\n")

    return explainer, shap_values


if __name__ == "__main__":
    generate_shap_explanations()