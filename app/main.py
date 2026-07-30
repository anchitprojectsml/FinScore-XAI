import os
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# 1. Initialize FastAPI Application
app = FastAPI(
    title="FinScore-XAI Credit Risk API",
    description="Enterprise Real-Time Credit Default Risk Scoring Engine with Explainable AI (SHAP)",
    version="1.0.0"
)

# 2. Paths to Artifacts
MODEL_PATH = "models/xgboost_model.joblib"

# Load Model and SHAP Explainer during startup
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    print("[INFO] Model and SHAP Explainer successfully loaded into API runtime.")
else:
    model = None
    explainer = None
    print("[WARNING] Model file not found. Ensure src/train.py is executed.")


# 3. Define Input Data Schema using Pydantic for validation
class CreditApplicantRequest(BaseModel):
    status: int = Field(..., example=2, description="Status of existing checking account")
    duration: int = Field(..., example=24, description="Duration in months")
    credit_history: int = Field(..., example=2, description="Credit history score")
    purpose: int = Field(..., example=1, description="Purpose of loan")
    amount: int = Field(..., example=2500, description="Credit amount")
    savings: int = Field(..., example=1, description="Savings account/bonds")
    employment_duration: int = Field(..., example=3, description="Present employment since")
    installment_rate: int = Field(..., example=4, description="Installment rate in percentage of disposable income")
    personal_status_sex: int = Field(..., example=2, description="Personal status and sex")
    other_debtors: int = Field(..., example=1, description="Other debtors / guarantors")
    present_residence: int = Field(..., example=2, description="Present residence since")
    property: int = Field(..., example=1, description="Property owned")
    age: int = Field(..., example=35, description="Age in years")
    other_installment_plans: int = Field(..., example=3, description="Other installment plans")
    housing: int = Field(..., example=1, description="Housing type")
    number_credits: int = Field(..., example=1, description="Number of existing credits at this bank")
    job: int = Field(..., example=2, description="Job category")
    people_liable: int = Field(..., example=1, description="Number of people being liable to provide maintenance for")
    telephone: int = Field(..., example=1, description="Telephone registered")
    foreign_worker: int = Field(..., example=1, description="Foreign worker status")


# 4. API Endpoints
@app.get("/health", tags=["System Health"])
def health_check():
    """Health check endpoint to verify API deployment status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "FinScore-XAI Credit Scoring Engine"
    }


@app.post("/predict", tags=["Scoring Engine"])
def predict_credit_risk(applicant: CreditApplicantRequest):
    """
    Evaluates credit risk for a given applicant and returns:
    - Default Probability (%)
    - Risk Category (LOW, MEDIUM, HIGH)
    - Top SHAP Reason Codes for Audit/Explainability
    """
    if model is None or explainer is None:
        raise HTTPException(status_code=500, detail="Model runtime not initialized.")

    # Convert request data to DataFrame matching feature structure
    input_data = pd.DataFrame([applicant.model_dump()])

    # Generate Model Prediction Probability
    default_prob = float(model.predict_proba(input_data)[0, 1])
    risk_score = round(default_prob * 100, 2)

    # Classify Risk Tier
    if risk_score < 30.0:
        risk_tier = "LOW RISK (APPROVED)"
    elif risk_score < 60.0:
        risk_tier = "MEDIUM RISK (MANUAL REVIEW)"
    else:
        risk_tier = "HIGH RISK (REJECTED)"

    # Generate SHAP Explanation for this specific applicant
    shap_vals = explainer(input_data)
    feature_impacts = pd.DataFrame({
        "feature": input_data.columns,
        "value": input_data.values[0],
        "shap_impact": shap_vals.values[0]
    }).sort_values(by="shap_impact", key=abs, ascending=False)

    top_reasons = []
    for _, row in feature_impacts.head(3).iterrows():
        direction = "Increased Risk" if row["shap_impact"] > 0 else "Reduced Risk"
        top_reasons.append({
            "feature": row["feature"],
            "value": float(row["value"]),
            "impact_direction": direction,
            "score_contribution": round(float(row["shap_impact"]), 4)
        })

    return {
        "credit_risk_score_percentage": risk_score,
        "decision_tier": risk_tier,
        "audit_reason_codes": top_reasons
    }