# FinScore-XAI: Production-Ready Credit Default Risk Engine

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)
![SHAP](https://img.shields.io/badge/XAI-SHAP-brightgreen)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-success)

**FinScore-XAI** is an enterprise-grade, explainable credit risk assessment system built to predict loan default probability and generate transparent decision reasons for financial workflows.

Designed with **XGBoost, FastAPI, SHAP, and Docker**, this project combines machine learning performance, real-time inference, and audit-ready explainability in a production-style microservice architecture.

---

## 🚀 Project Highlights

- **Credit Risk Prediction:** Predicts the likelihood of loan default using supervised machine learning.
- **Explainable AI:** Uses SHAP to generate top reason codes behind every prediction.
- **Production Ready:** Exposes predictions through a FastAPI microservice.
- **Scalable Deployment:** Containerized with Docker for consistent setup across environments.
- **Compliance Friendly:** Built with interpretability in mind for banking and audit use cases.
- **Multi-Model Development:** Evaluated multiple baseline models before selecting the final production model.

---

## 🧠 Problem Statement

Financial institutions need more than just a prediction model. They need a system that can:

- assess credit default risk accurately,
- explain why a customer is high risk,
- support transparent lending decisions,
- and align with compliance and audit requirements.

A black-box model alone is not enough in regulated environments.  
**FinScore-XAI** solves this by combining **accuracy + interpretability + deployment readiness**.

---

## 🏗️ System Architecture

```text
[ Applicant Data ]
       │
       ▼
[ FastAPI Input Layer ]
       │
       ▼
[ Pydantic Validation ]
       │
       ▼
[ XGBoost Risk Model ]
       │
       ├──► Default Probability Score
       │
       ▼
[ SHAP Explanation Engine ]
       │
       ├──► Top 3 Reason Codes
       │
       ▼
[ Final JSON Response ]
```

---

## ✨ Key Features

- **Imbalanced Data Handling:** Uses `scale_pos_weight` to handle severe class imbalance in default prediction.
- **Real-Time Prediction:** FastAPI delivers low-latency responses for instant scoring.
- **Model Explainability:** SHAP identifies the most influential features for each prediction.
- **Structured Validation:** Pydantic ensures clean and reliable input payloads.
- **Dockerized Workflow:** Easy deployment across local, staging, and production environments.
- **Model Comparison:** Compared baseline ML algorithms to improve robustness and select the best-performing model.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.10 |
| ML Models | XGBoost, Logistic Regression, Random Forest, Scikit-learn |
| API Framework | FastAPI, Uvicorn |
| Explainability | SHAP |
| Validation | Pydantic |
| Containerization | Docker |
| Data Handling | Pandas, NumPy |

---

## 📌 Workflow

1. Applicant sends loan-related data to the API.
2. FastAPI validates the request payload.
3. ML model generates default risk probability.
4. SHAP extracts feature-level explanations.
5. API returns prediction score and reason codes in JSON format.

---

## 🔍 Sample Request

```json
{
  "age": 35,
  "income": 45000,
  "credit_score": 620,
  "loan_amount": 150000,
  "employment_years": 3
}
```

---

## 📤 Sample Response

```json
{
  "default_probability": 0.82,
  "risk_level": "High Risk",
  "reason_codes": [
    "High credit utilization ratio",
    "Low income stability relative to loan amount",
    "Recent missed or delayed payments"
  ]
}
```

---

## 🔗 API Endpoints

### `POST /predict`
Returns default probability and risk explanation for a given applicant.

### `GET /health`
Returns API health status.

### `GET /model-info`
Returns model version and basic metadata.

---

## ⚙️ Installation

```bash
git clone https://github.com/anchitprojectsml/FinScore-XAI.git
cd FinScore-XAI
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
uvicorn app.main:app --reload
```

Then open:

```bash
http://127.0.0.1:8000/docs
```

---

## 🐳 Run with Docker

```bash
docker build -t finscore-xai .
docker run -p 8000:8000 finscore-xai
```

---

## 📊 Model Impact

This project demonstrates the ability to build a complete machine learning solution that includes:

- data preprocessing,
- class imbalance handling,
- model training and evaluation,
- explainable AI outputs,
- API deployment,
- and containerized production setup.

It is a strong portfolio project for roles in **Machine Learning, Data Science, and AI Engineering**.

---

## 📈 Business Value

- Helps lenders make faster credit decisions.
- Improves trust through explainable predictions.
- Supports compliance and audit review.
- Reduces manual underwriting effort.
- Demonstrates real-world ML deployment skills.

---

## 🔮 Future Improvements

- Add model monitoring and drift detection.
- Introduce a retraining pipeline with new data.
- Deploy on cloud with CI/CD using AWS or GCP.
- Add role-based authentication for API security.
- Build a frontend dashboard for risk visualization.

---

## 📁 Folder Structure

```text
FinScore-XAI/
│
├── app/
│   ├── main.py
│   ├── model.py
│   ├── schemas.py
│   └── utils.py
│
├── models/
│   └── xgboost_model.pkl
│
├── notebooks/
│   └── training.ipynb
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 👨‍💻 Author

**Anchit Shrivastava**  
Aspiring Machine Learning Engineer | Data Analyst  
GitHub: [anchitprojectsml](https://github.com/anchitprojectsml)  
Focused on building practical, job-ready AI and data solutions.

---

## 📄 License

This project is open for learning and portfolio demonstration purposes.
