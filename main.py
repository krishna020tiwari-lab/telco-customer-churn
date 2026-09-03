import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Production REST API serving tuned XGBoost churn predictions.",
    version="1.0.0"
)

artifact = joblib.load("xgb_churn_model.joblib")
model = artifact['model']
model_features = artifact['feature_names']

class CustomerChurn(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaperlessBilling: str
    InternetService: str
    PaymentMethod: str
    TechSupport: str
    OnlineSecurity: str

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Telco Customer Churn Prediction API is active."
    }

@app.post("/predict")
def predict_churn(data:CustomerChurn):
    raw_dict = data.model_dump()
    raw_df = pd.DataFrame(raw_dict)

    encoded_df = pd.get_dummies(raw_df)
    final_input = encoded_df.reindex(columns=model_features, fill_value=0)

    churn_proba = float(model.predict(final_input[0][1]))
    is_high_risk = bool(churn_proba >= 0.5)

    return {
        "churn_probability": round(churn_proba, 4),
        "risk_status": "HIGH RISK" if is_high_risk else "LOW RISK",
        "recommended_action": "Trigger Retention Campaign" if is_high_risk else "No Action Required"
    }