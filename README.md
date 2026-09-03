# 🔮 Telco Customer Churn Prediction Engine
## 🔌 API Deployment (FastAPI)

The model is served as a production REST API with Pydantic data validation and an interactive Swagger UI.

### Running the API Locally
```bash
# Start Uvicorn ASGI server
uvicorn main:app --reload

An end-to-end Machine Learning project predicting customer churn using **XGBoost**, handling class imbalance via `scale_pos_weight`, and served via an interactive **Streamlit** dashboard.

## 📌 Business Objective
Retaining existing customers is 5-7x cheaper than acquiring new ones. This project identifies high-risk churners early, enabling targeted customer retention campaigns.

## 🚀 Key Performance Results
* **Best Model:** XGBoost Classifier (Tuned via `RandomizedSearchCV`)
* **ROC-AUC Score:** `0.846`
* **Recall (Churn Class):** `80%` *(Prioritized Recall over Accuracy to capture 80% of true churners)*

## 🛠️ Project Architecture & Features
* **EDA & Preprocessing:** Numerical median imputation, categorical One-Hot Encoding (`pd.get_dummies`).
* **Imbalance Handling:** Utilized native `scale_pos_weight=2.77` to balance class weights without synthetic resampling.
* **Regularization:** Early Stopping (`early_stopping_rounds=15`) on validation set to avoid overfitting.
* **Deployment:** Interface built with `Streamlit` utilizing pre-trained `joblib` artifacts.

## 💻 How to Run Locally
```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/Telco-Customer-Churn.git
cd Telco-Customer-Churn

# Install dependencies
pip install -r requirements.txt

# Run Streamlit Web Application
streamlit run app.py

 
