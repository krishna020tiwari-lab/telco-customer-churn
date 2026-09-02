import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Telco Churn Predictor", page_icon="🔮", layout="wide")
st.markdown("<h1 style='text-align: center; color: #6A0DAD;'>🔮 Telco Customer Churn Prediction Engine</h1>", unsafe_allow_html=True)
st.write("Use this interactive dashboard to estimate churn risk based on customer subscription details.")


@st.cache_resource
def load_artifact():
    return joblib.load('xgb_churn_model.joblib')

artifact = load_artifact()
model = artifact['model']
model_features = artifact['feature_names']


st.markdown("### 📋 Customer Information")
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("📅 Tenure (Months)", min_value=0, max_value=100, value=12)
    contract = st.selectbox("📑 Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("🧾 Paperless Billing", ["Yes", "No"])

with col2:
    monthly_charges = st.number_input("💵 Monthly Charges ($)", min_value=0.0, value=65.0)
    internet = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("💳 Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col3:
    total_charges = st.number_input("💰 Total Charges ($)", min_value=0.0, value=780.0)
    tech_support = st.selectbox("🛠 Tech Support", ["Yes", "No", "No internet service"])
    online_security = st.selectbox("🔒 Online Security", ["Yes", "No", "No internet service"])


if st.button("🚀 Analyze Churn Risk", type="primary"):
    raw_input = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'InternetService': internet,
        'PaymentMethod': payment,
        'TechSupport': tech_support,
        'OnlineSecurity': online_security
    }])

    encoded_input = pd.get_dummies(raw_input)
    final_input = encoded_input.reindex(columns=model_features, fill_value=0)

    churn_proba = model.predict_proba(final_input)[0][1]

    st.markdown("---")
    st.subheader("Results")

    col_res1, col_res2 = st.columns(2)
    col_res1.metric(label="Churn Probability", value=f"{churn_proba * 100:.1f}%")

    if churn_proba >= 0.5:
        col_res2.error("⚠️ Risk Status: HIGH RISK OF CHURN")
    else:
        col_res2.success("✅ Risk Status: LOW RISK (RETENTION LIKELY)")


    st.progress(int(churn_proba * 100))

    st.markdown("#### 📈 Probability Breakdown")
    st.bar_chart(pd.DataFrame({
        "Churn Risk": [churn_proba],
        "Retention Likely": [1 - churn_proba]
    }))
