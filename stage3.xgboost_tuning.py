import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv('Telco_Customer_churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = df.drop(columns=['customerID'])

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']


X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)


scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)
print(f"Calculated scale_pos_weight: {scale_weight:.2f}")


print("Training xgboost model with early stopping")
clf = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=15
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_pred = clf.predict(X_test)
y_proba_base = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC (Base): {roc_auc_score(y_test, y_proba_base):.3f}")
print(f"Optimal number of trees built before stopping: {clf.best_iteration}")

# Hyperparameter tuning
print("\n--- Running RandomizedSearchCV for XGBoost ---")
param_grid = {
    'n_estimators': [100, 250, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

clf = XGBClassifier(
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric="logloss"
)

random_search = RandomizedSearchCV(
    estimator=clf,
    n_iter=15,
    param_distributions=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
best_xgb = random_search.best_estimator_

print(f"\nBest Hyperparameters Found: {random_search.best_params_}")


y_proba_tuned = best_xgb.predict_proba(X_test)[:, 1]   # ✅ FIX here too
y_pred_tuned = best_xgb.predict(X_test)

print("\n=== FINAL TUNED XGBOOST PERFORMANCE ===")
print(f"ROC-AUC (Tuned): {roc_auc_score(y_test, y_proba_tuned):.3f}")
print(classification_report(y_test, y_pred_tuned))




