import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from preprocess import get_feature_groups

MODELS_PATH = 'models'
categorical, NUM_FEATURES = get_feature_groups()
ALL_FEATURES = categorical + NUM_FEATURES

from sklearn.preprocessing import StandardScaler
from preprocess import get_feature_groups

def train_and_save_model(X_train, X_test, y_train, y_test, model_name=""):
    # Получаем списки признаков
    categorical, numerical = get_feature_groups()

    # Инициализируем и применяем масштабирование
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical] = scaler.fit_transform(X_train[numerical])
    X_test_scaled[numerical] = scaler.transform(X_test[numerical])

    models = {}
    metrics = {}


    if model_name == "Logistic Regression":
        model = LogisticRegression(class_weight="balanced")
        model.fit(X_train_scaled[ALL_FEATURES], y_train)
    elif model_name == "Logistic Regression (default)":
        model = LogisticRegression()
        model.fit(X_train_scaled[ALL_FEATURES], y_train)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
        param_grid = {
            'n_estimators': [100],
            'max_depth': [5, 7],
            'min_samples_leaf': [50, 100]
        }
        grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
        grid.fit(X_train_scaled[ALL_FEATURES], y_train)
        model = grid.best_estimator_
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.3]
        }
        grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
        grid.fit(X_train_scaled[ALL_FEATURES], y_train)
        model = grid.best_estimator_
    else:
        model = DummyClassifier(strategy='stratified')
        model.fit(X_train_scaled[ALL_FEATURES], y_train)

    y_pred = model.predict(X_test_scaled[ALL_FEATURES])

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

    joblib.dump(model, os.path.join(MODELS_PATH, f"{model_name.replace(' ', '_')}.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))

    models[model_name] = model
    return model, metrics, models, X_test_scaled, y_test, y_pred, scaler

def visualize_model(models: dict, X_test, y_test, scaler):
    st.subheader("📈 Сравнение ROC-кривых моделей")
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        X_input = X_test.copy()
        categorical, numerical = get_feature_groups()
        if scaler:
            X_input[numerical] = scaler.transform(X_input[numerical])
        X_input = X_input[categorical + numerical]  # правильный порядок

        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_input)[:, 1]
        else:
            y_probs = model.decision_function(X_input)

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc = roc_auc_score(y_test, y_probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые моделей")
    plt.legend()
    st.pyplot(plt.gcf())

    if "XGBoost" in models:
        st.subheader("📊 SHAP-график (XGBoost)")
        explainer = shap.Explainer(models["XGBoost"])
        shap_values = explainer(X_input)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_input, show=False)
        st.pyplot(fig)

def load_model_and_scaler(model_file):
    model_path = os.path.join(MODELS_PATH, model_file)
    scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def run_prediction(model, scaler, df):
    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.lower().str.strip()
    categorical, numerical = get_feature_groups()

    for col in categorical + numerical:
        if col not in df_copy.columns:
            df_copy[col] = 0.0

    df_copy[numerical] = scaler.transform(df_copy[numerical])

    print("📉 Scaler mean:", scaler.mean_ if hasattr(scaler, 'mean_') else 'нет')
    print("📉 Scaler var:", scaler.var_ if hasattr(scaler, 'var_') else 'нет')

    df_copy = df_copy[categorical + numerical]

    preds = model.predict(df_copy)
    probs = model.predict_proba(df_copy)[:, 1]

    df_result = df.copy().reset_index(drop=True)
    df_result['prediction'] = preds
    df_result['probability'] = probs
    return df_result
