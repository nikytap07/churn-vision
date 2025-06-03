import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda import run_eda
from preprocess import preprocess_data
from model import train_and_save_model, visualize_model, load_model_and_scaler, run_prediction
import os
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title='ChurnVision App',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("ChurnVision: EDA, Обучение и Предсказания")

tab1, tab2 = st.tabs(["📊 Анализ и обучение", "🔮 Предсказания"])

# ---------- TAB 1: EDA и обучение ----------
with tab1:
    st.subheader("📁 Загрузка обучающего файла")
    uploaded = st.file_uploader("Загрузите CSV или Excel файл", type=["csv", "xlsx"], key="train")

    do_eda = st.sidebar.checkbox("Выполнить EDA", value=True)

    eda_options = st.sidebar.multiselect(
        "Выберите, какие графики отобразить:",
        [
            "Общая информация",
            "Гистограммы признаков",
            "QQ-графики",
            "Категориальные признаки по churn",
            "Распределения по churn",
            "PHIK корреляция",
            "Взаимная информация (MI)"
        ],
        default=[
            "Общая информация",
            "Гистограммы признаков",
            "QQ-графики",
            "Категориальные признаки по churn"
        ]
    )

    selected_model = st.sidebar.selectbox(
        "Выберите модель для обучения",
        ["Logistic Regression (balanced)", "Logistic Regression (default)", "Random Forest", "XGBoost"]
    )

    if "all_model_metrics" not in st.session_state:
        st.session_state.all_model_metrics = {}

    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        df.columns = df.columns.str.lower().str.strip()

        if 'churn' not in df.columns:
            st.error("Файл не содержит колонку 'churn'. EDA будет выполнен без обучения модели.")
            if do_eda:
                st.header("Exploratory Data Analysis (EDA) 🔍")
                run_eda(df, eda_options)
        else:
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column='churn')

            st.success("✅ Предобработка завершена")
            st.write("Размерность X_train:", X_train.shape)

            if do_eda:
                st.header("Exploratory Data Analysis (EDA) 🔍")
                run_eda(df, eda_options)

            if st.button("Обучить и сохранить модель"):
                st.info("⏳ Обучение модели...")

                model_name = selected_model
                if selected_model == "Logistic Regression (balanced)":
                    model_name = "Logistic Regression"

                model, metrics, models, X_test_scaled, y_test, y_pred, scaler = train_and_save_model(
                    X_train, X_test, y_train, y_test, model_name=model_name
                )

                st.success(f"🎯 Модель {selected_model} успешно обучена и сохранена!")

                st.subheader("📌 Метрики качества")
                for k, v in metrics.items():
                    st.write(f"**{k}:** {v:.3f}")

                st.session_state.all_model_metrics[selected_model] = metrics
                visualize_model(models, X_test_scaled, y_test, scaler)

                st.subheader("🔍 Матрица ошибок (Confusion Matrix)")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix: {selected_model}")
                st.pyplot(fig)

            if st.session_state.all_model_metrics:
                st.subheader("📊 Сравнение моделей по метрикам")
                df_metrics = pd.DataFrame(st.session_state.all_model_metrics).T.reset_index().rename(columns={"index": "Model"})
                st.dataframe(df_metrics.style.format(precision=2))

                df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
                plt.figure(figsize=(10, 5))
                sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model")
                plt.title("Сравнение моделей по метрикам")
                plt.ylim(0.5, 1.0)
                st.pyplot(plt.gcf())
    else:
        st.info("⬆ Загрузите файл с клиентскими данными (обязательно с колонкой `churn`)")

# ---------- TAB 2: Предсказания ----------
with tab2:
    st.markdown("### Загрузка данных для предсказания")
    predict_file = st.file_uploader("Загрузите CSV или Excel файл без колонки 'churn'", type=["csv", "xlsx"], key="predict")

    if predict_file:
        df_pred = pd.read_csv(predict_file) if predict_file.name.endswith(".csv") else pd.read_excel(predict_file)
        df_pred.columns = df_pred.columns.str.lower().str.strip()

        st.subheader("Пример данных для предсказания")
        st.dataframe(df_pred.head())

        model_files = [f for f in os.listdir("models") if f.endswith(".pkl") and f != "scaler.pkl"]
        selected_pkl = st.selectbox("Выберите модель (.pkl) для предсказания", model_files)

        if st.button("Сделать предсказание"):
            model, scaler = load_model_and_scaler(selected_pkl)
            result = run_prediction(model, scaler, df_pred)

            st.success("✅ Предсказания выполнены")
            st.dataframe(result)

            st.download_button(
                label="📥 Скачать результат",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )
