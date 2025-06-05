import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda import run_eda
from preprocess import preprocess_data
from model import train_and_save_model, visualize_model, load_model_and_scaler, run_prediction
import os
from clustering import run_clustering, plot_churn_by_cluster, plot_categorical_distributions, run_hierarchical_clustering

from preprocess import get_feature_groups
import os
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title='ChurnVision App',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("ChurnVision: EDA, Обучение и Предсказания")

tab1, tab2, tab3 = st.tabs(["\U0001F4CA Анализ и обучение", "\U0001F52E Предсказания", "\U0001F465 Кластеризация"])

# ---------- TAB 1: EDA и обучение ----------
with tab1:
    st.subheader("\U0001F4C1 Загрузка обучающего файла")
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
                st.header("Exploratory Data Analysis (EDA) \U0001F50D")
                run_eda(df, eda_options)
        else:
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column='churn')

            st.success("\u2705 Предобработка завершена")
            st.write("Размерность X_train:", X_train.shape)

            if do_eda:
                st.header("Exploratory Data Analysis (EDA) \U0001F50D")
                run_eda(df, eda_options)

            if st.button("Обучить и сохранить модель"):
                st.info("\u23F3 Обучение модели...")

                model_name = selected_model
                if selected_model == "Logistic Regression (balanced)":
                    model_name = "Logistic Regression"

                model, metrics, models, X_test_scaled, y_test, y_pred, scaler = train_and_save_model(
                    X_train, X_test, y_train, y_test, model_name=model_name
                )

                st.success(f"\U0001F3AF Модель {selected_model} успешно обучена и сохранена!")

                st.subheader("\U0001F4CC Метрики качества")
                for k, v in metrics.items():
                    st.write(f"**{k}:** {v:.3f}")

                st.session_state.all_model_metrics[selected_model] = metrics
                visualize_model(models, X_test_scaled, y_test, scaler)

                st.subheader("\U0001F50D Матрица ошибок (Confusion Matrix)")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix: {selected_model}")
                st.pyplot(fig)

            if st.session_state.all_model_metrics:
                st.subheader("\U0001F4CA Сравнение моделей по метрикам")
                df_metrics = pd.DataFrame(st.session_state.all_model_metrics).T.reset_index().rename(columns={"index": "Model"})
                st.dataframe(df_metrics.style.format(precision=2))

                df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
                plt.figure(figsize=(10, 5))
                sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model")
                plt.title("Сравнение моделей по метрикам")
                plt.ylim(0.5, 1.0)
                st.pyplot(plt.gcf())
    else:
        st.info("\u2B06 Загрузите файл с клиентскими данными (обязательно с колонкой `churn`)")

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

            st.success("\u2705 Предсказания выполнены")
            st.dataframe(result)

            st.download_button(
                label="\U0001F4E5 Скачать результат",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

# ---------- TAB 3: Кластеризация ----------
with tab3:
    st.markdown("### 👥 Кластеризация клиентов")
    n_clusters = st.slider("Выберите количество кластеров", 2, 10, 5)

    if uploaded:
        df = df.copy() if 'df' in locals() else None
        df.columns = df.columns.str.lower().str.strip()

        clustered_df = run_clustering(df, n_clusters=n_clusters)
        plot_churn_by_cluster(clustered_df)

        # plot_cluster_distributions больше не используется
        # визуализация признаков вынесена внутрь run_clustering()

        categorical, _ = get_feature_groups()
        plot_categorical_distributions(clustered_df, categorical)

        with st.expander("📌 Иерархическая кластеризация"):
            run_hierarchical_clustering(df)
    else:
        st.warning("Загрузите обучающий файл для кластеризации")
