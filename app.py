import streamlit as st
import pandas as pd
from eda import run_eda
from preprocess import preprocess_data

st.set_page_config(
    page_title='EDA – ChurnVision',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("EDA и подготовка данных")

uploaded = st.sidebar.file_uploader("Загрузите CSV/XLSX", ["csv", "xlsx"])
do_eda = st.sidebar.checkbox("Выполнить EDA", value=True)

# Выбор графиков
options = st.sidebar.multiselect(
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

if uploaded:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # Приведение всех заголовков к нижнему регистру
    df.columns = df.columns.str.lower().str.strip()

    if 'churn' not in df.columns:
        st.error("Файл не содержит колонку 'churn'. EDA будет выполнен без train/test.")
        if do_eda:
            st.header("Exploratory Data Analysis (EDA) 🔍")
            run_eda(df, options)
    else:
        # Предобработка
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df, target_column='churn', scale=True)

        st.success("Предобработка завершена: данные разбиты на train/test и масштабированы")
        st.write("X_train shape:", X_train.shape)

        if do_eda:
            st.header("Exploratory Data Analysis (EDA) 🔍")
            run_eda(df, options)

else:
    st.info("Загрузите файл для анализа")

