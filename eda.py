import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
import scipy.stats as stats
from phik.report import plot_correlation_matrix
from sklearn.feature_selection import mutual_info_classif
import io

sns.set_style('darkgrid')
current_palette = sns.color_palette('mako', 50)

def get_info(data):
    st.write(data.head())
    st.write(data.describe())
    st.markdown('**Информация о пропусках и типах данных:**')
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

def qq_plt(data, col):
    stats.probplot(data, dist="norm", plot=pylab)
    plt.title(f'QQ-график для {col}')
    plt.xlabel('Теоретические квантили')
    plt.ylabel('Упорядоченные значения')

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    fig, ax = plt.subplots()
    ax.barh(width, scores, ec='black', alpha=0.84)
    ax.set_yticks(width)
    ax.set_yticklabels(ticks)
    ax.set_title('Взаимная информация')
    st.pyplot(fig)

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI', index=X.columns).sort_values(ascending=False)
    return mi_scores

def remove_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def run_eda(df, options):
    df.columns = df.columns.str.lower()

    if "Общая информация" in options:
        st.subheader("Общая информация")
        st.markdown("Отображение первых строк и статистического описания датасета")
        get_info(df)
        st.write(f'**Количество дубликатов в данных:** {df.duplicated().sum()}')

    if "Гистограммы признаков" in options:
        st.subheader("Гистограммы признаков")
        st.markdown("Распределения числовых признаков с нормальной кривой KDE")
        df.hist(bins=50, edgecolor='black', linewidth=2, alpha=0.72, figsize=(15, 15))
        st.pyplot(plt.gcf())

    if "QQ-графики" in options:
        st.subheader("QQ-графики")
        st.markdown("Проверка отклонения распределения признаков от нормального закона")
        cols = ['avg_additional_charges_total', 'avg_class_frequency_current_month', 'avg_class_frequency_total']
        fig = plt.figure(figsize=(15, 5))
        for idx, col in enumerate(cols):
            fig.add_subplot(1, 3, idx+1)
            qq_plt(df[col], col)
        plt.tight_layout()
        st.pyplot(fig)

    if "Категориальные признаки по churn" in options:
        st.subheader("Категориальные признаки по churn")
        st.markdown("Доля клиентов с churn=1 и churn=0 в разных категориальных признаках")
        features = ['gender','near_location','partner','promo_friends','phone','group_visits']
        df.groupby('churn')[features].mean().T.plot(kind='bar', stacked=False, ec='black', alpha=0.72)
        plt.title('Признаки для оттока и лояльных клиентов')
        plt.xlabel('Признак')
        plt.ylabel('Доля клиентов')
        plt.xticks(rotation=42)
        st.pyplot(plt.gcf())

    if "Распределения по churn" in options:
        st.subheader("Распределения по churn")
        st.markdown("Сравнение плотностей распределения признаков между лояльными и ушедшими клиентами")
        real_features = ['lifetime', 'month_to_end_contract', 'contract_period', 'age',
                         'avg_class_frequency_total', 'avg_class_frequency_current_month', 'avg_additional_charges_total']
        hist_features = ['contract_period', 'month_to_end_contract','lifetime']
        churn = df[df['churn'] == 1]
        loyal = df[df['churn'] == 0]

        fig, axes = plt.subplots(3, 2, figsize=(20, 20), sharey=True)
        cnt = 20
        for col, ax in zip(real_features, axes.flatten()):
            if col not in hist_features and col != 'avg_additional_charges_total':
                sns.histplot(loyal[col], ax=ax, kde=True, stat="density", color=current_palette[cnt], label="Лояльные")
                sns.histplot(churn[col], ax=ax, kde=True, stat="density", color=current_palette[cnt + 5], label="Отток")
            else:
                min_ = int(min(df[col]))
                max_ = int(max(df[col]))
                sns.histplot(loyal[col], ax=ax, color=current_palette[cnt], bins=range(min_, max_+2), kde=False, stat="density")
                sns.histplot(churn[col], ax=ax, color=current_palette[cnt + 2], bins=range(min_, max_+2), kde=False, stat="density")
            ax.set_title(f'{col} по churn')
            ax.set_ylabel('Плотность')
            ax.legend()
            cnt += 2
        plt.tight_layout()
        st.pyplot(fig)

        fig = plt.figure(figsize=(20, 6))
        col = 'avg_additional_charges_total'
        sns.histplot(loyal[col], kde=True, stat="density", color=current_palette[cnt], label="Лояльные")
        sns.histplot(churn[col], kde=True, stat="density", color=current_palette[cnt + 8], label="Отток")
        plt.legend()
        plt.ylabel('Плотность')
        plt.title(f'Плотность для {col}')
        st.pyplot(fig)

    if "PHIK корреляция" in options:
        st.subheader("PHIK корреляция 🔗")
        st.markdown(
            "Phik (φK) — это обобщённая мера корреляции, которая работает как с числовыми, так и с категориальными признаками, "
            "и способна выявлять нелинейные зависимости между признаками. Особенно полезна в разведочном анализе данных (EDA)."
        )
        interval = [c for c in df.columns if c != 'churn']
        phik_overview = df.phik_matrix(interval_cols=interval)

        plot_correlation_matrix(
            phik_overview.values,
            x_labels=phik_overview.columns,
            y_labels=phik_overview.index,
            vmin=0, vmax=1,
            color_map="Blues",
            title=r"Матрица φK-корреляции (phik)",
            fontsize_factor=.9,
            figsize=(15, 12)
        )
        st.pyplot(plt.gcf())

    if "Взаимная информация (MI)" in options:
        st.subheader("Взаимная информация (MI)")
        st.markdown(
            "Нелинейная мера зависимости между признаками и целевой переменной `churn`. "
            "Позволяет определить, какие признаки дают наибольший вклад в предсказание."
        )

        new_df = remove_correlated_features(df, threshold=0.9)
        X = new_df.copy()
        y = X.pop('churn')
        discrete_features = X.dtypes == int
        mi_scores = make_mi_scores(X, y, discrete_features)
        plot_mi_scores(mi_scores)

        dropped = list(set(df.columns) - set(new_df.columns))
        st.markdown(f"🧹 **Удалены признаки из-за сильной корреляции**: `{dropped}`")

        st.subheader("Матрица корреляции (после очистки)")
        st.markdown(
            "Показывает линейную корреляцию между оставшимися признаками после фильтрации. "
            "Позволяет убедиться, что мультиколлинеарность устранена."
        )
        fig = plt.figure(figsize=(15, 12))
        sns.heatmap(new_df.corr(), square=True, annot=True, fmt=".2f", cmap='Blues', linewidths=.42, vmax=.95)
        plt.title('Матрица корреляции между признаками')
        st.pyplot(fig)

    return df
