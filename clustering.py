import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import streamlit as st
from preprocess import get_feature_groups
import umap


def run_clustering(df, n_clusters=3):
    churn_column_present = 'churn' in df.columns
    df_original = df.copy()

    categorical, numerical = get_feature_groups()
    df_clustering = df_original.copy()

    if churn_column_present:
        df_clustering = df_clustering.drop(columns=['churn'])

    scaler = StandardScaler()
    df_clustering[numerical] = scaler.fit_transform(df_clustering[numerical])

    # Метод локтя
    st.subheader("📉 Метод локтя")
    distortions = []
    for k in range(1, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(df_clustering[numerical])
        distortions.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(1, 8), distortions, 'bx-')
    ax_elbow.set_xlabel("Число кластеров")
    ax_elbow.set_ylabel("Значение целевой функции")
    ax_elbow.set_title("Метод локтя для наших признаков")
    st.pyplot(fig_elbow)

    # KMeans
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = model.fit_predict(df_clustering[numerical])
    df_original['cluster'] = cluster_labels + 1

    # Silhouette
    score = silhouette_score(df_clustering[numerical], cluster_labels)
    st.info(f"💡 Silhouette Score: {score:.2f}")

    # UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(df_clustering[numerical])
    df_original['x'] = embedding[:, 0]
    df_original['y'] = embedding[:, 1]

    fig_umap, ax_umap = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_original, x="x", y="y", hue="cluster", palette="tab10", s=60, edgecolor='black', ax=ax_umap)
    ax_umap.set_title("UMAP проекция с раскраской по кластерам")
    st.pyplot(fig_umap)

    # Количество по кластерам
    st.subheader("📦 Количество клиентов в каждом кластере")
    cluster_counts = df_original['cluster'].value_counts().sort_index()
    df_counts = cluster_counts.rename("Количество").reset_index()
    df_counts.columns = ["Кластер", "Количество"]
    st.dataframe(df_counts)

    # Интерактив по признакам
    st.subheader("📊 Распределение признаков по кластерам (выбор)")
    cluster_ids = sorted(df_original['cluster'].unique())
    selected_cluster = st.selectbox("Выберите кластер:", cluster_ids)

    _, numerical = get_feature_groups()
    default_features = ['lifetime', 'contract_period', 'age', 'avg_class_frequency_total', 'avg_additional_charges_total']
    selected_features = st.multiselect("Выберите признаки:", numerical, default=default_features)

    if selected_features:
        plot_cluster(df_original, selected_cluster, selected_features)

    return df_original


def plot_cluster(df, cluster: int, features: list):
    st.subheader(f"📊 Распределение признаков — кластер {cluster}")
    subset = df[df['cluster'] == cluster]
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    for i, feature in enumerate(features):
        sns.histplot(data=subset, x=feature, kde=True, ax=axes[i], color='steelblue')
        axes[i].set_title(feature)
    st.pyplot(fig)


def plot_churn_by_cluster(df):
    if 'churn' not in df.columns:
        st.warning("Колонка 'churn' отсутствует для анализа оттока.")
        return

    st.subheader("📉 Доля оттока по кластерам")
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby('cluster')['churn'].mean().plot(kind='bar', ec='black', alpha=.7, ax=ax)
    ax.set_title('Доля оттока по кластерам')
    ax.set_xlabel('Кластер')
    ax.set_ylabel('Доля оттока')
    st.pyplot(fig)


def plot_categorical_distributions(df, cat_features):
    filtered_features = [f for f in cat_features if f not in ['gender', 'phone']]
    summary = df.groupby('cluster')[filtered_features].mean().T
    fig, ax = plt.subplots(figsize=(10, 5))
    summary.plot(kind='bar', ax=ax, stacked=False, ec='black', alpha=.5)
    ax.set_title("Столбчатые гистограммы признаков по кластерам")
    ax.set_xlabel("Признак")
    ax.set_ylabel("Доля клиентов")
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig)


def run_hierarchical_clustering(df):
    _, numerical = get_feature_groups()
    df = df.copy()
    df_num = df[numerical].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)
    linked = linkage(X_scaled, method='ward')

    st.subheader("📐 Дендрограмма (иерархическая кластеризация)")
    fig, ax = plt.subplots(figsize=(15, 8))
    dendrogram(linked, orientation='top', ax=ax)
    ax.set_title("Агломеративная иерархическая кластеризация")
    st.pyplot(fig)

    st.markdown("### ✂️ Разделение дендрограммы на кластеры")
    k = st.slider("Выберите количество кластеров для обрезки:", 2, 10, 4)
    cluster_assignments = fcluster(linked, k, criterion='maxclust')
    df_clustered = df_num.copy()
    df_clustered['cluster'] = cluster_assignments

    st.write("📊 Распределение по кластерам:")
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    df_display = cluster_counts.rename("Количество").reset_index()
    df_display.columns = ["Кластер", "Количество"]
    st.dataframe(df_display)

    selected_feature = st.selectbox("Выберите признак для отображения:", numerical)
    fig_feat, ax_feat = plt.subplots()
    sns.boxplot(data=df_clustered, x="cluster", y=selected_feature, ax=ax_feat)
    ax_feat.set_title(f"Распределение признака '{selected_feature}' по кластерам")
    st.pyplot(fig_feat)
