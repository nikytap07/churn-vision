import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_clusters_on_plot(df, x_name, y_name, cluster_name):
    """Визуализация кластеров"""
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x=x_name, y=y_name, hue=cluster_name, palette='Paired')
    plt.title(f'{x_name} vs {y_name}')
    plt.show()

def plot_cluster(data, cluster, features):
    """Гистограммы признаков по кластерам"""
    current_palette = sns.color_palette('mako', 50)
    hist_features = ['contract_period', 'month_to_end_contract','lifetime']
    fig, axes = plt.subplots(1, 4, figsize=(20, 8), sharey=True)
    cnt = 20

    for col, ax in zip(features, axes.flatten()):
        cluster_data = data[data['cluster'] == cluster]
        if col in hist_features:
            sns.histplot(cluster_data[col], ax=ax, bins='auto', color=current_palette[cnt])
        elif col == 'avg_additional_charges_total':
            continue
        else:
            sns.kdeplot(cluster_data[col], ax=ax, color=current_palette[cnt])
        ax.set_title(f'Кластер {cluster}: {col}')
        cnt += 2

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 6))
    sns.kdeplot(data[data['cluster'] == cluster]['avg_additional_charges_total'],
                color=current_palette[cnt])
    plt.title(f'Кластер {cluster}: avg_additional_charges_total')
    plt.ylabel('Плотность')
    plt.show()
