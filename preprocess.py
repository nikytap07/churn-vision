import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, target_column='churn', scale=True):
    """
    Разделение признаков и целевой переменной, опциональное масштабирование.

    Parameters:
        df (pd.DataFrame): исходный датафрейм
        target_column (str): имя колонки целевой переменной
        scale (bool): выполнять ли масштабирование признаков

    Returns:
        X_train, X_test, y_train, y_test, scaler (если scale=True)
    """
    df = df.copy()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train, X_test, y_train, y_test, scaler


def get_feature_groups():
    """
    Возвращает списки категориальных и числовых признаков
    (на основе структуры проекта ChurnVision)
    """
    categorical = ['gender', 'near_location', 'partner', 'promo_friends', 'phone', 'group_visits']
    numerical = ['lifetime', 'month_to_end_contract', 'contract_period',
                 'age', 'avg_class_frequency_total',
                 'avg_class_frequency_current_month', 'avg_additional_charges_total']
    return categorical, numerical
