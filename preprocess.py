import pandas as pd
from sklearn.model_selection import train_test_split

# Единый список признаков
NUM_FEATURES = [
    'contract_period',
    'age',
    'avg_additional_charges_total',
    'lifetime',
        'avg_class_frequency_total',
    'month_to_end_contract',
    'avg_class_frequency_current_month'
]

CATEGORICAL_FEATURES = [
    'gender',
    'near_location',
    'partner',
    'promo_friends',
    'phone',
    'group_visits'
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUM_FEATURES


def get_feature_groups():
    return CATEGORICAL_FEATURES, NUM_FEATURES


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_column='churn'):
    df = df.copy()
    categorical, numerical = get_feature_groups()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test

