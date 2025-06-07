from sklearn.metrics import accuracy_score, precision_score, recall_score

def score_model(y_test, y_pred):
    """Оценка модели"""
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print(f'Precision: {precision_score(y_test, y_pred):.2f}')
    print(f'Recall: {recall_score(y_test, y_pred):.2f}')
