# Функция подсчета метрик (F1-macro, Accuracy)
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(eval_pred):
    """
    Вычисляет метрики для оценки модели во время обучения.
    Аргументы:
        eval_pred: Tuple(logits, labels)
    Возвращает:
        dict: Словарь с метриками {'f1': ..., 'accuracy': ...}
    """
    logits, labels = eval_pred
    
    predictions = np.argmax(logits, axis=-1)
    
    f1_macro = f1_score(labels, predictions, average='macro')
    
    acc = accuracy_score(labels, predictions)
    
    return {
        'f1': f1_macro,
        'accuracy': acc
    }