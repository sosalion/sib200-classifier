# Инициализация модели и токенизатора
from transformers import AutoModelForSequenceClassification
from configs.config import Config

def build_model(num_labels, id2label, label2id):
    """
    Загружает предобученную модель и заменяет голову классификации под задачу.
    """
    print(f"Loading model architecture: {Config.MODEL_NAME}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    return model