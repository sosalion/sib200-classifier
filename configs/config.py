# Конфигурация гиперпараметров (lr, batch_size, model_name)
import torch

class Config:
    # Основные параметры
    MODEL_NAME = "ai-forever/ruRoBERTa-large"
    
    # Путь сохранения чекпоинтов
    OUTPUT_DIR = "./checkpoints/ruroberta_sib200_best"

    # Данные
    DATASET_NAME = "Davlan/sib200"
    DATASET_LANGUAGE = "rus_Cyrl"
    MAX_LENGTH = 128

    # Гиперпараметры обучения
    SEED = 42
    NUM_EPOCHS = 15
    BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    
    # Система
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_TOTAL_LIMIT = 2    # только 2 лучших чекпоинта