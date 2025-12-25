import torch
import os

class Config:
    # Основные параметры
    MODEL_NAME = "ai-forever/ruRoBERTa-large" 
    OUTPUT_DIR = "./checkpoints/ruroberta_lora_best"
    
    METRICS_DIR = "./metrics_logs"
    METRICS_FILE = os.path.join(METRICS_DIR, "training_metrics.json")

    # Данные
    DATASET_NAME = "Davlan/sib200"
    DATASET_LANGUAGE = "rus_Cyrl"
    MAX_LENGTH = 128
    
    # Гиперпараметры
    SEED = 42
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 1
    
    LEARNING_RATE = 3e-4      
    
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.05
    LABEL_SMOOTHING = 0.05

    # Система
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_TOTAL_LIMIT = 1