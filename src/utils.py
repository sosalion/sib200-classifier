# Фиксация seed, настройки логгирования, девайс
import random
import os
import numpy as np
import torch
import json
from transformers import TrainerCallback

def seed_everything(seed: int):
    """
    Фиксирует все возможные источники случайности для полной воспроизводимости.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # для multi-GPU
    
    # делают вычисления на GPU детерминированными, 
    # могут немного замедлить обучение
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set to: {seed}")

class SaveMetricsCallback(TrainerCallback):
    """
    Колбек для сохранения метрик валидации в JSON файл после каждой эпохи.
    """
    def __init__(self, output_file):
        self.output_file = output_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Вызывается после завершения валидации (evaluation loop).
        metrics содержит: eval_loss, eval_f1, eval_accuracy, epoch и т.д.
        """
        # Читаем текущие данные
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []

        # Добавляем новые метрики
        # Округлим epoch для красоты
        if 'epoch' in metrics:
            metrics['epoch'] = round(metrics['epoch'], 2)
            
        data.append(metrics)

        # Перезаписываем файл
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print(f"\n[Metrics] Saved evaluation metrics to {self.output_file}")