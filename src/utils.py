# Фиксация seed, настройки логгирования, девайс
import random
import os
import numpy as np
import torch

def seed_everything(seed: int):
    """
    Фиксирует все возможные источники случайности для полной воспроизводимости.
    """
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