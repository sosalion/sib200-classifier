from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from configs.config import Config

def build_model(num_labels, id2label, label2id):
    """
    Загружает модель и оборачивает её в LoRA адаптеры.
    """
    print(f"Loading base model: {Config.MODEL_NAME}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # Конфигурация LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    
    print("Injecting LoRA adapters...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model