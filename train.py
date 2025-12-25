# Entry Point
import os
from transformers import TrainingArguments, Trainer
from configs.config import Config
from src.utils import seed_everything
from src.data import DataModule
from src.metrics import compute_metrics
from models.model_builder import build_model

def main():
    seed_everything(Config.SEED)
    
    # Подготовка данных
    data_module = DataModule()
    tokenized_datasets, data_collator = data_module.get_data()
    
    # Инициализация модели
    model = build_model(
        num_labels=data_module.num_labels,
        id2label=data_module.id2label,
        label2id=data_module.label2id
    )

    # Настройка аргументов обучения
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        
        # Параметры обучения
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.NUM_EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
        warmup_ratio=Config.WARMUP_RATIO,
        
        # Стратегия валидации и сохранения
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, # лучшая модель в конце
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        
        # Логгирование
        logging_dir=f"{Config.OUTPUT_DIR}/logs",
        logging_steps=10,
        report_to="tensorboard",
        
        # Оптимизация
        fp16=True if Config.DEVICE == "cuda" else False, # для ускорения на GPU
        dataloader_num_workers=4
    )
    
    # Инициализация Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=data_module.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Запуск обучения
    print("Starting training...")
    trainer.train()
    
    # Финальная оценка
    print("\nEvaluating on TEST set...")
    test_results = trainer.predict(tokenized_datasets["test"])
    
    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    print(f"Macro F1: {test_results.metrics['test_f1']:.4f}")
    print(f"Accuracy: {test_results.metrics['test_accuracy']:.4f}")
    print("="*30)
    
    # Проверка
    if test_results.metrics['test_f1'] > 0.89:
        print("✅ SUCCESS: You beat the baseline (0.89)!")
    else:
        print("⚠️ WARNING: Result is below 0.89. Try increasing epochs or tuning LR.")

if __name__ == "__main__":
    main()