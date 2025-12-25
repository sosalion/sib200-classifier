import os
from transformers import TrainingArguments, Trainer
from configs.config import Config
from src.utils import seed_everything, SaveMetricsCallback
from src.data import DataModule
from src.metrics import compute_metrics
from models.model_builder import build_model

def main():
    seed_everything(Config.SEED)
    
    data_module = DataModule()
    tokenized_datasets, data_collator = data_module.get_data()
    
    model = build_model(
        num_labels=data_module.num_labels,
        id2label=data_module.id2label,
        label2id=data_module.label2id
    )
    
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.NUM_EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
        warmup_ratio=Config.WARMUP_RATIO,
        label_smoothing_factor=Config.LABEL_SMOOTHING,
        
        eval_strategy="epoch",
        save_strategy="epoch",
        
        load_best_model_at_end=True, 
        metric_for_best_model="f1",  
        greater_is_better=True,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        
        logging_dir=f"{Config.OUTPUT_DIR}/logs",
        logging_steps=5,
        report_to="none",
        fp16=True if Config.DEVICE == "cuda" else False,
        dataloader_num_workers=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=data_module.tokenizer, 
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[SaveMetricsCallback(Config.METRICS_FILE)]
    )
    
    print("Starting LoRA training...")
    trainer.train()
    
    print("\nEvaluating on TEST set (Final Check)...")
    test_results = trainer.predict(tokenized_datasets["test"])
    
    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    print(f"Macro F1: {test_results.metrics['test_f1']:.4f}")
    print(f"Accuracy: {test_results.metrics['test_accuracy']:.4f}")
    print("="*30)
    
    import json
    try:
        with open(Config.METRICS_FILE, 'r') as f:
            data = json.load(f)
    except:
        data = []
    
    test_metrics = test_results.metrics
    test_metrics['stage'] = 'final_test_evaluation'
    data.append(test_metrics)
    with open(Config.METRICS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    if test_results.metrics['test_f1'] > 0.89:
        print("✅ SUCCESS: You beat the baseline (0.89)!")
    else:
        print("⚠️ WARNING: Result is below 0.89.")

if __name__ == "__main__":
    main()