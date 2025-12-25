from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorWithPadding
from configs.config import Config

class DataModule:
    def __init__(self):
        print(f"Loading dataset: {Config.DATASET_NAME} ({Config.DATASET_LANGUAGE})...")
        self.dataset = load_dataset(Config.DATASET_NAME, Config.DATASET_LANGUAGE)
        
        # Собираем категории
        all_categories = sorted(list(
            set(self.dataset['train']['category']) | 
            set(self.dataset['validation']['category']) | 
            set(self.dataset['test']['category'])
        ))
        
        self.label2id = {label: i for i, label in enumerate(all_categories)}
        self.id2label = {i: label for i, label in enumerate(all_categories)}
        self.num_labels = len(all_categories)
        
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _preprocess_function(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=Config.MAX_LENGTH
        )
        tokenized_inputs["label"] = [self.label2id[cat] for cat in examples["category"]]
        return tokenized_inputs

    def get_data(self):
        print("Tokenizing dataset...")
        tokenized_dataset = self.dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=self.dataset['train'].column_names
        )
        tokenized_dataset.set_format("torch")
        
        full_train_dataset = concatenate_datasets([
            tokenized_dataset['train'], 
            tokenized_dataset['validation']
        ])
        
        print(f"Merged Train+Val size: {len(full_train_dataset)} examples")
        
        return {
            "train": full_train_dataset,
            "test": tokenized_dataset["test"]
        }, self.data_collator