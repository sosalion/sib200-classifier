# Загрузка датасета, токенизация, создание DataCollator
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from configs.config import Config

class DataModule:
    def __init__(self):
        print(f"Loading dataset: {Config.DATASET_NAME} ({Config.DATASET_LANGUAGE})...")
        self.dataset = load_dataset(Config.DATASET_NAME, Config.DATASET_LANGUAGE)
        
        all_categories = sorted(list(
            set(self.dataset['train']['category']) | 
            set(self.dataset['validation']['category']) | 
            set(self.dataset['test']['category'])
        ))
        
        self.label2id = {label: i for i, label in enumerate(all_categories)}
        self.id2label = {i: label for i, label in enumerate(all_categories)}
        self.num_labels = len(all_categories)
        
        print(f"Found {self.num_labels} categories: {all_categories}")
        
        # токенизатор
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
        """
        Возвращает токенизированный датасет и коллатор
        """
        print("Tokenizing dataset...")
        tokenized_dataset = self.dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=self.dataset['train'].column_names
        )
        
        tokenized_dataset.set_format("torch")
        
        return tokenized_dataset, self.data_collator