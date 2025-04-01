import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_MODEL_PATH = "llama-3.1-8B-Instruct"
DATA_PATH = "data/diabetes/with_info/"
OUTPUT_DIR = "output/lora_finetuned_llama-3.1-8B-Instruct"

def load_diabetes_data():
    all_data = []
    
    for i in range(10):
        file_path = os.path.join(DATA_PATH, f"diabetes_{i}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
    
    processed_data = []
    for item in all_data:
        prompt = f"""<|im_start|>user{item['instruction']}{item['input']}<|im_end|><|im_start|>assistant"""
        completion = f"{item['output']}<|im_end|>"
        
        processed_data.append({
            "prompt": prompt,
            "completion": completion,
            "text": prompt + completion,
            "label": 1 if item['output'].strip() == "Yes" else 0
        })
    
    np.random.shuffle(processed_data)
    split_idx = int(len(processed_data) * 0.8)
    
    train_data = processed_data[:split_idx]
    eval_data = processed_data[split_idx:]
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    return train_dataset, eval_dataset

def tokenize_function(examples, tokenizer, max_length=512):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return tokenized_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Loading and processing data...")
    train_dataset, eval_dataset = load_diabetes_data()
    
    train_tokenized = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    eval_tokenized = eval_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=500,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        fp16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    print("Starting training...")
    trainer.train()
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Model saved to {OUTPUT_DIR}")
    
    print("Performing detailed evaluation...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

if __name__ == "__main__":
    main()