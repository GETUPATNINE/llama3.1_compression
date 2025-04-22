import argparse
import os
import sys
sys.path.append(".")
import time
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
from evaluation import evaluate_model


def load_diabetes_data(DATA_PATH):
    all_data = []
    
    for i in range(10):
        file_path = os.path.join(DATA_PATH, f"diabetes_{i}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
    
    processed_data = []
    for i, item in enumerate(all_data):
        text = f"""Below is an instruction that describes a task, along with input data. Write a response that appropriately completes the request.

### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
{item['output']}"""

        processed_data.append({
            "text": text,
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

def main(args):
    OUTPUT_DIR = args.output_dir
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    if args.pruned_model:
        print("Using LLMPruner pruned model with pruning ratio of {}...".format(args.pruning_ratio))

        pruned_dict = torch.load("prune_log/llama3.1_" + str(args.pruning_ratio) + "/pytorch_model.bin", map_location='cuda', weights_only=False)
        tokenizer, pruned_model = pruned_dict['tokenizer'], pruned_dict['model']

        print("Model architecture before finetuning:")
        print(pruned_model)
        model = get_peft_model(pruned_model, lora_config)
    else:
        print("Using base model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

        print("Model architecture before finetuning:")
        print(model)
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading and processing data...")
    train_dataset, eval_dataset = load_diabetes_data(DATA_PATH=args.data_path)
    
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
        eval_strategy="epoch",
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
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    if args.pruned_model:
        model = model.merge_and_unload()
    print("Model architecture after finetuning:")
    print(model)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    if args.pruned_model:
        with open("prune_log/llama3.1_" + str(args.pruning_ratio) + "/finetuning_information.txt", "w") as f:
            f.write("Fine-tuning time: {} seconds\n".format(end_time - start_time))
    else:
        with open("prune_log/llama3.1/finetuning_information.txt", "w") as f:
            f.write("Fine-tuning time: {} seconds\n".format(end_time - start_time))
    
    if args.eval_after_finetuning:
        if args.pruned_model:
            evaluate_model(base_model_path=OUTPUT_DIR, data_path=args.data_path, model=model, tokenizer=tokenizer, pruning_ratio=args.pruning_ratio, results_path=None)
        else:
            evaluate_model(base_model_path=OUTPUT_DIR, data_path=args.data_path, model=model, tokenizer=tokenizer, pruning_ratio=None, results_path=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a model")
    parser.add_argument("--pruned_model", action="store_true", help="Use the finetuned model with LLMPruner")
    parser.add_argument("--pruning_ratio", type=float, default=0.25, help="Pruning ratio for the model")
    parser.add_argument("--data_path", type=str, default="data/diabetes/with_info/", help="Path to the data directory")
    parser.add_argument("--base_model_path", type=str, default="llama-3.1-8B-Instruct", help="Path to the base model directory")
    parser.add_argument("--eval_after_finetuning", action="store_true", help="Evaluate the model after fine-tuning")
    parser.add_argument("--output_dir", type=str, default="llama3.1-8b-instruct_adapter", help="Output directory for the fine-tuned model")
    args = parser.parse_args()
    main(args)