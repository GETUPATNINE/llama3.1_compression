import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

PRUNED_MODEL_PATH = "llama-3.1-8B-Instruct"
FINE_TUNED_OUTPUT_DIR = "fine-tuned-llama"
SMS_DATA_PATH = "data/spam_classification.jsonl"

os.makedirs(FINE_TUNED_OUTPUT_DIR, exist_ok=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print(f"Loading pruned model from: {PRUNED_MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(PRUNED_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    PRUNED_MODEL_PATH,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files=SMS_DATA_PATH, split="train")

def preprocess_function(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

processed_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

training_args = TrainingArguments(
    output_dir=FINE_TUNED_OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    push_to_hub=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    peft_config=lora_config,
    dataset_text_field=None,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator
)

trainer.train()

trainer.model.save_pretrained(FINE_TUNED_OUTPUT_DIR)
tokenizer.save_pretrained(FINE_TUNED_OUTPUT_DIR)

print(f"LoRA fine-tuning complete! Model saved at {FINE_TUNED_OUTPUT_DIR}")
