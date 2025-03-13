import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import classification_report, accuracy_score

FINETUNED_MODEL_PATH = "fine-tuned-llama"
DATASET_PATH = "data/spam_classification.jsonl"

tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def predict_label(example):
    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            do_sample=False
        )

    decoded_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    decoded_output = decoded_output.lower()
    if "spam" in decoded_output:
        return "spam"
    elif "ham" in decoded_output:
        return "ham"
    else:
        return "unknown"

true_labels = []
pred_labels = []

for example in dataset:
    pred = predict_label(example)
    true = example["output"].strip().lower()

    pred_labels.append(pred)
    true_labels.append(true)

print("Classification Report:")
print(classification_report(true_labels, pred_labels, labels=["ham", "spam"]))

accuracy = accuracy_score(true_labels, pred_labels)
print(f"Accuracy: {accuracy:.4f}")
