import argparse
import os
import json
import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_test_data(DATA_PATH):
    """Load the test data (using the last 20% of each file)"""
    all_data = []
    
    for i in range(10, 15):
        file_path = os.path.join(DATA_PATH, f"diabetes_{i}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
    
    processed_data = []
    for item in all_data:
        processed_data.append({
            "instruction": item['instruction'],
            "input": item['input'],
            "true_output": item['output'],
            "label": 1 if item['output'].strip() == "Yes" else 0
        })
        
    return processed_data

def generate_predictions(model, tokenizer, test_data):
    """Generate predictions for the test data"""
    predictions = []
    raw_probabilities = []
    labels = []
    detailed_results = []
    
    model.eval()
    
    for i, item in enumerate(test_data):
        prompt = f"""Below is an instruction that describes a task, along with input data. Write a response that appropriately completes the request.

### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_return_sequences=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response = generated_text.split("### Response:")[-1].strip()
        
        scores = outputs.scores[0][0]
        yes_tokens = tokenizer(" yes", add_special_tokens=False).input_ids
        no_tokens = tokenizer(" no", add_special_tokens=False).input_ids
        
        if len(yes_tokens) > 0 and len(no_tokens) > 0:
            yes_prob = scores[yes_tokens[0]].item()
            no_prob = scores[no_tokens[0]].item()
            positive_score = yes_prob / (yes_prob + no_prob) if (yes_prob + no_prob) > 0 else 0.5
        else:
            positive_score = 0.5 if "yes" in response.lower() else 0.0
            
        raw_probabilities.append(positive_score)
        
        prediction = "Yes" if "yes" in response.lower() else "No"
        predicted_label = 1 if prediction == "Yes" else 0

        predictions.append(predicted_label)
        labels.append(item['label'])
        
        detailed_results.append({
            "input": item['input'],
            "true_output": item['true_output'],
            "predicted_output": prediction,
            "confidence_score": positive_score,
            "correct": item['true_output'] == prediction
        })
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(test_data)} examples")
    
    return predictions, raw_probabilities, labels, detailed_results

def evaluate_model(base_model_path=None, data_path=None, model=None, tokenizer=None, pruning_ratio=None, results_path=None):

    BASE_MODEL_PATH = base_model_path
    DATA_PATH = data_path

    print("Starting evaluation...")
    if model is None and BASE_MODEL_PATH is not None:
        print("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if pruning_ratio is not None:
        RESULTS_PATH = "prune_log/llama3.1_" + str(pruning_ratio) + "/detailed_information/"
        os.makedirs(RESULTS_PATH, exist_ok=True)
        print("Results directory created at:", RESULTS_PATH)
    else:
        RESULTS_PATH = "prune_log/llama3.1/detailed_information/"
        os.makedirs(RESULTS_PATH, exist_ok=True)
        print("Results directory created at:", RESULTS_PATH)
    
    print("Loading test data...")
    test_data = load_test_data(DATA_PATH=DATA_PATH)

    print("Start inference...")
    start_time = time.time()
    predictions, probabilities, labels, detailed_results = generate_predictions(model, tokenizer, test_data)
    MemoryRequirements = torch.cuda.memory_allocated() / (1024 ** 3)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")
            
    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, probabilities)
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    cm = confusion_matrix(labels, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    metrics = {
        "pruning sparsity": 0.5,
        "accuracy": float(accuracy),
        "auc": float(auc),
        "confusion_matrix": cm.tolist()
    }
    
    with open(os.path.join(RESULTS_PATH, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    with open(os.path.join(RESULTS_PATH, "detailed_results.json"), "w") as f:
        json.dump(detailed_results, f, indent=5)
    
    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist()
    }
    
    with open(os.path.join(RESULTS_PATH, "roc_data.json"), "w") as f:
        json.dump(roc_data, f, indent=3)
    
    # Create visualizations
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "confusion_matrix.png"))
    
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(RESULTS_PATH, "roc_curve.png"))

    if pruning_ratio is not None:
        with open("prune_log/llama3.1_" + str(pruning_ratio) + "/evaluation_information.txt", "w") as f:
            f.write(f"Inference time: {end_time - start_time:.2f} seconds\n")
            f.write(f"Memory Requirements: {MemoryRequirements:.2f} GB\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write(f"Confusion Matrix: {cm.tolist()}\n")
    else:
        with open("prune_log/llama3.1/evaluation_information.txt", "w") as f:
            f.write(f"Inference time: {end_time - start_time:.2f} seconds\n")
            f.write(f"Memory Requirements: {MemoryRequirements:.2f} GB\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write(f"Confusion Matrix: {cm.tolist()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--base_model_path", type=str, default="output/pruned_llama3.1-8binstruct_adapter-", help="Path to the base model directory")
    parser.add_argument("--data_path", type=str, default="data/diabetes/with_info/", help="Path to the data directory")
    parser.add_argument("--results_path", type=str, default="results", help="Path to save the results")
    args = parser.parse_args()

    evaluate_model(base_model_path=args.base_model_path, data_path=args.data_path, model=None, tokenizer=None, pruning_ratio=None, results_path=args.results_path)