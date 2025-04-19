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

# BASE_MODEL_PATH = "llama-3.1-8B-Instruct"
# BASE_MODEL_PATH = "merged_lora_llama"
BASE_MODEL_PATH = "output/lora_finetuned_llama-3.1-8B-Instruct2"
FINETUNED_MODEL_PATH = "output/lora_finetuned_llama-3.1-8B-Instruct"
DATA_PATH = "data/diabetes/with_info/"
RESULTS_PATH = "output/evaluation_results"

os.makedirs(RESULTS_PATH, exist_ok=True)

def load_test_data():
    """Load the test data (using the last 20% of each file)"""
    all_data = []
    
    for i in range(50, 55):
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
        prompt = f"""<|im_start|>user{item['instruction']}{item['input']}<|im_end|><|im_start|>assistant"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response = generated_text.split("<|im_start|>assistant")[-1].strip()
        
        # Get probability score for positive class (approximation)
        # Using the confidence of the first token as a proxy
        scores = outputs.scores[0][0]
        yes_tokens = tokenizer(" yes", add_special_tokens=False).input_ids
        no_tokens = tokenizer(" no", add_special_tokens=False).input_ids
        
        # Get probability for "yes" (simplified approach)
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

def evaluate_model(model=None, tokenizer=None):
    
    print("Loading test data...")
    test_data = load_test_data()

    if model is None:
        print("Loading base model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        print("Loading the base model...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
        # pruned_dict = torch.load("prune_log/llama3.1_prune_log/pytorch_model.bin", map_location='cuda', weights_only=False)
        # pruned_tokenizer, pruned_model = pruned_dict['tokenizer'], pruned_dict['model']
        # tokenizer.pad_token_id = 0
    
    print("Generating predictions...")
    start_time = time.time()
    print("GPU memory usage: " + str(torch.cuda.memory_allocated() / 1024**3) + " GB")

    predictions, probabilities, labels, detailed_results = generate_predictions(model, tokenizer, test_data)
    # predictions, labels, detailed_results = generate_predictions(pruned_model, pruned_tokenizer, test_data)

    end_time = time.time()
    print(f"Time taken for predictions: {end_time - start_time:.2f} seconds")

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

    # analyze_feature_importance(test_data, predictions, labels)

def analyze_feature_importance(test_data, predictions, labels):
    """Analyze which features contribute most to correct/incorrect predictions"""
    feature_names = [
        'Age', 'Pregnant', 'BloodPressure', 'SkinThickness', 
        'Glucose', 'Insulin', 'BMI', 'DiabetesPedigree'
    ]
    
    features = []
    for item in test_data:
        input_text = item['input']
        feature_values = {}
        
        for feature in feature_names:
            if feature == 'Age':
                value = float(input_text.split('Age is ')[1].split('.')[0])
            elif feature == 'Pregnant':
                value = float(input_text.split('Number of times pregnant is ')[1].split('.')[0])
            elif feature == 'BloodPressure':
                value = float(input_text.split('Diastolic blood pressure is ')[1].split('.')[0])
            elif feature == 'SkinThickness':
                value = float(input_text.split('Triceps skin fold thickness is ')[1].split('.')[0])
            elif feature == 'Glucose':
                value = float(input_text.split('Plasma glucose concentration at 2 hours in an oral glucose tolerance test (GTT) is ')[1].split('.')[0])
            elif feature == 'Insulin':
                value = float(input_text.split('2-hour serum insulin is ')[1].split('.')[0])
            elif feature == 'BMI':
                value = float(input_text.split('Body mass index is ')[1].split('.')[0])
            elif feature == 'DiabetesPedigree':
                value = float(input_text.split('Diabetes pedigree function is ')[1].split('.')[0])
            
            feature_values[feature] = value
        
        features.append(feature_values)
    
    df = pd.DataFrame(features)
    df['true_label'] = labels
    df['predicted_label'] = predictions
    df['correct'] = df['true_label'] == df['predicted_label']
    
    categories = [
        ('True Positives', (df['true_label'] == 1) & (df['predicted_label'] == 1)),
        ('False Positives', (df['true_label'] == 0) & (df['predicted_label'] == 1)),
        ('True Negatives', (df['true_label'] == 0) & (df['predicted_label'] == 0)),
        ('False Negatives', (df['true_label'] == 1) & (df['predicted_label'] == 0))
    ]
    
    stats = {}
    for name, mask in categories:
        if df[mask].shape[0] > 0:
            stats[name] = df[mask][feature_names].mean().to_dict()
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(feature_names):
        plt.subplot(2, 4, i+1)
        values = [stats[cat].get(feature, 0) for cat in ['True Positives', 'False Positives', 'True Negatives', 'False Negatives'] if cat in stats]
        categories_present = [cat for cat in ['True Positives', 'False Positives', 'True Negatives', 'False Negatives'] if cat in stats]
        
        plt.bar(categories_present, values)
        plt.title(feature)
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "feature_analysis.png"))
    
    with open(os.path.join(RESULTS_PATH, "feature_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    evaluate_model()