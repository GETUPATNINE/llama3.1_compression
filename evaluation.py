import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BASE_MODEL_PATH = "llama-3.1-8B-Instruct"
FINETUNED_MODEL_PATH = "output/lora_finetuned_llama-3.1-8B-Instruct"
DATA_PATH = "data/diabetes/with_info/"
RESULTS_PATH = "output/evaluation_results"

os.makedirs(RESULTS_PATH, exist_ok=True)

def load_test_data():
    """Load the test data (using the last 20% of each file)"""
    all_data = []
    
    for i in range(10):
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
    
    np.random.seed(42)
    np.random.shuffle(processed_data)
    split_idx = int(len(processed_data) * 0.8)
    test_data = processed_data[split_idx:]
    
    return test_data

def generate_predictions(model, tokenizer, test_data):
    """Generate predictions for the test data"""
    predictions = []
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
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text.split("<|im_start|>assistant")[-1].strip()
        
        prediction = "Yes" if "yes" in response.lower() else "No"
        predicted_label = 1 if prediction == "Yes" else 0
        
        predictions.append(predicted_label)
        labels.append(item['label'])
        
        detailed_results.append({
            "input": item['input'],
            "true_output": item['true_output'],
            "predicted_output": prediction,
            "correct": item['true_output'] == prediction
        })
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(test_data)} examples")
    
    return predictions, labels, detailed_results

def evaluate_model():
    print("Loading test data...")
    test_data = load_test_data()
    
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading the base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print("Loading the finetuned model...")
    peft_config = PeftConfig.from_pretrained(FINETUNED_MODEL_PATH)
    finetuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
    
    print("Generating predictions...")
    predictions, labels, detailed_results = generate_predictions(finetuned_model, tokenizer, test_data)
    # predictions, labels, detailed_results = generate_predictions(base_model, tokenizer, test_data)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }
    
    with open(os.path.join(RESULTS_PATH, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(RESULTS_PATH, "detailed_results.json"), "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "confusion_matrix.png"))
    
    # Plot feature importance (using coefficients analysis for false positives/negatives)
    analyze_feature_importance(test_data, predictions, labels)

def analyze_feature_importance(test_data, predictions, labels):
    """Analyze which features contribute most to correct/incorrect predictions"""
    feature_names = [
        'Age', 'Pregnant', 'BloodPressure', 'SkinThickness', 
        'Glucose', 'Insulin', 'BMI', 'DiabetesPedigree'
    ]
    
    # Extract features from input text
    features = []
    for item in test_data:
        input_text = item['input']
        feature_values = {}
        
        # Extract values using string parsing
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
    
    # Convert to DataFrame
    df = pd.DataFrame(features)
    df['true_label'] = labels
    df['predicted_label'] = predictions
    df['correct'] = df['true_label'] == df['predicted_label']
    
    # Analyze features for different prediction categories
    categories = [
        ('True Positives', (df['true_label'] == 1) & (df['predicted_label'] == 1)),
        ('False Positives', (df['true_label'] == 0) & (df['predicted_label'] == 1)),
        ('True Negatives', (df['true_label'] == 0) & (df['predicted_label'] == 0)),
        ('False Negatives', (df['true_label'] == 1) & (df['predicted_label'] == 0))
    ]
    
    # Calculate feature statistics for each category
    stats = {}
    for name, mask in categories:
        if df[mask].shape[0] > 0:  # Only if we have examples in this category
            stats[name] = df[mask][feature_names].mean().to_dict()
    
    # Plot feature comparisons
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
    
    # Save feature statistics
    with open(os.path.join(RESULTS_PATH, "feature_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    evaluate_model()