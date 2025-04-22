import matplotlib.pyplot as plt
import os
import json
import pandas as pd

RESULTS_PATH = "output/evaluation_results"

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
