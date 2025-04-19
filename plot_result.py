import matplotlib.pyplot as plt
import os
import json

RESULTS_PATH = "output/"

def plot_overall_result():
    with open(os.path.join(RESULTS_PATH, "metrics.json"), "r") as f:
        metrics = json.loads(f)

    accuracy = []
    auc = []
    pruning_sparsity = []
    for item in metrics:
        accuracy.append(item['accuracy'])
        auc.append(item['auc'])
        pruning_sparsity.append(item['pruning sparsity'])

    plt.figure(figsize=(10, 7))

    plt.plot(pruning_sparsity, accuracy, marker='o', linestyle='-', label='Accuracy', color='b')
    plt.plot(pruning_sparsity, auc, marker='s', linestyle='-.', label='AUC', color='r')

    plt.xlabel("Pruning Sparsity")
    plt.ylabel("Performance Metrics")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig("pruned_with_finetune.jpg")

if __name__ == '__main__':
    plot_overall_result()