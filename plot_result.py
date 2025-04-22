import matplotlib.pyplot as plt
import os
import json

def plot_overall_result():

    pruning_ratio = [0, 0.25, 0.5, 0.75]
    accuracy = []
    auc = []
    pruning_sparsity = []
    for ratio in pruning_ratio:
        with open(f"prune_log/llama3.1_{ratio}/evaluation_information.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Accuracy" in line:
                    accuracy.append(float(line.split(":")[1].strip()))
                elif "AUC" in line:
                    auc.append(float(line.split(":")[1].strip()))
        
        if ratio != 0:
            with open(f'prune_log/llama3.1_{ratio}/pruning_information.txt', "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "Pruning ratio" in line:
                        pruning_sparsity.append(1 - float(line.split(":")[1].strip()) / 100)
        else:
            pruning_sparsity.append(0)

    print("Accuracy:", accuracy)
    print("AUC:", auc)
    print("Pruning Sparsity:", pruning_sparsity)

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