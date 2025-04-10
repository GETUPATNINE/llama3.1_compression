import matplotlib.pyplot as plt

accuracy = [0.6880, 0.4960, 0.6600, 0.6960]
precision = [0.7146, 0.4934, 0.5997, 0.6985]
recall = [0.6069, 0.9919, 0.9246, 0.6701]
f1_score = [0.6564, 0.6590, 0.7276, 0.6840]
pruning_sparsity = [0, 1 - 0.837039, 1 - 0.674077, 1 - 0.511116]

plt.figure(figsize=(10, 7))

plt.plot(pruning_sparsity, accuracy, marker='o', linestyle='-', label='Accuracy', color='b')
plt.plot(pruning_sparsity, precision, marker='x', linestyle='--', label='Precision', color='g')
plt.plot(pruning_sparsity, recall, marker='s', linestyle='-.', label='Recall', color='r')
plt.plot(pruning_sparsity, f1_score, marker='^', linestyle=':', label='F1 Score', color='purple')

plt.title("Model Performance vs. Pruning Sparsity")
plt.xlabel("Pruning Sparsity")
plt.ylabel("Performance Metrics")
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig("pruned_with_finetune.jpg")
