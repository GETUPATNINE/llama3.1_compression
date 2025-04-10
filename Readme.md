## Pruning projects using LLM-Pruner

### Description

- This repository excludes `/data/diabetes/with_info`, pre-trained model, fine-tuned model and pruned_model.

### Quick Start

- For fine-tuning the pruned model ()

  - ```bash
    python finetuning.py
    ```

- For evaluation

  - ```bash
    python evaluation.py
    ```

- For pruning using LLM-Pruner

  - ```bash
    bash prune_llmpruner.sh
    ```

### Reference

- Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., … Ma, Z. (2024). *The Llama 3 Herd of Models*. http://arxiv.org/abs/2407.21783

- Ma, X., Fang, G., & Wang, X. (2023). *LLM-Pruner: On the Structural Pruning of Large Language Models*. http://arxiv.org/abs/2305.11627