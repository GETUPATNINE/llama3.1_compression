## Pruning projects using LLM-Pruner

### Description

- This repository excludes `/data/diabetes/with_info`, pre-trained model, fine-tuned model and pruned_model.

### Quick Start

- For pruning using LLM-Pruner (modify the --pruning_ratio and --save_ckpt_log_name for different ratio, and --calibration_dataset for different dataset)

  - ```bash
    bash prune_llmpruner.sh
    ```

- For fine-tuning and evaluating the pruned model (modify the --pruning_ratio)

  - ```bash
    bash finetune_pruned.sh
    ```

- For fine-tuning and evaluating the pretrained model

  - ```bash
    bash finetune_pretrained.sh
    ```

### Reference

- Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., … Ma, Z. (2024). *The Llama 3 Herd of Models*. http://arxiv.org/abs/2407.21783

- Ma, X., Fang, G., & Wang, X. (2023). *LLM-Pruner: On the Structural Pruning of Large Language Models*. http://arxiv.org/abs/2305.11627