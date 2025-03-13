python ./wanda/main.py \
    --model llama-3.1-8B-Instruct \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save /pruned-llama-3.1-8b-wanda 