python ./wanda/main.py \
    --model llama-3.1-8B-Instruct \
    --nsamples 1 \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --prune_method wanda \
    --save /output/wanda/log/ \
    --save_model output/wanda/pruned_model/wanda_pruned-llama-3.1-8b