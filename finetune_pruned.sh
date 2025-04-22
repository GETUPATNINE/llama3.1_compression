python ./finetuning.py --pruning_ratio 0.25 \
    --pruned_model \
    --eval_after_finetuning \
    --output_dir pruned_llama3.1-8b-instruct_adapter \