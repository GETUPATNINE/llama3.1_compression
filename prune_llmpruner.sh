python ./LLM-Pruner/llama3.py --base_model llama-3.1-8B-Instruct \
  --pruning_ratio 0.25 \
  --device cuda --eval_device cuda \
  --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 18 \
  --block_attention_layer_start 4 --block_attention_layer_end 18 \
  --save_ckpt_log_name tinyllama_prune_log \
  --pruner_type taylor  --taylor param_first \
  --save_model  --max_seq_len 2048 \
  --test_before_train --test_after_train