python ./LLM-Pruner/llama3.py --base_model llama-3.1-8B-Instruct \
  --pruning_ratio 0.75 \
  --device cuda --eval_device cuda \
  --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
  --block_attention_layer_start 4 --block_attention_layer_end 28 \
  --save_ckpt_log_name llama3.1_0.75 \
  --pruner_type taylor  --taylor param_first \
  --save_model  --max_seq_len 2048 \
  --num_examples 25 \
  --calibration_dataset c4 \