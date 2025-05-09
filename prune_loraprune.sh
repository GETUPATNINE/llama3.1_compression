CUDA_VISIBLE_DEVICES=0 python LoRAPrune/prune.py \
     --base_model "llama-3.1-8B-Instruct" \
     --data_path 'data/diabetes/with_info/diabetes_0.json' \
     --output_dir 'LoRAPrune_Adapter' \
     # --batch_size 128 \
     # --micro_batch_size 2 \
     # --num_epochs 2 \
     # --learning_rate 1e-4 \
     # --cutoff_len 512 \
     # --val_set_size 1000 \
     # --lora_r 8 \
     # --lora_alpha 16 \
     # --lora_dropout 0.05 \
     # --lora_target_modules '[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]' \
     # --train_on_inputs \
     # --group_by_length \
     # --ratio 0.5 \
     # --prune_metric 'lora' \
     # --prune_freq 10 \