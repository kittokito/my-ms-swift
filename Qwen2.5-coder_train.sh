nproc_per_node=8
lora_rank=16
lora_alpha=32
# target_modulesの設定例:
# all-linear: すべての線形層にLoRAを適用
# q_proj,k_proj,v_proj,o_proj: Attention層のQuery, Key, Value, Output射影にのみ適用
# gate_proj,up_proj,down_proj: FFN層のGate, Up, Down射影にのみ適用
# wqkv,wo: 統合QKV層と出力層にのみ適用
target_modules="all-linear"


# 最適化されたDeepSpeed設定を使用
# export DS_CONFIG="optimized_zero3.json"

# CUDA グラフの有効化
# export CUDA_LAUNCH_BLOCKING=0
# export CUDA_DEVICE_MAX_CONNECTIONS=4



# CUDA・メモリ設定
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
# export TRANSFORMERS_OFFLINE=1


NPROC_PER_NODE=$nproc_per_node \
swift pt \
    --model ./models/Qwen2.5-Coder-14B-Instruct \
    --train_type lora \
    --dataset  ./dataset/~ \
    --torch_dtype bfloat16 \
    --do_eval true \
    --val_dataset ./dataset/~ \
    --streaming true \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 32 / $nproc_per_node) \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 3 \
    --deepspeed zero3 \
    --max_length 1024 \
    --max_steps  10 \
    --attn_impl 'sdpa' \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --target_modules $target_modules \
    --lr_scheduler_type cosine \
    --gradient_checkpointing true \
    --dataloader_num_workers 1 \
    --acc_strategy 'seq' \
    --logging_first_step true
    
    # --dataloader_pin_memory true \
    # --remove_unused_columns false \

# lr_scheduler_type --> [cosine(デフォルト), linear, constant, constant_with_warmup, polynomial]
# save_steps --> チェックポイントが不要な場合：-1、エポックごとの保存のみ必要な場合：0
