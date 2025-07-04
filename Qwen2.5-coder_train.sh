# 訓練に使用するGPUの数（1〜利用可能なGPU数）
nproc_per_node=8
# LoRAのランク値：低ランク行列の次元数を指定（通常は8〜256の範囲で設定）
lora_rank=16
# LoRAのスケーリング係数：LoRA行列の出力をスケーリングする値
# 通常はrank×2の値を設定（値が大きいほどLoRAの影響が強くなる）
lora_alpha=32
# target_modulesの設定オプション:
# all-linear: すべての線形層にLoRAを適用（最も広範囲に適用）
# q_proj,k_proj,v_proj,o_proj: Attention層のQuery, Key, Value, Output射影にのみ適用
# gate_proj,up_proj,down_proj: FFN層のGate, Up, Down射影にのみ適用
# wqkv,wo: 統合QKV層と出力層にのみ適用
target_modules="all-linear"


# 最適化されたDeepSpeed設定を使用（コメント解除して有効化可能）
# export DS_CONFIG="optimized_zero3.json"

# CUDA グラフの有効化（パフォーマンス向上のためのオプション）
# export CUDA_LAUNCH_BLOCKING=0  # 0に設定するとCUDAカーネルの非同期実行が有効になる
# export CUDA_DEVICE_MAX_CONNECTIONS=4  # GPUデバイス間の最大接続数


# CUDA・メモリ設定
# 使用するGPUのIDを指定（カンマ区切りで複数指定可能）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024  # CUDAメモリアロケータの設定
# export TRANSFORMERS_OFFLINE=1  # オフラインモードの有効化（インターネット接続なしで動作）


# 訓練コマンド
NPROC_PER_NODE=$nproc_per_node \
swift pt \
    --model ./models/Qwen2.5-Coder-14B-Instruct \
    --train_type lora \
    --dataset  ./train_data/plc_normal_05-3_train.jsonl \
    --val_dataset ./train_data/plc_normal_05-3_val.jsonl \
    --torch_dtype bfloat16 \
    --do_eval true \
    --streaming false \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --eval_steps 50 \
    --save_strategy  epoch \
    --save_total_limit 2 \
    --logging_steps 20 \
    --deepspeed zero3 \
    --max_length 16384 \
    --attn_impl 'sdpa' \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --target_modules $target_modules \
    --lr_scheduler_type cosine \
    --gradient_checkpointing true \
    --dataloader_num_workers 1 \
    --acc_strategy 'seq' \
    --logging_first_step true
 
    # 追加オプション（コメント解除して使用可能）
    # --val_dataset ./dataset/~ \
    # --dataloader_pin_memory true \    # データローダーのピンメモリ機能（GPUへの転送を高速化）
    #   選択肢: true（高速化、メモリ使用量増加）, false
    # --remove_unused_columns false \   # 未使用の列をデータセットから削除するかどうか
    #   選択肢: true（メモリ効率化）, false（すべての列を保持）
