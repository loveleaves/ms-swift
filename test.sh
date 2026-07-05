PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model ~/model/Qwen3.5-2B \
    --tuner_type lora \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --add_non_thinking_prefix true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --group_by_length true \
    --max_length 1024 \
    --gradient_checkpointing true \
    --output_dir output/Qwen3.5-2B