#!/bin/bash
swift sft \
    --model ~/model/Qwen3.5-2B \
    --tuner_type lora \
    --dataset \
        'swift/self-cognition#600' \
        'AI-ModelScope/alpaca-gpt4-data-zh#500' \
        'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --model_name '小叶' 'XiaoYe' \
    --model_author '叶子' 'loveleaves' \
    --add_non_thinking_prefix true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \

    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \

    --learning_rate 2e-4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \

    --group_by_length true \
    --max_length 1024 \
    --gradient_checkpointing true \

    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --warmup_ratio 0.03 \
    --output_dir output/Qwen3.5-2B-self-cognition