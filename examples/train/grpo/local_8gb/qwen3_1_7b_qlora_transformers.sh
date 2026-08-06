#!/usr/bin/env bash
#
# 单张 8GB NVIDIA GPU 的保守 GRPO QLoRA 配置。
# - 模型：Qwen3-1.7B，4-bit NF4 加载，LoRA rank 8
# - rollout：TransformersEngine，不安装/不使用 vLLM
# - 数据：GSM8K 子集，使用仓库内置插件，无需 math_verify
# - 默认关闭 Qwen3 thinking，避免 8GB 环境中生成过长
#
# 直接运行：
#   bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_transformers.sh
#
# 先只检查并打印命令：
#   DRY_RUN=1 bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_transformers.sh
#
# 若仍然 OOM，优先降为 0.6B：
#   MODEL=Qwen/Qwen3-0.6B bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_transformers.sh
#
# 跑通后扩大数据：
#   DATASET='modelscope/gsm8k#2000' bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_transformers.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 本脚本不使用 vLLM，因此阻止 TRL 在导入 GRPOTrainer 时加载该可选依赖，
# 让 Transformers rollout 的环境和调用链与 vLLM 完全隔离。
export SWIFT_DISABLE_VLLM_IMPORT=1

# 强制单卡，避免外部残留的分布式环境变量让 CLI 意外启动多个进程。
unset NPROC_PER_NODE NNODES NODE_RANK

MODEL="${MODEL:-/home/cb/model/Qwen3-1.7B}"
DATASET="${DATASET:-modelscope/gsm8k#500}"
MAX_LENGTH="${MAX_LENGTH:-768}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-384}"
MAX_STEPS="${MAX_STEPS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-output/grpo-qwen3-1.7b-qlora-8gb}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-${PROJECT_ROOT}/.venv/bin/swift}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ ! -x "${SWIFT_BIN}" ]]; then
    SWIFT_BIN="$(command -v swift || true)"
fi
if [[ -z "${PYTHON_BIN}" || -z "${SWIFT_BIN}" ]]; then
    echo "未找到可用的 Python 或 swift。请先激活工程虚拟环境。" >&2
    exit 1
fi

"${PYTHON_BIN}" -c \
    "import torch, bitsandbytes; assert torch.cuda.is_available(), 'CUDA 不可用'; assert torch.cuda.is_bf16_supported(), '当前 GPU/PyTorch 不支持 BF16'" \
    || {
        echo "环境检查失败：需要带 CUDA 的 PyTorch 和 bitsandbytes。" >&2
        exit 1
    }

SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem concisely and put the final numerical answer within \boxed{}.'

ARGS=(
    rlhf
    --rlhf_type grpo
    --model "${MODEL}"
    --external_plugins examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py
    --reward_funcs gsm8k_accuracy gsm8k_format
    --reward_weights 1.0 0.2
    --columns '{"answer": "solution"}'
    --dataset "${DATASET}"
    --system "${SYSTEM_PROMPT}"
    --enable_thinking false

    # 明确使用 TransformersEngine rollout。
    --use_vllm false

    # 4-bit QLoRA：8GB 显存下训练 1.7B 模型的关键。
    --tuner_type lora
    --target_modules all-linear
    --lora_rank 8
    --lora_alpha 16
    --lora_dropout 0.05
    --quant_method bnb
    --quant_bits 4
    --bnb_4bit_quant_type nf4
    --bnb_4bit_use_double_quant true
    --bnb_4bit_compute_dtype bfloat16
    --torch_dtype bfloat16
    --attn_impl sdpa
    --gradient_checkpointing true
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'

    # 单设备 micro-batch 为 1。G=2 是 GRPO 允许的最小候选数。
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --num_generations 2

    # 必须显式指定为 2：否则它会默认跟随梯度累积步数 8，
    # 使 Transformers rollout 一次生成 8 条 completion，显存峰值明显升高。
    --steps_per_generation 2
    --gradient_accumulation_steps 8
    --num_iterations 1

    # 控制 prompt + completion 的激活和 KV cache。
    --max_length "${MAX_LENGTH}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --truncation_strategy left
    --temperature 0.9
    --top_p 0.95

    --loss_type grpo
    --scale_rewards none
    --beta 0.01
    --epsilon 0.2
    --epsilon_high 0.28
    --learning_rate 1e-5
    --lr_scheduler_type cosine
    --warmup_ratio 0.05
    --max_grad_norm 1.0
    --num_train_epochs 1
    # 本地流程分析默认只跑 3 个 optimizer/global step。
    # 正式训练可用 MAX_STEPS=-1 恢复按 num_train_epochs 运行。
    --max_steps "${MAX_STEPS}"

    # 不切验证集，减少一次额外 rollout，也避免验证 batch 的整除约束。
    --split_dataset_ratio 0
    --load_from_cache_file true
    --dataset_num_proc 1
    --dataloader_num_workers 0

    --logging_steps 1
    --log_completions true
    --report_to none
    --save_steps 3
    --save_total_limit 2
    --output_dir "${OUTPUT_DIR}"
    --seed 42
)

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${SWIFT_BIN}" "${ARGS[@]}"
    printf '\n'
    exit 0
fi

exec "${SWIFT_BIN}" "${ARGS[@]}"
