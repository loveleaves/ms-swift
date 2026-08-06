#!/usr/bin/env bash
#
# 单张 8GB NVIDIA GPU：Qwen3-1.7B 4-bit QLoRA + vLLM colocate GRPO。
# 该脚本用于执行路径分析，默认仅训练 3 个 optimizer/global step。
#
# 运行：
#   bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_vllm_colocate.sh
#
# 仅检查并打印命令：
#   DRY_RUN=1 bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_vllm_colocate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset NPROC_PER_NODE NNODES NODE_RANK

VLLM_MODEL="${VLLM_MODEL:-/home/cb/model/Qwen3-1.7B}"
if [[ ! -e "${VLLM_MODEL}" ]]; then
    VLLM_MODEL="Qwen/Qwen3-1.7B"
fi

DATASET="${DATASET:-modelscope/gsm8k#500}"
MAX_LENGTH="${MAX_LENGTH:-768}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-384}"
MAX_STEPS="${MAX_STEPS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-output/grpo-qwen3-1.7b-qlora-vllm-8gb}"

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

# PyPI 的 CUDA 13 runtime 位于 site-packages/nvidia/cu13/lib。Torch 导入时会
# 自行加载它，但 vLLM 的 CuMemAllocator 是独立扩展，需要动态链接器也能找到该目录。
PYTHON_SITE="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${PYTHON_SITE}/nvidia/cu13/lib:${PYTHON_SITE}/torch/lib:${LD_LIBRARY_PATH:-}"

"${PYTHON_BIN}" -c \
    "import importlib.util, torch, bitsandbytes, vllm, vllm._C; from vllm import LLM, SamplingParams; from vllm.device_allocator.cumem import CuMemAllocator; assert importlib.util.find_spec('flash_attn') is None or __import__('flash_attn'); assert torch.cuda.is_available(), 'CUDA 不可用'; assert torch.cuda.is_bf16_supported(), '当前 GPU/PyTorch 不支持 BF16'; assert torch.ones(1, device='cuda').item() == 1; print('Torch', torch.__version__, 'CUDA', torch.version.cuda, 'vLLM', vllm.__version__)" \
    || {
        echo "环境检查失败：需要 ABI 兼容的 CUDA PyTorch、bitsandbytes 和 vLLM CUDA 扩展。" >&2
        exit 1
    }

SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem concisely and put the final numerical answer within \boxed{}.'

ARGS=(
    rlhf
    --rlhf_type grpo
    --model "${VLLM_MODEL}"
    --external_plugins examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py
    --reward_funcs gsm8k_accuracy gsm8k_format
    --reward_weights 1.0 0.2
    --columns '{"answer": "solution"}'
    --dataset "${DATASET}"
    --system "${SYSTEM_PROMPT}"
    --enable_thinking false

    # 单卡 colocate：训练模型与 vLLM 分时复用同一张 GPU。
    --use_vllm true
    --vllm_mode colocate
    --vllm_tensor_parallel_size 1
    --vllm_gpu_memory_utilization 0.35
    --vllm_max_model_len 768
    --vllm_max_num_seqs 2
    --vllm_enforce_eager true
    --vllm_enable_prefix_caching false
    --vllm_enable_lora true
    --vllm_max_lora_rank 8
    --sleep_level 1
    --offload_model true

    # 训练侧仍是 4-bit QLoRA。
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

    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --num_generations 2
    --steps_per_generation 2
    --gradient_accumulation_steps 8
    --num_iterations 1

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
    --max_steps "${MAX_STEPS}"

    --split_dataset_ratio 0
    --load_from_cache_file true
    --dataset_num_proc 1
    --dataloader_num_workers 0

    # 记录 vLLM rollout 与训练模型之间的概率偏差。
    --log_rollout_offpolicy_metrics true
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
echo "Model: ${VLLM_MODEL}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${SWIFT_BIN}" "${ARGS[@]}"
    printf '\n'
    exit 0
fi

exec "${SWIFT_BIN}" "${ARGS[@]}"
