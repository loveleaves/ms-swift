# Qwen3-1.7B LoRA SFT：Transformers + PEFT + Accelerate 最小完整实现

对应可执行文件：[`minimal_qwen3_1_7b_sft_lora_transformers.py`](./minimal_qwen3_1_7b_sft_lora_transformers.py)

本文说明这个单文件如何覆盖 ms-swift 当前主流 `sft + lora + peft` 链路、每一步为什么存在、如何训练/恢复/验证，以及它与完整 ms-swift 的边界。这里的“最小”是减少框架封装，不是删掉训练闭环：数据、模板、标签、LoRA 注入、优化、验证、checkpoint、精确恢复、adapter 导出、加载生成与合并导出均保留。

## 1. 最终得到什么

一次正常运行会在 `output_dir` 生成：

```text
output_dir/
├── checkpoint-N/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── rng_state.pth
│   └── trainer_state.json
├── final_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer ...
├── merged_model/                 # 仅 --merge_lora 时生成
│   ├── config.json
│   ├── model*.safetensors
│   └── tokenizer ...
├── train_results.json
├── eval_results.json
├── run_manifest.json
└── generation_smoke_test.json
```

这三类模型资产的用途不同：

| 资产 | 是否含 Qwen3 基座 | 是否含优化器状态 | 用途 |
|---|---:|---:|---|
| `checkpoint-N` | 否 | 是 | 从第 N 步精确续训 |
| `final_adapter` | 否 | 否 | 与原基座一起推理，或作为新训练的 LoRA 热启动 |
| `merged_model` | 是 | 否 | 不依赖 PEFT 的普通 Transformers 推理 |

不要把 `final_adapter` 当成完整模型，也不要把“从 adapter 热启动”当成“精确断点恢复”。后者必须恢复 optimizer、scheduler、RNG 和 global step。

## 2. SFT 与 LoRA 的核心数学

对对话 token 序列 \(x_1,\ldots,x_T\)，自回归语言模型学习：

\[
\mathcal L=-\sum_{t=1}^{T-1}m_{t+1}\log p_\theta(x_{t+1}\mid x_{\le t})
\]

`m` 是监督掩码。本实现把 system、user、assistant header 和 batch padding 的 label 设为 `-100`，因此它们不进入交叉熵；assistant 内容、Qwen3 非思考前缀和消息结束符进入损失。这是 instruction SFT 的关键，而不只是“把一段文字丢给 causal LM”。

LoRA 不直接更新某个线性层的基座权重 \(W_0\)，而是学习低秩增量：

\[
h=W_0x+\Delta Wx,\qquad \Delta W=\frac{\alpha}{r}BA
\]

其中 \(A\in\mathbb R^{r\times d_{in}}\)、\(B\in\mathbb R^{d_{out}\times r}\)，且 \(r\ll d\)。标准初始化令一个低秩矩阵为零，因此训练开始时 \(\Delta W=0\)，模型行为与原基座一致。`lora_alpha/r` 控制增量缩放；`lora_dropout` 只作用于 LoRA 分支。

当前 ms-swift PEFT LoRA 主流默认值为：

- `target_modules=all-linear`
- `lora_rank=8`
- `lora_alpha=32`
- `lora_dropout=0.05`
- `lora_bias=none`
- `learning_rate=1e-4`
- `weight_decay=0.1`
- `adam_beta1=0.9`、`adam_beta2=0.95`、`adam_epsilon=1e-8`
- `lr_scheduler_type=cosine`

Qwen3-1.7B 有 28 层，隐藏维 2048，中间维 6144，GQA 的 K/V 输出维 1024。`all-linear` 展开为：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

`lm_head` 被排除。rank=8 时这七类矩阵约增加 871.6 万个参数。若 LoRA 权重和梯度为 FP32、Adam 一阶/二阶状态为 FP32，adapter 训练状态约为 140 MB 量级；主要显存仍来自 BF16 基座、激活和临时张量。

## 3. 单文件流程与 ms-swift 源码地图

| 阶段 | 最小实现 | ms-swift 当前对应位置 |
|---|---|---|
| 参数契约 | `parse_args` | `swift/arguments/sft_args.py::SftArguments`；`swift/arguments/tuner_args.py::TunerArguments` |
| 本地数据读取 | `read_records` | `swift/dataset/loader.py::load_dataset` |
| schema 统一 | `normalize_record` | `swift/dataset/preprocessor/core.py::RowPreprocessor`、`MessagesPreprocessor` |
| train/eval 划分 | `split_records` | `swift/pipelines/train/sft.py::SwiftSft._get_dataset` |
| Qwen3 模板 | `encode_qwen3_messages` | `swift/template/templates/qwen.py`、`swift/template/templates/utils.py::CHATML_TEMPLATE_META` |
| labels / 动态 EOS | `encode_qwen3_messages` | `swift/template/base.py::_encode_context_list`、`_add_dynamic_eos` |
| 数据编码编排 | `TokenizedSFTDataset` | `swift/pipelines/train/sft.py::_encode_dataset` |
| 动态 padding | `DataCollatorForSeq2Seq` | `swift/template/base.py::Template.data_collator/_data_collator` |
| 模型/精度加载 | `choose_precision`、`from_pretrained` | `SwiftSft._prepare_model_tokenizer`、`swift/model/register.py::get_model_processor` |
| `all-linear` 展开 | `resolve_target_modules` | `swift/pipelines/train/tuner.py::get_target_modules`、`swift/utils/transformers_utils.py::find_all_linears` |
| LoRA 注入 | `create_or_load_lora_model` | `swift/pipelines/train/tuner.py::prepare_adapter`、`Swift.prepare_model` |
| adapter 加载 | `PeftModel.from_pretrained(..., is_trainable=True)` | `swift/pipelines/train/tuner.py::TunerMixin.prepare_model` |
| 可训练参数审计 | `audit_lora_model` | `TunerMixin.prepare_model` 冻结逻辑及 ms-swift model-info 输出 |
| 梯度检查点 | `enable_input_require_grads`、`TrainingArguments` | `swift/trainers/mixin.py::_prepare_gradient_checkpointing` |
| selective logits | `QwenSFTTrainer._prepare_inputs` | `swift/trainers/mixin.py::prepare_logits_to_keep`；`swift/trainers/seq2seq_trainer.py::_prepare_inputs` |
| 优化与 DDP | `Trainer.train` | `SwiftSft.run/train`、`swift/trainers/trainer_factory.py` |
| 验证 | `Trainer.evaluate` | `swift/trainers/seq2seq_trainer.py` |
| checkpoint / adapter 保存 | `save_final_adapter` | `swift/trainers/mixin.py::_save_model/_save` |
| 完整恢复 | `resolve_resume_checkpoint` + `Trainer.train(resume...)` | `SwiftSft._get_resume_checkpoint`、`TunerMixin.prepare_model`、Trainer |
| 推理烟测 | `generate_smoke_test` | `SwiftSft._prepare_generation_config`、`Seq2SeqTrainer.prediction_step` |
| LoRA 合并 | `merge_and_save_lora` | `swift/pipelines/export/merge_lora.py`、`swift/tuners/base.py::merge_and_unload` |

代码中的每一个主要流程旁还有“代码功能 / 所处 SFT 阶段 / ms-swift 对应文件与函数”的就地注释，可从 `main()` 顺序阅读。

## 4. 为什么训练模板没有直接调用官方 Jinja

官方 Qwen3 chat template 与 ms-swift 当前 SFT 模板对历史 assistant 的非思考前缀处理不同：官方 Jinja 主要为最后一轮 assistant 插入空的 `<think>...</think>`；ms-swift 的训练模板会为每个普通 assistant 回答加入该前缀。另外，ms-swift 的动态 EOS 会监督 `<|im_end|>` 后的换行。

因此训练路径显式按 segment 编码：

```text
<|im_start|>system\n...<|im_end|>\n      labels 全为 -100
<|im_start|>user\n...<|im_end|>\n        labels 全为 -100
<|im_start|>assistant\n                    labels 全为 -100
<think>\n\n</think>\n\n回答<|im_end|>\n   labels 等于 input_ids
```

生成路径仍调用 tokenizer 官方 `apply_chat_template(..., add_generation_prompt=True)`，因为此时只有待生成的最后一轮 assistant prompt，不存在训练多轮标签差异。

`--enable_thinking` 的含义是：

- 默认关闭：若 assistant 内容不以 `<think>` 开头，加入空思考块；
- 开启：不自动加入空块，训练数据应自己提供期望的思考内容；
- 数据内容禁止嵌入 `<|im_start|>` / `<|im_end|>`，避免模板控制 token 注入。

## 5. 数据格式

支持 ms-swift 常用 messages 形式：

```json
{"messages":[
  {"role":"system","content":"你是一个准确的助手。"},
  {"role":"user","content":"什么是 LoRA？"},
  {"role":"assistant","content":"LoRA 用低秩增量适配预训练权重。"}
]}
```

也支持简单 instruction 形式：

```json
{"system":"你是一个准确的助手。","instruction":"翻译","input":"早上好","output":"Good morning."}
```

约束：system 只能位于开头，之后 user/assistant 严格交替，训练记录必须以 assistant 结束。没有独立 `eval_file` 时按 `seed` 和 `val_ratio` 做确定性划分。

## 6. 安装与运行

核心依赖：

```bash
pip install "torch>=2.4" "transformers>=4.57,<5" "peft>=0.19" "accelerate>=1.10" safetensors
```

先只验证数据、模板和 label，不加载 1.7B 模型：

```bash
python docs/learning/sft/minimal_qwen3_1_7b_sft_lora_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file docs/learning/sft/qwen3_1_7b_sft_sample.jsonl \
  --max_length 1024 \
  --dry_run
```

单卡主流 LoRA 训练：

```bash
CUDA_VISIBLE_DEVICES=0 python \
  docs/learning/sft/minimal_qwen3_1_7b_sft_lora_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file /path/to/train.jsonl \
  --eval_file /path/to/eval.jsonl \
  --output_dir output/qwen3-1.7b-lora \
  --max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --precision bf16
```

多卡通过 Accelerate/Transformers 的 DDP 路径运行，无需改代码：

```bash
torchrun --standalone --nproc_per_node=4 \
  docs/learning/sft/minimal_qwen3_1_7b_sft_lora_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file /path/to/train.jsonl \
  --eval_file /path/to/eval.jsonl \
  --output_dir output/qwen3-1.7b-lora-ddp \
  --precision bf16
```

有效全局 batch 为：

```text
per_device_train_batch_size × gradient_accumulation_steps × WORLD_SIZE
```

DDP 会复制 BF16 基座权重，但只同步很小的 LoRA 梯度。若单卡连基座与激活都放不下，应该进入 QLoRA、FSDP 或 DeepSpeed 场景；它们不是这个“普通 PEFT LoRA 最小实现”的默认路径。

## 7. 两种恢复方式

### 7.1 精确恢复完整训练状态

自动选择 `output_dir` 下最新 checkpoint：

```bash
python docs/learning/sft/minimal_qwen3_1_7b_sft_lora_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file /path/to/train.jsonl \
  --eval_file /path/to/eval.jsonl \
  --output_dir output/qwen3-1.7b-lora \
  --resume_from_checkpoint
```

或指定路径：

```bash
--resume_from_checkpoint output/qwen3-1.7b-lora/checkpoint-1000
```

启动顺序是：加载相同基座 → 从 checkpoint 以 `is_trainable=True` 恢复 adapter → Trainer 恢复 optimizer/scheduler/RNG/global step → 跳过已经消费的数据。这与 ms-swift LoRA resume 的职责分工一致。

### 7.2 只把已有 adapter 当初始化

```bash
--adapter_name_or_path output/old-run/final_adapter \
--output_dir output/new-run
```

这种方式加载 adapter 权重及其 `adapter_config.json`，但创建全新 optimizer/scheduler，从 step 0 开始。恢复 adapter 时，adapter 文件内的 rank、alpha、targets、dropout 等拓扑优先；命令行上的创建参数不会伪装成覆盖已有配置。

两个恢复参数互斥，脚本会显式报错，而不是产生含义模糊的训练。

## 8. adapter 推理与合并模型推理

直接加载 adapter：

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_path = "Qwen/Qwen3-1.7B"
adapter_path = "output/qwen3-1.7b-lora/final_adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()
```

训练时加 `--merge_lora` 会生成 `merged_model`。它可像普通 Transformers 模型一样加载：

```python
model = AutoModelForCausalLM.from_pretrained(
    "output/qwen3-1.7b-lora/merged_model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

合并是部署形式转换，不是训练必需步骤。保留 `final_adapter` 才便于切换 adapter、继续 LoRA 训练或复用同一个基座。

## 9. 与 ms-swift 主流 SFT LoRA 的覆盖情况

### 9.1 已完整覆盖的核心链路

- 本地 JSON/JSONL 读取和两种常见 schema；
- 确定性 train/eval 划分；
- Qwen3 ChatML、多轮 assistant、思考/非思考前缀；
- assistant-only 交叉熵标签和动态 padding；
- BF16/FP16/FP32、gradient checkpointing、`use_cache=False`；
- `all-linear` 展开，或显式 target modules；
- PEFT `LoraConfig`、rsLoRA、DoRA、bias、modules-to-save、初始化方式；
- 基座冻结、LoRA FP16 安全提升、可训练参数和注入位置硬审计；
- Transformers Trainer + Accelerate 单卡/DDP；
- AdamW、梯度累积、裁剪、warmup、cosine；
- Qwen3 `logits_to_keep` 显存优化；
- teacher-forced eval、loss、perplexity；
- adapter-only checkpoint 与最终导出；
- optimizer/scheduler/RNG/global-step 精确恢复；
- adapter-only 热启动；
- 训练后真实生成烟测；
- LoRA `merge_and_unload` 和完整模型导出；
- 数据 hash、版本、有效 PEFT 配置与训练参数清单。

### 9.2 完整 ms-swift 仍然多出的生产能力

这些不是 LoRA 算法不可缺少的步骤，但在大规模或复杂业务中很重要：

| 能力 | 最小脚本状态 | 何时需要 ms-swift |
|---|---|---|
| ModelScope/HF Hub、流式数据、多数据混合采样 | 仅本地 JSON/JSONL | 数据很大或来自多个数据源 |
| 大量已注册模型和模板 | 只面向 Qwen3 文本 | 混用不同模型家族 |
| lazy tokenize、packing、padding-free | 未实现 | 长度分布很散、追求吞吐 |
| 多模态输入 | 未实现 | 图像、音频、视频 SFT |
| loss scale、自定义 loss、序列并行 | 未实现 | token 加权或超长上下文 |
| QLoRA/AWQ/GPTQ/BnB/HQQ/EETQ | 未实现 | 单卡无法容纳 BF16 基座 |
| FSDP2、DeepSpeed ZeRO、Megatron | 未实现 | 基座/激活或多机规模要求分片 |
| 多 adapter、额外 tuner、Unsloth backend | 单一默认 adapter | 多任务/多租户或专用加速 |
| LoRA+、LoRA-GA、PiSSA 等高级初始化/学习率 | 仅标准/gaussian、rsLoRA、DoRA | 做特定 LoRA 研究 |
| FlashAttention/Liger/sequence packing 联合优化 | 可选 FA2，未封装其他 | 极致吞吐和长序列 |
| 丰富 metrics、generation eval、外部评测器 | loss/PPL + 单 prompt | 正式质量验收 |
| Hub push、ModelScope 发布、回调和实验平台 | 仅本地 manifest | 团队化训练治理 |
| flash checkpoint、远端 checkpoint | 普通本地 HF checkpoint | 超大分布式容错 |

因此结论不是“最小实现缺了 LoRA 的关键算法”，而是：它已经覆盖普通 Qwen3 文本 SFT LoRA 的算法与工程闭环；ms-swift 的主要增量是多模型/多模态/大规模分布式/数据工程/评测发布等生产抽象。

## 10. 已执行验证

本仓库环境中使用：

```text
torch 2.11.0+cu130
transformers 4.57.6
peft 0.19.1
accelerate 1.14.0
```

完成了以下检查：

1. 用本地 Qwen3-1.7B 官方 tokenizer 对示例数据 dry-run，检查多轮 ChatML 和 assistant-only labels；并与当前 ms-swift `qwen3 + swift backend + train mode` 逐 token 对比，`input_ids` 与 `labels` 完全相同；
2. 用缩小但保持真实 `Qwen3ForCausalLM` 结构的模型运行 `all-linear` LoRA 注入；
3. 执行 1 step 训练、selective logits、eval 和 perplexity；
4. 检查 `checkpoint-1` 同时含 adapter、optimizer、scheduler、RNG 和 trainer state；
5. 保存并重新检查 `final_adapter`；
6. 使用活动 adapter 执行 `generate`；
7. 执行 `merge_and_unload(safe_merge=True)` 并验证普通 `model.safetensors`；adapter 与 merged 模型对同一输入的 logits 最大绝对差为 0；
8. 从 `checkpoint-1` 精确恢复到 step 2，生成 `checkpoint-2`，最终 `global_step=2`；
9. 分别执行 batch size 1 和 2 的 selective-logits 分支，并通过 Python 编译和 Ruff 静态检查。

缩小模型的回答没有语义意义；该测试验证的是接口、张量、梯度、保存恢复和加载导出的完整性。真实 Qwen3-1.7B 的质量验证还应使用足量训练数据、独立验证集、固定业务评测集，并对基座与 adapter 做同 prompt 对照。

## 11. 建议的正式验收清单

训练前：

- 检查首个 rendered sample 和 supervised target；
- 检查 target ratio 不是 0%，也没有把 user/system 放进 target；
- 检查打印的 target modules 正好是预期的七类投影；
- 检查 trainable ratio 远小于 100%；
- 记录代码、数据 hash、模型 revision 和依赖版本。

训练中：

- loss 有限且总体下降，无 NaN/Inf；
- gradient norm 有限；
- 有效全局 batch、学习率和 warmup 符合设计；
- checkpoint 可在另一进程恢复，global step 连续；
- 观察显存峰值和 tokens/s，而不只观察 step/s。

训练后：

- `final_adapter` 可由 `PeftModel.from_pretrained` 加载；
- 固定 prompt 比较 base、adapter 和 merged 输出；
- adapter 与 merged 输出在确定性解码下基本一致；
- 独立 eval loss/PPL 不恶化，业务评测有可解释提升；
- 保留 adapter、manifest、数据版本、评测结果和可恢复 checkpoint。
