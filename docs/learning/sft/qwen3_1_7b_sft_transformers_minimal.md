# Qwen3-1.7B：基于 Transformers 的最小完整 SFT 实现

> 实现目标：不用 ms-swift、TRL、PEFT 和自定义训练循环，只依赖 PyTorch 与 Transformers，把 SFT 从原始对话数据一直走到训练、评估、断点、最终模型和生成验收；显存受限的 Full 场景可选用 PyTorch FSDP2 做参数、梯度和优化器状态全分片。
>
> 本地核验环境：PyTorch 2.11.0、Transformers 4.57.6、Accelerate 1.14.0；Qwen3-1.7B 官方模型至少要求 Transformers 4.51。

配套文件：

- `minimal_qwen3_1_7b_sft_transformers.py`：完整单文件实现；
- `qwen3_1_7b_sft_sample.jsonl`：12 条可直接用于 dry-run 的示例数据。

## 1. 这个“最小版本”覆盖了什么

脚本覆盖一条完整的 SFT 闭环：

```text
JSON/JSONL
  -> 数据格式标准化与角色校验
  -> 训练/验证切分
  -> Qwen 官方 Chat Template
  -> input_ids / attention_mask
  -> assistant-only labels（prompt 与 padding 为 -100）
  -> DataCollatorForSeq2Seq 动态 padding
  -> AutoModelForCausalLM
  -> 可选 FSDP2 FULL_SHARD（参数、梯度、优化器状态分片）
  -> Trainer 前向、loss、反向传播、梯度累积、优化器与调度器
  -> epoch 验证、eval_loss、perplexity
  -> checkpoint-* 普通/分片断点与训练状态
  -> final 完整推理模型或可离线合并的最终分片
  -> 固定 prompt 贪心生成验收
```

它有意采用**全参数 SFT**，让训练目标和 Transformers 主流程保持最透明。LoRA/QLoRA 需要 PEFT、bitsandbytes 等额外抽象，不属于这个“只观察 SFT 本体”的最小实现。

需要同时明确：“代码最小”不等于“显存最小”。Qwen3-1.7B 的 bf16 模型权重约为数 GB，但全参数 AdamW 还需要梯度、优化器状态和激活，实际峰值通常远高于 8GB。脚本现已提供 FSDP2 Full 分片路径；它能显著降低每卡的模型状态占用，但前向时仍需按 FSDP 单元临时 AllGather 参数，也无法消除激活与 logits 占用。单张 8GB 显卡不能通过 FSDP2 自己和自己分片；此时仍需更多 GPU、CPU offload/ZeRO 方案，或者 LoRA/QLoRA。普通 DDP 只复制模型，并不会降低单卡模型状态显存。

## 2. SFT 的核心数学目标

一条对话被模板编码为 token 序列：

$$x=(x_1,x_2,\ldots,x_T).$$

因果语言模型在位置 $t$ 的输出预测下一个 token $x_{t+1}$。SFT 只要求模型模仿 assistant 回复，因此定义监督位置集合 $A$：

$$
\mathcal L_{SFT}
=-\frac{1}{|A|}\sum_{t+1\in A}
\log p_\theta(x_{t+1}\mid x_{\le t}).
$$

代码层面不需要自己写交叉熵。我们令：

- assistant 回复及其 `<|im_end|>` 对应的 `labels[t] = input_ids[t]`；
- system、user、assistant 角色头和 padding 对应 `labels[t] = -100`。

`AutoModelForCausalLM` 收到 `labels` 后会在内部完成 next-token shift 和 `ignore_index=-100` 的交叉熵。随后 `Trainer` 对这个标量 loss 执行反向传播、梯度累积和优化器更新。

## 3. 为什么不能直接复制 input_ids 到 labels

许多因果语言模型教程使用：

```python
labels = input_ids.copy()
```

这适合预训练式的“每个非 padding token 都是目标”，但不是典型 assistant-only SFT。若直接复制，模型还会学习预测 system、user、角色控制符和用户问题，相当于浪费梯度去复述提示词。

这个实现也没有使用 `DataCollatorForLanguageModeling(mlm=False)` 自动制造 labels，因为它同样倾向于把全部非 padding 输入作为目标。脚本在预处理阶段精确构造 labels，再让 `DataCollatorForSeq2Seq` 只负责动态 padding，并用 `-100` 补齐 label。

## 4. Qwen3 Chat Template 的特殊陷阱

Transformers 提供：

```python
tokenizer.apply_chat_template(
    messages,
    return_assistant_tokens_mask=True,
    return_dict=True,
)
```

但这个功能只有 Chat Template 用 `{% generation %}` / `{% endgeneration %}` 标出 assistant 生成区间时才有效。当前 Qwen3-1.7B 官方模板不包含该标签；在本地 Transformers 4.57.6 上调用会产生警告，并返回全 0 的 `assistant_masks`。如果不审计 mask，训练样本会变成“没有任何有效 label”。

本实现仍然复用官方 `apply_chat_template`，但在 token 化结果中识别：

```text
<|im_start|>assistant\n  ...assistant 生成内容...  <|im_end|>
```

只把 header 之后到 `<|im_end|>`（含结束 token）的区间复制到 labels。这样既不自行重写 Qwen 的复杂 Jinja 模板，又能得到准确的 assistant-only mask。为防止数据内容伪造边界，脚本拒绝包含 `<|im_start|>` 或 `<|im_end|>` 的原始 content。

Qwen3 支持 thinking/non-thinking 两种模式。本示例默认 `--no-enable_thinking`，适合普通短回答数据。官方模板在非思考回答前仍会呈现空的 `<think>...</think>` 前缀；它属于 assistant 需要生成的序列，因此会被纳入监督。如果数据本身包含高质量推理过程，可显式传 `--enable_thinking` 并统一训练与推理约定。

## 5. 数据格式

推荐标准 messages JSONL，每行一条完整对话：

```json
{"messages":[{"role":"system","content":"你是一个准确的助手。"},{"role":"user","content":"什么是 SFT？"},{"role":"assistant","content":"SFT 是使用带目标回答的数据继续训练模型。"}]}
```

多轮对话可以写成：

```json
{"messages":[{"role":"user","content":"2+3 等于多少？"},{"role":"assistant","content":"等于 5。"},{"role":"user","content":"再乘以 4？"},{"role":"assistant","content":"等于 20。"}]}
```

这个最小实现也接受：

```json
{"system":"可选系统提示","instruction":"任务指令","input":"可选输入","output":"目标回答"}
```

约束如下：

1. 只支持纯文本 `system/user/assistant`，不处理图片、tool call 和 `reasoning_content` 独立字段；
2. system 只能位于首条，此后 user/assistant 必须交替；
3. 每条训练对话必须以 assistant 结束；
4. assistant 不能为空；
5. 默认对超长样本使用 `drop`，避免粗暴截断破坏对话结构。`--overlength_strategy left` 可保留最后 `max_length` 个 token，但可能丢失 system 或早期轮次，必须审计。

生产实验应提供独立 `--eval_file`。未提供时，脚本按 `seed` 固定随机切分；这只适合教学或快速验收。正式数据应先按用户、文档、时间或任务实体去重分组，再切分，避免近重复泄漏。

## 6. 环境准备

最小依赖：

```bash
pip install "transformers>=4.51,<5" accelerate torch
```

仓库的 `swift/config/fsdp2.json` 描述仍写着 PyTorch 2.4 起，但当前本地 Accelerate 1.14.0 的实际运行时常量要求 PyTorch 2.6 起；脚本按后者做早期检查。本文代码还依赖当前 Transformers/Accelerate 对 FSDP2 checkpoint 和 CPU-RAM-efficient loading 的集成，建议使用本地已验证组合 PyTorch 2.11.0、Transformers 4.57.6、Accelerate 1.14.0，且通过 NCCL/CUDA 多进程启动。

在当前仓库已经准备好的虚拟环境中可以直接使用：

```bash
.venv/bin/python -c \
  "import torch, transformers, accelerate; print(torch.__version__, transformers.__version__, accelerate.__version__)"
```

官方模型名为 `Qwen/Qwen3-1.7B`。若已经下载到本地，可把 `--model_name_or_path` 指向本地目录并加 `--local_files_only`。

## 7. 先执行不加载模型的 dry-run

这一步会验证 tokenizer、数据格式、模板、assistant mask、长度过滤和动态 padding，但不会加载 1.7B 权重：

```bash
.venv/bin/python \
  docs/learning/sft/minimal_qwen3_1_7b_sft_transformers.py \
  --model_name_or_path /home/cb/model/Qwen3-1.7B \
  --train_file docs/learning/sft/qwen3_1_7b_sft_sample.jsonl \
  --max_length 512 \
  --local_files_only \
  --dry_run
```

输出中必须看到：

- train/eval 都有保留下来的样本；
- `target_tokens > 0`；
- `first supervised target` 只包含 assistant 内容和 `<|im_end|>`，不包含 system/user；
- collator 后 `input_ids`、`attention_mask`、`labels` 形状完全一致；
- 最后一行是 `tokenizer, data validation, assistant-only labels and collation: OK`。

如果这一步不通过，不应加载模型开始训练。

## 8. 最小完整训练命令

用本地 Qwen3-1.7B 做三步链路验收：

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python \
  docs/learning/sft/minimal_qwen3_1_7b_sft_transformers.py \
  --model_name_or_path /home/cb/model/Qwen3-1.7B \
  --train_file docs/learning/sft/qwen3_1_7b_sft_sample.jsonl \
  --output_dir output/qwen3-1.7b-transformers-sft-smoke \
  --max_length 512 \
  --max_steps 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --precision bf16 \
  --attn_implementation sdpa \
  --local_files_only
```

这仍是全参数训练，8GB 显卡通常无法承载。三步命令用于“有足够显存时快速验证”，不是效果训练配置。

正式训练示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python \
  docs/learning/sft/minimal_qwen3_1_7b_sft_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file /path/to/train.jsonl \
  --eval_file /path/to/eval.jsonl \
  --output_dir output/qwen3-1.7b-full-sft \
  --max_length 1024 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.03 \
  --precision bf16 \
  --attn_implementation sdpa
```

全局 batch 为：

$$
B_{global}=B_{device}\times gradient\_accumulation\_steps\times world\_size.
$$

上述单卡配置的全局 batch 是 16。

### 8.1 两卡及以上的 FSDP2 Full 分片命令

假设一台机器有两张可见 GPU，使用 `torchrun` 启动两个进程，每个进程绑定一张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
.venv/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  docs/learning/sft/minimal_qwen3_1_7b_sft_transformers.py \
  --model_name_or_path /home/cb/model/Qwen3-1.7B \
  --train_file /path/to/train.jsonl \
  --eval_file /path/to/eval.jsonl \
  --output_dir output/qwen3-1.7b-full-sft-fsdp2 \
  --max_length 1024 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --precision bf16 \
  --attn_implementation sdpa \
  --local_files_only \
  --fsdp2
```

这个例子的全局 batch 是 `1 × 8 × 2 = 16`。不能用普通 `python ... --fsdp2`：脚本会检查 `WORLD_SIZE`、`LOCAL_RANK`、CUDA 和进程数，并在未真正进入多卡任务时立即报错，避免用户误以为模型已经分片。

`--fsdp2` 的有效配置与当前 `swift/config/fsdp2.json` 对齐：

| 配置 | 本实现值 | 含义 |
|---|---:|---|
| `fsdp_version` | `2` | 使用基于 DTensor 和 `fully_shard` 的 FSDP2 |
| Trainer `fsdp` | `full_shard auto_wrap` | 参数、梯度和优化器状态全分片，并按 Transformer 层自动包装 |
| `reshard_after_forward` | `true` | 每个 FSDP 单元前向完成后立即重新分片，降低驻留参数显存 |
| `auto_wrap_policy` | `TRANSFORMER_BASED_WRAP` | 使用 Qwen3 的 `_no_split_modules=["Qwen3DecoderLayer"]` 按 decoder layer 分片 |
| `cpu_ram_efficient_loading` | `true` | 每个节点只让 local rank 0 读取完整预训练权重，其余 local rank 先构造 meta 参数，再接收自己的 shard；单机时就是 global rank 0 |
| `state_dict_type` | `SHARDED_STATE_DICT` | 训练 checkpoint 按 rank 保存，避免保存时聚合整个模型 |
| `activation_checkpointing` | `true` | 使用 FSDP 原生的非重入式激活重计算 |

环境配置和 `TrainingArguments` 必须在 `from_pretrained` 之前完成。否则进程组尚未初始化，Transformers 不知道当前是 FSDP CPU-RAM-efficient loading，结果会是每个 rank 都从磁盘读取并在 CPU 放置一份完整模型。本实现为此调整了主流程顺序，这与 ms-swift 在 `SftArguments._init_fsdp/_init_device` 完成分布式初始化、随后才由 `SwiftSft._prepare_model_tokenizer` 加载模型的顺序一致。这里优化的是同一节点内的重复加载；多机时每个节点的 local rank 0 仍可能读取一次模型文件。

FSDP2 原生 `activation_checkpointing=true` 时，代码会把 `TrainingArguments.gradient_checkpointing` 的有效值自动设为 `false`。二者同时开启不仅语义重复，还会导致 Transformers 直接拒绝启动或引入额外 backward AllGather。若传 `--no-fsdp2_activation_checkpointing`，则普通 `--gradient_checkpointing`（默认开启）重新生效。

### 8.2 FSDP2 保存、续训与最终导出

FSDP2 下有两类用途不同的产物：

- `checkpoint-*` 始终使用 `SHARDED_STATE_DICT`，包含模型 shard、optimizer shard、scheduler、Trainer state 和各 rank RNG，适合低峰值保存及完整续训；续训必须继续使用相同的 `torchrun ... --fsdp2 --resume_from_checkpoint ...` 方式。
- 默认的 `final/` 会在训练结束后临时切换到 `FULL_STATE_DICT`，所有 rank 参加 collectives，完整权重被 offload/gather 到 rank 0 CPU 并写成 Hugging Face 可直接加载的模型。这个过程不再占用每卡一份完整权重，但 rank 0 必须有足够主机内存；1.7B FP32 master state 的导出峰值不能按 bf16 权重大小估算。

如果主机内存不足，训练命令可改为：

```bash
... --fsdp2 --fsdp2_final_state_dict_type sharded
```

此时 `final/pytorch_model_fsdp_0/` 仍是分布式 checkpoint，不可直接传给 `AutoModelForCausalLM.from_pretrained`。训练进程退出后，在 CPU 内存充足的机器上离线合并，并把权重写回已经包含 config/tokenizer 的 `final/`：

```bash
.venv/bin/accelerate merge-weights \
  output/qwen3-1.7b-full-sft-fsdp2/final/pytorch_model_fsdp_0 \
  output/qwen3-1.7b-full-sft-fsdp2/final
```

合并后应出现 `final/model.safetensors`，再用新进程执行 `from_pretrained(final_dir)` 验证。不要把任意一个 `.distcp` shard 当作完整模型。

### 8.3 FSDP2 启动后必须看到的审计信息

Trainer 构造后以及第一次训练结束后，代码各检查一次分布式状态。第二次必须输出类似：

```text
[fsdp2] verified: version=2, full_shard=True, activation_checkpointing=True,
state_dict_type=StateDictType.SHARDED_STATE_DICT, wrapped=True
```

其中 `wrapped=True` 会用 `torch.distributed.fsdp.FSDPModule` 做实际类型断言，证明这不是设置了参数却仍走 DDP。还应检查每个 `checkpoint-*` 内存在 `pytorch_model_fsdp_0/` 和 `optimizer_0/`，并实际执行一次从该 checkpoint 恢复的短训练。

## 9. 代码各阶段的职责

### 9.1 数据读取与可复现切分

`read_records` 读取 JSON/JSONL；`normalize_record` 把两种输入格式收敛到 messages，并严格校验角色。`split_records` 使用独立的 `random.Random(seed)` 固定切分，不依赖 Trainer 的 DataLoader shuffle。

脚本在 `run_manifest.json` 中记录训练文件绝对路径和 SHA-256。仅记录参数而不记录数据版本，不能真正复现实验。

### 9.2 模板与标签

`TokenizedSFTDataset` 在训练前一次性调用官方 Chat Template，然后用 `build_assistant_only_labels` 构造 `-100` mask。这种预编码方式适合最小教学代码和中小数据集；海量或多模态数据应改成 lazy/streaming 编码。

每条数据都检查：

- 模板扫描出的 assistant span 数量等于 messages 中 assistant 轮数；
- 超长处理后至少剩余一个有效目标 token；
- `input_ids`、`attention_mask`、`labels` 等长。

### 9.3 Collator

`DataCollatorForSeq2Seq` 虽然名字包含 Seq2Seq，但它可以安全地为已经构造好 labels 的 Causal LM batch 做动态 padding：

- `input_ids` 用 tokenizer 的 pad token；
- `attention_mask` 在 padding 处为 0；
- `labels` 用 `-100` padding；
- CUDA 下可补齐到 8 的倍数，提高 Tensor Core 友好度。

### 9.4 模型与 loss

`AutoModelForCausalLM.from_pretrained` 加载 Qwen3。没有传 `device_map=auto`，因为训练时应让 Trainer/Accelerate 管理设备和分布式进程。脚本断言 `trainable_parameters == total_parameters`，防止本来要演示 Full SFT 却意外冻结部分参数。

梯度检查点开启时把 `model.config.use_cache=False`，避免 KV cache 与训练重计算冲突。推理验收前再恢复 `use_cache=True`。

### 9.5 Trainer

`TrainingArguments` 控制：

- bf16/fp16；
- 梯度累积与裁剪；
- AdamW/Adafactor；
- cosine 调度与 warmup；
- 每个 epoch 验证和保存；
- checkpoint 数量；
- 确定性、DataLoader 与日志。

`Trainer` 接收已经准备好的模型、数据集、collator 和 tokenizer。模型前向返回 loss 后，Trainer 负责 backward、优化器 step、scheduler step、分布式梯度同步和训练状态保存。

启用 FSDP2 时，`TrainingArguments(fsdp="full_shard auto_wrap", fsdp_config=...)` 让 Trainer 创建带 FSDP2 plugin 的 Accelerator。优化器创建和模型分片由当前 Transformers/Accelerate 协同处理；脚本没有手动调用 `fully_shard`，但会在训练后断言根模型已经成为 `FSDPModule`。这种接法与 ms-swift 相同：ms-swift 负责参数预处理和配置，最终仍由其 Transformers Trainer 基类与 Accelerate 执行分片。

### 9.6 评估、保存与生成

训练后显式调用 `trainer.evaluate()`，记录 `eval_loss` 和普通有效目标 token 上的 perplexity：

$$PPL=\exp(eval\_loss).$$

产物分为：

- `checkpoint-*`：包含用于断点续训的 Trainer/优化器/调度器状态；FSDP2 时是分布式 model/optimizer state；
- `final/`：普通模式或 FSDP2 默认 `full` 导出时是推理就绪模型和 tokenizer；FSDP2 `sharded` 导出时需先执行 `accelerate merge-weights`；
- `run_manifest.json`：代码环境、参数、数据哈希和数据统计；
- `generation_smoke_test.json`：固定 prompt 的训练后生成结果。

生成测试使用与训练相同的 Chat Template，`add_generation_prompt=True`，并只解码 prompt 之后的新 token。

## 10. 断点续训

从明确 checkpoint 恢复：

```bash
.venv/bin/python docs/learning/sft/minimal_qwen3_1_7b_sft_transformers.py \
  --model_name_or_path /home/cb/model/Qwen3-1.7B \
  --train_file /path/to/train.jsonl \
  --eval_file /path/to/eval.jsonl \
  --output_dir output/qwen3-1.7b-full-sft \
  --resume_from_checkpoint output/qwen3-1.7b-full-sft/checkpoint-100
```

自动选择 `output_dir` 下编号最大的 checkpoint：

```bash
... --resume_from_checkpoint
```

恢复后应检查日志中的 global step 和学习率是否接续，而不是只看“成功加载”。`final/` 是推理交付物，不保证包含 optimizer/scheduler，续训应使用 `checkpoint-*`。

## 11. 如何证明全流程正确

最小验收顺序如下：

1. **静态检查**：脚本能编译，CLI help 正常；
2. **dry-run**：标签中只出现 assistant 内容，padding 后三个张量同形；
3. **微型训练**：用 8～32 条数据和少量 step，train loss 能下降；
4. **过拟合测试**：重复训练极小数据，固定 prompt 能复现训练答案；
5. **验证集测试**：eval loss 有限，不含训练集重复样本；
6. **参数测试**：Full SFT 的 trainable/total 参数相等，更新后至少一个参数变化；
7. **断点测试**：连续训练 N 步，与 N/2 保存后续训到 N 步的曲线连续；
8. **新进程加载**：从 `final/` 加载并用相同模板生成；
9. **基线对比**：Base 与 SFT 模型使用相同 prompt、生成参数和测试集；
10. **回归评测**：除目标任务外，还检查通用问答、格式遵循和安全能力是否退化。

FSDP2 还必须额外通过四项系统验收：日志中 `version=2/full_shard=True/wrapped=True`；checkpoint 同时具有 model/optimizer 分片；从中断 checkpoint 续训后 global step 连续；完整导出或离线 merge 后能在不初始化分布式的新进程里 `from_pretrained`。少任何一项，都不能只凭多卡显存下降认定链路完整。

“命令没有报错”和“loss 下降”不能单独证明流程正确。mask、模板、checkpoint 与生成验收都必须通过。

## 12. 最小实现的边界与升级方向

这个实现故意没有隐藏复杂度，因此边界也很清楚：

- 数据一次性放入内存，不适合超大数据集；
- 只支持 system/user/assistant 纯文本；
- 不做 Packing；
- 不做 LoRA/QLoRA；
- 已覆盖 FSDP2 Full 分片，但不覆盖 FSDP CPU offload、混合分片网格、DeepSpeed、TP/PP/SP 或多机容错编排；
- eval 主要监控 NLL/PPL，不等价于业务回答质量；
- Qwen assistant span 识别依赖 Qwen3 当前控制 token 契约，切换模型家族时必须替换或重新验证 mask 构造。

掌握本代码后，升级顺序建议是：先增加任务指标与数据审计，再加入 lazy/streaming 与 Packing，然后根据资源选择 PEFT、DeepSpeed 或更复杂的并行网格。每增加一层，都保留本脚本的单进程路径作为 correctness baseline，并为 FSDP2 单独保留保存/恢复回归测试。

## 13. 一手资料

- [Qwen3-1.7B 官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)：模型类型、参数量、上下文长度、Transformers 版本要求与推理方式；
- [Qwen3 官方技术博客](https://qwenlm.github.io/blog/qwen3/)：thinking/non-thinking 模式与 `enable_thinking`；
- [Transformers Chat Template 官方文档](https://huggingface.co/docs/transformers/chat_templating)：训练预处理时使用 Chat Template，并设置 `add_generation_prompt=False`；
- [Transformers Tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer)：`return_assistant_tokens_mask` 仅对含 `{% generation %}` 的模板有效；
- [Transformers Trainer 官方文档](https://huggingface.co/docs/transformers/main_classes/trainer)：完整训练、评估、混合精度与分布式循环；
- [Transformers Causal Language Modeling 教程](https://huggingface.co/docs/transformers/tasks/language_modeling)：next-token 目标、Trainer、评估与 perplexity。
