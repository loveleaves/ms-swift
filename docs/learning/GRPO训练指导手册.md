# ms-swift 4.4.0 GRPO 训练指导手册
## Qwen3-32B（Dense）与 Qwen3.6-35B-A3B（MoE）· LoRA / Full 全参 · GSM8K 数据集 · FSDP2 分片训练收敛保障

> 版本说明：本手册基于 ms-swift 4.4.0（对齐 4.4.x～4.5.0.dev 系列文档与社区实践）、PyTorch 2.x FSDP2（DTensor 版）、vLLM 0.7+ colocate/server 两种 rollout 模式编写，聚焦"如何保证 GRPO 训练稳定收敛"这一核心工程问题。手册中所有命令行参数均以 `swift rlhf --rlhf_type grpo` 为主线，覆盖 LoRA（`--train_type lora`）与全参（`--train_type full`）两种训练方式，分布式后端聚焦 `--fsdp fsdp2`（区别于 DeepSpeed ZeRO 路线）。
>
> 适用读者：具备一定 LLM 训练工程经验、需要在多卡/多机环境下用 GSM8K（或结构相似的可验证数学推理数据集）对 Qwen3-32B 稠密模型、Qwen3.6-35B-A3B（35B 总参数/约 3B 激活参数的 MoE 架构）进行 GRPO 强化学习训练的算法工程师、平台工程师。
>
> **重要声明**：Qwen3.6-35B-A3B 在公开渠道信息有限（可能为社区/内部命名或较新发布的 MoE 变体），本手册按"Qwen3 系列 A3B 型 MoE 架构（30B～35B 总参数、约 3B 激活参数，结构类比 Qwen3-30B-A3B）"的通用规律给出配置建议；实际字段名（如专家数、`num_experts`、`moe_intermediate_size`）请以你本地 `config.json` 为准，手册中给出的是可迁移的方法论而非写死的数字。

---

## 目录

1. 引言：为什么 GRPO 训练"不收敛"是工程常态
2. GRPO 算法原理与 ms-swift 实现机制
3. 环境搭建与版本锁定
4. 硬件规划：32B Dense 与 35B-A3B MoE 的资源画像差异
5. FSDP2 分片技术详解与 ms-swift 中的配置方法
6. GSM8K 数据集准备、格式化与奖励函数设计
7. 训练前的"收敛保障"检查清单（Convergence Readiness Checklist）
8. Qwen3-32B GRPO LoRA 训练完整方案
9. Qwen3-32B GRPO Full 全参训练完整方案
10. Qwen3.6-35B-A3B（MoE）GRPO 训练特殊性与 FSDP2 适配
11. 关键超参数详解与调优方法论
12. 训练过程监控：指标体系与健康区间
13. 收敛失败模式诊断手册（含典型 case）
14. 显存与吞吐优化：offload、vLLM colocate/server、sleep_level
15. 多机多节点分布式训练配置
16. Checkpoint 管理、断点续训与灾难恢复
17. 评估方法论：从训练内 reward 到线下 GSM8K Acc
18. LoRA → Full 的渐进式迁移路径
19. 完整可运行脚本合集
20. 常见问题 FAQ
21. 附录：参数速查表 / 术语表 / 参考资料

---

## 第一章 引言：为什么 GRPO 训练"不收敛"是工程常态

在 SFT（监督微调）中，"收敛"是一个相对简单的概念：训练 loss 单调下降、验证 loss 不明显反弹，基本可以判断训练是健康的。但 GRPO（Group Relative Policy Optimization）作为一种在线强化学习算法，其"收敛"含义完全不同，且更容易在工程上"看起来在跑，实际上已经跑偏"。这是因为：

1. **loss 本身不是核心观测指标**。GRPO 的 policy loss 在健康训练中往往在 0 附近小幅波动（尤其是 on-policy 部分），不会像 SFT 那样持续下降。如果你盯着 loss 曲线判断收敛，大概率会得出错误结论。真正该盯的是 **reward 曲线、KL 散度、response 长度、熵（entropy）** 这几组指标的联合走势。
2. **rollout（采样）质量直接决定训练质量**。GRPO 依赖模型自身生成一组（`num_generations` 个）候选答案，再通过组内相对优势（advantage）做策略梯度更新。如果 rollout 阶段的推理引擎（vLLM）配置不当、采样温度不合理、或者 prompt 模板与训练模板不一致，会导致"训练用的数据本身就是错的"，无论怎么调学习率都无法收敛。
3. **奖励函数设计的鲁棒性问题**。GSM8K 是一个有确定性数值答案的数据集，天然适合用 rule-based（规则式）奖励（如答案精确匹配、格式匹配）。但奖励函数如果对答案抽取的正则表达式过于脆弱、对格式奖励和正确性奖励的权重设置不合理，会诱发"奖励黑客"（reward hacking）——模型学会用奇怪的格式骗过奖励函数，而不是真正提升推理能力。
4. **分布式训练框架（FSDP2）与 RL 特有的"训练-推理引擎切换"叠加的工程复杂度**。GRPO 训练循环中，策略模型需要在"用 FSDP2 做全参/LoRA 梯度更新"和"把参数同步给 vLLM 推理引擎做 rollout"之间反复切换，这个同步过程如果配置不当（尤其是 MoE 模型的专家权重同步），会导致 rollout 用的是旧权重、训练出现"滞后策略"（stale policy）问题，从而看起来像不收敛。
5. **大模型（32B/35B）叠加分片训练（FSDP2）会放大所有上述问题**。相比 7B/8B 模型上的 GRPO demo，32B/35B 级别模型显存和通信开销更大，工程师往往被迫开启梯度检查点（activation checkpointing）、CPU offload、混合精度等一系列显存优化手段，这些手段中的任何一个配置错误都可能间接影响数值稳定性（如 offload 导致的精度损失、梯度累积与 FSDP2 分片规则冲突等），使得"到底是算法层面不收敛，还是工程层面配置错误"变得难以区分。

本手册的核心目标，就是把"保证收敛"拆解成可执行、可检查、可复现的工程动作，而不是停留在"调调超参数试试"的经验主义层面。全文遵循以下方法论：

- **先正确，后收敛**：任何超参数调优之前，先确保数据格式、奖励函数、模板对齐、FSDP2 分片配置这四个"正确性"前提被满足。90% 的"不收敛"问题本质是配置错误，而非算法调参问题。
- **小规模验证，大规模复现**：所有新配置（新奖励函数、新 FSDP2 policy、新学习率）都应先在小模型（如 Qwen3-4B/8B）+ 小数据子集（GSM8K 200～500 条）上跑通，确认指标健康后再迁移到 32B/35B。这是控制试错成本的关键。
- **指标先行，直觉在后**：本手册第 12、13 章会给出一套完整的指标监控体系和异常模式对照表，帮助你在训练出问题的第一时间（而不是训练完成后）发现并定位问题。

---

## 第二章 GRPO 算法原理与 ms-swift 实现机制

### 2.1 GRPO 的核心思想

GRPO（Group Relative Policy Optimization）由 DeepSeek 团队在 DeepSeekMath 论文中提出，后在 DeepSeek-R1 中发扬光大，核心思想是**用组内相对奖励代替 PPO 中的价值函数（Critic）**，从而省去一个和策略模型同等规模的 Critic 网络，大幅降低强化学习训练的显存和工程复杂度。

其基本流程为：

1. 对于一个 prompt（如一道 GSM8K 数学题），策略模型（policy model）采样生成 `G`（对应 ms-swift 中的 `num_generations` 参数）个候选回答（completion）。
2. 用奖励函数（reward function）对这 `G` 个回答分别打分，得到奖励值 `r_1, r_2, ..., r_G`。
3. 计算组内归一化优势：`A_i = (r_i - mean(r)) / (std(r) + eps)`，即每个回答相对于同组其他回答的相对好坏。
4. 用这个优势值 `A_i` 加权策略梯度，同时引入与参考模型（reference model，通常是训练开始时的初始策略，或额外指定的 `--ref_model`）之间的 KL 散度惩罚项，防止策略偏离过远导致语言能力退化。
5. 使用类似 PPO 的 clip 机制（`--epsilon`、`--epsilon_high` 等）限制单步更新幅度，保证训练稳定性。

其优化目标可以直观理解为：

```
L(θ) = E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ] - β * KL(π_θ || π_ref)
```

其中 `ratio = π_θ(a|s) / π_θ_old(a|s)` 是新旧策略的概率比。

### 2.2 GRPO 相比 PPO 的工程优势与代价

优势：
- 不需要训练 Critic（价值网络），节省约 1 倍的模型显存和计算。
- 组内相对奖励天然对奖励的绝对尺度不敏感，一定程度上降低了奖励函数设计的难度。

代价（也是本手册要重点解决的"收敛难点"来源）：
- 组内样本数 `G`（`num_generations`）必须足够大，否则组内方差估计不准，advantage 信号噪声很大。GSM8K 场景经验值一般在 8～16 之间。
- 由于没有 Critic 做长期价值估计，GRPO 对**奖励函数的即时性和准确性**要求更高——每一个 rollout 的奖励都必须直接、准确地反映这个回答的好坏。
- rollout 阶段需要策略模型本身做推理生成，这意味着**训练进程需要同时承担"训练"和"推理"两种角色**，这是 ms-swift 在工程上引入 vLLM 加速、`--sleep_level`、`--offload_model` 等一系列机制的根本原因，也是分布式配置（尤其是 FSDP2 + vLLM 协同）复杂度的来源。

### 2.3 ms-swift 中 GRPO 的训练循环与关键组件

ms-swift 通过 `swift rlhf --rlhf_type grpo` 命令启动 GRPO 训练，内部主要包含以下组件的协同：

1. **Trainer（训练侧）**：基于 Transformers Trainer 扩展，负责前向/反向传播、优化器更新、FSDP2/DeepSpeed 分片管理。
2. **Rollout 引擎（推理侧）**：默认可选 `--use_vllm true`，vLLM 负责高效地为每个 prompt 生成 `num_generations` 个候选。vLLM 有两种运行模式：
   - **colocate 模式**（`--vllm_mode colocate`）：vLLM 与训练进程共享同一批 GPU，通过 `--sleep_level`、显存 `--vllm_gpu_memory_utilization` 参数控制训练/推理阶段的显存切换。适合单机场景，部署简单，但训练/推理阶段串行执行，GPU 利用率有取舍。
   - **server 模式**（`--vllm_mode server`）：vLLM 作为独立服务进程（可以单独占用部分 GPU，或部署在独立节点），训练进程通过 HTTP/gRPC 请求 rollout。适合多机场景，可以让推理和训练并行（异步 GRPO），吞吐更高，但引入了服务可用性、权重同步延迟等新的稳定性问题。
3. **奖励函数（Reward Function）**：可以是内置的规则奖励（如 `accuracy`、`format`），也可以通过 `--reward_funcs` 指定自定义 plugin，或者用 `--reward_model` 指定一个独立的奖励模型。GSM8K 场景以规则奖励为主。
4. **参考模型（Reference Model）**：用于计算 KL 惩罚。当 `--train_type lora` 时，参考模型天然可以是"关闭 LoRA adapter 后的 base 模型"，不需要额外显存；当 `--train_type full` 全参训练时，需要额外维护一份参考模型权重（可以 offload 到 CPU，见 `--ref_model` 与 offload 相关参数），这是 Full 训练比 LoRA 训练显存开销大得多的原因之一。
5. **权重同步机制**：每次训练更新完策略模型参数后，需要把最新参数同步给 vLLM 推理引擎，供下一轮 rollout 使用。这一步在 FSDP2 分片场景下，需要先做参数的 all-gather（把分片的参数临时聚合成完整参数），再传给 vLLM。这是 FSDP2 + GRPO 组合中**最容易出现显存尖峰（OOM）和同步 bug 的环节**，第 5、10 章会详细展开。

理解了这个循环，你就能明白为什么本手册反复强调"rollout 配置的正确性"和"权重同步的正确性"是 GRPO 收敛的两大工程基石——他们不是标准 SFT 训练里存在的问题，是 GRPO/RLHF 训练特有的、也是最容易被低估的风险点。


## 第三章 环境搭建与版本锁定

### 3.1 为什么"版本锁定"是收敛保障的第一步

在实践中，相当比例的"训练不收敛"或"训练中途崩溃"问题，根源是 **ms-swift、transformers、vLLM、torch、flash-attn 之间的版本不匹配**。GRPO 训练涉及的组件链条比普通 SFT 更长（多了 vLLM 推理引擎、可能还有 DeepSpeed/FSDP2 的版本适配），因此版本管理必须比 SFT 场景更严格。建议采用"锁定 + 记录"的方式管理环境，任何环境变更都应该先在小规模实验上验证不影响收敛，再推广到大规模训练。

### 3.2 推荐环境组合

```bash
# 基础环境：建议使用官方 PyTorch 容器或 conda 隔离环境，Python 3.10/3.11
conda create -n grpo-swift python=3.10 -y
conda activate grpo-swift

# 安装 ms-swift 4.4.0（指定版本号，避免自动拉取最新可能引入的不兼容变更）
pip install ms-swift==4.4.0

# 或从源码安装以便随时 patch（推荐用于生产环境，便于自定义 reward plugin）
# git clone https://github.com/modelscope/ms-swift.git
# cd ms-swift && git checkout v4.4.0 && pip install -e .

# transformers：跟随 ms-swift 4.4.0 要求的区间安装，不要盲目升级到最新版
pip install "transformers>=4.51,<4.56" -U

# torch：FSDP2 依赖较新的 PyTorch（建议 2.4+，官方更推荐 2.5/2.6，DTensor 与 FSDP2 API 在 2.4~2.6 之间有持续增强）
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# vLLM：用于 GRPO 的 rollout 加速，版本需要和 transformers/torch 匹配，建议 0.7.x～0.8.x 区间，
# 具体以 ms-swift 4.4.0 release note 标注的兼容版本为准
pip install vllm==0.7.3

# flash-attention：训练与 vLLM 推理都强烈建议开启，大幅降低长序列显存占用
pip install flash-attn --no-build-isolation

# 其他常用组件
pip install deepspeed          # 即使主用 FSDP2，也建议装上，便于灵活切换/对比实验
pip install liger-kernel       # 融合算子，节省显存、加速训练，Qwen3 系列支持较好
pip install wandb              # 训练可视化推荐，也可用 swanlab / tensorboard
pip install math_verify        # 数学答案校验常用库，用于 GSM8K 等奖励函数实现
```

### 3.3 环境自检脚本

在正式训练前，建议先运行以下自检，确认所有组件版本、CUDA 可见性、NCCL 通信正常：

```bash
python - << 'PYEOF'
import torch, transformers, vllm
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("gpu count:", torch.cuda.device_count())
print("bf16 support:", torch.cuda.is_bf16_supported())
import torch.distributed as dist
print("nccl available:", dist.is_nccl_available())
PYEOF

# 检查 ms-swift CLI 是否正常
swift --version

# 简单的 NCCL 多卡通信自检（避免大规模训练时才发现网络配置问题）
python -m torch.distributed.run --nproc_per_node=8 -c "import torch;import torch.distributed as dist;dist.init_process_group('nccl');print(dist.get_rank(), 'ok');dist.destroy_process_group()"
```

### 3.4 版本变更管理建议

- 每次升级 ms-swift / transformers / vllm 版本后，先在 **Qwen3-4B + GSM8K 200条子集 + 单机 2 卡** 的最小配置上跑 50～100 step，对比升级前后的 reward 曲线、KL 曲线是否出现系统性偏移。如果出现明显偏移（如 reward 突然无法上升、KL 突增），应先怀疑版本兼容性问题，而不是急于调超参数。
- 建议在训练启动脚本中，将 `pip freeze` 的输出连同训练配置一起归档（写入 `output_dir/env.txt`），确保每次实验环境可追溯、可复现。

```bash
mkdir -p ${OUTPUT_DIR}
pip freeze > ${OUTPUT_DIR}/env.txt
nvidia-smi > ${OUTPUT_DIR}/gpu_info.txt
cp $0 ${OUTPUT_DIR}/launch_script_backup.sh
```

---

## 第四章 硬件规划：32B Dense 与 35B-A3B MoE 的资源画像差异

### 4.1 参数量与显存的基本估算

GRPO 训练同时涉及"策略模型训练"（需要参数 + 梯度 + 优化器状态）和"策略模型推理"（rollout，仅需要参数，但需要 KV Cache），二者的显存需求特性完全不同，必须分开估算。

**训练侧显存（以 bf16 混合精度、AdamW 优化器为例，Full 全参训练）**：

- 参数（bf16）：约 2 bytes/参数
- 梯度（bf16 或 fp32，取决于配置）：约 2～4 bytes/参数
- AdamW 一阶矩 + 二阶矩（通常 fp32）：约 8 bytes/参数
- 合计（不含 activation）：约 **12～16 bytes/参数**

对 Qwen3-32B（约 320 亿参数）估算：`32e9 * 14 bytes ≈ 448GB`，这远超单机 8×80GB=640GB 的裕量（尤其还要留给 activation、KV cache、vLLM rollout 引擎），因此 **Full 全参训练 32B 模型几乎必须使用 FSDP2/DeepSpeed ZeRO-3 做参数分片**，且大概率需要结合 CPU offload 或多机训练。

对 Qwen3.6-35B-A3B（MoE，约 350 亿总参数，约 30 亿激活参数）：**训练侧显存开销主要由"总参数量"决定**（因为所有专家的参数都需要梯度和优化器状态，即便前向时只激活了部分专家），而**计算量（FLOPs）由"激活参数量"决定**。这意味着：MoE 模型在同等总参数规模下，训练显存需求与 Dense 模型接近甚至更高（专家路由等额外结构），但训练速度（吞吐）会显著优于同等总参数量的 Dense 模型。这是 MoE 训练"省计算不省显存"的核心特征，在做资源规划时极易被低估。

LoRA 训练可以大幅降低上述开销：由于 base 模型参数被冻结（`requires_grad=False`），只有 LoRA adapter 参数（通常占总参数量的 0.1%～2%）需要梯度和优化器状态，因此：

- 参数（bf16，含冻结的 base + 少量 adapter）：约 2 bytes/参数（仅 base 部分）
- LoRA 梯度 + 优化器状态：可忽略不计（相对 base 而言）
- 合计：约 **2～3 bytes/参数**（如果 base 模型本身也做 FSDP2 分片，可进一步降低单卡占用）

因此 Qwen3-32B 用 LoRA 做 GRPO，理论上单机 8×80GB 甚至 4×80GB 就有可能跑起来（需结合 rollout 引擎显存一起规划）；而 Full 全参 GRPO 训练 32B/35B 级别模型，建议规划**至少 2 机 16 卡 80GB 起步**，具体见下表。

### 4.2 资源规划建议表（经验值，需结合实际序列长度、num_generations 调整）

| 场景 | 模型 | 训练方式 | 建议最小配置 | FSDP2 分片策略 | 备注 |
|---|---|---|---|---|---|
| 验证/调试 | Qwen3-4B/8B | LoRA | 1×80GB | 可选 fsdp2 或不用 | 用于跑通 reward/pipeline |
| 生产 | Qwen3-32B | LoRA | 单机 8×80GB | fsdp2 全分片（FULL_SHARD） | vLLM colocate 需预留显存 |
| 生产 | Qwen3-32B | Full | 2机16×80GB 起 | fsdp2 全分片 + 可选 CPU offload | 建议 server 模式 vLLM 独立部署 |
| 生产 | Qwen3.6-35B-A3B | LoRA | 单机 8×80GB（紧张）或 8×96GB/8×141GB | fsdp2 全分片，专家层需正确识别 | 注意 MoE 层 wrap policy |
| 生产 | Qwen3.6-35B-A3B | Full | 2～4机 16～32×80GB 起 | fsdp2 全分片 + CPU offload | 强烈建议 EP（专家并行）路线，见第10章 |

> 上表为经验性资源下限，实际需求随 `max_completion_length`、`num_generations`、`per_device_train_batch_size` 增大而显著上升，务必先按第7章检查清单做小规模压测，再决定最终资源申请规模。

### 4.3 计算 GPU 数量时应包含的"隐藏消耗者"

规划集群规模时，容易只计算"训练模型"的显存，而遗漏：

1. **vLLM rollout 引擎的显存**（colocate 模式下与训练共享 GPU，需要用 `--vllm_gpu_memory_utilization` 显式限制其占比，如 0.3～0.5）；
2. **参考模型（ref model）的显存**（Full 训练时尤其显著，可通过 `--offload_model true` 或 LoRA 复用机制规避）；
3. **权重同步过程中的临时 all-gather 显存尖峰**（FSDP2 场景下，把分片参数聚合为完整参数发送给 vLLM 时会有短暂的显存翻倍风险，需要预留 buffer）；
4. **长 completion 情况下的 KV Cache 显存**（`max_completion_length` 越大，rollout 阶段 KV Cache 占用越高，GSM8K 虽然答案不长，但如果模型是"思考模型"（thinking model），中间推理链可能很长，需要按思维链长度而非最终答案长度规划）。


## 第五章 FSDP2 分片技术详解与 ms-swift 中的配置方法

### 5.1 FSDP2 与 FSDP1、DeepSpeed ZeRO 的关键差异

FSDP2（Fully Sharded Data Parallel v2）是 PyTorch 官方在 2.4+ 版本推出的新一代原生分布式训练方案，相较于 FSDP1，核心变化：

- **底层数据结构从 `FlatParameter` 改为 `DTensor`（分布式张量）**：每个参数被表示为一个按维度切分的 DTensor，语义更清晰，与 `torch.compile`、张量并行（TP）等技术组合更自然，也更容易调试（可以直接查看某个参数分片的 shape/device_mesh，而不是 FSDP1 中被打平拼接的 flat buffer）。
- **通信调度更细粒度**：FSDP2 以"每个 module"为粒度做 all-gather/reduce-scatter，相比 FSDP1 的整体 flat buffer 调度，更容易与 activation checkpointing、CPU offload 组合，也更容易针对特定模块（如 MoE 的专家层）定制分片策略。
- **与 DeepSpeed ZeRO-3 的关系**：二者思想相似（都是参数/梯度/优化器状态分片 + 需要时 all-gather），FSDP2 是 PyTorch 原生实现，DeepSpeed ZeRO 是第三方库实现。ms-swift 同时支持两者，**不建议同时开启**（`--fsdp` 与 `--deepspeed` 二选一），选择建议见 5.4 节。

### 5.2 ms-swift 中启用 FSDP2 的方式

ms-swift 4.4.0 通过 `--fsdp` 参数启用 FSDP2 系列的分布式训练：

```bash
--fsdp fsdp2
```

该参数支持传入：
1. 字符串 `fsdp2`：使用 ms-swift 内置的默认 FSDP2 配置（对 Qwen 系列模型的 decoder layer 做自动 wrap，已经过官方基础验证，适合大多数 Dense 模型场景）。
2. 自定义 FSDP 配置文件路径：适用于需要精细控制 wrap policy（比如 MoE 模型需要单独处理专家层）、mixed precision policy、CPU offload 策略等高级场景，见 5.3 节。

结合 `torchrun`/`swift` CLI 的标准多卡启动方式：

```bash
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-32B \
    --train_type lora \
    --fsdp fsdp2 \
    ... # 其余 GRPO 参数见第 8 章完整脚本
```

**注意**：使用 `--fsdp fsdp2` 时，不要同时传 `--deepspeed`；同时因为 FSDP2 依赖 PyTorch 原生分布式通信组，请确保 `NPROC_PER_NODE`、`NNODES`、`MASTER_ADDR`、`MASTER_PORT` 等环境变量正确设置（多机场景见第 15 章）。

### 5.3 自定义 FSDP2 配置文件（精细控制，推荐生产环境使用）

对于 32B/35B 级别模型，强烈建议不要完全依赖默认配置，而是显式编写 FSDP2 配置 yaml/json，明确以下几个关键决策点，这些决策点直接关系到显存是否够用、以及数值稳定性（进而关系到收敛）：

```yaml
# fsdp2_config.yaml（示例，实际字段名请以你所用 ms-swift 4.4.0 版本的 FSDP 配置 schema 为准）
fsdp_version: 2
sharding_strategy: FULL_SHARD          # 全参数/梯度/优化器状态分片，显存最省；
                                        # 如显存充裕可用 SHARD_GRAD_OP（只分片梯度和优化器状态，参数不分片，通信更少但显存更高）
auto_wrap_policy: transformer_based    # 按 transformer decoder layer 自动 wrap，MoE 模型需要额外指定专家层 wrap（见第10章）
transformer_layer_cls:
  - Qwen3DecoderLayer                  # 需与所用模型的具体 decoder layer 类名一致
backward_prefetch: BACKWARD_PRE        # 提前预取下一层参数，降低通信-计算气泡，提升吞吐
forward_prefetch: true
mixed_precision:
  param_dtype: bfloat16
  reduce_dtype: float32                # 梯度 reduce 使用 fp32 累积，显著提升大模型训练数值稳定性，
                                        # 是保证收敛的重要一环，不建议为了省显存而设为 bfloat16
  buffer_dtype: bfloat16
cpu_offload: false                     # 是否将参数/梯度 offload 到 CPU；32B LoRA 一般不需要，
                                        # 32B/35B Full 全参在显存紧张时可设 true，但会显著降低吞吐，需配合梯度累积使用
activation_checkpointing: true         # 强烈建议开启，大幅降低 activation 显存，代价是约 20~30% 的额外前向计算
state_dict_type: SHARDED_STATE_DICT    # checkpoint 保存方式，见第16章
```

在 `swift rlhf` 命令中通过路径引用：

```bash
--fsdp /path/to/fsdp2_config.yaml
```

### 5.4 FSDP2 vs DeepSpeed ZeRO-3：GRPO 场景下的选择建议

| 维度 | FSDP2 | DeepSpeed ZeRO-3 |
|---|---|---|
| 生态成熟度 | PyTorch 原生，较新，个别模型结构（尤其 MoE）wrap 策略需要手工验证 | 社区验证案例多，尤其在 ms-swift 早期 GRPO 最佳实践文档中大量使用 |
| 与 vLLM 权重同步 | 需要通过 DTensor 的 `full_tensor()`/`state_dict` API 聚合，ms-swift 已封装，但排查问题时需要理解 DTensor 语义 | ms-swift 中对 ZeRO-3 的权重同步路径打磨更久，一般更稳定 |
| CPU Offload 精细度 | 可按 module 粒度控制 | 参数/优化器 offload 是 DeepSpeed 的强项，配置项更丰富（如 `offload_optimizer`、`offload_param` 分开配置） |
| torch.compile 兼容性 | 更好（DTensor 设计上更兼容 compile） | 较弱 |
| 推荐场景 | 单一训练框架内追求原生 PyTorch 生态一致性、需要与 TP 等高级并行策略组合的场景 | 追求"已被最广泛验证过的 GRPO 稳定路径"、快速上生产的场景 |

**给"保证收敛"的建议**：如果你的团队此前没有 FSDP2 + GRPO 的成熟经验，建议先用**较小规模（如 Qwen3-8B）在 FSDP2 和 DeepSpeed ZeRO-3 两条路径上分别跑一遍同样的 GSM8K GRPO 配置**，对比 reward 曲线、训练速度、显存占用，确认 FSDP2 路径在你的环境（模型结构、PyTorch/CUDA 版本组合）下数值行为与 DeepSpeed 一致后，再迁移到 32B/35B 生产训练。这是本手册反复强调的"小规模验证，大规模复现"方法论在分布式配置层面的具体应用。

### 5.5 FSDP2 下常见的收敛相关陷阱

1. **`reduce_dtype` 误设为 bfloat16**：梯度 reduce（跨卡梯度平均）如果用 bf16 累积，在 32B/35B 大模型、大 batch 场景下会引入不可忽视的数值误差，表现为训练后期 loss/reward 出现小幅震荡或缓慢发散。**务必将 `reduce_dtype` 设为 `float32`**。
2. **auto_wrap_policy 粒度过粗或过细**：如果整个模型被 wrap 成一个 FSDP unit（粒度过粗），起不到分片省显存的效果；如果 wrap 粒度过细（比如按 Linear 层逐层 wrap），通信次数暴增，训练速度大幅下降，且容易在 MoE 专家层的动态路由场景下引发额外的同步开销甚至死锁。建议按 decoder layer（`Qwen3DecoderLayer` 等）为单位 wrap，MoE 模型需要单独考虑专家层的 wrap 粒度（见第10章 10.3 节）。
3. **activation checkpointing 与 packing/变长序列的交互问题**：开启 `--packing true` 做序列拼接训练时，需确认 activation checkpointing 的重计算边界与 packing 边界一致，否则可能导致某些拼接样本的梯度计算错误。GRPO 场景由于每个 completion 长度不同，这个问题比 SFT 更容易触发，建议先关闭 packing 跑通，确认收敛后再谨慎开启。
4. **权重同步时的精度不一致**：训练侧用 bf16 存储参数，同步给 vLLM 推理引擎时如果类型转换不当（例如 fp32 中间态转换出错），会导致 rollout 阶段模型行为与训练侧模型出现细微但系统性的偏差，长期累积后会出现"reward 涨到一定程度就卡住不动"的现象。建议训练初期打开 `--log_completions true`，人工抽查 rollout 生成内容是否符合预期。


## 第六章 GSM8K 数据集准备、格式化与奖励函数设计

### 6.1 GSM8K 数据集结构回顾

GSM8K（Grade School Math 8K）是一个包含约 7473 条训练样本、1319 条测试样本的小学数学应用题数据集，每条样本包含：

- `question`：题目文本（英文）
- `answer`：包含完整推理过程（用 `<<...>>` 标注计算步骤）和最终数值答案（用 `####` 分隔符标注，如 `#### 18`）

原始格式示例：

```
question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
```

在 GRPO 训练中，我们通常**只需要 `question` 作为 prompt，`#### ` 后的数字作为 ground truth 用于奖励函数计算**，中间的推理过程不作为监督信号（这正是 RL 相比 SFT 的核心差异——不要求模型模仿某个特定的推理路径，而是通过奖励引导模型自己探索出正确路径）。

ms-swift 内置支持通过 `--dataset AI-MO/NuminaMath-TIR` 等方式直接引用 ModelScope/HuggingFace Hub 上的数据集；对于 GSM8K，可以直接使用：

```bash
--dataset 'modelscope/gsm8k'
# 或
--dataset 'openai/gsm8k'
```

也可以自行下载后转换为本地 jsonl，格式化为 ms-swift 期望的对话格式：

```json
{"messages": [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"}], "solution": "72"}
```

其中 `solution` 字段（字段名可通过 reward plugin 自定义读取）保存最终数值答案，供奖励函数在训练时读取比对，**不会**作为模型的训练目标（GRPO 不使用 `solution` 做 teacher forcing）。

### 6.2 GSM8K 场景下的数据预处理建议

1. **划分训练/验证集**：即使 GRPO 是在线强化学习，也建议保留一小部分（如 200～500 条）GSM8K 训练数据或全部测试集作为训练期间的验证集，通过 `--split_dataset_ratio` 或显式指定 `--val_dataset` 实现，用于监控"训练集 reward 提升是否能泛化到验证集"，避免过拟合到训练集的表面模式。
2. **过滤过长/过短样本**：GSM8K 本身题目都不长，一般不需要复杂过滤，但如果混入了其他数据源（多任务 GRPO 训练），务必对 prompt 长度做统一截断/过滤，防止个别超长样本拖慢整个 batch 甚至导致 OOM。
3. **系统提示词（system prompt）的一致性**：GRPO 训练效果对 prompt 格式高度敏感。建议固定一个清晰的系统提示词，明确要求模型按固定格式输出（例如"请在最后用 `#### <数字>` 的格式给出最终答案"），并且**这个格式约定必须与奖励函数中的答案抽取逻辑完全对应**，这是最容易被忽视但极其关键的一致性要求。

```python
SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "and provide the final numeric answer at the end in the exact format:\n"
    "#### <answer>\n"
    "Do not include units or extra text after the final answer."
)
```

4. **是否开启 thinking 模式**：Qwen3 系列是"混合思考"模型，训练时可通过 `--enable_thinking` 控制是否在 completion 中包含 `<think>...</think>` 推理链。GSM8K 场景下开启 thinking 通常有助于模型探索更长的推理路径、提升最终准确率，但会显著增加 `max_completion_length` 需求和 rollout 显存/时间开销。建议先在小规模上分别测试开启/关闭 thinking 的收敛速度和最终效果，再决定生产配置。

### 6.3 奖励函数设计：GSM8K 场景的最佳实践

ms-swift 内置了 `accuracy`（正确性）等通用奖励函数，但生产环境强烈建议针对 GSM8K 场景**自定义奖励函数插件**（plugin），组合以下几类信号，这是保证 GRPO 收敛的核心工程环节之一：

#### 6.3.1 正确性奖励（Accuracy Reward）——核心信号

从 completion 中抽取最终数值答案，与 ground truth 精确比对（考虑浮点误差、单位、逗号分隔符等归一化处理），正确给 1 分，错误给 0 分（或 -1，视是否需要负奖励而定）。

```python
import re

def extract_answer(text: str):
    # 优先匹配约定的 "#### <answer>" 格式
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    # 兜底：匹配最后出现的数字
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None

def accuracy_reward(completions, solutions, **kwargs):
    rewards = []
    for completion, solution in zip(completions, solutions):
        pred = extract_answer(completion)
        try:
            correct = pred is not None and abs(float(pred) - float(solution)) < 1e-4
        except ValueError:
            correct = False
        rewards.append(1.0 if correct else 0.0)
    return rewards
```

#### 6.3.2 格式奖励（Format Reward）——辅助信号，权重需谨慎控制

用于约束模型严格遵守输出格式（如是否包含 `#### ` 标记、是否包含合法的 `<think>...</think>` 结构），一般给较小的权重（如 0.1～0.2），**核心原则：格式奖励绝不能大到足以让模型"为了拿格式分而放弃思考正确性"**，即避免"reward hacking"。

```python
def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        has_marker = "####" in completion
        rewards.append(0.1 if has_marker else 0.0)
    return rewards
```

#### 6.3.3 长度惩罚 / overlong 惩罚——防止 completion 无限增长

GRPO 训练中一个常见的退化模式是模型逐渐学会生成越来越长的推理链以"碰运气"提高正确率，导致训练/推理成本失控，且超出 `max_completion_length` 的回答会被截断，截断的回答往往拿不到正确性奖励，从而产生噪声梯度。ms-swift 提供 `--overlong_filter true` 参数，将超长（被截断）的样本从优势计算中排除，避免因截断导致的错误负反馈污染训练信号，这是官方最佳实践脚本中反复出现的参数，**GSM8K 训练建议默认开启**。

```bash
--overlong_filter true
```

也可以在自定义奖励函数中加入长度惩罚项（可选，非必须，需要谨慎调参避免压制模型必要的思考长度）：

```python
def length_penalty(completions, max_len=1024, **kwargs):
    rewards = []
    for completion in completions:
        token_len = len(completion.split())  # 简化示例，生产中应使用真实 tokenizer 计数
        penalty = -0.001 * max(0, token_len - max_len)
        rewards.append(penalty)
    return rewards
```

#### 6.3.4 多奖励组合与权重配置

ms-swift 支持通过 `--reward_funcs` 传入多个奖励函数名（对应 plugin 中注册的函数），并通过 `--reward_weights` 指定各自权重：

```bash
--reward_funcs accuracy format \
--reward_weights 1.0 0.1
```

**权重设计原则（GSM8K 场景经验）**：
- 正确性奖励应占绝对主导地位（权重 ≥ 0.8～1.0），格式类辅助奖励权重不超过 0.1～0.2，否则容易诱发模型专注刷格式分而放弃真实推理能力提升，训练曲线表现为 reward 总分上升但 accuracy（正确率）不涨甚至下降——**这是 GRPO 训练中最典型的"看起来在收敛，实际没在学正确的东西"的陷阱，务必对 accuracy 和 format 两个子奖励分别打点监控，而不只看加权总和**。

### 6.4 奖励函数的鲁棒性测试（上线前必做）

在正式大规模训练前，建议对奖励函数做单元测试和"对抗样本"测试：

1. 构造几十条"人工已知答案对错"的 completion 样本，跑一遍奖励函数，确认打分符合预期。
2. 构造边界 case：答案带单位（如"72 clips"而非"72"）、答案带逗号分隔符（"1,200"）、模型输出多个数字（防止正则误抽取中间计算过程中的数字而非最终答案）、答案格式与约定不符（模型没有输出 `####` 标记）等，逐一验证奖励函数的鲁棒性。
3. 在真实小规模 rollout（如用未训练的 base 模型对 50 条 GSM8K 生成 completion）上跑一遍奖励函数，人工抽查 20～30 条打分结果，确认没有系统性误判。

这一步骤看似繁琐，但根据社区大量实践反馈，**奖励函数抽取逻辑的 bug 是导致 GRPO"训练很久 reward 就是不涨"最常见的根因之一**，值得在正式训练前投入充分时间验证。


## 第七章 训练前的"收敛保障"检查清单（Convergence Readiness Checklist）

在启动 32B/35B 规模的正式 GRPO 训练前，强烈建议逐项核对以下清单。清单按"如果这一项错了，会导致什么后果"组织，方便你理解每一项检查的必要性，而不是机械打勾。

### 7.1 数据与奖励函数

- [ ] GSM8K 数据的 prompt 格式（system prompt、few-shot 与否、是否包含格式约定说明）与奖励函数的答案抽取逻辑**完全对应**。→ 否则会出现"模型答对了但奖励函数没识别出来"，训练信号系统性偏低。
- [ ] 奖励函数已完成 6.4 节的单元测试和边界 case 验证。→ 否则可能出现"奖励黑客"或"奖励恒为 0/恒为满分"导致 advantage 全为 0、梯度消失。
- [ ] 正确性奖励权重占主导，格式类奖励权重受控（建议 ≤ 0.2）。→ 否则容易训偏。
- [ ] 训练/验证集已正确划分，验证集不参与 rollout 训练。→ 否则无法及时发现过拟合。

### 7.2 模板与生成配置

- [ ] 训练模板（chat template）与推理（vLLM rollout）使用的模板完全一致，`enable_thinking` 设置在训练和推理阶段一致。→ 模板不一致会导致训练看到的 token 序列和推理阶段生成的 token 序列存在系统性差异，是最隐蔽的一类 bug。
- [ ] `--temperature`、`--top_p` 等采样参数设置合理（GSM8K 场景经验值 `temperature=0.8~1.0`，过低会导致组内样本同质化、advantage 方差趋近于 0，过高会导致生成质量下降、正确率整体偏低）。
- [ ] `--num_generations`（组大小 G）设置在 8～16 之间（GSM8K 这类相对简单的任务，G=8 通常已经足够；更难的任务可以适当增大）。→ G 过小会导致 advantage 估计噪声大，训练不稳定。
- [ ] `--max_completion_length` 结合是否开启 thinking 设置合理余量（非 thinking 模式 512～1024 通常足够，thinking 模式建议 2048～4096），并开启 `--overlong_filter true`。

### 7.3 FSDP2 与分布式配置

- [ ] 已在小模型（Qwen3-4B/8B）上验证过当前 FSDP2 配置（wrap policy、mixed precision policy）跑通且指标健康。
- [ ] `reduce_dtype` 设置为 `float32`（而非 bfloat16）。
- [ ] `--fsdp` 与 `--deepspeed` 未同时设置。
- [ ] 多机场景下 `MASTER_ADDR`/`MASTER_PORT`/`NNODES`/`NODE_RANK` 已正确配置并做过 NCCL 通信自检（见第15章）。
- [ ] Full 全参训练场景下，已确认参考模型（ref model）的显存/精度处理方式（offload 或独立分片）不会引入数值误差。

### 7.4 学习率与关键超参数

- [ ] 学习率量级符合训练方式：LoRA 场景经验值 `1e-5~2e-4`（视 lora_rank 而定），Full 全参场景经验值 `1e-6~5e-6`（比 LoRA 低 1～2 个数量级，这是新手最常见的踩坑点之一）。
- [ ] `--beta`（KL 惩罚系数）设置合理，经验起点 `0.001~0.04`，过大会限制策略探索导致 reward 涨不动，过小会导致策略过快偏离参考模型、语言能力退化甚至输出乱码。
- [ ] `warmup_ratio` 已设置（建议 0.03～0.1），避免训练初期学习率过高导致的早期发散。
- [ ] 梯度裁剪（`--max_grad_norm`，默认通常为 1.0）已确认生效。

### 7.5 监控与可观测性

- [ ] 已接入 wandb / swanlab / tensorboard 等可视化工具（`--report_to wandb`），并确认关键指标（reward 均值/方差、KL、entropy、completion length、clip fraction）都已正确上报。
- [ ] `--log_completions true` 已开启，训练初期能够人工抽查生成内容。
- [ ] 已设置合理的 `--eval_steps`、`--save_steps`，确保出问题时能及时发现并回滚到健康 checkpoint。

### 7.6 小规模冒烟测试（Smoke Test）——正式训练前的最后一道关卡

**在投入 32B/35B 全量资源之前，务必用以下最小配置跑一次"冒烟测试"**，确认整条 pipeline（数据→模板→rollout→奖励→FSDP2 训练→权重同步）没有工程 bug：

```bash
# 冒烟测试：Qwen3-4B + GSM8K 200条 + 单机2卡 + 跑30~50 step
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-4B \
    --train_type lora \
    --fsdp fsdp2 \
    --dataset 'modelscope/gsm8k#200' \
    --max_steps 50 \
    --num_generations 8 \
    --use_vllm true \
    --vllm_mode colocate \
    --reward_funcs accuracy format \
    --reward_weights 1.0 0.1 \
    --log_completions true \
    --report_to wandb \
    --output_dir output/smoke_test
```

冒烟测试通过标准：
1. 训练无报错、无 OOM，30～50 step 能正常跑完；
2. reward 均值有明显上升趋势（哪怕幅度不大）；
3. KL 散度保持在合理范围内（不是单调爆炸增长）；
4. 抽查 `log_completions` 输出的生成内容格式正确、没有乱码或重复token（repetition）现象；
5. checkpoint 能正常保存和加载。

只有冒烟测试全部通过后，才应该将配置等比放大到 32B/35B + FSDP2 + 多机的正式训练规模。**这是保证收敛最重要、也是最容易被工程师因赶进度而跳过的一步——跳过冒烟测试直接上大规模训练，一旦出问题，排查成本会是冒烟测试的几十倍。**


## 第八章 Qwen3-32B GRPO LoRA 训练完整方案

### 8.1 方案概览

LoRA 训练是 32B 规模 GRPO 训练的推荐起点：显存开销小、训练速度快、迭代周期短，适合快速验证奖励函数设计、超参数选择是否合理，验证收敛后再考虑是否升级到 Full 全参训练（见第18章迁移路径）。

### 8.2 单机 8×80GB 完整训练脚本（colocate vLLM 模式）

```bash
#!/bin/bash
# qwen3_32b_grpo_lora_fsdp2.sh
export WANDB_PROJECT=grpo-qwen3-32b-gsm8k
export WANDB_RUN_NAME=qwen3-32b-lora-fsdp2-v1
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-32B \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --fsdp fsdp2 \
    --dataset 'modelscope/gsm8k' \
    --split_dataset_ratio 0.02 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --max_length 1024 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --temperature 0.9 \
    --top_p 0.9 \
    --beta 0.01 \
    --epsilon 0.2 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_max_model_len 4096 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --gc_collect_after_offload true \
    --reward_funcs accuracy format \
    --reward_weights 1.0 0.1 \
    --overlong_filter true \
    --gradient_checkpointing true \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --log_completions true \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --output_dir output/qwen3-32b-grpo-lora
```

### 8.3 关键参数说明（LoRA 场景专属）

- `--lora_rank 16 --lora_alpha 32`：GRPO 场景 LoRA rank 建议略高于普通 SFT（SFT 常用 8），因为 RL 信号相对稀疏，更大的 rank 有助于模型有足够容量学习策略调整。经验区间 `rank=8~32`，`alpha` 通常取 `2×rank`。
- `--target_modules all-linear`：覆盖所有线性层（含 attention 的 q/k/v/o 和 MLP 的 gate/up/down），GRPO 场景下比只训练 attention 部分收敛更稳定，因为数学推理任务对 MLP 层的知识调整同样敏感。
- `--offload_model true --offload_optimizer true`：LoRA 训练时 base 模型参数本身不需要梯度，训练阶段和 rollout（vLLM）阶段交替使用同一批 GPU（colocate 模式），通过 sleep_level 机制在两阶段间切换显存占用，offload 进一步降低训练阶段的显存峰值，为 vLLM 腾出空间。
- `--sleep_level 1`：vLLM colocate 模式下，训练阶段让 vLLM "休眠"释放显存给训练侧使用，rollout 阶段再唤醒，是 GRPO 单机训练的核心显存复用机制。

### 8.4 显存不足时的调整优先级

如果 8×80GB 仍然 OOM，按以下优先级调整（越靠前对训练速度影响越小）：

1. 降低 `--vllm_gpu_memory_utilization`（如 0.4→0.3），为训练侧腾出更多显存；
2. 降低 `--per_device_train_batch_size`，用 `--gradient_accumulation_steps` 补偿有效 batch size；
3. 降低 `--num_generations`（如 8→4），注意这会增大 advantage 估计噪声，需要相应更谨慎地观察训练稳定性；
4. 降低 `--max_completion_length` / `--vllm_max_model_len`；
5. 确认 `--gradient_checkpointing true` 已开启；
6. 最后考虑升级到 2 机 16 卡配置，而不是继续压缩上述参数导致训练质量下降。

## 第九章 Qwen3-32B GRPO Full 全参训练完整方案

### 9.1 与 LoRA 方案的核心差异

Full 全参训练在收敛保障上比 LoRA 需要额外关注：

1. **学习率必须显著更低**（经验区间 `1e-6~5e-6`，比 LoRA 低 1～2 个数量级），因为全参数更新对参数空间的扰动远大于 LoRA 低秩更新，过高的学习率极易导致语言能力快速退化（表现为生成内容重复、乱码，reward 断崖式下跌）。
2. **必须显式管理参考模型（ref model）**，全参训练时 policy 和 ref 是两份独立的完整模型权重，显存开销翻倍，通常需要 offload ref model 到 CPU 或使用 `--ref_model` 指定独立加载路径并配合分片。
3. **资源需求陡增**，建议至少 2 机 16×80GB 起步（见第4章资源规划表），且大概率需要 CPU offload 配合。
4. **训练稳定性风险更高**，需要更严格的梯度裁剪、更保守的 `--beta`（KL 惩罚）设置，防止策略过快漂移。

### 9.2 两机 16×80GB 完整训练脚本（server 模式 vLLM，推荐多机场景）

多机场景推荐使用 vLLM **server 模式**而非 colocate 模式：将 vLLM 部署为独立服务（可以独占若干张卡，或部署在独立节点），训练进程通过网络请求 rollout，训练和推理解耦，避免了 colocate 模式下"训练/推理串行切换"的效率损失，也简化了多机 FSDP2 训练与 vLLM 生命周期管理的耦合。

**第一步：单独启动 vLLM rollout server（可部署在专用节点，或训练节点预留的部分 GPU 上）**

```bash
# 在独立的 rollout 节点/GPU 上启动 vLLM server
CUDA_VISIBLE_DEVICES=6,7 \
swift rollout \
    --model Qwen/Qwen3-32B \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.85 \
    --port 8000
```

**第二步：在训练节点启动 FSDP2 全参 GRPO 训练，指向 vLLM server**

```bash
#!/bin/bash
# qwen3_32b_grpo_full_fsdp2_node0.sh  (在 node0 执行)
export WANDB_PROJECT=grpo-qwen3-32b-gsm8k
export WANDB_RUN_NAME=qwen3-32b-full-fsdp2-v1
export NNODES=2
export NODE_RANK=0
export NPROC_PER_NODE=6            # 假设每机8卡，其中2卡给rollout server，6卡用于训练
export MASTER_ADDR=<node0_ip>
export MASTER_PORT=29500

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-32B \
    --train_type full \
    --fsdp fsdp2_config.yaml \
    --dataset 'modelscope/gsm8k' \
    --split_dataset_ratio 0.02 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-6 \
    --warmup_ratio 0.1 \
    --max_length 1024 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --temperature 0.9 \
    --top_p 0.9 \
    --beta 0.005 \
    --epsilon 0.2 \
    --max_grad_norm 0.5 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host <rollout_server_ip> \
    --vllm_server_port 8000 \
    --reward_funcs accuracy format \
    --reward_weights 1.0 0.1 \
    --overlong_filter true \
    --gradient_checkpointing true \
    --offload_optimizer true \
    --eval_steps 30 \
    --save_steps 30 \
    --save_total_limit 3 \
    --save_only_model false \
    --logging_steps 2 \
    --log_completions true \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --output_dir output/qwen3-32b-grpo-full

# node1 执行相同脚本，仅需修改 NODE_RANK=1
```

### 9.3 Full 全参训练的关键收敛保障要点

1. **学习率务必从保守值起步再逐步尝试提高**：建议第一次正式训练直接使用 `2e-6` 起步而非贸然尝试更高学习率，先确认训练稳定，再在后续实验中小幅上调（如 `2e-6→3e-6→5e-6`）对比 reward 上升速度，切勿一次性设置过高学习率。
2. **`--max_grad_norm` 建议比默认 1.0 更保守（如 0.5）**：全参训练梯度分布更容易出现离群值（尤其训练初期），更严格的梯度裁剪有助于防止早期发散。
3. **`--beta`（KL 惩罚）建议比 LoRA 场景更小但不为 0**：全参训练由于更新自由度更高，理论上不需要像 LoRA 那样强的 KL 约束（LoRA 本身低秩更新已经是一种隐式正则化），但完全去掉 KL 惩罚（`beta=0`）在 32B 全参场景下风险很高，容易出现语言能力断崖式退化，建议保留一个较小但非零的值（如 `0.005~0.01`）。
4. **`--save_steps` 应设置得比 LoRA 场景更密集**：全参训练一旦出现发散，往往需要回滚到更早的 checkpoint，更密集的保存频率（配合 `--save_total_limit` 控制磁盘占用）能显著降低"训练崩溃后损失大量已完成训练进度"的风险。
5. **监控 reference model 的显存/精度处理**：确认 ref model 是否被正确 offload、是否用与 policy 模型一致的精度做 KL 计算，这部分如果处理不当，会导致 KL 散度估计本身就是错的，进而误导整个训练的稳定性判断。


## 第十章 Qwen3.6-35B-A3B（MoE）GRPO 训练特殊性与 FSDP2 适配

### 10.1 MoE 架构给 GRPO 训练带来的额外复杂度

Qwen3.6-35B-A3B（35B 总参数、约 3B 激活参数的 MoE 架构，结构上可类比 Qwen3-30B-A3B）相比 Qwen3-32B Dense 模型，在 GRPO 训练中有以下几个本质区别，这些区别直接影响 FSDP2 配置和收敛策略：

1. **路由（Router）的随机性/敏感性**：MoE 模型每个 token 经过一个门控网络（router）决定激活哪些专家（expert），训练过程中路由的选择本身也在随参数更新变化。GRPO 的策略梯度更新如果幅度过大，容易导致路由分布剧烈震荡（同一类输入在相邻训练步之间被路由到完全不同的专家），这是 MoE 模型比 Dense 模型更容易出现"训练不稳定"的根本原因之一。
2. **负载均衡损失（load balancing / auxiliary loss）**：为防止路由退化为"只用少数几个专家"（专家利用不均衡，导致其余专家训练不足、整体容量浪费），MoE 模型训练通常需要引入辅助损失项（`aux_loss_coeff`，常见经验值 `0.001~0.01`）。在 GRPO 场景下，这个辅助损失需要和策略梯度损失协同，权重设置不当会干扰 GRPO 本身的优化目标，需要重点关注。
3. **FSDP2 wrap policy 需要单独处理专家层**：MoE 层（通常包含多个专家的 FFN 子模块 + 路由器）在参数规模上远大于普通 attention/FFN 层，如果按照 Dense 模型的默认 wrap policy（仅按 decoder layer 整体 wrap）处理，会导致单个 FSDP unit 内专家参数量过大，显存分片效果不佳，甚至可能在 all-gather 阶段产生显存尖峰。
4. **专家并行（Expert Parallelism, EP）与 FSDP2 的组合**：ms-swift 的 Megatron-SWIFT 分支原生支持 EP（专家并行），可将不同专家分布到不同 GPU 上，大幅提升 MoE 模型训练效率（官方数据显示，MoE 模型用 Megatron 路线训练速度可提升近 10 倍）。但**如果你的场景要求必须使用 transformers 原生路线 + FSDP2（例如需要 GRPO 与 vLLM 深度耦合、Megatron-SWIFT 对 GRPO 的支持相对 SFT 更新较慢）**，则需要在纯 FSDP2 分片下处理 MoE 层，本章后续内容以此为主线。

### 10.2 权重同步（policy → vLLM）在 MoE 场景下的额外注意事项

FSDP2 + GRPO 的权重同步流程（训练侧参数聚合后同步给 vLLM 推理引擎）在 MoE 模型上尤其容易出问题：

- **专家参数的命名和 shape 必须与 vLLM MoE 实现严格对齐**：vLLM 对 MoE 模型有专门的融合 kernel（如 fused MoE），其内部对专家权重的存储布局（例如是否将多个专家的权重拼接成一个大矩阵）可能与 transformers/FSDP2 训练时的存储方式不同，ms-swift 内部已经封装了转换逻辑，但**升级 ms-swift/vLLM 版本时，务必重新跑一次第7章的冒烟测试**，确认权重同步后 rollout 生成内容仍然正常（而不是看起来能跑但生成内容是错乱的，这种 silent bug 很难通过报错发现，只能通过人工抽查 `log_completions` 发现）。
- **同步过程的显存尖峰在 MoE 场景下更严重**：由于 MoE 模型总参数量大，all-gather 聚合完整参数用于同步的瞬时显存开销比 Dense 模型更高，建议：
  - 优先使用**分层/分块同步**（如果 ms-swift 版本支持逐层同步而非一次性全量同步，检查 4.4.0 release note 中相关参数）；
  - 在权重同步阶段临时提高 `--offload_optimizer`、`--offload_model` 的调用时机，确保同步瞬间训练侧其他显存占用（优化器状态等）已被 offload，为聚合腾出空间。

### 10.3 MoE 专属 FSDP2 配置建议

```yaml
# fsdp2_config_moe.yaml —— 针对 Qwen3.6-35B-A3B 等 MoE 模型
fsdp_version: 2
sharding_strategy: FULL_SHARD
auto_wrap_policy: transformer_based
transformer_layer_cls:
  - Qwen3MoeDecoderLayer          # 具体类名以实际模型结构为准，一般能在 modeling_*.py 中找到
# 关键：对 MoE 层的专家子模块做更细粒度的 wrap，避免单个 FSDP unit 内专家参数量过大
moe_wrap_policy: true             # 若 ms-swift/transformers 版本支持该选项；
                                   # 若不支持，需要通过自定义 auto_wrap_policy（lambda 判断模块类型）手工实现，
                                   # 核心思路是把每个 expert 的 FFN 作为独立的 wrap 单元
mixed_precision:
  param_dtype: bfloat16
  reduce_dtype: float32
  buffer_dtype: bfloat16
cpu_offload: true                 # MoE 模型 Full 训练显存压力大，建议默认开启，用吞吐换稳定性
activation_checkpointing: true
state_dict_type: SHARDED_STATE_DICT
```

如果你的 ms-swift/transformers 版本对 MoE 层没有内置的细粒度 wrap 支持，可以通过自定义 `auto_wrap_policy` 函数实现（概念示例，具体 API 需对照你使用的 PyTorch 版本 FSDP2 文档）：

```python
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

def moe_aware_wrap_policy(module):
    # 对每个 expert 的 FFN 子模块单独 wrap，router 和其余部分跟随外层 decoder layer wrap
    return isinstance(module, (Qwen3MoeDecoderLayer, Qwen3MoeExpertFFN))

wrap_policy = ModuleWrapPolicy({Qwen3MoeDecoderLayer, Qwen3MoeExpertFFN})
```

### 10.4 MoE 场景的超参数调整建议

相比 Dense 模型的第11章通用超参数建议，MoE 模型需要额外关注：

- **学习率应比同等总参数量的 Dense 模型更保守**：由于路由的敏感性，MoE 模型对参数更新幅度更敏感，建议在第9章 Full 训练建议学习率基础上再下调 20%~50% 作为起点（如 Dense 32B Full 用 `2e-6`，35B-A3B MoE Full 可先尝试 `1e-6~1.5e-6`）。
- **辅助负载均衡损失系数**：如果训练框架暴露了 `aux_loss_coeff` 相关参数，建议保持较小值（`0.001~0.01`），并在训练过程中监控专家利用率分布（部分推理/训练框架支持输出每个专家的 token 分配比例），如果发现严重的专家利用不均衡（如某些专家几乎不被激活），适当上调该系数。
- **`--num_generations` 可考虑略微增大**：由于激活参数量小、推理速度相对 Dense 32B 更快，MoE 模型的 rollout 阶段吞吐更高，在显存允许的情况下可以尝试更大的组大小（如 12～16）以降低 advantage 估计噪声，这是 MoE 模型"计算换质量"的一个可利用特性。
- **更密集的 checkpoint 保存和更保守的 `--max_grad_norm`**：MoE 模型一旦训练发散（尤其路由坍塌），恢复成本比 Dense 模型更高（路由坍塌往往需要从更早的 checkpoint 重新开始，而不能像 Dense 模型那样简单调低学习率继续训练），因此建议 `save_steps` 设置得更密集，`max_grad_norm` 设置得更保守（如 0.3~0.5）。

### 10.5 MoE 训练路由健康度监控（收敛保障专属指标）

除了第12章通用的 GRPO 监控指标外，MoE 训练建议额外监控：

- **专家利用率分布（expert load distribution）**：理想情况下各专家的 token 分配比例应大致均衡（视具体路由 top-k 策略而定），如果监控到分布随训练严重右偏（少数专家占据绝大部分流量），是路由坍塌的早期信号，应及时介入（降低学习率、提高 aux_loss_coeff，或回滚 checkpoint）。
- **路由熵（router entropy）**：类似于第12章的 token 级熵监控，路由熵持续下降到接近 0 意味着路由决策变得非常确定（可能是好的收敛信号，也可能是路由坍塌的信号），需要结合专家利用率分布联合判断。

> 若你所用的 ms-swift 版本或推理/训练引擎暂未暴露专家利用率、路由熵这类细粒度 MoE 监控指标，建议在自定义 reward/logging plugin 中通过 hook 路由层的 forward 输出手动统计并上报到 wandb，这是保证大规模 MoE GRPO 训练可控性的重要工程投入，不应省略。


## 第十一章 关键超参数详解与调优方法论

### 11.1 学习率（`--learning_rate`）

学习率是影响 GRPO 收敛最敏感的超参数，没有之一。经验区间总结：

| 训练方式 | 模型规模 | 经验起点 | 可尝试上限 |
|---|---|---|---|
| LoRA | 8B 及以下 | 1e-4 | 3e-4 |
| LoRA | 32B/35B | 3e-5~5e-5 | 1e-4 |
| Full | 8B 及以下 | 1e-6~5e-6 | 1e-5 |
| Full | 32B Dense | 1e-6~3e-6 | 5e-6 |
| Full | 35B-A3B MoE | 1e-6~1.5e-6 | 3e-6 |

调优方法：**从表中经验起点开始，先跑通冒烟测试确认无发散迹象，再小步（如 1.5～2 倍）上调学习率对比 reward 上升速度，一旦观察到 KL 散度或 loss 出现异常波动（见第13章诊断表），立即回退到上一个稳定值。** 不建议一开始就用网格搜索大范围试探学习率，这在 32B/35B 规模下试错成本过高。

### 11.2 KL 惩罚系数（`--beta`）

`beta` 控制策略模型偏离参考模型的惩罚力度，是"探索能力"与"训练稳定性"之间的核心权衡旋钮：

- `beta` 过大：策略更新被强力约束，reward 提升缓慢甚至停滞，模型几乎不发生变化。
- `beta` 过小或为 0：策略可以自由偏离参考模型，短期 reward 可能上升更快，但风险显著增加——容易出现语言能力退化（生成不流畅、重复、乱码）、后期训练发散。

经验区间：LoRA 场景 `0.01~0.04`，Full 全参场景 `0.001~0.01`（全参训练由于更新自由度高，通常需要比 LoRA 更小的 beta 才能获得可比的探索空间，但同时风险也更高，需要用第16章的密集 checkpoint 策略兜底）。

**调优建议**：如果观察到 reward 上升但语言质量（人工抽查 `log_completions`）明显下降，应上调 `beta`；如果 reward 长期停滞不前、KL 散度也维持在很低水平（说明策略几乎没有更新），应下调 `beta` 或检查是不是学习率也同时设置得过低。

### 11.3 组大小（`--num_generations`）

组大小 `G` 决定了每个 prompt 采样多少个候选回答用于计算组内相对优势。GSM8K 场景经验：

- `G=4`：最低可用配置，advantage 估计噪声较大，仅建议在显存极度受限或快速调试场景使用。
- `G=8`：生产环境常用默认值，噪声与效率的平衡点。
- `G=16`：更难任务或对训练稳定性要求更高时使用，advantage 估计更准，但 rollout 计算量线性增加。

一个实用的诊断技巧：如果同一 batch 内某个 prompt 的 `G` 个回答**全部正确或全部错误**，该组的组内标准差为 0，advantage 计算会退化（`(r_i - mean)/std` 中分母趋近于 0，需要 `eps` 兜底，此时该组实际上不产生有效的训练信号）。GSM8K 训练初期，简单题目容易出现"全对"、难题目容易出现"全错"的情况，这属于正常现象，但**如果监控到绝大多数 batch 的组内方差都接近 0，说明当前数据难度与模型当前能力不匹配（要么模型已经太强导致简单题全对，要么模型太弱导致难题全错），需要考虑课程学习（curriculum learning）式的数据难度调整，或者增大 `G` 以提升在临界难度题目上采到差异化样本的概率**。

### 11.4 裁剪范围（`--epsilon` / `--epsilon_high`）

类比 PPO 的 clip 机制，限制新旧策略概率比 `ratio` 的变化范围，防止单步更新幅度过大。默认对称裁剪 `epsilon=0.2`（即 ratio 被限制在 `[0.8, 1.2]`）是常见起点。部分 GRPO 变体（如 DAPO）建议使用非对称裁剪（`epsilon_low` 更小、`epsilon_high` 更大），允许对"表现显著更好"的样本给予更大的更新幅度，同时对"表现更差"的样本保持保守裁剪，有助于加速正向学习信号的传播，同时抑制负向信号引入的不稳定性。ms-swift 支持通过 `--epsilon`、`--epsilon_high` 分别设置，GSM8K 场景可以尝试 `epsilon=0.2, epsilon_high=0.28` 一类的非对称配置作为改进实验。

### 11.5 采样温度与 top_p（`--temperature` / `--top_p`）

影响 rollout 阶段生成的多样性，直接决定组内样本的差异化程度：

- `temperature` 过低（如 <0.5）：生成内容趋同，组内样本高度相似，advantage 信号弱，训练效率低。
- `temperature` 过高（如 >1.2）：生成质量下降，正确率整体偏低，虽然组内差异大，但大多是"混乱的差异"而非"有意义的推理路径差异"。

GSM8K 场景经验值：`temperature=0.8~1.0`，`top_p=0.9~1.0`。建议训练过程中保持采样参数固定（不要动态调整），除非有充分的诊断依据（如第13章提到的"熵坍塌"现象）。

### 11.6 有效 Batch Size 与梯度累积

GRPO 中"有效 batch size"需要按 `per_device_train_batch_size × gradient_accumulation_steps × world_size` 计算，但请注意：**在 RLHF/GRPO 训练中，梯度累积与增大 per-device batch size 并不完全等价**（这一点与 SFT 不同，官方文档也特别指出这一点），因为每个 rollout batch 内的样本是同一批策略模型采样得到的（on-policy 程度相关），累积多个 mini-batch 再更新，实际上引入了轻微的 off-policy 成分。生产环境建议：

- 优先通过增大 `per_device_train_batch_size`（在显存允许范围内）而非 `gradient_accumulation_steps` 来扩大有效 batch size；
- 如果显存不足必须使用梯度累积，累积步数不建议设置过大（一般不超过 4~8），避免 on-policy 程度下降过多影响训练稳定性。

### 11.7 学习率调度与 Warmup

GRPO 训练建议使用带 warmup 的学习率调度（`--warmup_ratio 0.03~0.1`），训练初期策略模型的 rollout 质量还不稳定，过高的初始学习率容易在训练最初的几十步内就造成不可逆的语言能力损伤。学习率调度类型建议使用 cosine 或 constant with warmup，GRPO 场景下由于训练步数通常远少于 SFT（GSM8K 规模数据往往几百到一两千 step 即可看到明显效果），过于激进的 decay 策略（如线性衰减到 0）可能导致训练后期学习率过低、reward 提升停滞，建议优先尝试 `constant_with_warmup` 或较缓的 cosine decay。


## 第十二章 训练过程监控：指标体系与健康区间

### 12.1 核心指标一览表

| 指标 | 含义 | 健康表现 | 异常表现 |
|---|---|---|---|
| `reward/mean` | 每个 rollout batch 的平均奖励 | 随训练稳步上升，后期趋于平台但不下降 | 长期不涨、剧烈震荡、涨到一半突然崩溃 |
| `reward/accuracy` | 正确性子奖励均值（需自行分开打点） | 与 `reward/mean` 同步上升 | 与总 reward 走势背离（如总reward涨但accuracy不涨） |
| `reward/std`（组内） | 组内奖励标准差 | 训练初期较高，随训练收敛逐渐下降但不为0 | 过早降到接近0（多样性坍塌）或持续极高（模型能力未提升） |
| `kl` | 策略与参考模型的 KL 散度 | 平稳缓慢上升，不出现突变 | 单调爆炸增长（策略失控偏离）或恒为接近0（策略几乎未更新） |
| `entropy` | 生成 token 分布的熵 | 缓慢下降，保留一定探索性 | 快速坍塌到接近0（多样性丧失，见13.3节） |
| `completion_length` | 生成回答的平均长度 | 稳定或合理增长（尤其thinking模式） | 无限制增长（reward hacking via碰运气）或急剧缩短（模型放弃思考） |
| `clip_fraction` | 被 clip 机制截断的更新比例 | 较低且稳定（如<20%） | 持续走高，说明策略更新幅度频繁超出裁剪范围，学习率或epsilon需调整 |
| `grad_norm` | 梯度范数 | 平稳，无剧烈尖峰 | 出现频繁的尖峰（可能预示数值不稳定，需检查 reduce_dtype/学习率） |
| `loss` | GRPO policy loss | 在0附近小幅波动（非单调下降） | 持续增长或出现NaN/Inf |

### 12.2 指标的联合解读——不要孤立看单一指标

GRPO 收敛判断的核心原则是**联合多个指标做交叉验证**，单独看任何一个指标都可能得出错误结论。给出几个典型的联合解读场景：

**场景A：reward 上升，但 accuracy 子奖励不涨，completion_length 持续增长**
→ 高度怀疑"长度型 reward hacking"：模型学会了拉长输出但没有真正提升解题能力（例如输出重复的中间步骤、无意义的自言自语来"凑"格式奖励或规避截断惩罚）。应立即检查奖励函数的长度惩罚设置，并考虑降低 `max_completion_length` 或加强 `overlong_filter`。

**场景B：reward 上升，KL 快速增长，entropy 快速下降**
→ 策略正在快速收窄到少数几种"确定能拿分"的回答模式，短期看指标"健康"，但存在过拟合到训练集表面模式、丧失泛化能力的风险。应结合验证集 reward 走势判断——如果验证集 reward 没有同步上升，基本可以确认是过拟合，需要降低学习率或提高 KL 惩罚。

**场景C：reward 长期不涨，KL 也几乎为0，grad_norm 很小**
→ 策略几乎没有发生有效更新。常见原因：学习率设置过低、`beta` 设置过高压制了更新、或者 LoRA rank 过小导致模型容量不足以学到有意义的策略调整。

**场景D：reward 剧烈震荡，clip_fraction 持续走高，grad_norm 出现频繁尖峰**
→ 典型的"学习率过高/epsilon裁剪范围过窄导致的不稳定训练"。应下调学习率，或适当放宽 `epsilon_high`。

### 12.3 训练日志中应重点关注的自定义打点

除了 ms-swift 内置上报的指标，生产环境建议在自定义 reward plugin / callback 中额外打点：

```python
# 伪代码示意：在 reward function 或自定义 trainer callback 中额外记录
import wandb

def log_custom_metrics(completions, solutions, rewards, step):
    accuracy_only = [1.0 if extract_answer(c) == s else 0.0 for c, s in zip(completions, solutions)]
    lengths = [len(c.split()) for c in completions]
    format_ok_ratio = sum("####" in c for c in completions) / len(completions)
    wandb.log({
        "custom/accuracy_only_mean": sum(accuracy_only) / len(accuracy_only),
        "custom/completion_length_mean": sum(lengths) / len(lengths),
        "custom/completion_length_p95": sorted(lengths)[int(len(lengths)*0.95)],
        "custom/format_ok_ratio": format_ok_ratio,
    }, step=step)
```

这些细分指标是第13章诊断表能够落地的前提——没有这些打点，很多收敛问题只能等到训练完成、线下评估时才会被发现，届时已经浪费了大量计算资源。

### 12.4 验证集（held-out）监控的必要性

即使 GRPO 是在线 RL、理论上不存在传统意义的"过拟合训练集标签"问题（因为没有使用 ground truth 做监督），但**策略仍然可能过拟合到训练 prompt 分布的表面统计特征**（比如训练集中的题目风格、数字范围），导致在训练集 reward 持续上升，但独立同分布的验证集（或线下 GSM8K test set）reward 不再提升甚至下降。建议：

- 每隔固定 step（`--eval_steps`）在验证集上跑一次 rollout + 打分（不参与梯度更新），记录 `eval_reward/mean`；
- 如果条件允许，定期（如每 100～200 step）用完整 GSM8K test set（1319条）做一次独立评估，计算真实的 pass@1 准确率，作为最终判断训练是否真正有效提升模型能力的黄金标准（见第17章）。


### 12.5 训练结果图表判读：如何通过 loss.png / reward.png 等曲线图判断是否收敛

ms-swift 训练结束或训练过程中，`output_dir` 下通常会生成 `trainer_state.json`（逐 step 的原始指标数据），如果配置了 `--report_to wandb`/`tensorboard`/`swanlab`，则可以在对应平台导出/截图得到 `loss.png`、`reward.png`、`kl.png`、`entropy.png`、`grad_norm.png`、`completion_length.png` 等曲线图。本节给出一套**看图判断收敛与否**的可操作方法，可用于日常巡检，也可用于事后复盘。

#### 12.5.1 通用读图流程（无论看哪张图都适用）

1. **先看整体形状（trend），再看局部抖动（noise）**：GRPO 曲线天然比 SFT 曲线噪声大得多（因为每个 step 的 rollout 都是随机采样），不要因为某几个 step 的尖峰/毛刺就下结论，应该看 50～100 step 为窗口的滑动平均趋势。
2. **不要孤立看一张图，至少同时打开 reward.png + kl.png + entropy.png 三张图对照**（原因见12.2节的"联合解读"原则，图形判读同样需要交叉验证）。
3. **区分"训练早期正常的震荡期"和"训练中期不该出现的震荡"**：一般前 5%~10% 的 step（对应 warmup 阶段）曲线剧烈波动是正常的，如果训练过了大半程仍然剧烈震荡甚至越来越剧烈，才需要报警。
4. **对比训练集曲线与验证集曲线（如果有 eval_reward.png）**：只看训练集曲线容易被"过拟合到表面模式"的假象欺骗，训练集和验证集曲线的走势应该基本同步。

#### 12.5.2 loss.png 怎么看

**最容易被误读的一张图**。健康的 GRPO policy loss 曲线特征：
- 数值始终在 0 附近的一个小区间内浮动（如 -0.1~0.1，具体量级取决于 `epsilon`/`beta` 设置），**不是单调下降的曲线**，如果你期待它像 SFT loss 那样一路下降，会得出"没在学习"的错误结论。
- 健康表现：整条曲线呈现"围绕 0 的白噪声状态"，方差不随训练进程明显放大。
- 异常表现：① loss 绝对值持续系统性增长（不是偶尔的尖峰，而是趋势性抬升）——警惕策略发散；② loss 曲线中出现突变的阶梯跳变或 NaN/Inf 断点——几乎可以确定发生了数值问题（检查 `reduce_dtype`、学习率、混合精度配置）；③ loss 长时间恒为 0——警惕梯度未生效（检查 LoRA target_modules、优化器是否正确绑定了可训练参数）。

**结论**：loss.png 主要用于"排除异常"（有没有发散/数值错误），而不是用于"确认正在进步"，确认进步要看 reward.png。

#### 12.5.3 reward.png 怎么看

**判断"是否在进步"的第一图**，但要看对方式：
- 健康形态：整体呈现"阶梯式上升 + 局部震荡"，类似股票的震荡上行走势——每隔一段 step 有明显抬升，抬升之间夹杂正常回调，但不会跌破前一个平台的低点太多。到训练后期趋于一个相对平坦的高位平台（plateau），说明该数据/该模型规模下的能力已接近当前配置的上限。
- 危险形态一：**持续走平/横盘，且从训练开始就没有明显抬升**——大概率是 6.4 节的奖励函数 bug，或学习率/beta 设置导致策略几乎不更新，需要结合 kl.png（看 KL 是否也贴地不动）确认。
- 危险形态二：**先陡峭上升，冲到一个高点后断崖式下跌，且不再恢复**——典型的策略发散（13.2节），应立刻定位崩溃发生的 step，回滚到崩溃前的 checkpoint。
- 危险形态三：**持续单调上升、曲线异常平滑、几乎没有震荡**——反直觉地，这也是需要警惕的形态，正常 GRPO 由于采样噪声，reward 曲线不可能非常平滑，过于平滑往往说明：a) 你在看的是做了较大窗口平滑处理后的曲线（此时应换成看原始点或更小平滑窗口复核），或 b) 数据集过于同质/简单，模型很快就学会了固定套路（此时要交叉看 entropy.png 是否同步过快下降，警惕熵坍塌）。
- **拐点判断法**：如果 reward 曲线连续 100～200 个 step（视总训练步数而定，一般取总步数的 10%~20%）内滑动均值不再抬升（斜率接近 0），且验证集 reward 同步走平，基本可以判定模型在当前数据/超参数下已经**收敛到平台期**，继续训练边际收益很低，此时应该考虑：a) 判定训练完成，导出最终 checkpoint；b) 如果效果尚不满意，考虑调整数据难度分布/学习率/beta 后重新训练，而不是无意义地延长当前配置的训练时长。

#### 12.5.4 kl.png（KL 散度曲线）怎么看

- 健康形态：从 0 附近缓慢、近似线性或轻微上凸地爬升，全程没有陡峭的拐点。
- 危险形态一：**指数级/陡峭爬升，尤其是与 reward.png 的崩溃点时间对齐**——策略发散的确凿证据，此时 reward 崩溃不是巧合而是因果关系（策略跑得离参考模型太远，语言分布本身已经损坏）。
- 危险形态二：**长期趴在 0 附近几乎不动**——策略没有发生有效更新，若同时 reward 也不涨（危险形态一的对照），基本可判定是学习率过低或 beta 过大压制了更新，而非奖励函数问题（因为如果是奖励函数给不出有效信号，KL 仍然可能因为随机噪声缓慢爬升，只是 reward 不涨；而"KL和reward都不动"更指向优化器层面没有产生有效更新）。

#### 12.5.5 entropy.png 怎么看

- 健康形态：从一个较高的初始值缓慢下降，下降速率逐渐放缓（凹形曲线，即所谓"边际递减"），最终稳定在一个不为 0 的水平（保留一定探索性）。
- 危险形态：**熵在训练很靠前的阶段（如前10%~20% step内）就快速跌到接近0**——即13.3节所述的"熵坍塌"，应立刻结合 `reward/std`（组内标准差）曲线复核，若组内标准差同步跌到接近0，基本确诊，需要提高温度或beta并考虑从更早 checkpoint 重启。

#### 12.5.6 completion_length.png 怎么看

- 健康形态：训练初期可能有一定波动，随后趋于稳定或（尤其 thinking 模式下）随 reward 提升伴随温和增长后趋于平稳——说明模型在学习"用适当长度的推理换取正确率"。
- 危险形态：**曲线持续、无上限地增长，逼近甚至频繁触达 `max_completion_length` 设置的上限**——高度怀疑第13.1节讨论的"长度型 reward hacking"，需要立即检查 `overlong_filter` 是否生效、是否需要加入显式长度惩罚。

#### 12.5.7 grad_norm.png 与 clip_fraction.png 怎么看

- `grad_norm.png` 健康形态：整体幅度平稳，允许有限的随机尖峰（尤其训练早期），但不应出现频率越来越高、幅度越来越大的尖峰簇。若尖峰簇集中出现在 reward 崩溃前，可作为"崩溃前兆"的早期预警信号（比 reward 本身更早反映问题，值得作为自动化报警的首选指标）。
- `clip_fraction.png` 健康形态：维持在一个较低且稳定的比例（如 <20%）并轻微波动。若持续走高（如超过 40%~50%）且趋势性上升，说明大量更新被 clip 机制截断，等效于"模型想更新但被强行摁住"，是学习率过高或 epsilon 裁剪范围过窄的信号，应下调学习率或放宽 epsilon_high。

#### 12.5.8 一张速查判读表

| 观察到的图形组合 | 最可能的判断 | 建议动作 |
|---|---|---|
| reward↑ 稳定，kl 缓慢线性↑，entropy 缓慢↓不到0，completion_length 平稳 | **训练健康，正在收敛** | 继续训练，定期做17章线下评估确认 |
| reward 走平，kl 也趴地不动 | 策略未有效更新 | 检查学习率是否过低/beta过高，检查LoRA可训练参数是否绑定正确 |
| reward 走平，kl 却在缓慢增长 | 策略有更新但方向无效/信号噪声大 | 检查奖励函数是否存在系统性bug（6.4节），检查num_generations是否过小 |
| reward 陡升后断崖下跌，同时kl陡峭爬升 | 策略发散 | 回滚checkpoint，降学习率、提beta（13.2节） |
| reward 短期虚高，entropy快速跌至接近0，reward/std同步跌至接近0 | 熵坍塌/多样性丧失 | 提温度、提beta，检查数据难度分布（13.3节） |
| reward↑，但accuracy子奖励不涨，completion_length持续走高 | 长度型reward hacking | 检查overlong_filter，加长度惩罚（13.1/6.3.3节） |
| loss出现NaN/Inf或阶梯跳变 | 数值稳定性问题 | 检查reduce_dtype是否为float32，检查混合精度配置（5.5节） |
| grad_norm尖峰簇增多，clip_fraction趋势性走高 | 更新幅度过大的早期预警 | 提前介入，无需等reward真正崩溃才处理 |

**实践建议**：如果你已经把训练输出的 `loss.png`、`reward.png` 等图片准备好，可以直接发给我（上传图片），我可以结合上表逐张帮你读图判断当前训练所处的健康状态，并给出具体的下一步调参建议。

## 第十三章 收敛失败模式诊断手册（含典型 case）

本章以"症状 → 可能原因 → 排查步骤 → 解决方案"的结构，汇总 GRPO + FSDP2 + 32B/35B 大模型场景下最常见的收敛失败模式。建议将本章作为训练值班时的排障手册（runbook）使用。

### 13.1 症状：训练一开始 reward 就是 0 或恒定不变

**可能原因**：
1. 奖励函数 bug（答案抽取正则表达式匹配不到任何内容）；
2. 模板不一致（训练模板与 vLLM rollout 模板不一致，导致生成内容格式与奖励函数预期完全不符）；
3. 权重同步失败（vLLM 拿到的仍然是随机初始化权重或错误权重，而非当前训练权重）。

**排查步骤**：
1. 打开 `--log_completions true`，人工检查前几个 step 的生成内容，确认格式是否合理（有意义的文本还是乱码）；
2. 如果生成内容合理但格式与预期不符（如没有 `####` 标记），检查 system prompt 是否正确下发到 rollout 阶段；
3. 如果生成内容本身就是乱码/重复token，高度怀疑权重同步问题，检查 FSDP2 参数聚合与 vLLM 权重加载的日志，确认没有 shape mismatch 或静默失败；
4. 单独用几条样本手动跑一遍奖励函数（不经过训练框架），确认打分逻辑本身没有 bug。

**解决方案**：根据排查结果针对性修复；强烈建议在下次训练前，把这几个排查步骤固化为第7章冒烟测试的一部分，避免同类问题复现。

### 13.2 症状：训练中期 reward 突然断崖式下跌，且伴随生成内容变成乱码/重复

**可能原因**：**策略发散（policy collapse）**，通常由学习率过高、`beta` 过小（KL 约束不足）、或某个训练 step 出现了异常大的梯度（未被梯度裁剪有效控制）共同导致。

**排查步骤**：
1. 查看崩溃前几个 step 的 `grad_norm`、`kl`、`clip_fraction` 是否有异常尖峰或持续攀升的趋势（很多情况下崩溃不是瞬间发生的，而是有 5～20 step 的"劣化前兆期"）；
2. 检查是否恰好在崩溃前有 batch 内出现异常数据（如超长 prompt、特殊字符导致 tokenize 异常）；
3. 检查 FSDP2 的 `reduce_dtype` 是否被误设为 bf16（数值稳定性问题在大模型、长训练情况下更容易积累到临界点爆发）。

**解决方案**：
1. 从故障前的最近一个健康 checkpoint 恢复训练（这是第16章强调"密集保存 checkpoint"的核心价值所在）；
2. 降低学习率（建议降至原来的 1/2～1/3）、适当提高 `beta`、收紧 `max_grad_norm`；
3. 确认 `reduce_dtype=float32`；
4. 如果怀疑是数据异常触发，为数据管道增加长度/字符过滤和异常捕获，避免同类脏数据再次触发问题。

### 13.3 症状：熵坍塌（entropy collapse）——模型输出多样性迅速消失，reward 短期虚高后停滞

**现象**：`entropy` 指标在训练早期快速下降到接近 0，同时组内 `reward/std` 也快速趋近于 0（模型对几乎所有 prompt 都生成高度相似甚至完全相同的回答），训练集 reward 短期内可能因为"抓住了几个高频简单题型的固定答题套路"而看似上升，但验证集 reward 停滞，模型泛化能力实际上在变差。

**可能原因**：温度设置过低、`beta` 过小导致策略过快收窄到局部最优、训练数据难度分布过于集中（大量简单重复题型）。

**解决方案**：
1. 适当提高 `temperature`（如 0.9→1.0~1.1），增加探索性；
2. 适当提高 `beta`，约束策略更新幅度；
3. 检查数据集是否需要做难度均衡采样，避免模型过早收敛到"简单题套路"；
4. 部分 GRPO 变体引入显式的熵正则化项（entropy bonus）鼓励探索，若框架支持可以尝试加入。

### 13.4 症状：Full 全参训练在 FSDP2 下出现间歇性 OOM（并非每次都复现）

**可能原因**：这是 32B/35B FSDP2 训练中较难排查的一类问题，通常与以下因素相关：
1. 权重同步阶段（policy → vLLM）的 all-gather 显存尖峰，只有在 batch 内出现较长 completion（导致该阶段其他显存占用也偏高）时才会叠加触发 OOM；
2. `dataloader` 采样到的某个 batch prompt 长度异常，导致该 step 的 activation 显存超出预期；
3. CPU offload 与 GPU 显存回收的时序问题（比如上一 step 的 offload 还未完成，下一 step 已经开始分配显存）。

**排查步骤**：
1. 记录每次 OOM 发生时的 step 号，检查该 step 对应的 batch 数据（prompt/completion 长度分布）是否有异常；
2. 在训练脚本中加入显存监控日志（`torch.cuda.max_memory_allocated()`），逐 step 记录，定位显存尖峰出现的具体阶段（前向/反向/优化器更新/权重同步）；
3. 检查 `--gc_collect_after_offload true` 是否已开启（帮助及时回收 offload 后的显存碎片）。

**解决方案**：
1. 对数据管道增加更严格的长度过滤，避免个别异常长样本触发尖峰；
2. 为显存预留更多 buffer（降低 `per_device_train_batch_size` 或 `vllm_gpu_memory_utilization`）；
3. 确认 offload 相关参数（`offload_model`、`offload_optimizer`、`gc_collect_after_offload`）都已正确开启且顺序合理。

### 13.5 症状：MoE 模型（Qwen3.6-35B-A3B）训练中，reward 提升缓慢且伴随专家利用率严重不均衡

**可能原因**：路由坍塌（详见10.5节），少数专家承担了绝大部分计算，导致模型有效容量大幅萎缩，等效于一个远小于35B的模型在训练。

**解决方案**：
1. 提高负载均衡辅助损失系数（`aux_loss_coeff`）；
2. 降低学习率，MoE 路由对参数更新更敏感；
3. 若坍塌已经发生且难以通过继续训练恢复，建议直接回滚到路由分布健康的更早 checkpoint 重新开始，而不是尝试"训练出来"。

### 13.6 症状：LoRA 训练 reward 表现良好，但切换到 Full 全参训练后各项指标全面变差

**可能原因**：最常见的原因是**直接照搬 LoRA 的学习率/beta等超参数到 Full 训练**，未按第9章、第18章的建议做数量级调整。

**解决方案**：严格按第18章的"LoRA → Full 渐进式迁移路径"重新设置超参数，不要假设 LoRA 场景验证过的超参数可以直接复用到 Full 场景。


## 第十四章 显存与吞吐优化：offload、vLLM colocate/server、sleep_level

### 14.1 colocate 模式的显存生命周期管理

在 colocate 模式下，训练进程与 vLLM 推理进程共享同一批 GPU，二者的显存需求在时间上交替出现（训练阶段不需要 vLLM 的 KV Cache，rollout 阶段不需要训练侧的梯度/优化器状态）。ms-swift 通过以下机制管理这一生命周期：

- `--sleep_level`：控制训练阶段 vLLM 释放显存的程度。`sleep_level=1` 通常释放 KV Cache 等推理专用显存但保留模型权重在显存中（唤醒更快）；更高级别可能连模型权重也释放（唤醒更慢但训练阶段显存更充裕），需结合 ms-swift 4.4.0 文档中该参数的具体级别定义调整。
- `--vllm_gpu_memory_utilization`：限制 vLLM 在其"苏醒"阶段最多能使用的显存比例，需要与训练侧的显存需求联合估算，避免二者相加超过物理显存导致 OOM。
- `--offload_model` / `--offload_optimizer`：训练阶段结束后（进入 rollout 阶段前），将训练侧的模型参数/优化器状态 offload 到 CPU，为 vLLM 腾出显存；rollout 阶段结束后再加载回 GPU 继续训练。`--gc_collect_after_offload true` 确保 offload 后及时触发 Python/CUDA 的垃圾回收，避免显存碎片化导致的"看起来有空闲显存但分配失败"问题。

### 14.2 server 模式的吞吐优势与稳定性权衡

server 模式将 vLLM 部署为独立服务，训练和推理可以并行执行（尤其是异步 GRPO 场景，训练侧在用当前 batch 更新参数的同时，rollout 侧已经在为下一 batch 生成候选），吞吐显著优于 colocate 模式的"训练/推理严格串行"。但需要关注：

1. **网络稳定性**：训练进程通过 HTTP/gRPC 请求 rollout server，需要确保网络配置（尤其跨节点场景）稳定，超时/重试机制需要合理设置，避免因网络抖动导致 rollout 数据缺失或训练进程挂起。
2. **权重同步延迟**：训练侧更新完参数后，需要主动将最新权重推送/同步给 rollout server，这个同步过程如果耗时较长，会导致 rollout server 在一段时间内仍使用旧权重生成候选，引入"策略滞后"（off-policy 程度增加）。对于 GRPO 这种理论上要求较高 on-policy 程度的算法，滞后过多会影响收敛质量，需要监控同步延迟并控制在合理范围内（经验上不超过 1～2 个训练 step 的滞后）。
3. **Server 资源独占 vs 复用**：如果 rollout server 独占若干张 GPU（如第9章示例中的 2 卡），需要在集群资源规划阶段就把这部分算力计入总需求，不能只按"训练卡数"申请资源。

### 14.3 显存优化手段的组合优先级建议（收敛质量优先）

在显存不足时，不同优化手段对训练质量/收敛速度的影响程度不同，建议按以下优先级考虑（优先使用对收敛质量影响最小的手段）：

1. **优先**：开启/确认 `gradient_checkpointing`、`flash_attn`、`liger_kernel`（几乎不影响收敛质量，纯粹的工程优化）；
2. **次优先**：调整 vLLM 显存占用比例、offload 策略的精细化配置（不改变训练本身的数学过程，只改变显存的时间/空间分布）；
3. **谨慎**：降低 `per_device_train_batch_size` 并用梯度累积补偿（如11.6节所述，会引入轻微 off-policy 成分，需要监控）；
4. **最后手段**：降低 `num_generations`、`max_completion_length`（直接影响 advantage 估计质量和模型探索空间，对收敛质量影响最大，应优先考虑扩充硬件资源而非压缩这两个参数）。

### 14.4 Liger-Kernel 与 Flash-Attention 在 GRPO 场景下的适配注意事项

- Liger-Kernel 提供融合的 CrossEntropy、RMSNorm、RoPE 等算子，显著降低显存占用、提升训练速度，`--use_liger_kernel true` 在 32B/35B 规模训练中建议默认开启。需注意部分融合算子对精度计算路径有细微调整，建议在冒烟测试阶段对比开启/关闭 Liger-Kernel 前后的 reward/KL 曲线，确认数值行为一致后再用于正式训练。
- Flash-Attention（`--attn_impl flash_attn`）在 GRPO 场景下同时影响训练侧和 vLLM 推理侧的 attention 计算，务必确保两侧都正确启用，尤其是使用 `--packing true` 训练时是强制要求（否则 attention mask 处理会出错）。


## 第十五章 多机多节点分布式训练配置

### 15.1 环境变量与启动方式

ms-swift 多机训练依赖标准的 PyTorch 分布式环境变量组合，结合 FSDP2 使用时需要格外注意所有节点的环境、代码、数据路径完全一致：

```bash
# 以两机、每机8卡为例，node0 为 master
# node0 执行：
export NNODES=2
export NODE_RANK=0
export NPROC_PER_NODE=8
export MASTER_ADDR=192.168.1.10   # node0 自身IP或可被所有节点访问的地址
export MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-32B \
    --train_type full \
    --fsdp fsdp2_config.yaml \
    ... # 其余参数同第9章

# node1 执行相同脚本，仅修改：
# export NODE_RANK=1
```

若使用调度平台（如 Alibaba Cloud DLC、Slurm、Kubernetes Job），建议通过平台原生的多节点分布式训练模板自动注入上述环境变量，避免手工维护多份脚本导致的配置漂移风险（`NODE_RANK` 设置错误、`MASTER_ADDR` 不一致等是多机训练最常见的低级错误来源）。

### 15.2 网络与 NCCL 配置检查

- 确保节点间网络（尤其是否有 InfiniBand/RoCE 高速网络）配置正确，`NCCL_IB_DISABLE`、`NCCL_SOCKET_IFNAME` 等环境变量需要根据实际网络拓扑设置，配置错误会导致 all-gather/reduce-scatter 通信退化到极低带宽，表现为训练速度远低于预期（而非直接报错），容易被误判为"模型太大训练本来就慢"而被忽视。
- 训练启动前建议运行 NCCL 性能自检工具（如 `nccl-tests` 中的 `all_reduce_perf`），确认跨节点带宽符合预期（RoCE/IB 网络通常应达到数十~上百 GB/s 量级，具体视硬件而定），避免在正式训练中才发现网络配置问题。

```bash
# NCCL 环境变量示例（需根据实际集群网络配置调整）
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO   # 训练初期建议开启，确认通信路径正确，稳定后可关闭以减少日志量
```

### 15.3 多机场景下 vLLM Rollout 的部署策略

多机 GRPO 训练中，vLLM rollout 引擎的部署有两种典型模式：

1. **每个训练节点本地部署一个 vLLM 实例（colocate 或本地 server）**：优点是 rollout 数据不需要跨节点网络传输，延迟低；缺点是需要在每个节点预留 rollout 所需的 GPU 资源，且各节点 rollout 结果需要汇总同步给全局 FSDP2 训练组。
2. **集中部署独立的 vLLM rollout 集群（server 模式）**：所有训练节点统一向这个 rollout 集群请求生成，架构更清晰、资源利用率通常更高（rollout 集群可以根据训练侧的请求压力独立扩缩容），是大规模（4机以上）训练的推荐方式，但需要更成熟的服务治理能力（负载均衡、故障转移、权重同步的一致性保证）。

### 15.4 多机训练的容错与重启策略

32B/35B 规模的多机 GRPO 训练动辄运行数小时到数天，硬件/网络故障（掉卡、节点重启、网络抖动导致的通信超时）是常态而非例外，务必提前规划：

- 训练脚本应支持从 checkpoint **无缝续训**（见第16章），任何节点故障导致的训练中断，应能在故障节点恢复或替换后，用最近一次成功保存的 checkpoint 快速恢复，而不需要从头开始；
- 建议在调度平台层面配置**自动重试机制**，训练进程异常退出后自动重新拉起并从最新 checkpoint 续训，减少人工介入的时间成本；
- 分布式训练框架的超时参数（如 NCCL 的 `NCCL_TIMEOUT` 或 PyTorch 分布式的 `timeout` 配置）应设置合理阈值，过短容易在正常的 rollout 长耗时阶段（大模型生成较长 completion 可能需要数十秒到数分钟）被误判为故障触发不必要的重启，过长则会导致真实故障发生时恢复过慢。


## 第十六章 Checkpoint 管理、断点续训与灾难恢复

### 16.1 FSDP2 场景下的 checkpoint 保存方式

FSDP2 训练的模型参数以分片（sharded）形式分布在各卡上，保存 checkpoint 时有两种主要方式，通过 `state_dict_type` 配置：

- **`SHARDED_STATE_DICT`（推荐用于训练过程中的常规保存）**：每张卡只保存自己持有的参数分片，保存/加载速度快、不需要额外的显存做参数聚合，适合频繁保存（如每隔几十 step 保存一次）。缺点是保存出来的 checkpoint 是"分片格式"，不能直接用于单卡推理/部署，恢复训练时也需要用同样（或兼容）的分片配置加载。
- **`FULL_STATE_DICT`（推荐用于最终导出/部署）**：训练结束或需要导出可直接部署的完整权重时，将所有分片聚合成完整的 state_dict 再保存。聚合过程会有临时的显存/内存尖峰（相当于把整个模型完整加载一遍），32B/35B 规模建议在导出阶段使用 CPU 内存做聚合（`--save_only_model true` 配合相关 CPU 聚合参数），避免占用宝贵的 GPU 显存。

生产环境建议策略：**训练过程中使用 `SHARDED_STATE_DICT` 高频保存（便于快速续训和故障恢复），训练完成或需要发布模型时再单独跑一次导出流程转换为 `FULL_STATE_DICT`（或使用 `swift export` 一类的转换工具）**，二者不冲突。

### 16.2 LoRA 场景 checkpoint 的特殊性

LoRA 训练的 checkpoint 通常只需要保存 adapter 权重（远小于 base 模型），保存/加载效率远高于全参训练，`--save_only_model true` 场景下建议确认保存的 checkpoint 目录中只包含 adapter 权重和必要的配置文件，而不是意外保存了完整 base 模型副本（会造成不必要的存储浪费，尤其在 32B 模型、频繁保存的场景下累积存储成本很高）。

### 16.3 断点续训操作

```bash
# 从指定 checkpoint 恢复训练（保留 GRPO 训练状态，如 optimizer state、当前 step 数、reward 历史统计量等）
swift rlhf \
    --rlhf_type grpo \
    --resume_from_checkpoint output/qwen3-32b-grpo-full/checkpoint-150 \
    ... # 其余参数与原训练脚本保持完全一致
```

**关键原则：续训时除 `--resume_from_checkpoint` 外，其余超参数应尽量与原训练脚本保持一致**，除非你是刻意要在恢复训练的同时调整超参数（如第13.2节故障恢复场景中，故意降低学习率）。任何非必要的超参数变更都会引入新的不确定性，让"续训后 reward 曲线的变化"难以归因（究竟是恢复训练本身的正常波动，还是超参数变更导致的）。

### 16.4 灾难恢复演练建议

对于计划投入数天甚至数周的 32B/35B GRPO 训练任务，建议在正式训练开始前，**主动做一次"灾难恢复演练"**：故意在冒烟测试阶段模拟一次训练中断（如手动 kill 训练进程），验证：

1. 能否用最新保存的 checkpoint 成功恢复训练，且恢复后指标（reward、KL 等）曲线与中断前平滑衔接，没有出现指标跳变；
2. 恢复训练所需的时间（模型加载、优化器状态恢复、FSDP2 分片重建等）是否在可接受范围内；
3. 数据加载器（dataloader）是否能正确恢复到中断时的数据读取位置（避免恢复后重复训练部分数据或跳过部分数据，虽然 GRPO 对这种程度的数据顺序扰动通常不敏感，但仍建议确认行为符合预期）。

这个演练成本很低（几分钟到十几分钟），但能在正式大规模训练前暴露 checkpoint/续训机制中的潜在问题，避免真正故障发生时手忙脚乱、甚至因为续训机制本身有 bug 而被迫从头重新训练，造成大量计算资源浪费。

## 第十七章 评估方法论：从训练内 reward 到线下 GSM8K Acc

### 17.1 训练内 reward 与线下评估的关系与差异

训练过程中监控的 `reward/mean` 是基于当前采样温度（`temperature=0.8~1.0` 等探索性配置）下的 rollout 结果计算的，**不能直接等同于模型的真实能力水平**。生产环境判断模型能力是否真正提升，应该定期（如每隔几十~上百 step，或每个 epoch 结束）用**独立的线下评估流程**，在更贴近实际部署场景的采样配置下（通常 `temperature` 更低甚至贪心解码 `temperature=0`）在完整 GSM8K test set（1319条）上计算真实的 pass@1 准确率。

### 17.2 评估脚本示例

```bash
# 使用 ms-swift 的 infer/eval 能力对训练得到的 checkpoint 做线下评估
CUDA_VISIBLE_DEVICES=0,1 \
swift eval \
    --model output/qwen3-32b-grpo-full/checkpoint-300 \
    --infer_backend vllm \
    --eval_backend OpenCompass \
    --eval_dataset gsm8k \
    --eval_limit 1319

# 对于 LoRA checkpoint，需要同时指定 base 模型与 adapter：
CUDA_VISIBLE_DEVICES=0,1 \
swift eval \
    --model Qwen/Qwen3-32B \
    --adapters output/qwen3-32b-grpo-lora/checkpoint-300 \
    --infer_backend vllm \
    --eval_backend OpenCompass \
    --eval_dataset gsm8k
```

### 17.3 pass@1 与 pass@k 的取舍

GSM8K 场景下，除了标准的贪心解码 pass@1，建议同时评估 pass@k（k=4/8，采样多次取"至少一次正确"的比例），这能更全面地反映模型的探索能力上限，尤其对 GRPO 训练关注的"是否拓宽了模型的正确解题路径分布"这一目标有更直接的指示意义——如果 pass@1 提升有限但 pass@8 提升明显，说明模型的知识/能力边界确实在扩展，只是采样稳定性/置信度还需要进一步训练强化；反之如果 pass@1 和 pass@8 都没有提升，则需要更严肃地审视训练是否真正有效。

### 17.4 建立评估-训练的反馈闭环

建议将线下评估纳入训练流程的常规环节，形成"训练 N step → 自动触发线下评估 → 结果写入实验跟踪系统（wandb/飞书文档/内部平台）→ 人工或自动化规则判断是否继续/调整/终止训练"的闭环，而不是等训练全部跑完才做一次性评估。对 32B/35B 这种资源消耗巨大的训练任务，尽早发现"训练没有真正带来能力提升"的信号，能够显著降低无效计算的资源浪费。


## 第十八章 LoRA → Full 的渐进式迁移路径

### 18.1 为什么建议渐进迁移而非直接上 Full

大规模 Full 全参 GRPO 训练资源消耗巨大、试错成本极高，直接在未经 LoRA 验证的情况下上 Full 训练，一旦奖励函数设计、数据配置存在问题，损失的计算资源远超 LoRA 阶段的试错成本。本手册推荐的路径是：**先用 LoRA 快速验证 pipeline 正确性和超参数的大致有效区间，再迁移到 Full 全参训练获取更强的最终效果**（社区经验普遍认为，Full 全参在数据量充分、训练稳定的前提下，上限通常优于 LoRA，这是最终生产模型倾向选择 Full 的原因，但迁移过程需要谨慎）。

### 18.2 迁移步骤

1. **确认 LoRA 阶段的收敛质量**：LoRA 训练在验证集/线下 GSM8K test set 上的 pass@1 相比 base 模型有稳定、可复现的提升（建议至少跑 2～3 次独立实验确认提升的稳定性，而非单次实验的偶然结果）。
2. **超参数的数量级转换**：按第9章、第11章给出的经验区间，将学习率下调 1～2 个数量级、`beta` 相应调低、`max_grad_norm` 收紧，**不要直接复用 LoRA 阶段效果最好的超参数**。
3. **资源与 FSDP2 配置的重新验证**：Full 训练的显存特性与 LoRA 完全不同（详见第4章），需要重新做第7章的冒烟测试（可以用 Qwen3-8B Full 训练做冒烟测试，验证 FSDP2 全参分片配置、参考模型处理、checkpoint 保存等环节都正确后，再迁移到 32B/35B）。
4. **小规模 Full 训练验证**：在正式投入 32B/35B 全量资源前，建议先用 32B/35B 模型 + 较小数据子集（如 GSM8K 全量的 20%~30%）+ 较短训练步数，跑一次"中等规模"的 Full 训练，确认各项指标健康、资源评估准确后，再扩展到完整数据集和更长的训练步数。
5. **正式 Full 训练**：按第9章方案执行，全程按第12章监控体系跟踪指标健康度，按第16章策略做密集 checkpoint 保存。

### 18.3 迁移过程中的常见误区

- **误区一：认为 LoRA 验证过的奖励函数和数据配置可以完全照搬到 Full 训练，无需重新审视**——奖励函数和数据配置确实通常不需要因训练方式改变而调整，但仍建议在 Full 训练冒烟测试阶段重新走一遍第7章检查清单，因为 Full 训练下模型的行为模式（尤其是探索的广度和深度）可能与 LoRA 有所不同，个别边界 case 可能在 Full 训练中才会暴露。
- **误区二：认为 Full 训练"更强"所以直接跳过 LoRA 阶段验证，节省时间**——实践反复证明，跳过 LoRA 验证直接上 Full，一旦出问题（尤其是奖励函数设计缺陷这类需要多轮迭代才能发现的问题），排查和试错成本远高于先做 LoRA 验证所"节省"的时间，属于典型的"欲速则不达"。
- **误区三：LoRA 和 Full 训练使用完全相同的 `num_generations`、`max_completion_length` 等生成相关参数，未考虑 Full 训练资源需求更高、可能需要适当缩减这些参数以保证资源可行性**——建议按第4章资源规划重新核算。

## 第十九章 完整可运行脚本合集

本章汇总本手册涉及的核心场景脚本索引，均已在前述章节详细展开，此处作为速查合集：

| 场景 | 章节 | 脚本要点 |
|---|---|---|
| 环境自检 | 3.3 | Python 自检脚本 + NCCL 通信自检 |
| 冒烟测试（Qwen3-4B LoRA） | 7.6 | 单机2卡，200条数据，50 step |
| Qwen3-32B GRPO LoRA（单机8卡） | 8.2 | colocate vLLM，`--fsdp fsdp2` |
| Qwen3-32B GRPO Full（2机16卡） | 9.2 | server vLLM，独立 rollout 节点 |
| Qwen3.6-35B-A3B FSDP2 MoE 配置 | 10.3 | 自定义 wrap policy yaml |
| 多机分布式环境变量配置 | 15.1 | NNODES/NODE_RANK/MASTER_ADDR |
| 断点续训 | 16.3 | `--resume_from_checkpoint` |
| 线下 GSM8K 评估 | 17.2 | `swift eval --eval_dataset gsm8k` |

生产环境建议将上述脚本统一纳入版本控制（git），每次实验的脚本、环境快照（`env.txt`）、wandb 实验链接三者一一对应存档，形成可追溯、可复现的实验管理体系，这是保证团队协作下"收敛问题可复现、可排查"的重要工程基础设施投入。


## 第二十章 常见问题 FAQ

**Q1：为什么我的 GRPO 训练 loss 一直在 0 附近波动，看起来"没有在学习"，这正常吗？**

正常。GRPO 的 policy loss 与 SFT 的 cross-entropy loss 性质完全不同，不应期待它单调下降。判断训练是否有效，应看 `reward/mean`、验证集 reward、线下评估 accuracy 等指标，而非 loss 本身（详见第12章）。

**Q2：`--fsdp fsdp2` 和 `--deepspeed zero3` 可以同时使用吗？**

不可以，二者是互斥的两条分布式训练路径，同时设置会导致配置冲突或未定义行为，请二选一（选择建议见5.4节）。

**Q3：LoRA 训练时是否还需要单独维护参考模型（ref model）？**

通常不需要单独维护完整的参考模型权重副本。LoRA 场景下，"关闭 adapter 的 base 模型"天然可以作为参考模型（因为 base 模型参数本身被冻结、未发生变化），ms-swift 已对此做了工程优化，无需像 Full 训练那样额外承担参考模型的显存开销。

**Q4：GSM8K 数据量（约7500条训练样本）对 32B/35B 模型做 GRPO 训练是否足够？**

对于提升数学推理能力这一相对聚焦的任务目标，GSM8K 全量数据通常是充足的（甚至部分子集即可观察到明显效果），GRPO 作为在线 RL 算法，同一批 prompt 可以通过多个 epoch 反复采样训练（每次采样的 completion 都不同），数据"利用率"与监督学习的固定标签训练有本质区别。如果观察到训练早期（如1个epoch内）reward 就已经趋于平台期不再提升，可以考虑引入更高难度的数学数据集（如 MATH、NuminaMath 等）做混合训练或课程学习，而不是单纯归因于"GSM8K数据量不够"。

**Q5：Qwen3.6-35B-A3B 这样的 MoE 模型，是否一定要用 Megatron-SWIFT + 专家并行才能训练，纯 FSDP2 路线可行吗？**

纯 FSDP2 路线可行（本手册第10章即针对此路线给出方案），但需要额外关注 wrap policy、权重同步、路由稳定性等 MoE 特有问题（详见10.1~10.5节）。如果团队已有 Megatron-SWIFT 的 GRPO 训练经验（需确认所用版本对 GRPO 的支持程度，官方文档中 Megatron 路线的 GRPO 支持通常滞后于 SFT），且追求更极致的 MoE 训练效率（官方数据显示 EP 路线可带来近10倍的 SFT 训练加速，GRPO 场景加速比可能因 rollout 环节占比不同而有所差异），可以评估切换到 Megatron-SWIFT 路线，但需要重新走一遍本手册的冒烟测试和收敛验证方法论，不能假设两条技术路线的收敛行为完全一致。

**Q6：训练过程中人工抽查 `log_completions` 发现模型偶尔输出英文夹杂中文、或者语言风格突变，是否说明训练有问题？**

需要结合 KL 散度、entropy 等指标综合判断。GRPO 训练本身会持续调整策略偏离参考模型，轻微的语言风格变化（如更简洁、更结构化的表达）是正常的策略演化。但如果伴随语言能力明显退化（表达不连贯、语法错误增多）、且 KL 散度在同一时期出现异常增长，则属于第13.2节描述的策略发散前兆，需要及时干预。

**Q7：`--overlong_filter true` 具体起什么作用，什么情况下应该关闭？**

该参数会将因超出 `max_completion_length` 而被截断的回答从优势（advantage）计算中排除，避免"回答本来可能是对的，但因为长度限制被截断而错误地拿到负奖励"这类噪声信号污染训练。GSM8K 场景（尤其开启 thinking 模式时）建议默认开启。极少数场景下，如果你的奖励函数本身已经妥善处理了截断情况（比如对截断样本给予中性而非负向奖励），或者你有意希望通过负反馈抑制模型生成过长回答的倾向，可以考虑关闭，但需要密切监控 `completion_length` 指标是否失控增长。

**Q8：多机训练时，如果某个节点的 GPU 型号与其他节点不同（异构集群），是否会影响 GRPO 训练收敛？**

强烈不建议在异构 GPU 集群上做单次 FSDP2 分布式训练（不同型号 GPU 的计算精度实现、显存带宽差异可能导致训练速度不均衡、甚至数值行为的细微差异累积成系统性问题）。如果只能使用异构集群，建议将训练侧和 rollout 侧分离部署在不同型号的 GPU 上（例如用较新型号做训练、较旧型号专门做 vLLM rollout server），而不是让同一个 FSDP2 训练组内混合不同型号的 GPU。

## 第二十一章 附录

### 21.1 GRPO 核心参数速查表

| 参数 | 含义 | GSM8K LoRA 经验值 | GSM8K Full 经验值 |
|---|---|---|---|
| `--learning_rate` | 学习率 | 3e-5~5e-5 | 1e-6~3e-6 |
| `--beta` | KL惩罚系数 | 0.01~0.04 | 0.001~0.01 |
| `--num_generations` | 组大小G | 8 | 8 |
| `--temperature` | 采样温度 | 0.8~1.0 | 0.8~1.0 |
| `--top_p` | 采样top-p | 0.9~1.0 | 0.9~1.0 |
| `--epsilon` / `--epsilon_high` | clip裁剪范围 | 0.2 / 0.2~0.28 | 0.2 / 0.2~0.28 |
| `--max_grad_norm` | 梯度裁剪阈值 | 1.0 | 0.3~0.5 |
| `--warmup_ratio` | 学习率warmup比例 | 0.05 | 0.1 |
| `--overlong_filter` | 超长截断过滤 | true | true |
| `--fsdp` | 分布式后端 | fsdp2 | fsdp2 |

### 21.2 术语表

- **GRPO（Group Relative Policy Optimization）**：一种通过组内相对奖励替代价值函数（Critic）的策略梯度强化学习算法。
- **Rollout**：策略模型针对给定 prompt 采样生成候选回答的过程。
- **Advantage（优势）**：某个回答相对于同组其他回答的相对好坏程度，GRPO 中通过组内奖励归一化计算。
- **KL 散度（KL Divergence）**：衡量当前策略与参考模型输出分布差异的指标，用作训练稳定性的正则化约束。
- **FSDP2（Fully Sharded Data Parallel v2）**：PyTorch 原生的新一代全分片数据并行训练方案，基于 DTensor 实现。
- **on-policy / off-policy**：训练所用数据是否由当前（或接近当前）策略采样得到，GRPO 理论上要求较高的 on-policy 程度。
- **Reward Hacking（奖励黑客）**：策略模型学会利用奖励函数设计缺陷获取高奖励，而非真正提升目标能力的现象。
- **专家并行（Expert Parallelism, EP）**：MoE 模型训练中将不同专家分布到不同设备上的并行策略。
- **路由坍塌（Router Collapse）**：MoE 模型训练中路由退化为只使用少数专家、导致模型有效容量萎缩的现象。

### 21.3 参考资料索引

- ms-swift 官方文档（Command-line Parameters / GRPO GetStarted）：https://swift.readthedocs.io/
- ms-swift GitHub 仓库与 Issues（包含大量社区实践反馈与踩坑记录）：https://github.com/modelscope/ms-swift
- Qwen 官方 ms-swift 训练指南：https://qwen.readthedocs.io/en/latest/training/ms_swift.html
- DeepSeekMath 论文（GRPO 算法原始出处）
- DeepSeek-R1 技术报告（GRPO 在大规模推理模型训练中的应用实践）

---

## 结语

本手册系统梳理了在 ms-swift 4.4.0 框架下，使用 FSDP2 分布式训练方案，对 Qwen3-32B（Dense）与 Qwen3.6-35B-A3B（MoE）模型进行 GRPO LoRA / Full 全参训练时，保证训练稳定收敛所需的完整工程方法论——从环境搭建、资源规划，到数据与奖励函数设计、FSDP2 精细化配置、超参数选择依据、训练过程监控体系，再到收敛失败模式的系统化诊断与应对，最后延伸到多机分布式、checkpoint 管理、线下评估闭环等生产级工程实践。

**核心方法论可以浓缩为四句话**：
1. 先正确，后收敛——90%的"不收敛"问题源于配置错误而非算法调参；
2. 小规模验证，大规模复现——任何新配置先在小模型/小数据上跑通再放大；
3. 指标先行，直觉在后——建立完整的多指标联合监控体系，而非仅凭经验判断；
4. 密集存档，随时回滚——checkpoint 策略是大规模训练风险控制的最后一道防线。

祝训练顺利，收敛稳定。