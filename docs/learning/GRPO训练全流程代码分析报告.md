# ms-swift GRPO 训练全流程代码分析报告

> 分析对象：当前仓库提交 7749cc643（2026-07-30）<br>
> 报告定位：面向源码阅读与二次开发，重点解释真实调用链、数据流、算法实现、分布式后端和扩展入口。<br>
> 分析方法：以静态代码追踪为主，并交叉检查配置、示例与测试；第 25、26 节还包含 Qwen3-1.7B 在当前 8GB GPU 环境的三步实际运行记录。第 27、28 节的 Qwen3.5-35B-A3B Megatron/FSDP2 场景由于当前环境无法运行，严格按源码、官方示例和同架构配置进行推导，不将推导性能当作实测结果。

## 0. 如何使用这份报告

如果目标是先建立全局认识，建议按以下顺序阅读：

1. 先读第 1～3 节，建立“三条实现路径”和主流程调用链。
2. 再读第 5～10 节，理解一条样本从 prompt 到 loss 的完整生命周期。
3. 根据实际后端选择第 12、13 或 14 节。
4. 调参时查第 15、17 节；排错时直接查第 19 节。
5. 要深入改代码时，从第 20、21 节选择专题和源码入口。
6. 要部署 Qwen3.5-35B-A3B MoE 时，先读第 27 节的 Megatron 四场景，再读第 28 节的 FSDP2 对应路径与逐项对照。

本文以最常用的 Transformers/TRL 路径为主线：

~~~bash
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset AI-MO/NuminaMath-TIR \
  --reward_funcs accuracy format
~~~

Megatron 和 Ray Megatron 并非简单包装，而是各自拥有训练循环；本文单独说明它们与主线共享的语义和不同的调度方式。

---

## 1. 核心结论

### 1.1 仓库中有三条 GRPO 训练路径

| 路径 | 命令/入口 | 训练循环 | rollout 方式 | 适用场景 |
|---|---|---|---|---|
| Transformers/TRL | swift rlhf | Hugging Face Trainer + Swift GRPOTrainer | Transformers、vLLM colocate、vLLM server | 默认主线，功能最全，便于调试和扩展 |
| Megatron | swift megatron rlhf | Megatron pipeline schedule | vLLM colocate/server | 大模型 TP/PP/CP/EP、MoE、高吞吐 |
| Ray Megatron | swift megatron rlhf --use_ray | Ray driver + Megatron workers + rollout replicas | vLLM 资源组 | 训练与推理解耦、异构资源、独立扩缩容 |

三条路径共享以下概念：

- 一个 prompt 生成 G 个 completion；
- 多个奖励按权重合并；
- 同组奖励构造 advantage；
- 当前策略、旧策略、参考策略和 rollout 策略的 token log-prob 各有明确用途；
- 通过 PPO/GRPO 风格比率裁剪、KL 约束和不同归一化方式计算 loss；
- 支持部分 DAPO、GSPO、RLOO、REINFORCE++、CISPO、SAPO、FIPO、REAL 等扩展。

但它们不共享同一训练循环，功能覆盖也不完全相同。阅读代码时，首先必须确认自己所在的后端。

### 1.2 GRPO 的真正主循环不是“生成一次、更新一次”

主线实现把一次较大的 generation batch 做 rollout、奖励和优势计算，然后切成若干训练 micro-batch 放入缓存。由 **steps_per_generation** 控制一次 rollout 覆盖多少优化步，由 **num_iterations** 控制同一批 completion 被复用多少轮。

因此调试时要区分：

- rollout step：产生新 response；
- optimizer/global step：对缓存样本进行梯度更新；
- iteration：同一 rollout 数据被策略重复消费；
- gradient accumulation step：Trainer 内部的梯度累积。

这些量混为一谈，会直接造成对吞吐、off-policy 程度、权重同步频率和日志步数的误判。

### 1.3 最重要的数据设计：先保留原始 messages，rollout 后再编码

GRPO 数据集不会像普通 SFT 那样在进入 Trainer 前统一 token 化。样本在 rollout 前保持为标准 messages 和额外字段，生成完成后才把精确的 response token ID 与 loss mask 注入模板编码。

这样做有四个关键作用：

1. 奖励函数可直接读取 completion、solution 和任意额外数据列；
2. 多模态、工具调用、多轮环境仍保留结构化消息；
3. 训练 token 与 vLLM 实际采样 token 对齐，避免字符串解码再编码造成漂移；
4. 可保留 rollout log-prob，用于训练/推理偏差诊断和重要性采样修正。

### 1.4 必须分清四类逐 token 概率

| 名称 | 产生时机 | 是否有梯度 | 主要用途 |
|---|---|---:|---|
| per_token_logps | loss 前向时，用当前模型重新计算 | 是 | 策略梯度主体 |
| old_per_token_logps | rollout 后、更新前，用当前策略无梯度计算 | 否 | PPO/GRPO 比率的分母 |
| ref_per_token_logps | 参考模型或禁用 LoRA 后的基座模型 | 否 | KL 约束 |
| rollout_per_token_logps | vLLM 实际采样时返回 | 否 | 监控和修正训练引擎与 rollout 引擎的分布偏差 |

前三者属于训练算法语义，第四个主要处理引擎差异与策略陈旧问题。不能把 old policy 与 rollout policy 简单视为同一对象。

---

## 2. 源码总地图

### 2.1 Transformers/TRL 主路径

| 层次 | 关键文件 | 作用 |
|---|---|---|
| 命令分发 | [swift/cli/main.py](../../swift/cli/main.py#L13) | 将 rlhf 路由到对应模块；需要时启动 torch distributed |
| RLHF CLI | [swift/cli/rlhf.py](../../swift/cli/rlhf.py#L1) | 调用 rlhf_main |
| 通用 pipeline | [swift/pipelines/base.py](../../swift/pipelines/base.py#L14) | 解析参数、设置随机种子、执行 run |
| RLHF pipeline | [swift/pipelines/train/rlhf.py](../../swift/pipelines/train/rlhf.py#L35) | 准备策略/参考/奖励模型、模板和 Trainer 参数 |
| 训练公共流程 | [swift/pipelines/train/sft.py](../../swift/pipelines/train/sft.py#L194) | 数据集、模型、Trainer 实例化、train/save |
| Trainer 工厂 | [swift/trainers/trainer_factory.py](../../swift/trainers/trainer_factory.py) | rlhf_type=grpo 映射到 GRPOTrainer/GRPOConfig |
| 参数定义 | [swift/arguments/rlhf_args.py](../../swift/arguments/rlhf_args.py#L170) | CLI 参数聚合、默认值、兼容性检查 |
| Trainer 配置 | [swift/rlhf_trainers/arguments.py](../../swift/rlhf_trainers/arguments.py#L97) | GRPOConfig 与 TRL 配置对接 |
| GRPO 主体 | [swift/rlhf_trainers/grpo_trainer.py](../../swift/rlhf_trainers/grpo_trainer.py#L73) | rollout、奖励、优势、逐 token loss、日志 |
| rollout 混入 | [swift/rlhf_trainers/rollout_mixin.py](../../swift/rlhf_trainers/rollout_mixin.py) | vLLM 初始化、权重同步、生成、offload |
| rollout 客户端 | [swift/rlhf_trainers/vllm_client.py](../../swift/rlhf_trainers/vllm_client.py) | 连接外部 rollout server、请求分发与权重传输 |
| vLLM engine | [swift/infer_engine/grpo_vllm_engine.py](../../swift/infer_engine/grpo_vllm_engine.py) | 统一 vLLM 输出为 GRPO 所需结构 |
| 奖励注册 | [swift/rewards/orm.py](../../swift/rewards/orm.py) | 内置规则奖励和 orms 注册表 |
| 奖励模型插件 | [swift/rewards/rm_plugin.py](../../swift/rewards/rm_plugin.py) | 分类 RM、生成式 RM 的统一接口 |
| 多轮调度 | [swift/rollout/agent_loop.py](../../swift/rollout/agent_loop.py) | 多轮 agent/environment 循环 |
| Gym 环境 | [swift/rollout/gym_env.py](../../swift/rollout/gym_env.py) | Env 接口、环境注册与 total_reward |

### 2.2 Megatron 与 Ray

| 路径 | 关键文件 | 作用 |
|---|---|---|
| Megatron CLI | [swift/cli/_megatron/rlhf.py](../../swift/cli/_megatron/rlhf.py) | 普通 Megatron 与 --use_ray 分流 |
| Megatron pipeline | [swift/megatron/pipelines/train/rlhf.py](../../swift/megatron/pipelines/train/rlhf.py) | 模型/数据/模板和 Trainer 组装 |
| Megatron GRPO | [swift/megatron/trainers/grpo_trainer.py](../../swift/megatron/trainers/grpo_trainer.py) | 替换数据迭代器、rollout、优势和 pipeline loss |
| Megatron 参数 | [swift/megatron/arguments/megatron_args.py](../../swift/megatron/arguments/megatron_args.py) | TP/PP/CP/EP、batch 与 GRPO 限制 |
| Ray pipeline | [swift/ray/megatron/pipeline.py](../../swift/ray/megatron/pipeline.py) | YAML、资源池、worker/rollout replica 编排 |
| Ray 基类 | [swift/ray/megatron/base_trainer.py](../../swift/ray/megatron/base_trainer.py) | driver 数据、生成上下文、offload 和生命周期 |
| Ray GRPO | [swift/ray/megatron/grpo_trainer.py](../../swift/ray/megatron/grpo_trainer.py) | driver 侧完整训练循环 |
| Ray loss | [swift/ray/megatron/loss/grpo.py](../../swift/ray/megatron/loss/grpo.py) | 复用 Megatron forward_step/loss_func |

---

## 3. Transformers/TRL 端到端调用链

### 3.1 总体流程图

~~~mermaid
flowchart TD
    A["swift rlhf --rlhf_type grpo"] --> B["CLI 路由与分布式启动"]
    B --> C["SwiftRLHF.run"]
    C --> D["加载并标准化原始数据集"]
    C --> E["准备 policy / ref / reward model / template"]
    D --> F["TrainerFactory 创建 GRPOTrainer"]
    E --> F
    F --> G["RepeatSampler: 每个 prompt 重复 G 次"]
    G --> H["_prepare_inputs"]
    H --> I["rollout 生成 completion"]
    I --> J["规则奖励 / RM / Gym 奖励"]
    J --> K["rollout 后编码和精确 token 对齐"]
    K --> L["计算 old/ref/rollout log-prob"]
    L --> M["组内奖励统计与 advantage"]
    M --> N["缓存并切分训练 micro-batch"]
    N --> O["当前策略前向 per_token_logps"]
    O --> P["ratio / clip / KL / mask / normalize"]
    P --> Q["Trainer 反向传播与 optimizer step"]
    Q --> R{"需要新 rollout?"}
    R -- 否 --> N
    R -- 是 --> I
    Q --> S["日志、评估、checkpoint"]
~~~

### 3.2 启动与参数解析

1. [swift/cli/main.py](../../swift/cli/main.py#L13) 将命令名 rlhf 映射到 swift.cli.rlhf。
2. CLI 支持把 YAML/JSON 配置展开为命令行参数。
3. 设置 NPROC_PER_NODE 或多节点参数后，CLI 通过 torch.distributed.run 重启为多进程训练。
4. [swift/cli/rlhf.py](../../swift/cli/rlhf.py) 调用 rlhf_main。
5. [SwiftPipeline](../../swift/pipelines/base.py#L14) 解析 RLHFArguments，并用 seed + rank 设置各进程随机种子。

这里的设计意味着：配置文件、CLI 和 Python API 最终都会进入同一套 dataclass 参数初始化逻辑。定位“参数为什么被改写”时，应先看 RLHFArguments 的 post-init，而不是只看 shell 命令。

### 3.3 Pipeline 组装

[SwiftRLHF](../../swift/pipelines/train/rlhf.py#L35) 继承通用 SFT pipeline，但为 GRPO 增加：

- 策略模型与参考模型；
- 一个或多个奖励模型及各自模板；
- vLLM 客户端或本地 engine 信息；
- reward_funcs、reward_weights；
- 可选 CHORD SFT 数据集；
- GRPO 使用的训练模板状态。

[SwiftSft.run](../../swift/pipelines/train/sft.py#L194) 的大体顺序是：

1. 准备数据集；
2. 保存解析后的参数；
3. 准备模型、tokenizer 和 tuner；
4. 由 TrainerFactory 选择 GRPOTrainer 和 GRPOConfig；
5. 构造 Trainer；
6. 执行 trainer.train；
7. 保存状态和最终 checkpoint。

### 3.4 Trainer 的继承关系

[GRPOTrainer](../../swift/rlhf_trainers/grpo_trainer.py#L73) 的继承结构为：

~~~text
RolloutTrainerMixin
    + SwiftMixin
    + trl.GRPOTrainer
~~~

它删除并覆盖了上游的部分方法，用 Swift 的模型/模板/日志/rollout 行为包裹 TRL Trainer。可以这样理解职责边界：

- TRL/HF Trainer：dataloader、训练 step、优化器、scheduler、checkpoint 基础框架；
- SwiftMixin：Swift 模型生态、模板、回调、指标和保存适配；
- RolloutTrainerMixin：推理后端、权重同步、生成和多轮调度；
- Swift GRPOTrainer：奖励、优势、逐 token loss、GRPO 扩展算法。

---

## 4. 参数初始化不是被动存储

GRPO 的 dataclass 初始化会主动推导和校验配置。最关键逻辑位于：

- [swift/arguments/rlhf_args.py](../../swift/arguments/rlhf_args.py#L334)
- [swift/rlhf_trainers/args_mixin.py](../../swift/rlhf_trainers/args_mixin.py#L99)
- [swift/rlhf_trainers/arguments.py](../../swift/rlhf_trainers/arguments.py#L97)

### 4.1 关键默认值

| 参数 | 典型默认/推导 | 代码含义 |
|---|---|---|
| num_generations | 8 | 每个 prompt 的采样数 G |
| beta | 0.04 | 参考策略 KL 系数 |
| loss_type | grpo | 默认逐 token GRPO loss |
| gradient_accumulation_steps | 1 | GRPO 未显式设置时的默认值 |
| truncation_strategy | left | 长 prompt 左截断；也允许 delete |
| remove_unused_columns | false | 保留 solution 等奖励所需额外列 |
| scale_rewards | GRPO: group | 组内标准差归一化 |
| kl_in_reward | GRPO: false | 默认把 KL 加入 loss，而非奖励 |
| generation_batch_size | global batch × steps_per_generation | 一次 rollout 的全局 completion 数 |

注意：num_generations 至少为 2；GRPOConfig 还要求较新的 TRL。代码不同位置存在不完全相同的最低版本提示，实际以 [GRPOConfig](../../swift/rlhf_trainers/arguments.py#L97) 的 trl>=0.26 约束为准。

### 4.2 batch 关系

定义：

- D：world size；
- B：per_device_train_batch_size；
- S：steps_per_generation；
- G：num_generations；
- GB：普通全局训练 batch，GB = B × D；
- R：一次 rollout 的全局 generation batch，R = GB × S。

约束：

~~~text
generation_batch_size = per_device_train_batch_size × world_size × steps_per_generation
generation_batch_size % num_generations = 0
~~~

因此一次 rollout 中不同 prompt 的数量约为 R / G。

例：D=8、B=2、S=4、G=8，则：

~~~text
GB = 2 × 8 = 16
R  = 16 × 4 = 64 completions
prompt 数 = 64 / 8 = 8
~~~

这 64 个 completion 会被切成 S 个全局训练 batch；若 num_iterations 大于 1，还会被重复用于更多优化步。

### 4.3 重要兼容性约束

- use_vllm=true 时必须设置 vllm_mode；
- async_generate 只允许 vLLM server 模式；
- async_generate 当前不支持 multi-turn；
- vLLM 与 device_map 多进程模型切分不兼容，应使用正常的分布式进程数；
- 不使用 vLLM 时，vllm_tensor_parallel_size 会被校正为 1；
- Liger 路径当前不支持 delta、sequence parallel、padding-free、entropy mask、off-policy sequence mask 等若干功能；
- REAL 强制关闭 reward normalization；
- cached_dataset/cached_val_dataset 不支持 GRPO，因为 rollout 后编码依赖训练期动态生成结果。

---

## 5. 数据生命周期

### 5.1 输入数据格式

典型数学数据可写成：

~~~json
{
  "messages": [
    {"role": "system", "content": "按要求推理并作答"},
    {"role": "user", "content": "题目内容"}
  ],
  "solution": "标准答案",
  "difficulty": "hard"
}
~~~

要求和行为：

- messages 最后通常是 user prompt；
- solution、ground_truth、difficulty 等额外列不会被删除；
- 奖励函数会收到 completion、trainer_state 和所有额外列；
- 若输入已有 assistant response，rollout 预处理会移除旧 response，再生成新 response。

### 5.2 为什么 GRPO 跳过训练前统一编码

[SwiftSft._prepare_dataset](../../swift/pipelines/train/sft.py#L125) 对 GRPO 设置 pre_process=false。GRPOTrainer 的 data collator 也是 identity collator。

这意味着 dataloader 先传递 Python 结构化样本，而不是 input_ids。直到生成与奖励结束，[GRPOTrainer._prepare_batch_inputs](../../swift/rlhf_trainers/grpo_trainer.py#L795) 才：

1. 将 rollout 实际返回的 response_token_ids 写回消息；
2. 将 response_loss_mask 写回模板输入；
3. 执行 template.encode；
4. collate 为张量；
5. 由 labels != -100 得到 completion_mask；
6. 计算训练所需的旧策略与参考策略概率。

### 5.3 字段演化表

| 阶段 | 核心字段 | 说明 |
|---|---|---|
| 数据集输出 | messages、solution、其他列 | 原始结构化样本 |
| 重复采样 | 同一 prompt 重复 G 次 | 保证组内比较 |
| rollout 预处理 | prompt_id、request_id | prompt_id 标识同 prompt；request_id 标识一次请求 |
| rollout 输出 | messages + assistant、response_token_ids | 实际生成内容与精确 token |
| 多轮输出 | response_loss_mask、rollout_infos | 区分模型动作/环境观察，记录回合信息 |
| vLLM 输出 | rollout_per_token_logps | 实际采样分布概率 |
| 编码后 | input_ids、attention_mask、labels | 模型前向输入 |
| mask | completion_mask、truncated_mask | 仅 completion token 参与 RL loss；可过滤超长样本 |
| 概率 | old_per_token_logps、ref_per_token_logps | 比率和 KL |
| 优势 | advantages | 每条 completion 一个标量，loss 中广播到 token |
| 全局归一化 | num_items_in_batch | DAPO/CISPO/FIPO 使用的全局 token 数 |

### 5.4 RepeatSampler 的职责

训练 sampler 以 mini_repeat_count=num_generations 重复每个 prompt，并让一个组在全局 generation batch 中保持可恢复的顺序。普通路径主要复用 TRL 的 RepeatSampler；启用 sequence parallel 时 Swift 显式构造 sampler，并将 SP world size 纳入 repeat_count。

它解决的不是普通的数据增强，而是 GRPO 的统计前提：同一 prompt 的 G 个结果必须能被 reshape 或按 prompt_id 重组，才能计算组均值和组标准差。

---

## 6. rollout：从 prompt 到 completion

### 6.1 生成入口

[GRPOTrainer._prepare_inputs](../../swift/rlhf_trainers/grpo_trainer.py#L186) 在训练状态下决定：

- 当前 step 是否需要产生一批新 rollout；
- 或从 buffered_inputs 中返回当前 micro-batch。

需要新数据时进入：

~~~text
_generate_and_score_completions
  ├─ resample_encode_failed_inputs
  ├─ _generate_completions
  ├─ _score_completions
  ├─ _dynamic_sampling（可选）
  ├─ _prepare_batch_inputs
  ├─ _compute_advantages
  └─ split_by_mini_batches
~~~

[GRPOTrainer._generate_completions](../../swift/rlhf_trainers/grpo_trainer.py#L214) 先补 prompt_id/request_id 和系统消息，然后选择 fast infer 或 TransformersEngine。

### 6.2 三种同步 rollout 后端

| 后端 | use_vllm | vllm_mode | 模型/显存关系 | 特点 |
|---|---:|---|---|---|
| TransformersEngine | false | 无 | 训练模型直接生成 | 部署最简单，速度通常较慢，适合调试 |
| vLLM colocate | true | colocate | 训练和 vLLM 共享节点/GPU | 权重同步快，可 sleep/offload，资源竞争需要精调 |
| vLLM server | true | server | 独立 swift rollout 服务 | 训练/推理解耦，支持独立扩展和多机，需网络与权重同步 |

RolloutTrainerMixin 会构造统一的 RequestConfig，包含 max_tokens、temperature、top_p、top_k、repetition_penalty、stop words、结构化输出规则等。vLLM 路径还请求 logprobs。

### 6.3 colocate 模式

[RolloutTrainerMixin._prepare_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L195) 检查：

- world size 必须可被 vllm_tensor_parallel_size 整除；
- 按 TP 大小创建 rollout 子组；
- 需要时在生成前 offload 训练模型或优化器；
- 初始化 GRPOVllmEngine；
- sleep_level 大于 0 时让 engine 释放部分或全部占用。

生成前，策略权重被同步到 vLLM；生成结束后再 sleep 并恢复训练资源。LoRA 可选择只同步 adapter，full tuning 或特定场景会做全量权重同步。

### 6.4 server 模式

训练主进程通过 [VLLMClient](../../swift/rlhf_trainers/vllm_client.py) 连接一个或多个 swift rollout 服务：

1. 获取 server 能力：同步/异步 engine、多轮、LoRA、Gym；
2. 聚合各训练 rank 的请求；
3. 分配给 rollout server；
4. 收集并广播结果；
5. 通过 NCCL/PyNccl 或相应通信后端传输新权重。

外部服务入口位于 [swift/pipelines/infer/rollout.py](../../swift/pipelines/infer/rollout.py)，包含服务 API 和 WeightSyncWorkerExtension。

### 6.5 async_generate 的准确含义

server 模式不等于 async_generate。

- server 模式可以同步等待当前权重生成；
- async_generate=true 才启用训练与下一批生成的流水重叠；
- 这种模式可能用较旧策略生成，因此代码明确将其视为近似加速模式；
- 当前只支持 server，且不支持 multi-turn。

评估稳定性时，应把“引擎数值差异”和“策略陈旧”分开看。前者可由 rollout_per_token_logps 观测，后者与异步流水和重复训练轮数直接相关。

### 6.6 多轮 agent loop

[agent_loop.py](../../swift/rollout/agent_loop.py) 提供后端无关的循环：

~~~text
trajectory start
  → 模型生成一轮
  → scheduler.on_turn_end
  → 环境/工具 step
  → scheduler.check_finished
  → 未结束则把观察加入 messages 并继续
~~~

它累计：

- response_token_ids；
- response_loss_mask；
- rollout_logprobs；
- rollout_infos；
- num_turns 等轨迹信息。

环境观察通常可通过 loss mask 排除，保证只对模型动作 token 计算策略 loss。所有分布式 rank 必须一致结束，以防 collective 通信死锁。

---

## 7. 奖励系统

### 7.1 奖励计算链

[GRPOTrainer._score_completions](../../swift/rlhf_trainers/grpo_trainer.py#L305) 统一处理三类来源：

1. 规则或自定义 reward function；
2. reward model；
3. Gym 多轮环境的 total_reward。

每个 reward function 产生一个列向量，最终形成：

~~~text
rewards_per_func: [num_completions, num_reward_functions]
~~~

随后按 reward_weights 逐列加权并以 nansum 合并。某个奖励返回 None 会转换为 NaN，从而允许该奖励对特定样本不适用；如果一行所有奖励都为 None，代码会打印包含 completion 和输入字段的警告。

### 7.2 内置规则奖励

[swift/rewards/orm.py](../../swift/rewards/orm.py) 注册的主要奖励包括：

| 名称 | 作用 | 典型输入 |
|---|---|---|
| accuracy / math | 数学答案解析与等价验证 | completion、solution |
| format | 检查 think/answer 等输出格式 | completion |
| react_format | 检查 ReAct 格式 | completion |
| toolbench | 工具调用/ReAct 轨迹评分 | completion、任务字段 |
| cosine | 将正确性与 completion 长度组合成余弦型奖励 | solution、response_token_ids |
| repetition | n-gram 重复惩罚 | completion |
| soft_overlong | 接近最大长度时的平滑惩罚 | response_token_ids |

内置名称最终由 orms 注册表解析。阅读某个 reward 的真实取值范围时，应直接看对应类的 __call__，不要仅凭名称推断。

### 7.3 Reward model

[rm_plugin.py](../../swift/rewards/rm_plugin.py) 将奖励模型包装成与规则奖励一致的调用方式：

- 分类式 RM：通常读取 logits[:, 0]；
- 生成式 RM：构造 judge prompt，通过推理 engine 生成，再解析 Reward；
- 每个 RM 可使用独立模板；
- 分布式下会按 DeepSpeed/FSDP/Accelerate 的模型准备方式包装。

在 Transformers 主路径中，reward model 和规则奖励可以同时使用。Megatron/Ray 的支持范围不同，选后端前需要核对第 13、14 节。

### 7.4 自定义奖励插件

参考 [examples/train/grpo/plugin/plugin.py](../../examples/train/grpo/plugin/plugin.py)。插件通过 external_plugins 在参数初始化时导入，再向 orms 注册名称。

基本模式：

~~~python
class MyReward:
    def __call__(self, completions, solution=None, trainer_state=None, **kwargs):
        return [score_one(x, y) for x, y in zip(completions, solution)]

orms["my_reward"] = MyReward
~~~

实现时应注意：

- 返回长度必须等于 completion 数；
- 允许返回 None，但不要让同一样本的全部奖励都为 None；
- 额外数据列会原样传入，应显式处理缺失值；
- 奖励最好保持可解释的范围，并分别记录均值/标准差；
- async reward function 会放入后台事件循环并并发执行，适合网络 judge 或 I/O 密集评分；
- CPU 密集 Python 逻辑不一定因 async 获益。

### 7.5 Gym 环境奖励

[gym_env.py](../../swift/rollout/gym_env.py) 定义 reset、step、close 等环境接口。Gym scheduler 在多轮 rollout 中执行环境交互，并把累计奖励放入 rollout_infos.total_reward。

GRPOTrainer 会把 total_reward 作为额外奖励列，可再与其他 reward function 通过 reward_weights 混合。此时“奖励是在生成完后一次性打分”已不完全成立：环境每一轮都可能产生局部回报，最后再汇总。

---

## 8. rollout 后编码与概率计算

### 8.1 精确 token 对齐

字符串 completion 不是最终训练真值；response_token_ids 才是。代码将 rollout engine 返回的 token ID 直接交给模板编码，避免：

~~~text
vLLM token IDs
  → decode 为字符串
  → tokenizer 再 encode
  → token 边界变化
~~~

多轮时 response_loss_mask 与 token 同步拼接，使工具结果、环境观察或不希望训练的片段被排除。

### 8.2 四类 log-prob 的时间关系

~~~mermaid
sequenceDiagram
    participant R as "rollout engine"
    participant P0 as "更新前 policy"
    participant Ref as "reference"
    participant P as "训练中的 policy"
    R->>R: "采样并记录 rollout_per_token_logps"
    P0->>P0: "no_grad 计算 old_per_token_logps"
    Ref->>Ref: "no_grad 计算 ref_per_token_logps"
    P->>P: "有梯度前向计算 per_token_logps"
    P->>P: "current/old 构造策略比率"
    P->>P: "current/ref 构造 KL"
    P->>P: "old/rollout 诊断或修正偏差"
~~~

#### per_token_logps

在 _compute_loss_and_metrics 内用当前策略前向得到，保留梯度。所有策略 loss 最终都必须通过它把梯度传回模型。

#### old_per_token_logps

rollout 准备阶段用当前策略 no_grad 计算。若训练完全 on-policy 且不需要缓存旧概率，代码可用 per_token_logps.detach 代替。

它参与：

~~~text
log_ratio = per_token_logps - old_per_token_logps
ratio = exp(log_ratio)
~~~

#### ref_per_token_logps

由独立 ref_model，或 LoRA 场景下暂时禁用 adapter 的基座模型产生。beta=0 时可完全跳过参考模型。

#### rollout_per_token_logps

由 vLLM 在采样时返回。它可能与 old_per_token_logps 不同，原因包括：

- vLLM 与 Transformers 内核/精度差异；
- 权重同步时点不同；
- async_generate 使用旧权重；
- LoRA merge、量化、MoE 路由等实现差异。

代码可记录 KL、PPL、chi-square、ESS 等指标，也可用 token/sequence truncate 或 mask 做重要性采样修正。

---

## 9. 奖励、优势与 KL

### 9.1 奖励聚合

对第 i 个 completion：

~~~text
r_i = Σ_j weight_j × reward_ij
~~~

NaN 奖励列被忽略。若 kl_in_reward=true 且 beta 非零，再执行：

~~~text
r_i ← r_i - beta × Σ_t mask_it × (old_logp_it - ref_logp_it)
~~~

注意这个 reward-side KL 使用的是 old 与 ref 的简单 log-ratio 求和；默认 GRPO 则在 loss 中使用 K3 KL 估计。

### 9.2 GRPO advantage

同一 prompt 的 G 个奖励为 r_1 … r_G：

~~~text
mean_g = mean(r_1 … r_G)
A_i = r_i - mean_g
~~~

默认 scale_rewards=group 时：

~~~text
A_i ← A_i / (std(r_1 … r_G) + 1e-4)
~~~

这带来两个直接结论：

- 同组奖励完全相同时，优势为 0，无法产生有效策略梯度；
- reward 的绝对平移不改变 advantage，但 reward 方差和 normalization 会影响梯度尺度。

日志中的 frac_reward_zero_std 是判断奖励是否缺少区分度的关键指标。

### 9.3 RLOO 与 REINFORCE++

RLOO 使用 leave-one-out baseline：

~~~text
A_i = G/(G-1) × (r_i - group_mean)
~~~

默认不做 reward 标准差缩放，并默认把 KL 放进 reward。

REINFORCE++ 仍使用 group mean baseline，但默认使用 batch 级 advantage 标准差做 whitening；它归一化的是 advantage，而标准 GRPO/RLOO 归一化使用原 reward 的标准差。

### 9.4 GDPO 多奖励归一化

scale_rewards=gdpo 时，各 reward function 先在组内独立标准化，再按 reward_weights 合并，最后对总 advantage 做 batch 标准化。

它适合多个奖励尺度差异很大的情况，但会改变各奖励原始量纲对加权的意义。调试时必须同时观察每个 rewards/name/mean、std，不能只看总 reward。

### 9.5 动态多轮分组

普通模式直接按 num_generations reshape。多轮或动态返回条数变化时，代码使用 request_id 去重，并按 prompt_id 重组统计，避免假设每个请求严格产生固定数量结果。

这也是 prompt_id 与 request_id 同时存在的原因：

- prompt_id：说明哪些轨迹属于同一个原始问题；
- request_id：说明哪条结果来自哪次具体 rollout 请求。

---

## 10. loss、反向传播与样本复用

### 10.1 标准 GRPO/PPO 风格 loss

对每个 token：

~~~text
ratio_t = exp(logp_t - old_logp_t)
clipped_ratio_t = clip(ratio_t, 1-epsilon_low, 1+epsilon_high)
policy_loss_t = -min(ratio_t × A, clipped_ratio_t × A)
~~~

如果设置 delta，还会对未裁剪分支增加上界。优势是 sequence 标量，在 token 维度广播。

### 10.2 loss-side KL

默认 GRPO 的 kl_in_reward=false。代码使用数值截断后的 K3 形式：

~~~text
x_t = clamp(ref_logp_t - current_logp_t)
KL_t = exp(x_t) - x_t - 1
loss_t ← policy_loss_t + beta × KL_t
~~~

它与简单的 current_logp-ref_logp 不是同一个估计式。排查 KL 时，必须先确认 kl_in_reward 的位置。

### 10.3 importance sampling 粒度

| importance_sampling_level | 权重构造 | 含义 |
|---|---|---|
| token | 每个 token 独立 ratio | 标准逐 token PPO/GRPO |
| sequence | completion token 平均 log-ratio，再广播 | GSPO 风格序列级权重 |
| sequence_token | detached 序列权重 + 当前 token 梯度路径 | 序列权重、token 级梯度估计 |

sequence 模式降低逐 token 比率波动，但会让一条序列的所有 token 共享更新权重。

### 10.4 loss_type 的差异

| loss_type | 策略项 | 归一化/特殊行为 |
|---|---|---|
| grpo | PPO 风格 min clip | 每条 completion 先 token 平均，再 batch 平均 |
| bnpo | 同上 | 所有有效 token 全局平均 |
| dr_grpo | 同上 | 除以 batch_size × max_completion_length |
| dapo | 同上 | 使用跨进程 completion token 总数归一化 |
| cispo | 上界截断 ratio 后乘 logp 与 advantage | 跨进程 token 总数归一化 |
| sapo | 正负 advantage 使用不同 sigmoid soft gate | 每条 completion token 平均 |
| fipo | GRPO 项再乘未来影响权重 | 跨进程 token 总数归一化 |
| real | 按组构造正负序列 score 的 pairwise/logsumexp loss | 强制 scale_rewards=none |

同样的 ratio/advantage，在不同分母下梯度尺度会随输出长度产生不同偏差。这是 loss_type 不应被当作“一个名字参数”的根本原因。

### 10.5 额外 mask 与修正的执行顺序

核心顺序可概括为：

1. 计算 current log-prob 和可选 entropy；
2. overlong_filter 清除截断 completion；
3. 构造 KL；
4. 构造 current/old 策略比率；
5. 计算选定 loss_type 的逐 token loss；
6. 应用 top entropy mask；
7. 加入 loss-side KL；
8. 应用 rollout importance sampling 权重；
9. 应用 off-policy negative-advantage sequence mask；
10. 依 loss_type 归一化；
11. 可选混入 CHORD SFT loss；
12. 交给 Trainer 反向传播、梯度累积和 optimizer step。

mask 的先后顺序会影响有效 token 数和归一化分母。新增算法时不应只插入一行 loss 公式，还要核对 mask、metrics 和 padding-free/SP 的形状语义。

### 10.6 rollout 缓存和 num_iterations

一次 _generate_and_score_completions 返回多个已编码 micro-batch，并存入 _buffered_inputs。之后 _prepare_inputs 按 step 取切片。

- steps_per_generation 决定一批 rollout 被切成多少训练 step；
- num_iterations 决定同一批数据被复用多少轮；
- 使用旧样本越多，current 与 old 的差异通常越大，clip ratio 和 off-policy 指标更重要；
- 权重只需在真正生成新 rollout 前同步到 vLLM，避免每个 gradient accumulation 子步重复传输。

### 10.7 DAPO 相关机制

- dynamic_sample：组内 reward std 为 0 时丢弃该组并重新采样，最多 max_resample_times；
- overlong_filter：completion 因长度上限被截断时，不让其 token 进入策略 loss；
- soft_overlong：在 soft_cache_length 到 soft_max_length 区间逐步施加长度惩罚；
- dapo loss normalization：按所有进程的有效 completion token 总数归一化；
- epsilon_high 可与 epsilon_low 非对称。

dynamic_sample 提升有效梯度比例，但也改变真实采样分布并增加 rollout 成本；必须同时观察重采样率、reward 方差和生成吞吐。

---

## 11. 参考模型与权重同步

### 11.1 ref model 的来源

- full tuning：默认需要一份冻结参考模型；未显式给 ref_model 时，pipeline 可从策略初始权重构造；
- LoRA：通常不复制完整模型，而是在计算 ref log-prob 时禁用 adapter，使用冻结基座；
- beta=0：不需要 ref log-prob，能节省显存与计算；
- ref_adapter 可指定专用参考 adapter。

### 11.2 sync_ref_model

启用后，回调按 ref_model_sync_steps 同步参考模型：

~~~text
ref ← (1 - alpha) × ref + alpha × policy
~~~

其中 alpha 对应 ref_model_mixup_alpha。它把固定参考策略变成缓慢移动的锚点，会改变 KL 的解释，尤其应在长期训练或分布漂移场景中谨慎使用。

### 11.3 训练权重到 vLLM 的同步

[rollout_mixin.py](../../swift/rlhf_trainers/rollout_mixin.py#L449) 区分：

- full training：全量参数；
- LoRA 且 vLLM 原生 LoRA 可用：只传 adapter；
- 需要 merge 的 PEFT：临时 merge/unmerge；
- ZeRO-3：先 gather 参数；
- FSDP2：取 full tensor；
- MoE：处理专家参数和可能的路由重放；
- server：可按 bucket 分块并传输 flattened weights；
- colocate：直接把状态加载进本地 vLLM engine。

_last_loaded_step 用于避免同一个训练 step 内重复同步。遇到“生成结果没有随训练变化”时，优先检查权重同步 step、adapter 名称、merge 状态和 server 通信，而不是先怀疑优化器。

---

## 12. Transformers 路径的分布式与内存模型

### 12.1 DDP / DeepSpeed / FSDP

HF Trainer/Accelerate 负责数据并行与优化器状态分片；Swift 在 rollout 阶段额外处理：

- 收集各 rank 请求；
- 广播生成结果与奖励统计；
- ZeRO-3/FSDP 权重 gather；
- reward model 的分布式包装；
- 全局 token normalizer；
- 日志指标 gather。

GRPO 比 SFT 多了一套“训练并行 + 推理并行”的组合。训练 world size 与 vLLM tensor parallel size 不一定相同，但 colocate 时后者必须整除前者。

### 12.2 Sequence Parallel 与 padding-free

逐 token log-prob 支持 sequence parallel、padding-free 和多模态分支。启用 SP 后：

- sampler 的重复次数纳入 SP world size；
- completion/token 序列需要在并行组内重组；
- 某些 fused/Liger 路径不可用。

修改 loss 或 mask 时，必须测试：

- 普通 padded batch；
- padding-free 扁平 token；
- SP 切分后的局部序列；
- 多模态额外输入；
- 只有 completion token 的有效区间。

### 12.3 显存生命周期

colocate 模式下最重要的资源循环：

~~~text
训练状态
  → 可选 offload optimizer/model
  → wake vLLM
  → 同步权重
  → rollout
  → 清 cache / sleep vLLM
  → reload model/optimizer
  → 训练状态
~~~

sleep_level、offload_model、offload_optimizer、vllm_gpu_memory_utilization 共同决定是否 OOM 和切换开销。更高 offload 并非总是更快：它以 PCIe/CPU 内存传输换取 GPU 容量。

---

## 13. Megatron GRPO 路径

### 13.1 启动和总体结构

CLI 进入 [swift/cli/_megatron/rlhf.py](../../swift/cli/_megatron/rlhf.py)，未指定 --use_ray 时调用 MegatronRLHF pipeline。

[MegatronGRPOTrainer](../../swift/megatron/trainers/grpo_trainer.py) 不依赖 HF Trainer 的 _prepare_inputs 缓存机制，而是在 Megatron 训练循环中替换数据迭代器：

~~~text
原始 prompt iterator
  → 收集 generation batch
  → rollout / reward / dynamic sample
  → 精确 token 编码
  → old/ref log-prob
  → KL / advantage
  → 切成 Megatron micro-batch
  → 替换 data_iterator
  → forward_step / loss_func
  → pipeline parallel schedule 优化
~~~

关键入口是 _replace_data_iterator。理解 Megatron 路径时，应从它向前追 rollout，向后追 forward_step/loss_func。

### 13.2 并行组

Megatron 支持：

- TP：tensor parallel；
- PP：pipeline parallel；
- CP：context parallel；
- EP：expert parallel；
- DP：data parallel。

GRPO 额外创建 rollout group，把拥有相同 DP 索引、不同 TP/PP/CP rank 的进程组织起来。只有必要阶段/rank 持有完整生成或 loss 所需数据，再通过组内通信同步。

PP 最后一阶段负责基于 logits 计算逐 token log-prob 和 loss；CP 路径还需重组被切开的序列与 mask。

### 13.3 与 Transformers 路径的功能对齐

Megatron 实现保持了大部分核心语义：

- 组内 GRPO/RLOO/REINFORCE++ advantage；
- old/ref/rollout log-prob；
- PPO clip 与主要 loss_type；
- rollout 重要性采样；
- overlong、dynamic sample、entropy 等部分扩展；
- 多轮 agent loop；
- vLLM 权重 bridge；
- MoE 路由信息处理。

但它拥有独立的 loss 实现。修改核心公式时，要同时检查：

- [Transformers GRPO loss](../../swift/rlhf_trainers/grpo_trainer.py#L1090)
- [Megatron GRPO loss](../../swift/megatron/trainers/grpo_trainer.py)
- [共享工具](../../swift/rlhf_trainers/utils.py)

这是仓库中最值得关注的一致性风险之一。

### 13.4 当前限制

以当前代码检查为准，Megatron 参数初始化明确限制：

- 非 Ray Megatron GRPO 需要 use_vllm；
- async_generate 不支持；
- sync_ref_model 不支持；
- num_iterations 大于 1 不支持；
- reward model 支持不如 Transformers 主路径完整；
- 某些算法、padding-free、并行维度组合有额外约束。

不要直接把 Transformers 示例参数复制到 Megatron；应从 [Megatron 示例](../../tests/megatron/test_grpo.py) 或对应 examples 配置起步。

---

## 14. Ray Megatron GRPO 路径

### 14.1 资源编排

--use_ray 将流程交给 [swift/ray/megatron/pipeline.py](../../swift/ray/megatron/pipeline.py)。YAML 把资源分为 train 与 rollout 两类：

- colocate：两组 actor 共享 GPU placement；
- separate：训练和推理使用独立 GPU。

示例：

- [ray_grpo_colocate.yaml](../../examples/ray/grpo/ray_grpo_colocate.yaml)
- [ray_grpo_separate.yaml](../../examples/ray/grpo/ray_grpo_separate.yaml)

### 14.2 driver 侧循环

[Ray GRPO trainer](../../swift/ray/megatron/grpo_trainer.py) 的逻辑比 HF Trainer 更显式：

~~~text
同步 policy 权重
  → 进入 generation context
  → driver 取 prompt batch
  → 每个 prompt 扩展 G 份
  → rollout replicas 生成
  → driver 计算奖励/动态采样
  → 每个 steps_per_generation 分块编码
  → Megatron workers 计算 old/ref log-prob
  → driver CPU 计算 advantage
  → 将 advantage 附到 batch
  → workers.train_step
~~~

BaseRayTrainer 负责循环数据、资源 wake/sleep、训练模型和优化器 offload，以及 worker/replica 生命周期。

### 14.3 loss 复用

[swift/ray/megatron/loss/grpo.py](../../swift/ray/megatron/loss/grpo.py) 通过轻量适配对象复用 MegatronGRPOTrainer 的 forward_step 和 loss_func。因此 Ray 路径的算法一致性主要跟随 Megatron，而不是 Transformers 主路径。

### 14.4 权重同步

- colocate：可利用同机 IPC/共享资源做较直接的权重传输；
- separate：通过 NCCL 等通信把训练权重传给 rollout replicas；
- generation context 会协调训练资源 offload 和 rollout engine wake/sleep。

### 14.5 当前边界

- 多轮当前以 driver-side agent loop 为主，server-side 模式仍有代码限制；
- reward model 能力不像 Transformers 路径那样完整；
- Ray 配置增加了 placement、actor 数、每组 GPU、资源共置等新的失败面；
- 调试应先确认 actor/资源拓扑，再排查 GRPO 算法本身。

---

## 15. 算法扩展功能地图

| 功能 | 主要参数 | 核心修改点 | 深入入口 |
|---|---|---|---|
| 标准 GRPO | advantage_estimator=grpo | 组均值 baseline，默认组标准差缩放 | _compute_advantages |
| RLOO | advantage_estimator=rloo | leave-one-out baseline | _compute_advantages |
| REINFORCE++ | advantage_estimator=reinforce_plus_plus | batch/group advantage whitening | _compute_advantages |
| DAPO | loss_type=dapo、dynamic_sample、overlong_filter | 动态采样、长度处理、全局 token 归一化 | _dynamic_sampling、loss |
| Dr. GRPO | loss_type=dr_grpo | 固定 max completion length 分母 | loss normalization |
| GDPO | scale_rewards=gdpo | 多奖励分别归一化 | _compute_advantages |
| GSPO | importance_sampling_level=sequence | 序列级重要性权重 | log_importance_weights |
| GSPO-token | importance_sampling_level=sequence_token | 序列权重、token 梯度路径 | log_importance_weights |
| CISPO | loss_type=cispo | ratio 上界截断后的策略目标 | loss 分支 |
| SAPO | loss_type=sapo | 正负优势的 soft gate | loss 分支 |
| FIPO | loss_type=fipo | future influence 权重与安全控制 | _compute_fipo_influence |
| REAL | loss_type=real | 组内正负 completion 序列排序 | REAL loss 分支 |
| entropy mask | top_entropy_quantile | 只保留高熵 token | entropy_mask |
| CHORD | chord_sft_dataset 等 | GRPO loss 混合 SFT loss | compute_chord_loss |
| ref EMA | sync_ref_model | 移动参考模型 | SyncRefModelCallback |
| rollout IS | rollout_importance_sampling_mode | 修正 vLLM/训练分布差异 | _apply_rollout_importance_sampling |
| off-policy mask | off_policy_sequence_mask_delta | 过滤偏差过大的负优势序列 | sequence mask |
| 多轮 | multi_turn_scheduler | agent/environment 轨迹 | agent_loop.py |
| Gym | gym_env | 环境累计回报 | gym_env.py |

官方专题文档位于 [docs/source/Instruction/GRPO](../source/Instruction/GRPO/index.rst)，其中 AdvancedResearch 目录可用来补充论文背景；源码仍应以上表入口为最终依据。

---

## 16. 日志、指标与 checkpoint

### 16.1 主要指标族

| 指标 | 应如何解释 |
|---|---|
| reward | 加权总奖励均值 |
| reward_std | 按 scale_rewards 语义统计的奖励波动 |
| frac_reward_zero_std | 同组无区分度的 prompt 比例 |
| rewards/name/mean、std | 单个奖励的健康度和尺度 |
| advantages | 最终进入 loss 的序列级优势 |
| kl | 当前/旧策略与参考策略的偏离，具体定义取决于 kl_in_reward |
| clipping/low、high、region | PPO 裁剪比例；过高常意味着更新过猛或数据过旧 |
| entropy | 输出分布不确定性 |
| completion length | 生成长度、截断和 overlong 行为 |
| rollout correction | rollout 与训练概率差异、ESS、截断/屏蔽比例 |

### 16.2 completion 日志

启用 log_completions 后，Trainer 会记录 prompt、completion、reward、solution 和可用的多轮信息，并写入 completions.jsonl；W&B/SwanLab 可显示表格。

需要注意：

- 分布式日志先 gather，行数与 generation_batch_size 相关；
- 自定义希望显示的额外列必须在所有 rollout 输出中存在，否则 collective 可能不一致；
- 生产数据可能包含敏感 prompt，不应无审计地打开完整 completion 日志。

### 16.3 输出文件

通常应关注：

- args.json：post-init 后的最终参数，是复现实验的第一依据；
- logging.jsonl：训练 step 指标；
- completions.jsonl：prompt/response/reward 级诊断；
- checkpoint-*：模型、adapter、Trainer 状态、优化器/scheduler 状态；
- 最终模型目录与 last_model_checkpoint。

排查“命令行参数未生效”时，先比较 args.json，而不是依赖启动命令的原始文本。

---

## 17. 配置阅读与调参清单

### 17.1 最小可用配置

~~~bash
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset AI-MO/NuminaMath-TIR \
  --reward_funcs accuracy format \
  --num_generations 8 \
  --max_completion_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1
~~~

### 17.2 建议按层调参

第一层：先验证数据和奖励。

- 用 num_generations=2、小数据和短 max_completion_length；
- 打开 log_completions；
- 检查每个 reward 的均值、标准差和 None/NaN；
- 确认 solution 与 completion 对齐。

第二层：再确认 rollout。

- TransformersEngine 跑通后再切 vLLM；
- 检查 response_token_ids、finish_reason、truncated_mask；
- vLLM 下先打开 log_rollout_offpolicy_metrics；
- server 模式先验证一次权重同步后生成确实变化。

第三层：再扩 batch 与复用。

- 满足 generation_batch_size 与 G 的整除；
- 逐步提高 steps_per_generation；
- num_iterations 大于 1 时重点看 clipping 与 ESS；
- 动态采样时记录有效组比例和实际 rollout 成本。

第四层：最后启用算法增强。

- 每次只引入一个主要变量；
- DAPO 同时改变采样、长度和归一化，最好拆开做消融；
- sequence importance sampling 与 rollout IS correction 是不同层次的权重，不要混淆；
- beta、kl_in_reward 和 sync_ref_model 共同决定参考约束。

### 17.3 常见参数组

| 目标 | 参数 |
|---|---|
| 生成行为 | temperature、top_p、top_k、max_completion_length、stop_words |
| 组采样 | num_generations、num_generations_eval、generation_batch_size |
| 样本复用 | steps_per_generation、num_iterations |
| PPO/GRPO 更新 | epsilon、epsilon_high、delta、beta |
| 优势 | advantage_estimator、scale_rewards、reward_weights |
| DAPO | dynamic_sample、max_resample_times、overlong_filter、soft_* |
| rollout 后端 | use_vllm、vllm_mode、vllm_tensor_parallel_size |
| 内存 | sleep_level、offload_model、offload_optimizer、vllm_gpu_memory_utilization |
| 偏差修正 | importance_sampling_level、rollout_importance_sampling_mode、threshold |
| 可观测性 | log_completions、log_entropy、log_rollout_offpolicy_metrics |

---

## 18. 测试与示例现状

### 18.1 现有测试

| 文件 | 覆盖内容 | 评价 |
|---|---|---|
| [tests/train/test_grpo.py](../../tests/train/test_grpo.py) | LLM、ZeRO-2、vLLM、vLLM+ZeRO、多模态 | GPU 集成/冒烟性质，成本高 |
| [tests/megatron/test_grpo.py](../../tests/megatron/test_grpo.py) | Megatron VL、LoRA、TP、vLLM、动态采样 | 当前是 __main__ 手工运行脚本，不是完整单元测试 |
| [tests/train/test_vllm_importance_sampling_basic.py](../../tests/train/test_vllm_importance_sampling_basic.py) | 四种 rollout IS、阈值、mask、ESS、KL、chi-square | 针对性较好，但使用 mock trainer |
| [tests/utils/test_rewards.py](../../tests/utils/test_rewards.py) | MathAccuracy 的 LaTeX、answer 标签、等价式和异常输入 | 奖励细节覆盖较充分 |
| [tests/utils/test_async_rewards.py](../../tests/utils/test_async_rewards.py) | daemon event loop、async ORM、并发性能 | 覆盖异步奖励基础设施 |
| [tests/test_align/test_rlhf_loss.py](../../tests/test_align/test_rlhf_loss.py) | 预期为 loss 对齐 | 当前文件为空，属于明显覆盖缺口 |

### 18.2 推荐补测优先级

1. 建立纯 CPU 的 advantage/loss 数值测试，覆盖 GRPO、RLOO、REINFORCE++ 和所有 normalization。
2. 同一固定张量同时跑 Transformers 与 Megatron loss，做数值对齐。
3. 测试 G 组跨 rank 边界时的 sampler 顺序和 advantage 分组。
4. 测试 num_iterations 与 steps_per_generation 的缓存复用/更新时点。
5. 测试 vLLM token IDs 与模板编码后的 completion_mask 精确对齐。
6. 测试多轮环境观察被 loss mask 排除。
7. 测试 reward None/NaN、多奖励 GDPO、Gym 混合权重。
8. 测试 ZeRO-3/FSDP/LoRA 的权重同步是否改变 rollout 输出。

### 18.3 示例导航

| 需求 | 示例 |
|---|---|
| Transformers rollout | [examples/train/grpo/internal/transformers.sh](../../examples/train/grpo/internal/transformers.sh) |
| vLLM colocate | [examples/train/grpo/internal/vllm_vl7b.sh](../../examples/train/grpo/internal/vllm_vl7b.sh) |
| 外部 server | [examples/train/grpo/external/grpo_7b.sh](../../examples/train/grpo/external/grpo_7b.sh) |
| 多轮 | [examples/train/grpo/external/vllm_multi_turn.sh](../../examples/train/grpo/external/vllm_multi_turn.sh) |
| Gym | [examples/train/grpo/external/vllm_gym.sh](../../examples/train/grpo/external/vllm_gym.sh) |
| 自定义奖励 | [examples/train/grpo/plugin](../../examples/train/grpo/plugin) |
| RLOO | [examples/train/grpo/internal/rloo.sh](../../examples/train/grpo/internal/rloo.sh) |
| REINFORCE++ | [examples/train/grpo/internal/reinforce_plus_plus.sh](../../examples/train/grpo/internal/reinforce_plus_plus.sh) |
| GSPO | [examples/train/grpo/internal/gspo.sh](../../examples/train/grpo/internal/gspo.sh) |
| CHORD | [examples/train/grpo/internal/chord.sh](../../examples/train/grpo/internal/chord.sh) |
| Ray colocate | [examples/ray/grpo/ray_grpo_colocate.yaml](../../examples/ray/grpo/ray_grpo_colocate.yaml) |
| Ray separate | [examples/ray/grpo/ray_grpo_separate.yaml](../../examples/ray/grpo/ray_grpo_separate.yaml) |

---

## 19. 排错地图

### 19.1 reward 一直不变或 advantage 接近 0

检查顺序：

1. completions.jsonl 中实际答案是否变化；
2. rewards/name/std 是否为 0；
3. solution/ground_truth 列是否存在且顺序对齐；
4. 自定义 reward 是否错误地广播了单个分数；
5. 同一 prompt 的 G 个 completion 是否被正确分组；
6. dynamic sampling 是否频繁重采样；
7. _compute_advantages 的 scale_rewards 分支。

源码入口：[奖励计算](../../swift/rlhf_trainers/grpo_trainer.py#L305)、[优势计算](../../swift/rlhf_trainers/grpo_trainer.py#L412)。

### 19.2 loss 为 NaN

优先检查：

- 全部 completion_mask 是否为 0；
- completion 是否全部被 overlong_filter 排除；
- reward 是否全为 None/NaN；
- num_generations=1 或组大小异常；
- 极端 log-ratio、beta、temperature；
- 自定义 loss/mask 是否在 padding-free/SP 后形状错位；
- REAL 组中是否没有有效正负样本。

代码已对若干分母 clamp(min=1)，但这只能避免除零，不能使空样本产生有效梯度。

### 19.3 clipping 比例持续很高

可能原因：

- 学习率过大；
- num_iterations 或 steps_per_generation 使样本过旧；
- async_generate 策略陈旧；
- 权重同步间隔不正确；
- sequence/token importance sampling 选择不匹配；
- reward 尺度或 advantage normalization 异常。

同时查看 current/old clip 指标和 old/rollout correction 指标，才能区分“优化步太大”与“推理策略太旧”。

### 19.4 vLLM 与训练结果偏差大

检查：

1. rollout_per_token_logps 是否存在；
2. adapter 是否已同步、是否加载到正确名称；
3. full/LoRA 权重同步路径；
4. server 的模型版本/step；
5. 量化、dtype、sampling 参数；
6. response token 是否经过 decode/re-encode；
7. MoE 路由是否需要重放；
8. rollout correction 的 KL、ESS 和 mask fraction。

源码入口：[权重同步](../../swift/rlhf_trainers/rollout_mixin.py#L449)、[IS 修正](../../swift/rlhf_trainers/grpo_trainer.py#L2441)。

### 19.5 分布式卡死

常见位置：

- 各 rank 进入不同数量的 reward/rollout 调用；
- 自定义奖励异常只发生在某个 rank；
- 多轮轨迹结束条件不一致；
- 自定义日志字段并非每个输出都存在；
- vLLM TP 不能整除 world size；
- server 请求/广播数量与本地切片不一致；
- Megatron rollout group 构造错误。

先以单卡、G=2、同步 rollout 复现，再逐层增加 DDP、vLLM TP、多轮。

### 19.6 OOM

按占用来源区分：

- 策略前向/反向：减小 per_device batch、序列长度，使用 checkpointing/ZeRO/FSDP；
- rollout KV cache：降低 vllm_gpu_memory_utilization、max_num_seqs、max model length；
- reference model：beta=0 或 LoRA base-as-ref；
- reward model：改为 server/更小 RM 或优化分片；
- colocate 峰值：启用 sleep/offload；
- completion 太长：缩短 max_completion_length，检查停止词和格式奖励。

不要只降低训练 batch：rollout KV cache 与 reward model 可能才是主因。

### 19.7 参数看似未生效

1. 查看输出目录 args.json；
2. 检查 post-init 是否强制改写，如 REAL 的 scale_rewards；
3. 检查 use_vllm/vllm_mode 的联动；
4. 检查 YAML 字段拼写和 CLI 覆盖顺序；
5. 检查 Transformers 与 Megatron 同名参数是否语义完全一致。

---

## 20. 后续深入阅读路线

### 20.1 想改 reward

阅读顺序：

1. [swift/rewards/orm.py](../../swift/rewards/orm.py)
2. [GRPOTrainer._prepare_rewards](../../swift/rlhf_trainers/grpo_trainer.py#L2297)
3. [GRPOTrainer._compute_rewards_per_func](../../swift/rlhf_trainers/grpo_trainer.py#L343)
4. [插件示例](../../examples/train/grpo/plugin/plugin.py)
5. [reward 测试](../../tests/utils/test_rewards.py)

重点验证返回长度、None/NaN、额外列、多进程 gather、异步调用和 reward_weights。

### 20.2 想改 advantage 或 normalization

阅读：

1. [GRPOTrainer._compute_advantages](../../swift/rlhf_trainers/grpo_trainer.py#L412)
2. [共享 RLHF 工具](../../swift/rlhf_trainers/utils.py)
3. Megatron 中对应 advantage 调用
4. args_mixin 中 scale_rewards/advantage_estimator 定义

建议先写固定 reward 张量的数值单测，再改训练代码。

### 20.3 想新增 loss

至少要同步处理：

- 参数 Literal 和默认/校验；
- _prepare_algorithm_params；
- Transformers _compute_loss_and_metrics；
- Megatron loss_func；
- Ray 对 Megatron loss 的复用；
- normalization 分母；
- clip/KL/entropy/rollout IS/mask 的组合；
- metrics；
- padded、padding-free、SP/CP；
- Liger/fused 路径的支持或显式禁止；
- 数值对齐测试与文档示例。

### 20.4 想改 vLLM 权重同步

阅读：

1. [rollout_mixin.py](../../swift/rlhf_trainers/rollout_mixin.py)
2. [vllm_client.py](../../swift/rlhf_trainers/vllm_client.py)
3. [rollout server](../../swift/pipelines/infer/rollout.py)
4. [GRPOVllmEngine](../../swift/infer_engine/grpo_vllm_engine.py)
5. ZeRO/FSDP/PEFT/MoE 分支

需要建立“训练 step → sync step → server 已加载版本 → request_id”的可观测链。

### 20.5 想扩展多轮或工具调用

阅读：

1. [multi_turn.py](../../swift/rollout/multi_turn.py)
2. [agent_loop.py](../../swift/rollout/agent_loop.py)
3. [gym_env.py](../../swift/rollout/gym_env.py)
4. [多轮示例](../../examples/train/grpo/external/vllm_multi_turn.sh)

重点是结束条件、消息结构、环境观察 loss mask、轨迹奖励、最大轮数和跨 rank 一致性。

### 20.6 想做大规模 Megatron/Ray 优化

先画清资源拓扑：

~~~text
DP × TP × PP × CP × EP
rollout replicas × vLLM TP
train/rollout colocate 或 separate
~~~

随后分别追踪：

- prompt 在哪些 rank 上存在；
- completion 在哪个组内 gather；
- 哪个 PP stage 计算 log-prob/loss；
- 权重如何 bridge 到 vLLM；
- optimizer/model 在 generation context 如何 offload；
- driver CPU 是否成为 reward/advantage 瓶颈。

---

## 21. 关键函数索引

| 问题 | 首选函数/类 |
|---|---|
| GRPO Trainer 如何初始化 | GRPOTrainer.__init__ |
| 为什么输入还是 messages | SwiftSft._prepare_dataset、identity_data_collator |
| prompt 如何重复 G 次 | GRPOTrainer._get_train_sampler、TRL RepeatSampler |
| 什么时候重新 rollout | GRPOTrainer._prepare_inputs |
| 一次 rollout 全流程 | GRPOTrainer._generate_and_score_completions |
| 如何生成 | GRPOTrainer._generate_completions、RolloutTrainerMixin._fast_infer |
| 如何连接 vLLM server | VLLMClient、_rollout_server |
| 如何同步权重 | _move_model_to_vllm、_move_adapter_to_vllm |
| 奖励如何调用 | _score_completions、_compute_rewards_per_func |
| 动态采样如何做 | _dynamic_sampling |
| rollout 后如何编码 | _prepare_batch_inputs |
| old/ref log-prob 如何算 | _get_per_token_logps、_prepare_batch_inputs |
| advantage 如何算 | _compute_advantages |
| loss 如何算 | _compute_loss_and_metrics |
| rollout IS 如何做 | _apply_rollout_importance_sampling |
| reference 如何移动 | SyncRefModelCallback、_sync_ref_model_weights |
| completion 如何记录 | _prepare_metrics、log |
| Megatron 如何喂入 rollout 数据 | MegatronGRPOTrainer._replace_data_iterator |
| Megatron loss 在哪里 | MegatronGRPOTrainer.forward_step、loss_func |
| Ray 主循环在哪里 | Ray Megatron GRPOTrainer._train_loop |

---

## 22. 一条样本的完整时序

~~~mermaid
sequenceDiagram
    participant DS as "Dataset"
    participant Sampler as "RepeatSampler"
    participant T as "GRPOTrainer"
    participant E as "Rollout Engine"
    participant RW as "Reward"
    participant Ref as "Reference"
    participant P as "Policy"
    participant Opt as "Optimizer"

    DS->>Sampler: "messages + solution"
    Sampler->>T: "同一 prompt × G"
    T->>E: "同步权重并发送 prompts"
    E-->>T: "response token IDs + rollout log-probs"
    T->>RW: "completion + 数据额外列"
    RW-->>T: "各奖励列"
    T->>P: "no_grad 计算 old log-probs"
    T->>Ref: "no_grad 计算 ref log-probs"
    T->>T: "组内统计并计算 advantages"
    T->>T: "编码、切分并缓存 micro-batches"
    loop "steps_per_generation × num_iterations"
        T->>P: "有梯度计算 current log-probs"
        P-->>T: "ratio / KL / token loss"
        T->>Opt: "backward / accumulate / step"
    end
~~~

---

## 23. 术语表

| 术语 | 本工程中的含义 |
|---|---|
| prompt group | 同一 prompt 产生的 G 个 completion |
| generation batch | 一次集中 rollout 的全局 completion 批次 |
| micro-batch | 单设备一次前向/反向消费的批次 |
| rollout policy | 真正执行采样的 engine 权重与数值实现 |
| old policy | rollout 后冻结用于 PPO ratio 分母的策略快照 |
| reference policy | KL 锚点，不参与策略更新 |
| on-policy | current、old、rollout 接近；严格程度取决于缓存和同步 |
| off-policy gap | current/old 或 old/rollout 的分布偏差 |
| completion mask | 标明哪些 response token 参与 RL loss |
| rollout_infos | 多轮/Gym/finish reason 等轨迹元数据 |
| dynamic sampling | 丢弃零方差组并补采样 |
| colocate | 训练和 rollout 共享 GPU 资源 |
| server | rollout 为独立服务，训练通过客户端请求 |

---

## 24. 最终建议

若要快速掌握项目，最有效的实践顺序是：

1. 用 TransformersEngine、小模型、G=2 跑一个短任务；
2. 对照 completions.jsonl 手工复算一组 reward 和 advantage；
3. 在 _compute_loss_and_metrics 处核对四类 log-prob 的形状与数值；
4. 切换 vLLM colocate，并观察 rollout off-policy 指标；
5. 再尝试 server、multi-turn 或 Gym；
6. 最后进入 Megatron/Ray，并先验证同一小 batch 的算法数值对齐。

从代码维护角度，后续最值得优先建设的是：Transformers/Megatron loss 的共享数值测试、跨 rank prompt 分组测试，以及“权重版本—rollout 请求—日志记录”的端到端追踪。它们能显著降低新增算法和大规模后端优化时的回归风险。

---

## 25. 8GB 单卡、Qwen3-1.7B、无 vLLM 的三步实测案例

本节不是静态推演，而是对以下脚本进行真实运行后得到的实例分析：

- 脚本：[qwen3_1_7b_qlora_transformers.sh](../../examples/train/grpo/local_8gb/qwen3_1_7b_qlora_transformers.sh)
- 命令：bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_transformers.sh
- 运行时间：2026-08-04 23:51:34 至 23:57:34
- 训练范围：3 个 optimizer/global step
- 训练结果：[checkpoint-3](../../output/grpo-qwen3-1.7b-qlora-8gb/v5-20260804-235131/checkpoint-3)
- 完整参数：[args.json](../../output/grpo-qwen3-1.7b-qlora-8gb/v5-20260804-235131/args.json)
- 逐步指标：[logging.jsonl](../../output/grpo-qwen3-1.7b-qlora-8gb/v5-20260804-235131/logging.jsonl)
- completion 快照：[completions.jsonl](../../output/grpo-qwen3-1.7b-qlora-8gb/v5-20260804-235131/completions.jsonl)

这些 output 链接是本机运行产物，不是仓库源码依赖。清理 output 目录后链接会失效。

### 25.1 实际硬件和软件条件

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti |
| 显存 | 8192 MiB |
| 启动前空闲显存 | 约 7252 MiB |
| Compute Capability | 8.6 |
| BF16 | PyTorch 检查为支持 |
| PyTorch | 2.6.0+cu124 |
| ms-swift | 4.3.0 |
| TRL | 0.29.1 |
| bitsandbytes | 已安装 |
| vLLM | 不使用 |
| 运行 Python | 工程 .venv |

本次进程实际解析到的模型路径为 /home/cb/model/Qwen3-1.7B。脚本默认值是 Qwen/Qwen3-1.7B；当前运行环境中的 MODEL 变量使其使用了已存在的本地模型目录，因此没有等待完整模型下载。

### 25.2 最终生效配置

训练脚本经过两层参数对象：

1. Pipeline 层的 RLHFArguments；
2. Trainer 层的 GRPOConfig。

训练 checkpoint 中的 training_args.bin 证实 Trainer 最终使用：

| 参数 | 生效值 | 作用 |
|---|---:|---|
| per_device_train_batch_size | 1 | 一个训练 micro-batch 只有一条 completion |
| world_size | 1 | 单卡 |
| gradient_accumulation_steps | 8 | 8 个 micro-batch 后更新一次 |
| num_generations | 2 | 每个 prompt 生成两个候选 |
| steps_per_generation | 2 | 一次 rollout 缓存切成两个 micro-batch |
| generation_batch_size | 2 | 1 × 1 × 2 |
| num_iterations | 1 | rollout 数据不做额外迭代复用 |
| max_steps | 3 | 只执行三个 optimizer step |
| max_length | 768 | prompt 与 completion 编码总长度上限 |
| max_completion_length | 384 | 单条生成长度上限 |
| use_vllm | false | 使用 TransformersEngine |
| vllm_mode | None | 没有外部或 colocate vLLM |
| beta | 0.01 | loss-side KL 系数 |
| scale_rewards | none | advantage 不除以奖励标准差 |
| reward_weights | 1.0、0.2 | accuracy 和 format 的权重 |

Pipeline 打印的 RLHFArguments 中 generation_batch_size 仍可能显示 None，因为它是进入 Trainer 配置后才根据 batch 关系推导为 2。要分析真实训练行为，应以 GRPOConfig 或 checkpoint 中的 training_args.bin 为准。

### 25.3 从 shell 到 Trainer 的真实启动路径

~~~mermaid
flowchart TD
    A["bash qwen3_1_7b_qlora_transformers.sh"] --> B["工程 .venv/bin/swift rlhf"]
    B --> C["swift/cli/main.py 路由"]
    C --> D["单卡：直接运行 swift/cli/rlhf.py"]
    D --> E["rlhf_main → SwiftRLHF"]
    E --> F["RLHFArguments post-init"]
    F --> G["加载 Qwen3-1.7B 为 BNB 4-bit"]
    G --> H["加载并标准化 GSM8K 500 条数据"]
    H --> I["注入 PEFT LoRA"]
    I --> J["TrainerFactory → GRPOTrainer"]
    J --> K["GRPOTrainer 创建 TransformersEngine"]
    K --> L["transformers.Trainer.train max_steps=3"]
    L --> M["checkpoint-3 + JSONL 日志"]
~~~

对应源码入口：

1. [CLI 路由](../../swift/cli/main.py)
2. [RLHF CLI](../../swift/cli/rlhf.py)
3. [SwiftRLHF pipeline](../../swift/pipelines/train/rlhf.py#L23)
4. [SwiftSft.run](../../swift/pipelines/train/sft.py#L193)
5. [TrainerFactory](../../swift/trainers/trainer_factory.py)
6. [GRPOTrainer.__init__](../../swift/rlhf_trainers/grpo_trainer.py#L93)
7. [TransformersEngine](../../swift/infer_engine/transformers_engine.py#L50)

因为 NPROC_PER_NODE、NNODES 和 NODE_RANK 在脚本中被清除，本次没有进入 torchrun，也没有创建 DDP 进程组。整个训练只有 rank=-1、world_size=1。

### 25.4 模型加载和 LoRA 注入

模型加载阶段的真实顺序是：

~~~text
模型元信息解析
  → 构造 BitsAndBytesConfig
  → 4-bit NF4 权重加载到 cuda:0
  → 设置 Qwen3 模板和 generation config
  → 加载原始 GSM8K 数据
  → PEFT 为目标线性层注入 LoRA
  → 构造 GRPOTrainer
~~~

实际量化配置：

- quant_method=bnb；
- quant_bits=4；
- quant_type=nf4；
- double_quant=true；
- compute_dtype=bfloat16；
- attention implementation=sdpa。

all-linear 最终展开为七类模块：

- Attention：q_proj、k_proj、v_proj、o_proj；
- MLP：gate_proj、up_proj、down_proj。

每层原始 Linear 被包装为 lora.Linear4bit：

~~~text
输入
  ├─ 4-bit base_layer：冻结
  └─ LoRA A(rank=8) → LoRA B → alpha/rank 缩放：可训练
         两支结果相加
~~~

训练日志报告：

~~~text
1024.6482M counted parameters
8.7163M trainable parameters
0.8507% trainable
~~~

这里的 1024.6482M 不能直接当作 Qwen3-1.7B 的架构参数量。bitsandbytes 会以打包形式保存 4-bit 权重，通用 numel 统计会受物理存储布局影响。对本次训练更有意义的是 8.7163M 个 LoRA 可训练参数，以及约 34.9MB 的 adapter_model.safetensors。

### 25.5 数据路径：为什么进入 Trainer 的仍是 messages

GSM8K 加载结果：

~~~text
train_dataset:
  features = [messages, solution]
  num_rows = 500
val_dataset = None
~~~

脚本通过 columns 把原始 answer 映射为 solution。外部插件在参数初始化阶段注册：

- GSM8KAccuracy；
- GSM8KFormat。

[SwiftSft._prepare_dataset](../../swift/pipelines/train/sft.py#L125) 检测到 rlhf_type=grpo，设置 pre_process=false，因此此处不把 500 条数据预编码为 input_ids。

这次实测验证了主报告中的数据设计：

~~~text
Dataset row
  {messages, solution}
      ↓ RepeatSampler
  同一个 prompt 的两份采样位置
      ↓ Transformers rollout
  两个 assistant completion + response_token_ids
      ↓ 奖励函数
  accuracy / format / weighted reward / advantage
      ↓ _prepare_batch_inputs
  input_ids / completion_mask / old log-prob / ref log-prob
      ↓ Trainer
  current log-prob / loss / backward
~~~

solution 始终不参与模型前向，只供 reward plugin 使用。

### 25.6 一个 global step 究竟包含多少工作

这是本次实例最重要的时序结论。

已知：

~~~text
B = per_device_train_batch_size = 1
D = world_size = 1
S = steps_per_generation = 2
G = num_generations = 2
A = gradient_accumulation_steps = 8
~~~

一次 rollout 的 completion 数：

~~~text
generation_batch_size = B × D × S = 1 × 1 × 2 = 2
~~~

每个 prompt 的候选数也是 2，因此：

~~~text
每次 rollout 的唯一 prompt 数 = generation_batch_size / G = 1
~~~

一次 rollout 生成两个 completion，并被切成两个 batch size 为 1 的 micro-batch。一个 optimizer step 需要累计 8 个 micro-batch，所以：

~~~text
每个 optimizer step 的 rollout 次数 = A / S = 8 / 2 = 4
每个 optimizer step 的 prompt group 数 = 4
每个 optimizer step 的 completion 数 = 4 × 2 = 8
三步合计 prompt group 数 = 3 × 4 = 12
三步合计 completion 数 = 12 × 2 = 24
~~~

训练结束时 epoch=0.024，也与 12 / 500 = 0.024 完全吻合。

### 25.7 一个 global step 的逐 micro-step 时序

[GRPOTrainer._prepare_inputs](../../swift/rlhf_trainers/grpo_trainer.py#L195) 使用内部 self._step，而不是 Trainer 的 global_step，决定何时重新生成。

本配置 num_iterations=1、steps_per_generation=2，因此 generate_every=2：

| 内部 micro-step | self._step 进入值 | 行为 | 返回给 loss 的数据 |
|---:|---:|---|---|
| 1 | 0 | 对 prompt A 做 rollout，生成 A1/A2；评分并缓存 | A1 |
| 2 | 1 | 不生成，复用缓存 | A2 |
| 3 | 2 | 对 prompt B 做新 rollout | B1 |
| 4 | 3 | 复用缓存 | B2 |
| 5 | 4 | 对 prompt C 做新 rollout | C1 |
| 6 | 5 | 复用缓存 | C2 |
| 7 | 6 | 对 prompt D 做新 rollout | D1 |
| 8 | 7 | 复用缓存 | D2 |
| optimizer | 8 次 backward 后 | 梯度裁剪、optimizer.step、scheduler.step、zero_grad | global_step + 1 |

随后第二、第三个 global step 重复同样结构。

self._step 的递增不在 Swift 的 _prepare_inputs 中，而发生在上游 TRL GRPOTrainer.training_step。Swift 的 [training_step](../../swift/rlhf_trainers/grpo_trainer.py#L1949) 调用 super 后，由 TRL：

1. 执行 HF Trainer.training_step；
2. HF Trainer 内部调用当前对象覆盖后的 _prepare_inputs 和 compute_loss；
3. 完成 backward；
4. self._step 加一；
5. 每满 current_gradient_accumulation_steps 记录 step_time。

因此这里同时存在两个 step：

- self._step：每个 forward/backward micro-batch 增加；
- state.global_step：每次 optimizer 更新后增加。

日志中的 global_step/max_steps=1/3 指后者。

### 25.8 每次 rollout 的内部调用链

每当 self._step 为 0、2、4、6……时，进入：

~~~text
GRPOTrainer._prepare_inputs
  → _generate_and_score_completions
      → _generate_completions
          → _preprocess_inputs
          → _infer_single_or_multi_turn
              → TransformersEngine.infer
                  → model.generate
      → _score_completions
          → _compute_rewards_per_func
              → GSM8KAccuracy
              → GSM8KFormat
      → _prepare_batch_inputs
          → 注入真实 response_token_ids
          → template.encode
          → data_collator
          → completion_mask
          → old_per_token_logps
          → ref_per_token_logps
      → _compute_advantages
      → split_by_mini_batches
  → 返回当前缓存 micro-batch
~~~

主要源码：

- [rollout 总入口](../../swift/rlhf_trainers/grpo_trainer.py#L243)
- [无 vLLM 生成分支](../../swift/rlhf_trainers/grpo_trainer.py#L223)
- [TransformersEngine.infer](../../swift/infer_engine/transformers_engine.py#L563)
- [奖励计算](../../swift/rlhf_trainers/grpo_trainer.py#L315)
- [GSM8K 插件](../../examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py)
- [rollout 后编码](../../swift/rlhf_trainers/grpo_trainer.py#L804)
- [优势计算](../../swift/rlhf_trainers/grpo_trainer.py#L421)

本次 use_fast_infer=false，所以没有执行：

- RolloutTrainerMixin._fast_infer；
- vLLM 权重同步；
- rollout_per_token_logps；
- rollout importance sampling correction；
- vLLM sleep/offload。

### 25.9 本实例中的三类概率

因为没有 vLLM，本实例只有三类训练概率：

| 概率 | 实际如何得到 | 用途 |
|---|---|---|
| old_per_token_logps | rollout 后，启用 LoRA 的当前模型 no_grad 前向 | current/old ratio |
| ref_per_token_logps | 同一个 PEFT 模型临时 disable_adapter 后 no_grad 前向 | loss-side KL |
| per_token_logps | micro-batch 训练时启用 LoRA、有梯度前向 | 策略梯度 |

rollout_per_token_logps 为 None，因为普通 TransformersEngine 不返回 vLLM 采样 log-prob 链路。

本配置没有单独复制一份 reference model。LoRA 的冻结 base model 同时承担参考模型角色：

~~~text
同一份 4-bit Qwen3 base
  ├─ adapter enabled：policy / old policy
  └─ adapter disabled：reference policy
~~~

这正是 8GB 显存能够保留 beta=0.01 KL 约束而无需第二份模型的关键。

每次 rollout 的两条 completion 被拆成两个 micro-batch。对每个 micro-batch，逻辑上至少有：

1. 一次 old policy 全序列 no_grad scoring；
2. 一次 adapter-disabled reference 全序列 no_grad scoring；
3. 一次 current policy 有梯度 scoring；
4. current scoring 后的 backward；
5. 此外还有 rollout 的逐 token 自回归 decode。

三步总计：

- 12 次双候选 rollout；
- 24 条 completion；
- 24 次 old-policy 序列评分；
- 24 次 reference 序列评分；
- 24 次 current-policy 训练前向与 backward；
- 3 次 optimizer.step。

这里说的是逻辑模型调用次数，不等于 CUDA kernel 次数。

### 25.10 奖励和 advantage 的真实样例

completion 日志中最清楚的一组来自电费题。

两个候选：

1. 候选 A 错误地把 60 kWh 当成 60 美元；
2. 候选 B 正确计算为 6 美元。

奖励：

| 候选 | accuracy | format | 加权总奖励 |
|---|---:|---:|---:|
| A | 0 | 1 | 0 × 1.0 + 1 × 0.2 = 0.2 |
| B | 1 | 1 | 1 × 1.0 + 1 × 0.2 = 1.2 |

组均值为 0.7。因为 scale_rewards=none：

~~~text
A_A = 0.2 - 0.7 = -0.5
A_B = 1.2 - 0.7 = +0.5
~~~

completions.jsonl 中实际记录的 advantages 正是：

~~~text
[-0.5000000596, 0.5]
~~~

另一组游泳题的两个候选都得到：

~~~text
accuracy = [1, 1]
format = [1, 1]
weighted reward = [1.2, 1.2]
advantage = [0, 0]
~~~

这组数据虽然完成了生成、两次奖励、编码、old/ref/current scoring 和 backward，但不会贡献有效策略梯度。

### 25.11 从 advantage 到 backward

HF Trainer 对缓存中的单条 completion 调用：

~~~text
GRPOTrainer.compute_loss
  → _compute_loss
  → _compute_loss_single
  → _compute_loss_and_metrics
      → current per_token_logps
      → log_ratio = current - old
      → PPO/GRPO ratio clip
      → K3 reference KL
      → completion_mask
      → completion token 平均
  → Accelerate backward
~~~

源码入口：

- [compute_loss](../../swift/rlhf_trainers/grpo_trainer.py#L1015)
- [核心 loss](../../swift/rlhf_trainers/grpo_trainer.py#L1099)

本实例的 standard GRPO 分支使用：

~~~text
ratio_t = exp(current_logp_t - old_logp_t)
clipped_t = clip(ratio_t, 0.8, 1.28)
policy_loss_t = -min(ratio_t × A, clipped_t × A)
loss_t = policy_loss_t + 0.01 × K3_KL_t
~~~

每条 completion 先对有效 token 平均，batch size 为 1，再进行梯度累积。

### 25.12 为什么 loss 接近 0，但 grad_norm 不一定为 0

三步日志：

| global step | loss | grad_norm | reward | reward_std | zero-std group 比例 | KL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 1.20 | 0 | 1.00 | 0 |
| 2 | 2.98e-8 | 0.2096 | 0.95 | 0.3536 | 0.50 | 0 |
| 3 | 1.4436e-4 | 0.0616 | 1.05 | 0.2121 | 0.75 | 5.1564e-4 |

第 1 步：

- 四个 prompt group 的两条候选奖励都相同；
- 所有 advantage 都为 0；
- LoRA 的 B 矩阵初始为零，policy 与禁用 adapter 后的 reference 相同；
- 因此 policy loss、KL、gradient 全为 0。

第 2 步：

- 一半 group 出现非零 reward std；
- 正负 advantage 成对出现；
- current 与 old 在该 optimizer step 内仍非常接近；
- 组内标量 loss 可以因正负项抵消到接近 0；
- 但不同 completion 对参数的梯度方向不同，不能按 loss 数值直接抵消；
- 所以 loss 约 3e-8 时 grad_norm 仍为 0.2096。

这说明 GRPO 不能只监控 loss。至少要联合观察：

- reward_std；
- frac_reward_zero_std；
- grad_norm；
- KL；
- clip ratio；
- 每个 reward function 的均值和标准差。

### 25.13 三步指标的具体解释

#### Step 1

- completion 平均长度约 137.9 token；
- accuracy=1、format=1；
- 所有组无奖励差异；
- 没有参数更新信号；
- LoRA policy 与 base reference 相同，因此 KL=0。

#### Step 2

- completion 平均长度约 138.9 token；
- accuracy 均值 0.75；
- format 均值 1；
- 50% group 有奖励差异；
- grad_norm=0.2096，首次出现有效更新；
- PPO token clip ratio 仍为 0。

#### Step 3

- completion 平均长度约 215.1 token；
- accuracy 和 format 均值都是 0.875；
- 75% group 无奖励差异，仅 25% group 提供相对优势；
- KL 上升到约 5.16e-4，说明 LoRA policy 已开始偏离 base；
- PPO low-region clip token 比例约 3.26e-4，极低；
- completions/clipped_ratio=0.125，说明聚合的 8 条 completion 中约一条触及生成截断；它不是 PPO clip ratio。

必须区分两个名字相近的指标：

| 指标 | 含义 |
|---|---|
| completions/clipped_ratio | completion 因生成长度上限被截断的比例 |
| clip_ratio/region_mean | 策略概率 ratio 触发 PPO 裁剪的 token 比例 |

### 25.14 学习率时序

总训练步数只有 3，cosine scheduler 产生：

~~~text
step 1: 1.0e-5
step 2: 5.0e-6
step 3: 0
~~~

因此这次运行只能验证流程和调用链，不能用于判断最终训练效果。仅三步时 scheduler 很快衰减到 0；做正式实验应增加 max_steps，或重新设计 warmup_steps 和 scheduler。

### 25.15 性能和显存实测

| 项目 | 实测 |
|---|---:|
| Trainer train_runtime | 343.60 秒 |
| 总训练 wall time | 约 5 分 44 秒 |
| 全 pipeline 时间 | 约 6 分钟 |
| train_steps_per_second | 0.009 |
| train_samples_per_second | 0.070 |
| 平均 optimizer step | 约 114.5 秒 |
| Step 1 step_time | 100.96 秒 |
| Step 2 step_time | 112.56 秒 |
| Step 3 step_time | 151.51 秒 |
| Trainer 记录峰值显存 | 2.46 GiB |
| nvidia-smi 观察到的最高总占用 | 约 3.47 GiB |
| 训练结束后的系统基线占用 | 约 0.77 GiB |

nvidia-smi 总占用包含桌面/WSL/CUDA context，Trainer 指标更接近训练进程内部统计。两者口径不同，不应直接当作矛盾。

本配置在 8GB 上有明显显存余量，但速度较慢。主要瓶颈不是 LoRA backward，而是无 vLLM 时的自回归 rollout，以及每条 completion 随后的 old、reference、current 三次全序列 scoring。

Step 3 更慢，与该步 completion 平均长度从约 139 增长到约 215 有直接关系。纯 Transformers rollout 的计算成本会明显随生成长度增加。

### 25.16 checkpoint 与日志产物

checkpoint-3 中主要文件：

| 文件 | 大小约 | 用途 |
|---|---:|---|
| adapter_model.safetensors | 34.9 MB | LoRA 参数 |
| optimizer.pt | 70.1 MB | AdamW 状态，用于续训 |
| adapter_config.json | 1.1 KB | LoRA 结构配置 |
| trainer_state.json | 3.3 KB | global step、指标、callback 状态 |
| scheduler.pt | 1.0 KB | 学习率调度器 |
| rng_state.pth | 14.2 KB | 随机状态 |
| training_args.bin | 10.4 KB | Trainer 最终配置 |
| args.json | 28.0 KB | Swift 参数快照 |

整个运行目录约 101 MB。由于保存的是 LoRA adapter，而不是完整 4-bit base 模型，checkpoint 较小；后续推理仍需要原始 Qwen3-1.7B base。

### 25.17 completions.jsonl 不是完整轨迹审计日志

本次实际产生 24 条 completion，但 completions.jsonl 只有 4 行 JSON，每行包含两个候选。

原因位于 [GRPOTrainer._prepare_metrics](../../swift/rlhf_trainers/grpo_trainer.py#L2220)：

~~~text
prompt / completion / reward / advantage
使用 maxlen = generation_batch_size = 2 的 deque
~~~

每次 global step 写日志时，只保留最近一次 rollout 的两个 completion，而不是该 optimizer step 内四次 rollout 的全部八条 completion。结束阶段再次调用 log，导致最后一组可能重复写入。

因此：

- logging.jsonl 适合查看 optimizer step 聚合指标；
- completions.jsonl 适合抽样查看最近 completion；
- 当前默认配置不适合做完整的 24 条轨迹审计；
- 如果需要全量轨迹，应增加独立 append-only rollout trace，不能只增大展示表格。

这一点对调试 reward 和数据污染尤其重要。

### 25.18 框架设计在本实例中的落地

#### 设计一：Pipeline 负责组装，Trainer 负责算法

Pipeline 完成：

- 参数解析；
- 模型、模板、数据集准备；
- 量化加载；
- LoRA 注入；
- Trainer 创建；
- checkpoint 汇总。

GRPOTrainer 完成：

- prompt 分组；
- rollout；
- 奖励；
- old/ref 概率；
- advantage；
- GRPO loss；
- completion 日志。

这样新增模型主要影响模型/模板层，新增算法主要影响 Trainer 层。

#### 设计二：InferEngine 隔离 rollout 后端

本次 GRPOTrainer 构造 TransformersEngine。若将 use_vllm 改为 true，算法层的 _generate_and_score_completions 仍保留，主要替换生成和权重同步部分。

这种设计使 reward、advantage 和 loss 不依赖具体推理后端。

#### 设计三：原始数据延迟编码

GSM8K 的 messages 和 solution 保持到 rollout 后。这样同一个 reward plugin 同时适用于：

- TransformersEngine；
- vLLM；
- 多轮轨迹；
- 单卡或分布式。

#### 设计四：LoRA base 复用为 reference

同一份 4-bit 模型通过 adapter enable/disable 切换 policy/reference，避免复制模型。这是小显存 GRPO 的核心结构性优化。

#### 设计五：rollout batch 与 optimizer batch 解耦

steps_per_generation=2 决定一次生成两个 completion；gradient_accumulation_steps=8 决定四个 prompt group 后才更新。生成吞吐、奖励统计和优化 batch 可以分别调节，但参数关系也更难理解。

### 25.19 本实例暴露的训练问题

#### 奖励组零方差比例高

三步 zero-std group 比例为 1.0、0.5、0.75。G=2 时，只要两个候选同对或同错，组内 advantage 就全为 0。

改善方向：

- 正式训练增加 num_generations，但会提高显存和生成时间；
- 调高 temperature 增加候选差异；
- 增加过程质量、长度或部分正确性奖励；
- 启用 dynamic_sample，但会增加 rollout 成本；
- 扩大训练步数，不能用三步判断长期比例。

#### completion 日志不完整

现有 deque 更偏向可视化抽样，不是全量追踪。深入研究 reward 时应增加 request_id、prompt_id、policy version 和每次 rollout 的独立文件。

#### 三步 cosine scheduler 过短

第三步学习率已归零，所以这只是架构验证。正式实验应重新设置总步数与 scheduler。

#### 纯 Transformers rollout 吞吐低

约 0.009 optimizer step/s。8GB 显存不是瓶颈，rollout 自回归和三次 scoring 才是主要时间成本。

### 25.20 如何复现实例和继续实验

重新执行同样的三步流程：

~~~bash
bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_transformers.sh
~~~

只检查最终命令：

~~~bash
DRY_RUN=1 bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_transformers.sh
~~~

若桌面占用导致 OOM：

~~~bash
MODEL=Qwen/Qwen3-0.6B \
MAX_LENGTH=512 \
MAX_COMPLETION_LENGTH=256 \
bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_transformers.sh
~~~

进行正式长训练：

~~~bash
MAX_STEPS=-1 \
DATASET='modelscope/gsm8k#2000' \
bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_transformers.sh
~~~

正式训练前建议先调整脚本中的 save_steps，并根据目标总步数重新选择 lr_scheduler_type、warmup_steps 或 warmup_ratio。

### 25.21 对本实例最值得继续深挖的代码点

| 研究问题 | 首选入口 |
|---|---|
| 为什么一个 global step 有四次 rollout | _prepare_inputs、TRL training_step、RepeatSampler |
| 为什么只有最近两个 completion 被写出 | _prepare_metrics、log |
| 如何保留完整 rollout trace | _generate_and_score_completions 的 request_id/prompt_id 日志 |
| 为什么 loss 近零但 gradient 非零 | _compute_loss_and_metrics 的组 advantage 和 token 梯度 |
| LoRA reference 如何切换 | null_ref_context、disable_adapter |
| 4-bit 层如何注入 LoRA | SwiftRLHF.prepare_model、PEFT LoraModel |
| Transformers rollout 为什么慢 | TransformersEngine._infer、model.generate |
| 如何减少 old/ref scoring 成本 | _prepare_batch_inputs 与 KL/ratio 语义 |
| 如何降低零方差组比例 | reward plugin、num_generations、dynamic_sample |
| 如何做完整可复现实验 | args.json、training_args.bin、trainer_state.json、RNG state |

### 25.22 实测结论

这次三步运行验证了以下完整闭环：

~~~text
单卡脚本
  → 参数与模型解析
  → 4-bit Qwen3 加载
  → LoRA 注入
  → 原始 GSM8K messages
  → RepeatSampler 组采样
  → TransformersEngine 双候选生成
  → accuracy/format 奖励
  → 组相对 advantage
  → rollout 后 token 编码
  → old/reference/current 三类概率
  → GRPO + KL loss
  → 8 次梯度累积
  → optimizer step
  → 三步日志和 LoRA checkpoint
~~~

在该硬件上，Qwen3-1.7B 4-bit LoRA 的显存是可行的；真正昂贵的是无 vLLM rollout 的时间。更重要的是，实例说明“训练 step、micro-step、rollout 次数和 completion 数”是四个不同概念。掌握它们的换算关系，才算真正理解这套 GRPO 框架的执行全貌。

## 26. 8GB 单卡、Qwen3-1.7B、vLLM colocate 的三步实测案例

本节是在第 25 节 TransformersEngine 实例基础上增加的 vLLM 对照实验。它不是只根据源码推演，而是创建脚本、修复当前环境中的二进制兼容问题，并在同一张 8GB GPU 上真实完成 3 个 optimizer/global step 后得到的结果。

- 脚本：[qwen3_1_7b_qlora_vllm_colocate.sh](../../examples/train/grpo/local_8gb/qwen3_1_7b_qlora_vllm_colocate.sh)
- 命令：bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_vllm_colocate.sh
- 成功运行时间：2026-08-05 22:46:26 至 22:51:56
- 训练范围：3 个 optimizer/global step
- 训练结果：[checkpoint-3](../../output/grpo-qwen3-1.7b-qlora-vllm-8gb/v4-20260805-224623/checkpoint-3)
- 完整参数：[args.json](../../output/grpo-qwen3-1.7b-qlora-vllm-8gb/v4-20260805-224623/args.json)
- 逐步指标：[logging.jsonl](../../output/grpo-qwen3-1.7b-qlora-vllm-8gb/v4-20260805-224623/logging.jsonl)
- completion 快照：[completions.jsonl](../../output/grpo-qwen3-1.7b-qlora-vllm-8gb/v4-20260805-224623/completions.jsonl)
- 成功运行控制台日志：[console-3steps-run2.log](../../output/grpo-qwen3-1.7b-qlora-vllm-8gb/console-3steps-run2.log)

这些 output 链接是本机运行产物，不是源码依赖。清理 output 目录后链接会失效。

### 26.1 最终结论先行

本次运行验证了下面这条完整闭环：

~~~text
4-bit QLoRA 训练模型加载
  → GRPOTrainer.prepare_rollout
  → 临时把训练模型 offload 到 CPU
  → 创建进程内 GRPOVllmEngine
  → vLLM 以 bitsandbytes 4-bit 再加载同一 base model
  → vLLM sleep，训练模型回到 GPU
  → 第一次 rollout：唤醒权重、全量同步 base、同步 LoRA
  → rollout 时把训练模型 offload 到 CPU
  → 唤醒 KV cache、vLLM 生成并返回 token/log-prob
  → vLLM sleep、训练模型回到 GPU
  → accuracy/format reward
  → old/reference log-prob 与 group-relative advantage
  → current log-prob、GRPO ratio clip、reference KL
  → 8 次梯度累积后 optimizer step
  → 后续 global step 只同步 LoRA adapter
  → 3 步结束，保存 adapter checkpoint
~~~

结果表明：Qwen3-1.7B、训练侧 BNB 4-bit QLoRA、rollout 侧 vLLM BNB 4-bit、colocate、单卡 8GB 的组合在当前环境中能够完整运行。Trainer 报告峰值显存 6.37 GiB，未发生 OOM。

### 26.2 实际硬件和软件条件

| 项目 | 成功运行时的实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti |
| 物理显存 | 8192 MiB |
| 启动前可用显存 | 约 7.1 GiB |
| Compute Capability | 8.6 |
| PyTorch | 2.11.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| vLLM | 0.21.0 |
| Transformers | 4.57.6 |
| TRL | 0.29.1 |
| bitsandbytes | 0.50.0 |
| ms-swift | 4.3.0 |
| Python | 工程 .venv，Python 3.12 |

成功运行依赖的关键不是“能 import vllm”这一条，而是下面四层同时匹配：

1. NVIDIA 驱动能运行目标 CUDA runtime；
2. PyTorch 与 vLLM 扩展使用兼容的 CUDA、libtorch ABI；
3. vLLM 的 Python 依赖完整；
4. 可选 CUDA 扩展不能残留旧 PyTorch ABI。

当前仓库安装文档给出的新版本镜像组合包含 Torch 2.11、CUDA 13、vLLM 0.21。本环境最初是 Torch 2.6+cu124 与 vLLM 0.21 混装，必须先修复为一致组合。

TRL 0.29.1 会警告其显式测试的 vLLM 范围是 0.10.2～0.12.0。本次 ms-swift 自有 GRPOVllmEngine 路径在 vLLM 0.21 上完成了三步闭环，但这条实测证据不能替代生产级长跑验证。正式训练应固定完整版本矩阵，并先做更长的回归测试。

### 26.3 环境兼容修复及其工程含义

本次执行依次暴露了四层环境问题：

| 层级 | 现象 | 根因 | 处理 |
|---|---|---|---|
| CUDA/libtorch ABI | libcudart.so.13 找不到 | vLLM 0.21 按 Torch 2.11/CUDA 13 构建，环境仍是 Torch 2.6/CUDA 12.4 | 将 Torch 三件套升级为 2.11.0 |
| Python 依赖 | No module named cloudpickle | vLLM 原先以不完整依赖方式安装 | 按 vLLM 0.21 metadata 补齐依赖 |
| 动态库搜索 | libnvrtc.so.13 找不到 | CUDA 13 wheel 的库位于 site-packages/nvidia/cu13/lib，独立扩展未找到 | 在脚本设置 LD_LIBRARY_PATH |
| 可选扩展 ABI | flash_attn undefined symbol | flash-attn 是按旧 Torch 2.6 编译，vLLM 检测到后主动使用 | 移除旧 flash-attn；本脚本训练侧使用 SDPA |

实际使用的环境修复命令为：

~~~bash
uv pip install --python .venv/bin/python \
  'torch==2.11.0' 'torchaudio==2.11.0' 'torchvision==0.26.0'

uv pip install --python .venv/bin/python \
  'vllm==0.21.0' 'transformers>=4.56,<5.0' \
  'fsspec<=2026.2.0' 'pillow<12'

uv pip uninstall --python .venv/bin/python flash-attn
~~~

脚本的 preflight 不只导入 vllm 顶层包，还验证：

- 一个实际 CUDA tensor 运算；
- vllm._C 二进制扩展；
- vllm.LLM 高层 API；
- vLLM CuMemAllocator；
- bitsandbytes；
- 如果安装了 flash-attn，它必须能在当前 Torch 下成功导入。

这样可以把 ABI 错误提前到模型加载之前，避免先花时间加载 1.7B 模型再失败。

需要注意，uv pip check 仍报告两个与本次纯文本训练路径无关的问题：decord 平台不匹配，以及 Gradio 与 vLLM Web 依赖选出的 Starlette 版本冲突。这不影响本次 CLI GRPO，但说明“训练环境可运行”不等于“同一环境中的所有 UI/视频功能都完全一致”。正式工程更适合把训练、vLLM 服务和 Web UI 放在各自固定的环境或镜像中。

### 26.4 vLLM 脚本相对 Transformers 脚本改了什么

两份脚本保留相同的模型、数据、奖励、QLoRA 和 GRPO 算法参数。关键差异如下：

| 维度 | TransformersEngine 脚本 | vLLM colocate 脚本 |
|---|---|---|
| rollout 开关 | use_vllm=false | use_vllm=true |
| 部署模式 | 无 | vllm_mode=colocate |
| TP | 无 | tensor_parallel_size=1 |
| vLLM 显存池 | 无 | gpu_memory_utilization=0.35 |
| vLLM 上下文 | 无 | max_model_len=768 |
| vLLM 并发 | 无 | max_num_seqs=2 |
| 图执行 | Transformers eager generate | vLLM enforce_eager=true |
| prefix cache | 无独立 vLLM cache | 显式关闭 |
| LoRA 同步 | 同一个训练模型直接 generate | vLLM 原生 LoRA，rank=8 |
| 分时显存 | 不需要双模型切换 | sleep_level=1、offload_model=true |
| rollout log-prob | 没有独立 rollout backend 概率 | vLLM 返回 processed log-prob |
| off-policy 诊断 | 不记录 | log_rollout_offpolicy_metrics=true |

保留不变的关键项：

- Qwen3-1.7B；
- BNB 4-bit NF4 double quant；
- LoRA rank=8、alpha=16；
- B=1、G=2、steps_per_generation=2；
- gradient_accumulation_steps=8；
- beta=0.01、epsilon=[0.2, 0.28]；
- accuracy + 0.2 × format reward；
- max_steps=3。

因此两次实测可用于理解后端替换带来的调用链和资源变化；但由于 PyTorch/Transformers 版本已经变化，性能数字只能做方向性比较，不能当作严格 benchmark。

### 26.5 从 shell 到 vLLM engine 的真实启动路径

~~~mermaid
flowchart TD
    A["bash vLLM colocate 脚本"] --> B["环境 preflight"]
    B --> C[".venv/bin/swift rlhf"]
    C --> D["swift/cli/main.py 路由"]
    D --> E["单卡直接运行 swift/cli/rlhf.py"]
    E --> F["rlhf_main → SwiftRLHF"]
    F --> G["RLHFArguments / GRPOConfig"]
    G --> H["加载训练侧 BNB4 Qwen3"]
    H --> I["加载 GSM8K 500 条"]
    I --> J["注入 PEFT LoRA"]
    J --> K["TrainerFactory → GRPOTrainer"]
    K --> L["GRPOTrainer.prepare_rollout"]
    L --> M["RolloutTrainerMixin._prepare_vllm"]
    M --> N["offload_context：训练模型到 CPU"]
    N --> O["_prepare_vllm_engine"]
    O --> P["GRPOVllmEngine → VllmEngine"]
    P --> Q["EngineArgs → vLLM LLMEngine"]
    Q --> R["vLLM BNB4 加载相同 base"]
    R --> S["sleep level 1"]
    S --> T["训练模型回 GPU"]
    T --> U["Trainer.train max_steps=3"]
~~~

源码入口：

1. [CLI 路由](../../swift/cli/main.py#L90)
2. [RLHF CLI](../../swift/cli/rlhf.py#L3)
3. [SwiftRLHF pipeline](../../swift/pipelines/train/rlhf.py#L245)
4. [TrainerFactory](../../swift/trainers/trainer_factory.py#L12)
5. [GRPOTrainer.__init__](../../swift/rlhf_trainers/grpo_trainer.py#L96)
6. [prepare_rollout](../../swift/rlhf_trainers/rollout_mixin.py#L108)
7. [_prepare_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L195)
8. [_prepare_vllm_engine](../../swift/rlhf_trainers/rollout_mixin.py#L266)
9. [VllmEngine.__init__](../../swift/infer_engine/vllm_engine.py#L105)
10. [VllmEngine._prepare_engine](../../swift/infer_engine/vllm_engine.py#L225)

脚本清除了 NPROC_PER_NODE、NNODES、NODE_RANK，因此 CLI 不走 torchrun。vLLM 的 distributed_executor_backend 仍设置为 external_launcher，但 TP=1，实际没有第二个 tensor-parallel rank。

### 26.6 Trainer 构造阶段的顺序为何重要

[GRPOTrainer.__init__](../../swift/rlhf_trainers/grpo_trainer.py#L96) 的关键顺序是：

~~~text
_prepare_algorithm_params
  → super().__init__
  → prepare_rollout
      → _prepare_rollout_params
      → _prepare_scheduler
      → _prepare_vllm
      → _prepare_async_generate
      → split_batches
  → _prepare_rewards
  → _prepare_metrics
~~~

这说明：

- 数据集、训练模型、Accelerate/Trainer 状态先存在；
- vLLM engine 是 GRPOTrainer 内部的 rollout 组件，不是脚本外另起的服务；
- reward plugin 与 vLLM 彼此独立，engine 创建后才整理 reward 列表；
- split_batches 为后续 base/LoRA 权重同步预先建立参数名分组。

无 vLLM 时，GRPOTrainer 在构造末尾创建 TransformersEngine。使用 vLLM 时，这个分支不执行，self.engine 已由 prepare_rollout 设置为 GRPOVllmEngine。

### 26.7 vLLM engine 的配置如何从训练模型自动派生

[_prepare_vllm_engine](../../swift/rlhf_trainers/rollout_mixin.py#L266) 做了几个容易被 args.json 误导的自动推导。

#### 自动启用 vLLM 原生 LoRA

当 tuner_type=lora 且 vllm_enable_lora=true 时，传入 engine：

~~~text
enable_lora = true
max_loras = 1
max_lora_rank = args.lora_rank = 8
~~~

虽然脚本同时写了 vllm_max_lora_rank=8，当前 colocate 构造代码实际从训练侧 lora_rank 取 max_lora_rank。两者保持一致仍然是良好实践，尤其便于迁移到 server 模式和避免配置歧义。

#### 自动选择 bitsandbytes quantization

当训练模型的 model_info 表示 quant_method=bnb、quant_bits=4 时：

~~~text
vllm_quantization = bitsandbytes
~~~

这是 engine 构造时的局部变量，所以输出 args.json 中 vllm_quantization 仍是 null。不能据此误判 vLLM 使用了 BF16 全量基座；成功日志中的 vLLM checkpoint shard 加载和源码分支共同证明它走了 BNB 4-bit。

#### 自动请求 processed log-prob

vLLM >= 0.10.2 时设置：

~~~text
logprobs_mode = processed_logprobs
RequestConfig.logprobs = true
~~~

这些概率用于比较 vLLM rollout policy 与训练侧重算 policy 的偏差。

#### 最终 EngineArgs 的关键值

| EngineArgs | 值 | 功能 |
|---|---:|---|
| model | /home/cb/model/Qwen3-1.7B | rollout base |
| dtype | bfloat16 | 非量化计算 dtype |
| quantization | bitsandbytes | rollout base 4-bit 加载 |
| max_model_len | 768 | 限制 context/KV 规模 |
| max_num_seqs | 2 | 同时处理两条重复 prompt 行 |
| gpu_memory_utilization | 0.35 | vLLM 自身显存池规划参数 |
| enforce_eager | true | 避免 CUDA graph/compile 额外显存 |
| enable_lora | true | 动态加载训练 adapter |
| max_lora_rank | 8 | 与训练 rank 一致 |
| enable_sleep_mode | true | 支持阶段性 sleep/wake |
| enable_prefix_caching | false | 避免本实例的 cache 一致性复杂度 |
| tensor_parallel_size | 1 | 单卡 |

vllm_gpu_memory_utilization 只约束 vLLM 的规划，不是整个 Python 进程的总显存硬上限。训练模型、临时同步 tensor、CUDA context 和 optimizer/activation 都不在这一个比例里。

### 26.8 engine 初始化时的显存时序

colocate 的核心不是让两套模型永远同时常驻，而是分阶段交换 GPU 所有权。

~~~mermaid
sequenceDiagram
    participant T as "训练模型"
    participant M as "RolloutTrainerMixin"
    participant V as "vLLM engine"
    participant G as "GPU"

    T->>G: BNB4 Qwen3 + LoRA 已加载
    M->>T: 进入 offload_context
    T->>G: 参数迁到 CPU，释放 GPU
    M->>V: 创建 GRPOVllmEngine
    V->>G: BNB4 加载 Qwen3 base
    M->>V: reset cache + sleep(level=1)
    V->>G: 让出当前 rollout 阶段资源
    M->>T: 退出 offload_context
    T->>G: 训练参数重新加载
~~~

相关代码：

- [_prepare_vllm 的初始化 offload](../../swift/rlhf_trainers/rollout_mixin.py#L242)
- [GRPOTrainer.offload_context](../../swift/rlhf_trainers/grpo_trainer.py#L1965)
- [offload_model / load_model](../../swift/rlhf_trainers/rollout_mixin.py#L1200)

offload_model 会逐参数把训练权重迁到 CPU，退出 context 再迁回当前 GPU。因此 GPU 压力下降的代价是：

- CPU RSS 增大；
- PCIe/WSL 内存传输增加；
- 每次 rollout 都有模型迁移延迟。

实跑过程中 CPU RSS 采样最高约 6.3 GiB（约 6.6 GB）。这正是“用 CPU 内存和传输时间换 8GB GPU 可行性”的体现。

### 26.9 一个 global step 的工作量换算

本实例参数：

~~~text
B = per_device_train_batch_size = 1
D = world_size = 1
S = steps_per_generation = 2
G = num_generations = 2
A = gradient_accumulation_steps = 8
μ = num_iterations = 1
~~~

Trainer 配置推导：

~~~text
generation_batch_size = B × D × S = 2
~~~

因为 generation_batch_size=2 且 G=2，一次 vLLM infer 接收同一个 prompt 的两份重复采样位置，每个请求 n=1，最终得到两个 completion。框架没有把单个请求设置为 n=2，而是让 sampler 产生两行，再保证每行与 reward、request_id、token IDs 一一对应。

一次 rollout buffer 被两个 micro-step 消费：

~~~text
micro-step 0：生成 2 条 → 使用 buffer[0]
micro-step 1：复用同一生成批 → 使用 buffer[1]
micro-step 2：重新生成 2 条 → 使用 buffer[0]
micro-step 3：复用 → buffer[1]
micro-step 4：重新生成 2 条
micro-step 5：复用
micro-step 6：重新生成 2 条
micro-step 7：复用
optimizer.step → global_step + 1
~~~

因此：

~~~text
每个 global step 的 micro-step 数 = 8
每个 global step 的 vLLM 调用数 = 8 / 2 = 4
每个 global step 的 completion 数 = 4 × 2 = 8
三步总 vLLM 调用数 = 12
三步总 completion 数 = 24
~~~

但每个 global step 只同步一次权重。原因是 _fast_infer 用 state.global_step 与 _last_loaded_step 比较；同一个 accumulation cycle 内 global_step 不变，后面三次 rollout 会跳过同步。

### 26.10 每次 vLLM rollout 的精确阶段时序

[_fast_infer](../../swift/rlhf_trainers/rollout_mixin.py#L934) 是 colocate 的核心状态机：

~~~mermaid
sequenceDiagram
    participant G as "GRPOTrainer"
    participant T as "训练 QLoRA"
    participant V as "vLLM"
    participant R as "Reward/Scoring"

    G->>V: wake_up(tags=[weights])
    alt 本 global step 第一次 rollout
        G->>V: _move_model_to_vllm
        G->>V: reset prefix/encoder cache
    else 同一 global step 后续 rollout
        G->>G: _last_loaded_step 命中，跳过同步
    end
    G->>T: 进入 offload_context，训练模型到 CPU
    G->>V: wake_up(tags=[kv_cache])
    G->>V: infer(batch=2, n=1)
    V-->>G: text + token_ids + processed log-probs
    G->>V: reset prefix cache
    G->>V: sleep(level=1)
    G->>T: 退出 offload_context，训练模型回 GPU
    G->>R: reward、old/ref log-prob、advantage
    G->>T: current log-prob、loss、backward
~~~

日志中多次出现 Memory cleanup attempt，以及 expandable_segments 的切换，正是 wake/sleep、offload/reload 与 allocator 清理的外部表现。它们不能单独视为显存泄漏。

### 26.11 第一次为何全量同步，后续为何只同步 LoRA

[_move_model_to_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L449) 的判定是：

~~~text
如果 full tuning
或 base_sync_done=false
或 sleep_level=2
或没有启用 vLLM 原生 LoRA
    → _move_full_model_to_vllm
否则
    → _move_adapter_to_vllm
~~~

本实例使用 LoRA、sleep_level=1、vllm_enable_lora=true：

#### Global step 0 的第一次 rollout

1. base_sync_done 初始为 false；
2. 进入 _move_full_model_to_vllm；
3. 收集训练模型 base 参数并通过 vLLM inner_model.load_weights 更新；
4. 完成全部分组后执行 process_weights_after_loading；
5. 设置 base_sync_done=true；
6. 再调用 _move_adapter_to_vllm 加载 LoRA。

即使 vLLM engine 刚从同一本地 checkpoint 加载过 base，首次全量同步仍确保 rollout engine 与 Trainer 当前持有的参数和命名映射一致。

#### 后续 global step

base 冻结且已经同步，只走 _move_adapter_to_vllm：

1. 遍历训练参数分组；
2. 在需要时 gather 参数；
3. 临时 merge_adapter；
4. 用 get_peft_model_state_dict 提取 adapter tensor；
5. unmerge_adapter 恢复训练模型；
6. 构造 TensorLoRARequest；
7. engine.add_lora 原地更新 vLLM adapter。

本实例只有约 8.7163M 个可训练 LoRA 参数，远小于完整 base。checkpoint 中 adapter_model.safetensors 为 17,484,288 字节，说明 adapter-only 同步和保存都显著小于全模型。

实跑中的 PEFT 警告“4-bit linear merge/unmerge 可能因舍入造成生成差异”来自这条同步路径。实际 rollout/training 概率差较小，但在更长训练中仍应监控。

### 26.12 vLLM 请求、engine.step 和输出协议

调用链：

~~~text
GRPOTrainer._generate_completions
  → RolloutTrainerMixin._fast_infer
  → _infer_single_or_multi_turn
  → _rollout
  → _colocate_rollout
  → _engine_infer
  → GRPOVllmEngine.infer
  → VllmEngine.infer
      → template vLLM 编码
      → engine.add_request × 2
      → while engine.has_unfinished_requests
          → engine.step
      → ChatCompletionResponse
  → RolloutOutput
  → _postprocess_rollout_outputs
~~~

源码入口：

- [_generate_completions](../../swift/rlhf_trainers/grpo_trainer.py#L223)
- [_colocate_rollout](../../swift/rlhf_trainers/rollout_mixin.py#L1069)
- [_engine_infer](../../swift/rlhf_trainers/rollout_mixin.py#L1091)
- [GRPOVllmEngine.infer](../../swift/infer_engine/grpo_vllm_engine.py#L24)
- [VllmEngine.infer](../../swift/infer_engine/vllm_engine.py#L776)
- [_postprocess_rollout_outputs](../../swift/rlhf_trainers/rollout_mixin.py#L1128)

GRPOVllmEngine 在 infer 前检查固定 ID 的 LoRA 是否已经加载。如果存在，就为请求构造 LoRARequest；这使生成真正使用当前 adapter，而不是只用冻结 base。

输出统一为：

~~~text
RolloutOutput
  └─ ChatCompletionResponse
       ├─ choices[0].message.content
       ├─ choices[0].token_ids
       ├─ choices[0].logprobs
       ├─ finish_reason
       └─ prompt_token_ids
~~~

_postprocess_rollout_outputs 把它合回原始数据行，增加：

- assistant message；
- response_token_ids；
- rollout_logprobs；
- finish_reason；
- is_truncated；
- add_eos=false。

上层 reward、advantage 和 loss 只依赖统一的数据结构，因此不需要知道底层是 vLLM 还是 TransformersEngine。

### 26.13 reward 与 advantage 路径没有因 vLLM 改变

生成后仍执行：

~~~text
_score_completions
  → _compute_rewards_per_func
      → GSM8KAccuracy
      → GSM8KFormat
  → reward = accuracy + 0.2 × format
  → _compute_advantages
      → 按每 2 个 completion 分组
      → A_i = r_i - group_mean
~~~

插件源码：[gsm8k_plugin.py](../../examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py)

因为 scale_rewards=none，本实例不再除以组内标准差：

~~~text
两个候选奖励相同 → advantage = [0, 0]
两个候选奖励不同 → 一正一负
~~~

这解释了第 1 步 reward=1.2、但 loss 和 grad_norm 都为 0：四个 prompt group 中的两条候选都同奖，没有组内相对信号。

### 26.14 vLLM 实例中必须区分四类概率

无 vLLM 实例主要讨论 old、reference、current 三类概率。使用 vLLM 后多出 rollout probability：

| 概率 | 产生位置 | 作用 |
|---|---|---|
| rollout_per_token_logps | vLLM 生成时返回 | 记录真正采样后端的 token 概率 |
| old_per_token_logps | rollout 后，训练模型 no_grad 重算 | PPO/GRPO ratio 的分母策略 |
| ref_per_token_logps | 禁用 LoRA 后重算 base | reference KL |
| current per_token_logps | compute_loss 中有梯度前向 | ratio 分子、反向传播 |

本实例没有单独 ref_model。[_prepare_batch_inputs](../../swift/rlhf_trainers/grpo_trainer.py#L804) 通过 null_ref_context 暂时禁用 LoRA，在同一个 4-bit base 上计算 reference 概率，避免再复制一个 1.7B 模型。

为什么 rollout 和 old 不完全相同：

- 一个由 vLLM kernel 产生，一个由 Transformers/训练模型前向产生；
- 两边都有 BNB 4-bit，但权重装载和算子实现不同；
- LoRA 在 vLLM 侧动态应用；
- 4-bit adapter merge/extract 路径存在舍入；
- token 级 log-prob 的处理模式也可能产生微小数值差。

脚本只设置 log_rollout_offpolicy_metrics=true，rollout_importance_sampling_mode=null。因此本次只记录诊断指标，不把 rollout IS weight 乘入 loss。换言之，概率偏差被观测，但没有被算法修正。

### 26.15 rollout off-policy 指标如何阅读

三步的主要诊断值：

| step | training log-PPL | rollout log-PPL | log-PPL diff | PPL ratio | rollout k3_kl |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.07372 | 0.07177 | 0.00195 | 1.00196 | 0.000895 |
| 2 | 0.09432 | 0.08766 | 0.00667 | 1.00669 | 0.002290 |
| 3 | 0.07345 | 0.06667 | 0.00678 | 1.00681 | 0.001825 |

这说明本次 vLLM 与训练模型重算的平均概率非常接近，但不是逐 token 完全相同。三步都没有启用 rollout correction，所以上表只用于发现 backend drift。

不要把 rollout_correction/kl 与普通 kl 混为一谈：

- rollout_correction/kl：vLLM rollout 与训练侧 old policy 的差异诊断；
- kl：当前 policy 与禁用 LoRA 后 reference policy 的 K3 KL，用于 beta=0.01 的正则项。

### 26.16 从 current probability 到 GRPO loss

核心代码仍是 [_compute_loss_and_metrics](../../swift/rlhf_trainers/grpo_trainer.py#L1099)：

~~~text
log_ratio_t = current_logp_t - old_logp_t
ratio_t = exp(log_ratio_t)
clipped_t = clip(ratio_t, 1 - 0.2, 1 + 0.28)
policy_loss_t = -min(ratio_t × advantage, clipped_t × advantage)
K3_KL_t = exp(ref_logp_t - current_logp_t)
            - (ref_logp_t - current_logp_t) - 1
token_loss_t = policy_loss_t + 0.01 × K3_KL_t
sequence_loss = completion token 平均
batch loss = sequence 平均
~~~

由于一个 optimizer step 内还没有更新参数，current 与 old 通常非常接近；但 scalar loss 可以因正负 advantage 抵消到接近 0，梯度仍不一定为 0。第 2 步就是具体例子：loss≈7.45e-9，但 grad_norm=0.1172。

### 26.17 三步真实指标

| global step | loss | grad_norm | reward | reward_std | zero-std group 比例 | reference KL | completion 均长 | 生成截断比 | 峰值显存 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 1.2000 | 0 | 1.00 | 0 | 141.0 | 0 | 6.03 GiB |
| 2 | 7.45e-9 | 0.1172 | 0.8250 | 0.1768 | 0.75 | 0 | 127.375 | 0 | 6.06 GiB |
| 3 | -3.81e-6 | 0.01025 | 0.9250 | 0.03536 | 0.75 | 4.753e-4 | 203.25 | 0.125 | 6.37 GiB |

每项 reward function：

| step | accuracy mean | accuracy std | format mean | format std |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0 | 1.000 | 0 |
| 2 | 0.625 | 0.1768 | 1.000 | 0 |
| 3 | 0.750 | 0 | 0.875 | 0.1768 |

解释：

- Step 1：所有 group 同奖，没有学习信号；
- Step 2：部分 group 的 accuracy 不同，第一次出现有效梯度；
- Step 3：accuracy 在组内没有差异，格式奖励在部分 group 中不同，因此仍有较弱梯度；
- Step 3 在 Step 2 的有效更新之后才出现非零 reference KL；
- Step 3 completion 更长，所以该步耗时和峰值显存最高；
- Step 3 的 clip_ratio/high_mean≈0.000379，只有约 0.038% 有效 token 触发高侧 ratio clip。

三步 cosine scheduler 仍是 1e-5、5e-6、0。因此它只适合验证流程，不适合判断长期训练质量。

### 26.18 为什么 completion 快照与聚合指标看起来不一致

本次 completions.jsonl 每个日志事件只展示两个 completion，而每个 global step 实际产生八个。

原因是 [_prepare_metrics](../../swift/rlhf_trainers/grpo_trainer.py#L2220) 把 prompt、completion、reward、advantage 保存到 maxlen=generation_batch_size=2 的 deque。一个 global step 中执行四次 rollout，前面三次的明细会被后续 rollout 覆盖，但 reward 指标在日志前已经聚合。

例如 Step 2 的 completion 快照显示最后一个 group 两条都答错，advantages=[0, 0]；但 Step 2 聚合指标是 accuracy=0.625、reward_std=0.1768、grad_norm=0.1172。这不是冲突，而是：

~~~text
logging.jsonl：四个 group、八条 completion 的聚合指标
completions.jsonl：该步最后一个 group 的两条快照
~~~

文件末尾还会因训练结束时再次 log 而重复写入最后一个 step。它适合人眼抽样，不适合做完整轨迹审计。

若要研究每次权重同步后的 rollout，应在 _generate_and_score_completions 或 _postprocess_rollout_outputs 增加独立 append-only trace，至少记录：

- global_step；
- accumulation micro-step；
- rollout index；
- prompt_id / request_id；
- policy version；
- completion token IDs；
- rollout/old/ref log-prob；
- 每个 reward；
- advantage；
- finish_reason。

### 26.19 性能与显存实测

vLLM 三步运行：

| 项目 | 实测 |
|---|---:|
| Trainer train_runtime | 297.895 秒 |
| 主 pipeline 时间 | 约 330.25 秒 |
| train_steps_per_second | 0.010 |
| train_samples_per_second | 0.081 |
| Trainer 平均 optimizer step | 约 99.30 秒 |
| Step 1 step_time | 105.38 秒 |
| Step 2 step_time | 85.36 秒 |
| Step 3 step_time | 124.53 秒 |
| Trainer 报告峰值显存 | 6.37 GiB |
| 轮询 nvidia-smi 常见运行占用 | 约 5.1～5.2 GiB |
| 训练结束后系统基线 | 约 0.82 GiB |

与第 25 节 TransformersEngine 三步运行对比：

| 项目 | TransformersEngine | vLLM colocate | 观察 |
|---|---:|---:|---|
| train_runtime | 343.60 秒 | 297.895 秒 | vLLM 约快 13.3%，约 1.15× |
| 平均 optimizer step | 约 114.5 秒 | 约 99.3 秒 | 有改善，但不是数量级提升 |
| Trainer 峰值显存 | 2.46 GiB | 6.37 GiB | vLLM 高约 2.59× |
| sampled 总占用 | 约 3.47 GiB | 约 5.2 GiB | vLLM 多一套 engine/KV/同步状态 |
| 是否有 rollout backend drift | 无独立后端 | 有，已记录 | log-PPL diff 很小 |

为什么 vLLM 只获得约 1.15×，而不是常见的高并发大幅加速：

1. 每次只有 2 条 sequence，并发太小；
2. 单 prompt、G=2，无法充分发挥 continuous batching；
3. 每个 global step 有 4 次小 rollout；
4. 每次都有 wake/sleep 和训练模型 CPU/GPU 往返；
5. 每个 global step 还要同步 LoRA；
6. enforce_eager 和保守显存参数优先可运行性，不追求峰值吞吐；
7. reward 后仍要用训练模型做 old/ref/current 三类全序列 scoring；
8. Step 3 较长 completion 主导了自回归时间。

vLLM 的优势通常随更大 generation_batch_size、更多并发 prompt、较长 rollout 和独立 rollout GPU 增强。8GB 单卡 colocate 的主要价值是“在有限显存中验证加速后端的完整架构”，不是展示 vLLM 的峰值吞吐。

### 26.20 checkpoint 与运行产物

成功输出目录：

~~~text
output/grpo-qwen3-1.7b-qlora-vllm-8gb/v4-20260805-224623/
├── args.json
├── logging.jsonl
├── completions.jsonl
├── README.md
└── checkpoint-3/
    ├── adapter_model.safetensors   17,484,288 bytes
    ├── adapter_config.json
    ├── optimizer.pt                35,200,315 bytes
    ├── scheduler.pt
    ├── trainer_state.json
    ├── training_args.bin
    ├── rng_state.pth
    ├── args.json
    └── README.md
~~~

checkpoint 保存的是 LoRA adapter 和训练状态，不含完整 Qwen3 base。恢复或推理时仍需要原始 /home/cb/model/Qwen3-1.7B。

vLLM engine 本身不会保存额外 checkpoint。它是运行时副本，下一次训练启动时重新从 base 加载，再由 Trainer 同步当前 adapter。

### 26.21 失败尝试为何也值得保留

成功前的四个失败日志构成一条很清晰的分层诊断路径：

| 输出版本 | 停止位置 | 说明 |
|---|---|---|
| v0 | 导入 TRL GRPOTrainer / vLLM _C | CUDA/libtorch ABI 不一致，尚未构造 engine |
| v1 | TRL 导入 vLLM LLM | vLLM Python 依赖不完整 |
| v2 | _prepare_vllm_engine → CuMemAllocator | CUDA 13 动态库路径缺失 |
| v3 | vLLM 构造 Qwen3 rotary layer | 旧 flash-attn ABI 不匹配 |
| v4 | 完成 3 steps | 训练、rollout、同步、保存闭环成功 |

对应日志：

- [ABI 失败](../../output/grpo-qwen3-1.7b-qlora-vllm-8gb/console-3steps.log)
- [依赖失败](../../output/grpo-qwen3-1.7b-qlora-vllm-8gb/console-3steps-cuda13.log)
- [NVRTC 路径失败](../../output/grpo-qwen3-1.7b-qlora-vllm-8gb/console-3steps-final.log)
- [flash-attn 失败](../../output/grpo-qwen3-1.7b-qlora-vllm-8gb/console-3steps-run.log)

这也给出一个通用排错原则：

~~~text
先判断失败属于哪一层
  L0 Python package 是否存在
  L1 CUDA/libtorch ABI 是否匹配
  L2 动态库能否被 loader 找到
  L3 engine 能否构造模型层
  L4 base/LoRA 权重能否同步
  L5 rollout 能否返回 token/log-prob
  L6 reward/advantage/loss 是否正常
  L7 checkpoint 是否完整
~~~

不要把 engine 构造前的 ImportError 误判为 8GB OOM，也不要在没有进入 rollout 前调整 GRPO 超参数。

### 26.22 本实例体现的框架设计

#### 设计一：算法层与生成后端解耦

_generate_and_score_completions、reward、advantage、loss 没有因 vLLM 改写。后端差异被限制在 _fast_infer、engine 和权重同步层。

#### 设计二：统一输出协议

TransformersEngine 和 GRPOVllmEngine 最终都返回 ChatCompletionResponse/RolloutOutput，保证 token IDs、文本、finish reason 和 log-prob 可以进入相同后处理。

#### 设计三：训练态与 rollout 态分时复用 GPU

sleep/wake 管理 vLLM，offload/load 管理训练模型。两者组成一个显式状态机，而不是依赖 CUDA allocator 偶然释放。

#### 设计四：冻结 base 只同步一次

base_sync_done 与 rollout_enable_lora 让后续更新只传 adapter；_last_loaded_step 又避免同一 gradient accumulation cycle 内重复同步。

#### 设计五：采样概率可观测

vLLM 返回 processed log-prob，训练侧重算 old log-prob，从而能量化 backend drift，而不是默认两者严格相等。

#### 设计六：生成 batch 与 optimizer batch 解耦

steps_per_generation=2 控制每次生成 2 条；gradient_accumulation_steps=8 控制 8 个 micro-step 后更新。这样可以独立调节 rollout 并发和优化 batch，但也要求理解四种计数。

### 26.23 当前配置的局限与正式训练建议

1. **三步太短**：只能验证架构，不能判断 reward 趋势。
2. **G=2 零方差高**：三步 zero-std group 比例为 1.0、0.75、0.75，多数组没有学习信号。
3. **cosine 三步归零**：第三步 learning rate 已为 0。
4. **vLLM 并发过小**：max_num_seqs=2 主要满足可行性，吞吐优势有限。
5. **CPU offload 有代价**：如果 GPU 更大，关闭 offload_model 可能更快。
6. **TRL 有版本警告**：长训练前需要 50～100 step 回归，以及 checkpoint 恢复测试。
7. **ModelScope 在线 HEAD 不稳定**：本次出现 SSL EOF 后成功回退缓存；可复现实验应固定本地数据快照。
8. **QLoRA LoRA merge/unmerge 有舍入警告**：持续监控 rollout gap，必要时对比普通 LoRA。
9. **completions.jsonl 非全量**：正式研究 reward 必须增加轨迹日志。

8GB 上的调参优先级：

~~~text
如果 OOM：
  1. max_completion_length 384 → 256
  2. vllm_max_model_len 768 → 512
  3. vllm_gpu_memory_utilization 0.35 → 0.30 / 0.25
  4. 保持 max_num_seqs=2
  5. 改用 Qwen3-0.6B

如果显存稳定但太慢：
  1. 评估是否减少每个 global step 的小 rollout 次数
  2. 在更小模型上提高 generation_batch_size/max_num_seqs
  3. 比较 offload_model true/false
  4. 有第二张 GPU 时改用 server 模式，隔离训练与 rollout
~~~

正式训练还应把 MAX_STEPS 设置为足够长的值，并重新设计 scheduler、save_steps、eval 频率，不能直接沿用三步演示配置。

### 26.24 后续深入代码的导航地图

| 想研究的问题 | 首选代码入口 | 建议观测 |
|---|---|---|
| vLLM 参数如何构造 | [_prepare_vllm_engine](../../swift/rlhf_trainers/rollout_mixin.py#L266) | EngineArgs、quantization、LoRA kwargs |
| vLLM 如何加载模型 | [VllmEngine](../../swift/infer_engine/vllm_engine.py#L105) | from_engine_args、load_model |
| sleep/wake 如何切换 | [_fast_infer](../../swift/rlhf_trainers/rollout_mixin.py#L934) | is_sleeping、tags、显存曲线 |
| offload 如何实现 | [offload_model](../../swift/rlhf_trainers/rollout_mixin.py#L1200) | CPU RSS、PCIe 时间 |
| 为什么首次全量同步 | [_move_model_to_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L449) | base_sync_done |
| 全量参数如何映射 | [_move_full_model_to_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L773) | state_dict 名称、load_weights |
| LoRA 如何同步 | [_move_adapter_to_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L478) | TensorLoRARequest、add_lora |
| 为何一步只同步一次 | [_fast_infer](../../swift/rlhf_trainers/rollout_mixin.py#L946) | global_step、_last_loaded_step |
| vLLM 如何执行 batch | [VllmEngine.infer](../../swift/infer_engine/vllm_engine.py#L776) | add_request、engine.step |
| LoRA 如何随请求应用 | [GRPOVllmEngine.infer](../../swift/infer_engine/grpo_vllm_engine.py#L24) | list_loras、LoRARequest |
| rollout log-prob 如何保存 | [_postprocess_rollout_outputs](../../swift/rlhf_trainers/rollout_mixin.py#L1128) | choice.logprobs |
| rollout/old 如何对齐 | [_prepare_batch_inputs](../../swift/rlhf_trainers/grpo_trainer.py#L804) | completion_mask、token count |
| backend drift 指标 | [_compute_rollout_offpolicy_metrics](../../swift/rlhf_trainers/grpo_trainer.py#L2542) | PPL、KL、chi-square |
| IS correction 如何启用 | [_apply_rollout_importance_sampling](../../swift/rlhf_trainers/grpo_trainer.py#L2450) | mode、threshold、weights |
| 每步为何有 4 次 rollout | [_prepare_inputs](../../swift/rlhf_trainers/grpo_trainer.py#L195) | _step、buffer、SPG |
| reward 与 advantage | [_compute_advantages](../../swift/rlhf_trainers/grpo_trainer.py#L421) | group mean、zero std |
| GRPO loss 与 clip | [_compute_loss_and_metrics](../../swift/rlhf_trainers/grpo_trainer.py#L1099) | ratio、KL、mask |
| 如何做全量轨迹日志 | [_generate_and_score_completions](../../swift/rlhf_trainers/grpo_trainer.py#L242) | request_id、policy version |

### 26.25 复现命令

执行默认三步：

~~~bash
bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_vllm_colocate.sh
~~~

只验证环境和最终命令：

~~~bash
DRY_RUN=1 \
bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_vllm_colocate.sh
~~~

桌面占用较高时的保守配置：

~~~bash
MAX_LENGTH=512 \
MAX_COMPLETION_LENGTH=256 \
bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_vllm_colocate.sh
~~~

如果本地 1.7B 不存在，脚本会回退到 Qwen/Qwen3-1.7B。也可以显式指定：

~~~bash
VLLM_MODEL=Qwen/Qwen3-0.6B \
MAX_LENGTH=512 \
MAX_COMPLETION_LENGTH=256 \
bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_vllm_colocate.sh
~~~

进行正式长训练前，应先修改 save_steps，并根据总步数重新设置 scheduler：

~~~bash
MAX_STEPS=100 \
DATASET='modelscope/gsm8k#2000' \
bash examples/train/grpo/local_8gb/qwen3_1_7b_qlora_vllm_colocate.sh
~~~

### 26.26 vLLM 实例的整体全貌

这次实测说明，“rollout 使用 vLLM”并不只是把 model.generate 换成另一个函数。它新增了一套完整的运行时子系统：

~~~text
环境 ABI
  → 第二份量化 base 的 engine 生命周期
  → 训练/推理 GPU 所有权切换
  → base 与 adapter 权重同步协议
  → KV/prefix cache 一致性
  → 统一请求/响应协议
  → rollout probability 采集
  → backend drift 诊断或 IS correction
~~~

而 GRPO 算法主干仍保持稳定：

~~~text
同 prompt 多候选
  → reward functions
  → group-relative advantage
  → old/current ratio
  → reference KL
  → token mask 与 clip
  → 梯度累积和 optimizer step
~~~

掌握这两层的边界最重要：遇到 libcudart、engine load、LoRA add_lora、sleep/wake 问题时，应在 rollout 基础设施层排查；遇到 zero-std、advantage、KL、clip、reward 不上升时，才进入 GRPO 算法和数据层排查。

---

## 27. Qwen3.5-35B-A3B GRPO：LoRA/Full × colocate/external 标准场景源码推导

### 27.1 分析边界、型号校正与证据等级

用户所述“Qwen3-35B-A3B”在当前仓库中的准确官方型号是 `Qwen/Qwen3.5-35B-A3B`。仓库同时注册了 `Qwen/Qwen3.6-35B-A3B`，两者都走 `qwen3_5_moe` 模型类型和 `qwen3_5` 模板；但当前有可直接交叉检查的 Megatron GRPO 实例是 Qwen3.5，所以本节以 Qwen3.5 为主线。若切换为 Qwen3.6，调用链不变，但必须重新核对实际 `config.json`、mcore-bridge 版本、vLLM 模型支持和显存。

本节所说的 **external** 是部署拓扑名称：训练进程与 rollout 服务进程/机器分离。当前 CLI 中的真实参数不是 `--vllm_mode external`，而是：

~~~bash
--use_vllm true \
--vllm_mode server
~~~

后文统一写成“external/server”，避免部署概念和参数名混淆。

因当前环境不能加载并训练 35B MoE，本节没有执行命令，也不给出伪造的 step time、tokens/s 或峰值显存。结论按下列证据等级区分：

| 等级 | 含义 | 本节的代表内容 |
|---|---|---|
| A：型号直接实例 | 仓库就是该型号、该后端的脚本 | Qwen3.5-35B-A3B Megatron GRPO LoRA colocate |
| B：同类架构迁移 | 当前型号没有脚本，但有 Qwen3-30B-A3B MoE 对应场景 | full colocate 的 TP/PP/EP、offload 和 optimizer 配置 |
| C：实现逻辑推导 | 由通用 external 脚本与当前源码组合得到 | 多机 external/server 的客户端、NCCL 同步和时序 |
| D：容量规划 | 只能通过公式与目标机器实测确认 | 能否在某类 8 卡机器上跑通 full colocate |

四种组合的证据强度如下：

| 训练方式 | rollout 拓扑 | 证据 | 定位 |
|---|---|---|---|
| LoRA | colocate | A | 仓库官方主路径，最适合先验证 |
| Full | colocate | B+C | 高显存单机的容量压力测/研究路径 |
| LoRA | external/server | A 模型 + C 拓扑 | 训练显存与 rollout 显存隔离，但 MoE 原生 LoRA 同步有兼容性边界 |
| Full | external/server | B+C | 生产级资源隔离的主要候选，但整模权重同步是核心瓶颈 |

直接证据入口：

- [Qwen3.5-35B-A3B GRPO LoRA 官方实例](../../examples/models/qwen3_5/mcore_grpo_moe.sh)；
- [Qwen3 MoE LoRA colocate 实例](../../examples/megatron/grpo/moe_colocate_lora.sh)；
- [Qwen3 MoE full colocate 实例](../../examples/megatron/grpo/moe_colocate_full.sh)；
- [Transformers external MoE LoRA 实例](../../examples/train/grpo/external/moe_lora.sh)；
- [Transformers external MoE full 实例](../../examples/train/grpo/external/moe_full.sh)；
- [external 模式 README](../../examples/train/grpo/external/README.md)。

### 27.2 模型特性：A3B 只代表激活计算，不代表只需 3B 显存

[Qwen3.5 模型注册](../../swift/model/models/qwen.py#L1419) 表明：

- `model_type=qwen3_5_moe`；
- template 为 `qwen3_5`；
- Transformers 类为 `Qwen3_5MoeForConditionalGeneration`；
- loader 继承 Qwen3 VL loader，模型标记包含 vision/video；
- 依赖包含 `transformers>=5.2.0`、`qwen_vl_utils>=0.0.14`和 `decord`；
- loader 还会为 Qwen3.5 gated-delta/linear-attention 路径安装 sequence-parallel patch。

对 GRPO 容量规划最重要的是：

~~~text
35B 总参数：决定基础权重、全参梯度、优化器状态、reference copy、权重同步量
A3B 激活参数：主要影响每 token 前向/反向的计算量
KV cache：由层数、hidden/head 结构、序列长度、并发数和 dtype 决定
~~~

所以“A3B”不能用来估算训练权重只占 3B 模型的显存。以 BF16 只计一份 35B 参数为下界，原始权重字节量约为 `35e9 × 2 ≈ 70 GB`；这还没包含视觉组件差异、padding/alignment、梯度、master weights、Adam 状态、activation、KV cache 和内存碎片。

文本数学 GRPO 的标准配置应显式保持：

~~~bash
--freeze_llm false \
--freeze_vit true \
--freeze_aligner true \
--enable_thinking false
~~~

这表示 LoRA 或 full 主要训练语言模型，而不是在文本 reward 下更新视觉塔。如果要做字面意义上的“所有模态全参”，必须把后两项设为 false，并同时更换为多模态数据、reward 和 rollout 容量规划；这不属于本节的标准数学 GRPO 场景。

### 27.3 为什么标准主线选 Megatron-SWIFT

Qwen3.5-35B-A3B 同时需要 MoE 专家并行和大模型分片，所以本节不以 Transformers/DeepSpeed 为主线，而使用：

~~~bash
megatron rlhf --rlhf_type grpo
~~~

Megatron 训练侧与 rollout 侧有两套独立的并行视图：

| 侧 | 并行维度 | 作用 |
|---|---|---|
| Megatron policy/ref | TP、PP、CP、EP、DP | 切分训练权重、层、序列、专家和数据 |
| vLLM colocate | TP 为主，多个 TP group 形成生成副本 | 在同一批 rank/GPU 上重新组织推理并行 |
| vLLM external | TP、DP，可选 expert parallel | 在独立 rollout 节点上组织推理；rollout 不支持 PP>1 |

训练 TP/PP/EP 不需要与 vLLM TP/DP 相同。两者通过 mcore-bridge 的 HF/vLLM 名称导出与 `load_weights` 边界解耦。这是框架能够使用“训练 EP=8，vLLM TP=2”这类配置的根本原因。

但需要注意，下列公式中 EP 不直接出现在 Megatron GRPO 用于 batch 对齐的 DP 分母里：

~~~text
training_world_size = NNODES × NPROC_PER_NODE
DP = training_world_size / (TP × PP × CP)
generation_batch_size = global_batch_size × steps_per_generation
rollout_prompt_count = generation_batch_size / num_generations
per_device_generation_batch_size = generation_batch_size / training_world_size
~~~

[Megatron GRPO 参数校验](../../swift/megatron/arguments/megatron_args.py#L257) 还要求：

1. `generation_batch_size` 能被 `num_generations` 整除；
2. `rollout_prompt_count` 能被 DP 整除；
3. 每个 DP rank 的 prompt 数能被 `micro_batch_size` 整除；
4. `per_device_generation_batch_size >= 1`。

这些是脚本“能否启动”的整数条件，与显存是两类独立问题。

### 27.4 共享的 Megatron GRPO 端到端调用链

四种组合都先走同一条 policy 训练主干，只在 rollout engine 初始化、生成请求和权重同步三处分叉：

~~~mermaid
flowchart TD
    A["megatron rlhf --rlhf_type grpo"] --> B["MegatronRLHFArguments 解析与 batch/并行校验"]
    B --> C["MegatronRLHF.run / MegatronSft 通用 pipeline"]
    C --> D["加载 dataset、qwen3_5 template 和 processor"]
    C --> E["get_mcore_model + mcore-bridge 加载 policy"]
    E --> F{"tuner_type"}
    F -- "LoRA" --> G["prepare_mcore_model 注入 adapter"]
    F -- "full" --> H["构造独立 frozen ref_models"]
    G --> I["MegatronGRPOTrainer"]
    H --> I
    I --> J["_init_rollout_engine"]
    J --> K{"vllm_mode"}
    K -- "colocate" --> L["本地 GRPOVllmEngine"]
    K -- "server" --> M["VLLMClient 连接 swift rollout"]
    L --> N["_replace_data_iterator"]
    M --> N
    N --> O["每 steps_per_generation 次触发新 rollout"]
    O --> P["权重同步 + G 个 completion"]
    P --> Q["reward_funcs 评分"]
    Q --> R["后编码 + old/ref/rollout log-prob"]
    R --> S["组内 reward 归一化 + advantage"]
    S --> T["切分为 steps_per_generation 个训练缓冲"]
    T --> U["Megatron forward/backward pipeline"]
    U --> V["GRPO ratio/clip/KL/mask loss"]
    V --> W["optimizer step / log / checkpoint"]
    W --> O
~~~

具体入口如下：

| 阶段 | 函数/文件 | 功能 |
|---|---|---|
| CLI pipeline | [MegatronRLHF](../../swift/megatron/pipelines/train/rlhf.py) | GRPO trainer 工厂、server client 准备 |
| 训练/参考模型 | [MegatronRLHFTrainer.prepare_model](../../swift/megatron/trainers/rlhf_mixin.py#L27) | LoRA 用 disable-adapter 做 ref；full 创建独立 ref copy |
| rollout 初始化 | [MegatronRolloutMixin._init_rollout_engine](../../swift/megatron/trainers/rollout_mixin.py#L195) | colocate/server 分流、sleep/offload 准备 |
| rollout 触发 | [MegatronGRPOTrainer._replace_data_iterator](../../swift/megatron/trainers/grpo_trainer.py#L275) | 新建 rollout batch 或取缓存 mini-batch |
| prompt 重复 | [get_local_rollout_batch](../../swift/megatron/trainers/grpo_trainer.py#L1745) | 每 prompt 复制 G 次并切给 rollout group |
| 生成总控 | [_generate_completions](../../swift/megatron/trainers/grpo_trainer.py#L509) | wake、同步、offload、rollout、sleep |
| 奖励 | [_score_completions](../../swift/megatron/trainers/grpo_trainer.py#L710) | 规则/RM/Gym reward |
| old/ref 概率 | [_maybe_compute_logps](../../swift/megatron/trainers/grpo_trainer.py#L1033) | 更新前 policy 和 reference 的逐 token log-prob |
| advantage | [_compute_advantages](../../swift/megatron/trainers/grpo_trainer.py#L797) | 按 G 分组、基线和标准化 |
| loss | [Megatron GRPO loss](../../swift/megatron/trainers/grpo_trainer.py#L1228) | importance ratio、clip、KL、长度和 entropy mask |

### 27.5 LoRA 与 full 的 reference policy 实际差异

这是 35B 场景最容易被忽略的显存差异之一。

| 方式 | reference policy 实现 | 是否额外构造 35B ref model | `beta=0` 的影响 |
|---|---|---:|---|
| LoRA | `null_ref_context()` 临时 disable policy adapter | 否 | 不计算 ref log-prob |
| Full | `prepare_model()` 通过 `get_mcore_model()` 构造 frozen `ref_models` | 是 | 不计算 ref log-prob，但当前代码仍已构造/加载 ref model |

当前 [MegatronRLHFTrainer.prepare_model](../../swift/megatron/trainers/rlhf_mixin.py#L27) 的条件是 `tuner_type == 'full' and rlhf_type not in ['rm', 'gkd']`，没有把 `beta != 0` 放进构造条件。因此，对当前版本代码而言：

> 把 full GRPO 设为 `--beta 0.0` 能省去 reference 前向计算，但不能假设它会自动省去 reference 模型权重显存。

这也是 full colocate 比 LoRA colocate 难得多的原因：policy 训练态、frozen ref、optimizer state 和 vLLM runtime copy 需要同时进行容量规划。

### 27.6 rollout 权重同步决策树

[MegatronRolloutMixin._move_model_to_vllm](../../swift/megatron/trainers/rollout_mixin.py#L324) 的真实决策可简化为：

~~~text
如果 tuner_type != lora
  → 整模同步
否则如果 base 还没同步
  → 整模同步，然后可选同步 adapter
否则如果 sleep_level == 2
  → 整模同步
否则如果 rollout engine 没有启用原生 LoRA
  → 合并 LoRA 到 base 视图 → 整模同步 → 取消合并
否则
  → 只同步 LoRA adapter
~~~

| tuner/rollout 状态 | 首次 rollout | 后续新 rollout | 网络/本机数据量级 |
|---|---|---|---|
| Full | full weights | full weights | 整模 |
| LoRA + `vllm_enable_lora=false` | merge 后 full weights | merge 后 full weights | 整模 |
| LoRA + 原生 vLLM LoRA + sleep<2 | base + adapter | adapter only | 首次整模，之后 LoRA 量级 |
| LoRA + 原生 vLLM LoRA + sleep=2 | base + adapter | 仍触发 full + adapter | 基本失去增量同步优势 |

两个容易混淆的参数必须分开：

- `--merge_lora true` 在 Megatron trainer 的 checkpoint 逻辑中控制是否另外导出 merged checkpoint；
- rollout 前是否临时 merge 由 `rollout_enable_lora` 和 `_move_full_model_to_vllm()` 决定，并不等同于 checkpoint 参数。

官方 Qwen3.5-35B-A3B GRPO LoRA 脚本没有设置 `--vllm_enable_lora true`，因而使用默认 false：它会在每次新 rollout 前临时 merge LoRA，导出整模权重给 vLLM，再 unmerge。这是一条传输更重、但对 MoE LoRA 更保守的路径。

### 27.7 Qwen3.5 MoE 的原生 vLLM LoRA 边界

Transformers rollout mixin 在检测到 MoE + `vllm_enable_lora` 时明确警告：vLLM 对 MoE 专家层中的 LoRA 支持可能报错；对 multimodal model 也有额外警告。当前 Qwen3.5 官方 Megatron 实例通过不开原生 LoRA 来避开该边界。

因此标准建议是：

1. **首先跑可靠路径**：`target_modules=all-linear` + `vllm_enable_lora=false`，接受每次新 rollout 同步整模的代价；
2. **再做增量同步实验**：只对当前 vLLM 确认支持的非专家模块注入 LoRA，启动 server/engine 的 LoRA，校验名称映射、第二次 adapter-only 同步和 rollout 数值；
3. 不要在没有对当前 vLLM 版本做实验时，直接宣称 `all-linear + MoE + native LoRA` 是可生产使用的组合。

对 external/server 模式，是 rollout server 启动参数中的 `--vllm_enable_lora true` 决定 server 能力。Trainer 会调用 `/get_engine_type/` 读取 `enable_lora`，然后广播给其他训练 rank；不能只在 trainer 侧写一个 LoRA 开关就假设服务器已启用。

### 27.8 单机共卡 colocate 的进程与显存状态机

colocate 不是“policy 和 vLLM 同时常驻并同时计算”的简单结构，而是同一组 rank/GPU 在训练态与 rollout 态之间切换。

~~~mermaid
stateDiagram-v2
    [*] --> TrainerResident
    TrainerResident --> VllmInit: "初始化 engine 前进入 offload_context"
    VllmInit --> VllmSleeping: "GRPOVllmEngine 创建后 sleep(level)"
    VllmSleeping --> WeightsAwake: "新 rollout: wake_up(weights)"
    WeightsAwake --> WeightSync: "mcore-bridge export + vLLM load_weights"
    WeightSync --> TrainerOffloaded: "policy/ref/optimizer 按配置 offload"
    TrainerOffloaded --> KVReady: "wake_up(kv_cache)"
    KVReady --> Rollout: "vLLM infer"
    Rollout --> VllmSleeping: "reset cache + sleep"
    VllmSleeping --> TrainerResident: "policy/ref/optimizer reload"
    TrainerResident --> ForwardBackward: "old/ref 预计算与后续 current-policy 训练"
    ForwardBackward --> TrainerResident: "optimizer step"
~~~

当 `--offload_model true --offload_optimizer true` 时，[offload_context](../../swift/megatron/trainers/rollout_mixin.py#L777) 会在 rollout 期间把 policy/ref 和 optimizer 移到 CPU，结束后再加载回 GPU。这个状态机能降低同时常驻量，但会带来 CPU RAM 占用、PCIe/NVLink 数据搬运和 allocator 同步代价。

一个尤其重要的峰值点是：`_move_model_to_vllm()` 在进入 rollout 的 `offload_context` 之前执行。也就是说，mcore-bridge 导出和 vLLM 重载权重时，policy 训练态仍可能在 GPU 上。对 35B full 而言，“权重导出峰值”往往比稳态 forward 显存更需要优先实测。

### 27.9 场景一：单机共卡 LoRA + colocate

#### 27.9.1 标准基线脚本

这是四种组合中证据最强的一条。仓库已给出完整脚本 [mcore_grpo_moe.sh](../../examples/models/qwen3_5/mcore_grpo_moe.sh)，核心配置是：

~~~bash
SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --enable_thinking false \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --pipeline_model_parallel_size 1 \
    --moe_permute_fusion true \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --global_batch_size 64 \
    --micro_batch_size 1 \
    --steps_per_generation 2 \
    --num_generations 8 \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 9192 \
    --max_length 1000 \
    --max_completion_length 8192 \
    --tuner_type lora \
    --target_modules all-linear \
    --lr 5e-5 \
    --bf16 true \
    --beta 0.00 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --loss_type grpo \
    --overlong_filter true \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune \
    --freeze_vit true \
    --freeze_aligner true \
    --attention_backend flash \
    --sequence_parallel true
~~~

这里只省略了部分日志和保存参数，原脚本应作为实际执行基准。原脚本的 `--merge_lora true` 主要为 checkpoint 额外产出 merged 权重，不代表启用了 vLLM 原生 LoRA。

#### 27.9.2 并行与 batch 算术

~~~text
training world = 8
TP=1, PP=1, CP=1
DP = 8 / (1×1×1) = 8
EP = 8

global_batch_size = 64 completions
steps_per_generation = 2
generation_batch_size = 64×2 = 128 completions
num_generations = 8
rollout prompts = 128/8 = 16 prompts
prompts per DP rank = 16/8 = 2
per_device_generation_batch_size = 128/8 = 16 completions
~~~

`_replace_data_iterator()` 每逢 `_step % 2 == 0` 从 data iterator 取每个 DP rank 的 2 个 prompt，每个重复 8 次，产生 16 个本地 completion。这些结果完成 reward、后编码、old/ref log-prob 和 advantage 后，被分成 2 个训练缓冲，分别供两个 optimizer step 消费。

vLLM TP=2 与 Megatron TP=1 无关。8 个 rank 在 rollout 视图中形成 4 个 vLLM TP group。每个 TP group 先 `all_gather_object` 合并本组请求，执行 infer，然后每个 rank 只取回属于自己的 output slice。

#### 27.9.3 推导的三步时序

假设只跑 3 个 optimizer step，原脚本的 `steps_per_generation=2` 导致：

| optimizer step | 是否新 rollout | 权重同步 | 训练数据 |
|---:|---:|---|---|
| 0 | 是 | 官方保守配置为 merge LoRA 后整模同步 | rollout batch A 的第 1/2 个缓冲 |
| 1 | 否 | 无 vLLM 同步 | rollout batch A 的第 2/2 个缓冲 |
| 2 | 是 | 再次 merge LoRA 后整模同步 | rollout batch B 的第 1/2 个缓冲 |

对 step 0/2，阶段顺序是：

~~~text
wake vLLM weights
  → merge LoRA
  → mcore-bridge export full weights
  → patch vLLM MoE loader + load_weights
  → finish_vllm_weight_reload
  → unmerge LoRA
  → reset prefix/encoder cache
  → offload policy/optimizer
  → wake KV cache
  → vLLM rollout
  → vLLM sleep
  → reload policy/optimizer
  → reward + encode + old log-prob
  → advantage + buffer
  → current-policy forward/backward/update
~~~

step 1 直接消费已经完成 old/ref/reward/advantage 的缓冲，不再做 vLLM rollout。因此正常情况下，step 0/2 比 step 1 重；具体比例由 completion 长度、CPU offload 带宽和权重导出决定，本节不给出未实测数字。

#### 27.9.4 该场景的主要调参方向

| 瓶颈 | 首选方向 | 代价 |
|---|---|---|
| KV cache OOM | 降 `max_completion_length`、`vllm_max_model_len`、`vllm_gpu_memory_utilization` | rollout 更短/并发更低 |
| 权重切换慢 | 在显存允许时对比关闭 model/optimizer offload | 驻留显存增加 |
| 整模同步慢 | 实验非专家层 native vLLM LoRA | 需额外兼容性和数值校验 |
| rollout 太频繁 | 增大 `steps_per_generation` | policy staleness/off-policy 程度增加 |
| 生成吞吐低 | 在显存许可时增大 generation batch/max_num_seqs | KV cache 增加 |

### 27.10 场景二：单机共卡 Full + colocate

#### 27.10.1 配置来源与可行性定位

当前仓库没有直接的 Qwen3.5-35B-A3B full GRPO colocate 脚本。下面的基线由 [Qwen3-30B-A3B full colocate](../../examples/megatron/grpo/moe_colocate_full.sh) 迁移型号，并保留 Qwen3.5 官方脚本的文本长度和模型设置。

> 这是“高显存单机的容量验证模板”，不是当前环境已证明可运行的脚本。它同时存在 policy、ref、optimizer 以及 vLLM 权重/KV 压力，可行性取决于单卡显存、主机 RAM、PCIe/NVLink 和当前 mcore/vLLM 版本。

推导配置：

~~~bash
SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --enable_thinking false \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 2 \
    --moe_permute_fusion true \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --global_batch_size 8 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 8 \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 8 \
    --vllm_max_model_len 9192 \
    --max_length 1000 \
    --max_completion_length 8192 \
    --tuner_type full \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --lr 1e-6 \
    --bf16 true \
    --beta 0.0 \
    --loss_type grpo \
    --overlong_filter true \
    --sleep_level 2 \
    --offload_model true \
    --offload_optimizer true \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer \
    --recompute_granularity selective \
    --finetune \
    --attention_backend flash \
    --sequence_parallel true
~~~

#### 27.10.2 并行和三步时序

~~~text
world=8, TP=4, PP=2, CP=1
DP = 8/(4×2×1) = 1
EP=4
global_batch=8, steps_per_generation=1
generation_batch=8 completions
G=8 → 1 prompt
per_device_generation_batch=1 completion
vLLM TP=8 → 一个 8-rank rollout engine group
~~~

8 个 Megatron rank 各持有该 prompt 的一个 completion 切片任务；vLLM TP group 聚合 8 个请求后生成。`steps_per_generation=1` 表示每个 optimizer step 都需要新 rollout。

| optimizer step | rollout | 权重同步 | ref/current 计算 |
|---:|---:|---|---|
| 0 | 新 rollout A | full weights | beta=0 跳过 ref log-prob，但 ref model 仍已构造；current old/loss 照常 |
| 1 | 新 rollout B | full weights | 同上 |
| 2 | 新 rollout C | full weights | 同上 |

full 路径的每一次新 rollout 都执行：

~~~text
bridge.export_weights(all policy weights)
  → vLLM MoE load_weights
  → process/rebuild kernel-format weights
  → cache reset
  → rollout
  → reward/old-logp/advantage
  → full-policy forward/backward
  → optimizer step
~~~

`sleep_level=2` 用于尽量释放 vLLM 占用，也意味着代码明确把下次 rollout 当作需要整模重载。对 full 训练这不会改变同步类型，因为 full 本来就要每次同步所有权重。

#### 27.10.3 容量闸门

不应通过不断缩小 batch 来假设任意单机都能容纳 full colocate。至少应分别记录下列峰值：

1. policy + ref + optimizer 加载后，vLLM 尚未初始化；
2. vLLM weights wake，policy 尚未 offload；
3. bridge 导出 full weights 并调用 `load_weights`；
4. policy/ref/optimizer 已 offload，vLLM KV cache 已唤醒；
5. vLLM sleep 后 policy/ref/optimizer 回迁 GPU；
6. full-policy backward 和 optimizer step。

如果在第 2/3 点 OOM，只减 completion batch 可能没用，因为核心峰值是权重副本/导出，而不是 KV cache。此时应转向更高显存节点、增加机器、更激进的 CPU offload，或直接使用 external/server 分离 rollout。

### 27.11 external/server 的真实系统架构

#### 27.11.1 资源分离不等于 Megatron 已自动异步重叠

一个标准多机分离拓扑可以是：

~~~text
Training node 0: 8 GPUs, Megatron rank 0..7
Training node 1: 8 GPUs, Megatron rank 8..15
Rollout node 0:  8 GPUs, swift rollout, vLLM TP/DP workers
~~~

训练和生成显存确实完全隔离，但当前普通 Megatron GRPO 参数明确不支持 `async_generate`。因此每次新 rollout 的默认时序仍是同步串行：

~~~text
训练权重导出/同步
  → 训练 rank 聚合请求
  → external vLLM 生成
  → 返回所有结果
  → reward/old-ref log-prob/advantage
  → policy 训练更新
~~~

external 的首要优势是资源隔离、rollout 可独立扩容以及不需要单机 sleep/offload 状态切换；不应在没有改用 Ray/真正 async pipeline 时把它解读为 rollout 与 backward 已自动并行。

#### 27.11.2 控制面与数据面

[VLLMClient](../../swift/rlhf_trainers/vllm_client.py#L173) 和 [SwiftRolloutDeploy](../../swift/pipelines/infer/rollout.py#L695) 把通信分成两层：

| 平面 | 协议 | 端口 | 内容 |
|---|---|---|---|
| 控制/请求面 | HTTP/FastAPI | 默认 8000 | health、engine type、权重 metadata、infer JSON、cache reset |
| 权重数据面 | StatelessProcessGroup + PyNcclCommunicator | 默认从 51216 开始 | 大 tensor/flattened bucket 由 trainer client rank 广播给 vLLM workers |
| Megatron 训练集群 | torch distributed/NCCL | 例如 29500 | 训练 TP/PP/EP/DP 通信，与 rollout group port 不是同一个端口 |

因此只打开 8000 端口并不足以完成权重同步。多机网络必须同时允许 server port 和 group port，并正确配置 NCCL 网卡/IB。

初始化时序：

~~~mermaid
sequenceDiagram
    participant R as "Rollout node: swift rollout"
    participant W as "vLLM TP/DP workers"
    participant T as "Training ranks"
    participant C as "Last training rank / VLLMClient"

    R->>W: "启动 worker_extension_cls 并加载 Qwen3.5 base"
    W-->>R: "workers ready"
    R->>R: "FastAPI /health 可用"
    T->>T: "初始化 Megatron policy/ref/optimizer"
    T->>C: "全局最后一个 rank 负责 client"
    C->>R: "GET /health, GET /get_world_size"
    C->>R: "POST /close_communicator"
    C->>R: "POST /init_communicator(host, group_port)"
    R->>W: "collective_rpc(init_communicator)"
    C->>W: "以 world_size=vLLM workers+1、rank=last 加入 PyNCCL group"
    C->>R: "POST /get_engine_type"
    R-->>C: "sync/async、LoRA、multi-turn 能力"
    C->>T: "将 rollout_enable_lora 等状态广播给训练 ranks"
~~~

对 Megatron 路径，不是 rank 0，而是全局 `world_size-1` 的最后一个训练 rank 创建 `VLLMClient`。这与 `is_last_rank()` 的 Megatron 日志/输出惯例一致。

> **当前提交的静态审计风险**：[MegatronRLHF._prepare_vllm_client](../../swift/megatron/pipelines/train/rlhf.py#L48) 只在全局最后 rank 构造 client，`_sync_bucket_to_server()` 也会对非主 rank 提前返回；但 [_export_and_load_weights](../../swift/megatron/trainers/rollout_mixin.py#L423) 在 server 分支结束时没有显式判断 `self.is_main_process`，而是直接在所有调用该函数的 rank 上执行 `self.vllm_client.process_weights_after_loading()`。按当前可见调用链，非最后 rank 的 client 为 `None`，存在 `NoneType` 调用风险。Transformers 对应路径在该调用外有 main-process 条件，因此这更像是 Megatron server 路径当前版本需要在目标集群先验证的 rank-guard 缺口。本报告不在未执行环境中自行修改训练源码；实际运行 external 模板前，应将此处列为第一个断点/版本检查点。

#### 27.11.3 rollout 请求的端到端路径

~~~mermaid
sequenceDiagram
    participant TR as "All Megatron ranks"
    participant LR as "Last training rank"
    participant VC as "VLLMClient"
    participant API as "SwiftRolloutDeploy API"
    participant VW as "vLLM worker groups"

    TR->>TR: "inputs2requests: messages/images/tools/uuid"
    TR->>LR: "gather_object 聚合全部 requests"
    LR->>VC: "infer(all_requests, RequestConfig)"
    VC->>VC: "按 server 数量切 chunk"
    VC->>API: "HTTP POST /infer"
    API->>API: "按 vLLM DP connections 再切 chunk"
    API->>VW: "infer / async_infer"
    VW-->>API: "RolloutOutput + token_ids + processed logprobs"
    API-->>VC: "JSON response"
    VC-->>LR: "合并多 server outputs"
    LR->>TR: "broadcast_object_list"
    TR->>TR: "各 rank 根据 lengths 取回本地 slice"
~~~

如果配置多个 rollout server URL，client 会把 requests 分块并用线程池并发 HTTP 请求。权重同步则会把相同的权重并发广播到每个 server，所以增加 server 能增大 rollout 并发，同时也按 server 数增大 trainer 侧的聚合出网带宽压力。

#### 27.11.4 full weight 的 bucket 协议

Megatron bridge 产生 `(name, tensor)` iterator 后，server 路径会按 `SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE` 分 bucket，默认是 512 MiB。每个 bucket 的步骤是：

~~~text
收集 name/tensor
  → FlattenedTensorBucket 转为 uint8 flat tensor + metadata
  → HTTP POST /update_flattened_params 发 metadata
  → server workers 分配接收 buffer
  → PyNCCL broadcast 发送 flat tensor
  → worker 重建 named tensors
  → patch MoE weight loader
  → model.load_weights
~~~

全部 bucket 完成后，trainer 只调一次 `/process_weights_after_loading/`，让 vLLM 在权重齐全后重建 FusedMoE/内核所需的运行时格式。之后还要 reset prefix/encoder cache，防止新权重继续使用旧 cache。

对约 35B BF16 整模，一次同步的原始权重数据下界是数十 GB，不能用普通小模型的“控制请求很小”心智模型估算。当 `steps_per_generation=1` 时，这个整模传输每个 optimizer step 都发生。

### 27.12 external rollout server 标准启动模板

以独立 8-GPU rollout 节点为例，保守的同步 engine 配置是：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rollout \
    --model Qwen/Qwen3.5-35B-A3B \
    --host 0.0.0.0 \
    --port 8000 \
    --vllm_tensor_parallel_size 8 \
    --vllm_data_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.90 \
    --vllm_max_model_len 9192 \
    --vllm_max_num_seqs 64 \
    --vllm_enable_prefix_caching true \
    --vllm_enable_lora false
~~~

关键点：

- `host=0.0.0.0` 只用于服务监听；trainer 必须使用 rollout 节点的可路由 IP，不能写 `0.0.0.0` 或跨机写 `127.0.0.1`；
- `vllm_max_model_len`、TP、DP、GPU utilization 是 server 的属性，external trainer 侧的同名 engine 参数不会重配已运行的 server；
- rollout 参数检查会拒绝 vLLM pipeline parallel size > 1；
- 启动 `vllm_enable_expert_parallel` 会迫使 `vllm_use_async_engine=true`，应作为单独的吞吐优化实验，不应与首次跑通权重同步同时开启。

如要测试原生 LoRA 增量同步，server 改为：

~~~bash
--vllm_enable_lora true \
--vllm_max_lora_rank 8
~~~

`vllm_max_lora_rank` 必须大于等于 trainer 的 `lora_rank`，建议相等。但对 Qwen3.5 MoE，此快速路径只能在已确认 LoRA target 不落到 vLLM 不支持的专家层后启用。

如使用两个独立 rollout server，训练侧可传列表：

~~~bash
--vllm_server_host 10.0.2.10 10.0.2.11 \
--vllm_server_port 8000 8000 \
--vllm_server_group_port 51216 51217
~~~

两个 server 必须加载同一 model revision，使用一致的 tokenizer/template、LoRA 开关和生成语义。

### 27.13 场景三：多机分离 LoRA + external/server

本场景的执行前置是先核对 27.11.2 标注的 Megatron server `process_weights_after_loading()` rank guard 风险；下列内容是在“最后 rank 负责 server client”的设计意图下展开的标准拓扑与时序。

#### 27.13.1 训练节点模板

下列模板使用 2 个训练节点、每节点 8 GPU，再加 1 个独立 rollout 节点。运行前必须按真实集群注入 `NODE_RANK`、`MASTER_ADDR`、`ROLLOUT_IP`，并设置网卡与端口。

两个训练节点执行相同命令，只是 `NODE_RANK` 分别为 0 和 1：

~~~bash
SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

: "${NODE_RANK:?Set NODE_RANK to 0 or 1 on each training node}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the training master IP}"
: "${ROLLOUT_IP:?Set ROLLOUT_IP to the routable rollout-server IP}"
export NNODES="${NNODES:-2}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# 生产环境还应设置 NCCL_SOCKET_IFNAME/NCCL_IB_HCA/GLOO_SOCKET_IFNAME。

megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --enable_thinking false \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --moe_permute_fusion true \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --global_batch_size 64 \
    --micro_batch_size 1 \
    --steps_per_generation 2 \
    --num_generations 8 \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "$ROLLOUT_IP" \
    --vllm_server_port 8000 \
    --vllm_server_group_port 51216 \
    --vllm_server_timeout 600 \
    --max_length 1000 \
    --max_completion_length 8192 \
    --tuner_type lora \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32 \
    --freeze_vit true \
    --freeze_aligner true \
    --lr 5e-5 \
    --bf16 true \
    --beta 0.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --loss_type grpo \
    --overlong_filter true \
    --sleep_level 0 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune \
    --attention_backend flash \
    --sequence_parallel true \
    --logging_steps 1
~~~

这是源码推导的起点，不是 16 张任意显存 GPU 都已验证可运行的容量承诺。PP=2 是为了把语言模型层跨两个 stage 分摊，同时保持：

~~~text
world=16, TP=1, PP=2, CP=1
DP=16/(1×2×1)=8
EP=8
generation_batch=64×2=128 completions
rollout prompts=128/8=16
prompts per DP rank=16/8=2
per_device_generation_batch=128/16=8 completions
~~~

多机下 PP stage 的物理分布和跨节点通信会受 rank 排序影响。上线前应通过实际 rank mapping 与 NCCL trace 确认 PP/EP 是否导致过多跨机 all-to-all，再调整 TP/PP/EP，而不是把该组合视为唯一正确答案。

#### 27.13.2 保守路径与快速路径

| 方案 | rollout server | trainer target | 后续同步 | 定位 |
|---|---|---|---|---|
| 保守 | `vllm_enable_lora=false` | `all-linear` | 每次 merge 后 full weights | 先跑通 Qwen3.5 MoE |
| 快速实验 | `vllm_enable_lora=true` | 只选已验证的非专家模块 | 首次 base+adapter，后续 adapter-only | 显著减少网络量，但需版本/模块兼容性测试 |

保守路径中，LoRA 虽然大幅减少了训练梯度和 optimizer 状态，却没有减少 rollout 的权重同步量。这是“LoRA 训练显存很省，external 网络却仍然很慢”的主要原因。

快速路径中，`_move_adapter_to_vllm()` 使用 `bridge.export_weights(peft_format=True)` 导出 LoRA delta，将全部 adapter tensor 打平后通过 `/update_adapter_flattened_param/` + PyNCCL 传输，server 再构造 `TensorLoRARequest` 并 `add_lora`。支持 in-place load 的 vLLM 版本会直接更新，否则先 remove 旧 adapter 再 add 新 adapter。

#### 27.13.3 推导的三步时序

`steps_per_generation=2` 仍然使得只在 step 0 和 step 2 请求 external rollout：

| step | rollout | 保守路径同步 | native LoRA 快速路径 |
|---:|---:|---|---|
| 0 | batch A | merge 后 full weights | base full weights + adapter |
| 1 | 无 | 无 | 无 |
| 2 | batch B | merge 后 full weights | adapter-only，前提是 server 仍保留 base 且 sleep<2 |

step 0/2 中 external 特有的路径是：

~~~text
all training ranks 停在 rollout 边界
  → all ranks 参与 mcore-bridge 分布式导出，last rank 持有 client 并发送 bucket
  → HTTP metadata + PyNCCL weight broadcast
  → server 完成权重重建和 cache reset
  → all ranks 的 request gather 到 last rank
  → HTTP /infer
  → server vLLM TP/DP generation
  → outputs 返回 last rank
  → broadcast 到 all training ranks
  → reward/encode/old-logp/advantage
  → 切成 2 个训练缓冲
~~~

external 分离后，Megatron rollout mixin 的 `enable_offload` 只在 colocate 分支由 `offload_model or offload_optimizer` 设置。因此不应照搬 colocate 心智模型，假设 external 生成时 trainer 一定会自动把 policy/optimizer 整体 offload。训练侧的 optimizer CPU offload 仍可单独使用，但它与 rollout `offload_context` 不是一件事。

### 27.14 场景四：多机分离 Full + external/server

本场景同样以 27.11.2 的 rank/client guard 已经在目标版本中验证或修复为前提。

#### 27.14.1 训练节点模板

仍以 2×8 training GPUs + 1×8 rollout GPUs 作为拓扑示例。该数量只是配置模板，不代表对特定 GPU 显存的最小保证。

~~~bash
SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

: "${NODE_RANK:?Set NODE_RANK to 0 or 1 on each training node}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the training master IP}"
: "${ROLLOUT_IP:?Set ROLLOUT_IP to the routable rollout-server IP}"
export NNODES="${NNODES:-2}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --enable_thinking false \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --moe_permute_fusion true \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --global_batch_size 16 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 8 \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "$ROLLOUT_IP" \
    --vllm_server_port 8000 \
    --vllm_server_group_port 51216 \
    --vllm_server_timeout 600 \
    --max_length 1000 \
    --max_completion_length 8192 \
    --tuner_type full \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --lr 1e-6 \
    --bf16 true \
    --beta 0.0 \
    --loss_type grpo \
    --overlong_filter true \
    --sleep_level 0 \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer \
    --recompute_granularity selective \
    --finetune \
    --attention_backend flash \
    --sequence_parallel true \
    --logging_steps 1
~~~

此配置的 batch 整数关系：

~~~text
world=16, TP=4, PP=2, CP=1
DP=16/(4×2×1)=2
EP=4
global_batch=16, steps_per_generation=1
generation_batch=16 completions
G=8 → 2 prompts
prompts per DP rank=1
per_device_generation_batch=1 completion
~~~

TP=4、PP=2 让每个 model-parallel replica 占用 8 ranks，16 ranks 上得到 DP=2。它主要沿用 Qwen3 MoE full 实例的 TP/PP/EP 结构并增加 DP；实际生产配置应根据每 rank 参数、expert all-to-all、pipeline bubble 和网络拓扑重新搜索。

#### 27.14.2 三步时序与主瓶颈

Full + `steps_per_generation=1` 时：

| step | 权重 | rollout | 训练 |
|---:|---|---|---|
| 0 | 整模 bucket 导出和跨机广播 | batch A | full forward/backward/update |
| 1 | 整模 bucket 导出和跨机广播 | batch B | full forward/backward/update |
| 2 | 整模 bucket 导出和跨机广播 | batch C | full forward/backward/update |

一次 step 的时间可拆成：

~~~text
T_step ≈ T_bridge_export
       + T_bucket_control
       + T_weight_NCCL_broadcast
       + T_vLLM_repack_and_cache_reset
       + T_request_gather_and_HTTP
       + T_autoregressive_rollout
       + T_output_broadcast
       + T_reward_and_post_encode
       + T_old_logp
       + T_full_forward_backward_optimizer
~~~

当 beta>0 时还要增加 `T_ref_logp`；当 beta=0 时可省这次前向，但如 27.5 所述，frozen ref model 当前仍已构造。

Full external 的工程上限往往不是 vLLM 的纯生成 tokens/s，而是“每个 policy version 需要向每个 rollout replica 发送整份新权重”。如果这部分占比过高，优化顺序应是：

1. 在收敛可接受范围内增大 `steps_per_generation`，摊薄每个 optimizer step 的同步成本；
2. 确保权重数据面使用正确的 IB/RDMA/NCCL 网卡，不要只看 HTTP 延迟；
3. 通过 profiling 区分 bridge export、NCCL broadcast 和 vLLM repack，不把三者混成一个“server 慢”；
4. 审查是否真的需要 full；如 LoRA 可达到目标，并且 native LoRA target 可兼容，adapter-only 同步的数据量级差异极大；
5. 如要真正重叠 rollout 和 training，转向支持该调度语义的 Ray Megatron/异步方案，而不是只增加 server 数量。

#### 27.14.3 `SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE` 的作用与边界

~~~bash
export SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE=512
~~~

该环境变量单位是 MiB，默认 512。

- bucket 太小：HTTP metadata、barrier、Python 对象和 `load_weights` 调用次数增加；
- bucket 太大：trainer/server 临时 flat buffer 峰值、单次广播时间和失败重试粒度变大；
- 它只改变分块，不改变一次 full sync 的总字节量。

调优时应同时观测 trainer GPU 临时内存、rollout worker 接收 buffer、NIC 吞吐、每 bucket barrier 时间和 `process_weights_after_loading` 时间。

### 27.15 四种场景的统一比较

| 维度 | LoRA colocate | Full colocate | LoRA external/server | Full external/server |
|---|---|---|---|---|
| 直接 Qwen3.5 官方实例 | 有 | 无 | 无 | 无 |
| 训练显存 | 最低 | 最高 | 较低 | 高 |
| rollout 显存与训练是否隔离 | 否 | 否 | 是 | 是 |
| 是否需 sleep/wake | 是 | 是 | 否 | 否 |
| 是否需 policy/optimizer rollout offload | 通常需 | 强烈需要 | 不由 server rollout 分支自动触发 | 不由 server rollout 分支自动触发 |
| 参考模型 | disable adapter | 独立 full ref | disable adapter | 独立 full ref |
| 默认保守权重同步 | merged full | full | 跨机 merged full | 跨机 full |
| 可否 adapter-only | 可，MoE target 需验证 | 否 | 可，MoE target 需验证 | 否 |
| 主要瓶颈 | GPU 状态切换/KV/导出 | policy+ref+optimizer+vLLM 峰值 | 保守模式的整模网络同步 | 每 policy version 整模网络同步 |
| 标准用途 | 先建立正确性基线 | 高显存单机研究/压测 | 分离显存、扩展 rollout | 大规模 full 生产候选 |

可以按下面的决策顺序选型：

~~~text
先确认是否必须 Full
  ├─ 否 → 先 LoRA
  │        ├─ 单机容量足且要最短路径 → colocate
  │        └─ rollout 需独立扩容/训练显存紧 → external
  │             ├─ 先保守 full-weight sync 跑通
  │             └─ 再验证非专家 target 的 adapter-only sync
  └─ 是 → Full
           ├─ 单机可同时容纳 policy/ref/optimizer/vLLM 峰值 → colocate 压测
           └─ 否/要生产扩展 → external，优先解决整模同步带宽
~~~

### 27.16 与 8GB Qwen3-1.7B 实例相比，哪些语义不变

第 25、26 节的 1.7B QLoRA 实测与本节 35B Megatron 推导在规模上相差很大，但下列框架语义不变：

1. prompt 在 rollout 前保持 messages 结构，生成后再用真实 response token ID 编码；
2. G 个 completion 按原 prompt 分组，reward 构造 group-relative advantage；
3. rollout log-prob、old log-prob、current log-prob 和 ref log-prob 的语义保持分离；
4. `steps_per_generation` 把生成 batch 与 optimizer step 解耦，决定新 rollout 和权重同步频率；
5. vLLM 更新后要重置 prefix/encoder cache；
6. reward/advantage/loss 是算法层，engine lifecycle/权重同步是 rollout 基础设施层。

变化的是：

- Transformers PEFT state dict 换成 Megatron partition + mcore-bridge export；
- 单 GPU 切换换成 TP/PP/CP/EP/DP 组合；
- 1.7B 可忽略的 full-weight copy 在 35B 上成为主要容量/带宽成本；
- Qwen3.5 是 MoE + multimodal 注册，还需处理专家 loader、MM cache 和 LoRA 专家层兼容性。

### 27.17 MoE rollout/training 偏差与 Router Replay 深入方向

MoE 比 dense 模型多一个潜在的 backend drift 来源：不同 kernel/并行实现可能在边界数值上导致 router 选中的专家不同。即使 token 一样，vLLM rollout 与 Megatron old/current-policy forward 的专家路由也可能不完全相同。

当前代码提供 `router_replay_mode` 作为研究入口：

| 模式 | 大致作用 | 实现入口 |
|---|---|---|
| disabled | 默认，各 backend 自行路由 | Megatron GRPO 默认 |
| R2 | Megatron old-policy 前向记录路由，后续训练复用 | `_maybe_compute_logps()` 中 RouterReplay RECORD |
| R3 | vLLM 返回 routed experts，Megatron 前向 replay | `_prepare_vllm_engine()` 的 `enable_return_routed_experts` 和 rollout 后处理 |

R3 路径对 vLLM 版本有明确要求，并增加 routed-expert 数据的捕获、广播、padding 和对齐处理。它不应与首次 full/external 跑通同时开启；应在基线稳定后，通过 rollout/current PPL、KL、router 一致率和收敛指标单独对比。

### 27.18 生产级启动顺序与检查清单

#### 27.18.1 colocate

1. 锁定同一份 model snapshot/revision、Transformers、mcore-bridge、Megatron、vLLM、PEFT 版本；
2. 先用极短 prompt/completion、`train_iters=1`、不保存 optimizer 做初始化闭环；
3. 记录六个显存峰值：Megatron load、vLLM init、weight export/load、KV wake、trainer reload、backward；
4. 检查首次和第二次 rollout 的同步类型，确认是 full 还是 adapter-only；
5. 确认 completion token IDs、finish_reason、rollout log-prob、old log-prob 形状和 mask 对齐；
6. 再增加序列长度、generation batch 和步数。

#### 27.18.2 external/server

1. rollout 节点先加载模型，直到 `/health/` 可用；
2. 从 trainer 节点检查 server port 和 group port 的双向连通性；
3. 确认 trainer 使用可路由 rollout IP，不是 localhost/0.0.0.0；
4. 确认两个训练节点的 NNODES、NODE_RANK、MASTER_ADDR、MASTER_PORT 一致且 rank 唯一；
5. 锁定 `NCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME`、IB HCA/GID，检查 NCCL 没有回退到错误网卡；
6. 观察 last training rank 是否完成 `/init_communicator/`，rollout workers 是否全部加入 `vllm_world_size+1` 的 group；
7. 先发送一个小 bucket/完成一次 full sync，再扩大 bucket 和并发；
8. 多 server 时检查每个 server 的 model state keys、LoRA 开关、TP×DP world size 和输出数量一致；
9. 做一次 trainer 故障重启测试：旧 communicator 关闭、新 communicator 建立、权重重新全量同步；
10. 生产训练不应盲目保留示例的 `--no_save_optim --no_save_rng`，否则 checkpoint 不是精确的 optimizer/RNG 恢复点。

### 27.19 必须分层观测的指标

| 层 | 指标 | 用途 |
|---|---|---|
| 权重导出 | `export_weights`、`export_adapter_weights` 时间 | 区分 mcore-bridge 转换成本 |
| 权重传输 | bucket 数、字节、每 bucket NCCL 时间、NIC GB/s | 确定 external 是否受带宽限制 |
| vLLM 重建 | `process_weights_after_loading` 时间 | 区分传输与 MoE kernel repack |
| rollout | prompt/completion tokens、TTFT、decode tok/s、finish reason | 生成吞吐和截断 |
| 训练/生成偏差 | rollout vs old log-PPL/KL/chi-square、router 一致性 | 检查 vLLM/Megatron backend drift |
| reward | 每 reward 分布、group std=0 比例 | 判断是否有有效 advantage |
| GRPO | advantage 均值/方差、clip ratio、KL、有效 completion tokens | 检查算法稳定性 |
| MoE | expert load、dropped tokens/capacity、all-to-all 时间 | 检查 EP 与 router 失衡 |
| 显存 | 按状态机分阶段 GPU allocated/reserved 与 CPU RSS | 不把稳态显存当成峰值 |

当 external full 慢时，应优先回答“慢在 export、NCCL、repack、decode 还是 backward”，再调参。如果没有分层时间，仅看一个 `step_time` 几乎无法做出正确的架构判断。

### 27.20 典型失败模式与定位层次

| 现象 | 优先检查 | 根因层 |
|---|---|---|
| Qwen3.5 model class 不存在 | Transformers 版本、模型 revision | 模型注册/依赖 |
| Megatron 不识别模型 | mcore-bridge 版本与 `get_model_meta` | bridge |
| batch 初始化直接 ValueError | generation/global/G/DP/micro 整除关系 | 参数校验 |
| colocate 创建 vLLM OOM | policy/ref/optimizer 是否在初始化上下文正确 offload | engine lifecycle |
| full 在第一次 export OOM | bridge 导出临时 tensor 和 policy/ref 峰值 | 权重转换 |
| Megatron external 非最后 rank 出现 `NoneType.process_weights_after_loading` | `_export_and_load_weights()` 的 server 分支是否缺少 `self.is_main_process` guard | 当前提交的 rank/client lifecycle |
| external health 通但同步挂起 | group port、NCCL NIC/IB、所有 vLLM worker 是否入组 | 权重数据面 |
| `/infer` 能通但结果数不对 | server 数、vLLM DP chunk、placeholder 请求处理 | 请求调度 |
| native LoRA `add_lora/load` 报错 | target 是否包含 MoE expert、rank 是否超过 max rank | LoRA/vLLM 兼容 |
| 同步后输出仍像旧 policy | prefix/encoder cache 是否 reset，权重是否 finish/repack | cache/权重一致性 |
| rollout/old log-prob 差异大 | processed logprobs、temperature、token ID、MoE router path | 数值语义 |
| reward 长期不学习 | group std=0、accuracy parser、overlong filter、G | 数据/算法 |
| full beta=0 仍显存高 | ref model 当前仍在 `prepare_model()` 构造 | reference lifecycle |

### 27.21 本节不能由静态代码回答的问题

下列问题必须在目标机器上实测，本节不对它们作结论性承诺：

- 某种 8-GPU 单机的确切显存是否足以 full colocate；
- Qwen3.5 当前 model revision 在特定 vLLM 版本上的最佳 TP/DP/EP；
- `all-linear` 中哪些精确层名在目标 vLLM 版本支持 native LoRA；
- NVLink/PCIe 机型上 sleep/offload 的净收益；
- IB/RoCE 网络下 512 MiB 是否是最优 bucket；
- `steps_per_generation` 增大后吞吐收益与 policy staleness 对收敛的平衡；
- R2/R3 router replay 对当前 Qwen3.5 reward/稳定性的实际影响。

静态代码能够给出的是“应在什么边界测什么”，而不是用小模型的经验数字替代 35B MoE 的真实容量与网络测量。

### 27.22 后续深入代码导航

| 想继续研究的问题 | 第一入口 | 第二入口/建议断点 |
|---|---|---|
| Qwen3.5 如何被识别为 MoE 多模态模型 | [Qwen3_5MoeLoader 与 ModelMeta](../../swift/model/models/qwen.py#L1419) | `get_model_info_meta`、template meta |
| Megatron 如何加载/分片 Qwen3.5 | [BaseMegatronTrainer.prepare_model](../../swift/megatron/trainers/base.py#L185) | `get_mcore_model`、`bridge.load_weights`、`wrap_model` |
| LoRA 在 mcore 哪些层注入 | `prepare_mcore_model` | 打印 peft model 的 trainable names，与 vLLM state keys 交叉 |
| full 为何多一份 ref | [MegatronRLHFTrainer.prepare_model](../../swift/megatron/trainers/rlhf_mixin.py#L27) | `null_ref_context`、`_maybe_compute_logps` |
| generation batch 如何分成多个 step | [_replace_data_iterator](../../swift/megatron/trainers/grpo_trainer.py#L275) | `get_num_iters_per_step`、`get_local_rollout_batch` |
| Megatron rollout group 怎么建 | [create_rollout_group](../../swift/megatron/trainers/rollout_mixin.py#L35) | TP×PP×CP rank 集合与 DP index |
| colocate vLLM 怎么建 | [_prepare_vllm_engine](../../swift/megatron/trainers/rollout_mixin.py#L253) | `GRPOVllmEngine`、external_launcher、vLLM TP group |
| full/LoRA 如何选同步路径 | [_move_model_to_vllm](../../swift/megatron/trainers/rollout_mixin.py#L324) | `base_sync_done`、`sleep_level`、`rollout_enable_lora` |
| mcore weights 怎么变成 vLLM weights | [_export_and_load_weights](../../swift/megatron/trainers/rollout_mixin.py#L424) | `bridge.export_weights`、name aliases、MoE loader patch |
| external full bucket 怎么传 | [_load_weights_to_server_in_buckets](../../swift/megatron/trainers/rollout_mixin.py#L463) | `_sync_bucket_to_server`、`SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE` |
| external LoRA adapter 怎么传 | [_move_adapter_to_vllm](../../swift/megatron/trainers/rollout_mixin.py#L386) | `update_adapter_flattened_param`、`TensorLoRARequest` |
| HTTP/NCCL client 如何建立 | [VLLMClient.init_communicator](../../swift/rlhf_trainers/vllm_client.py#L199) | `StatelessProcessGroup`、`PyNcclCommunicator` |
| rollout server 的 API 有哪些 | [SwiftRolloutDeploy._register_rl_rollout_app](../../swift/pipelines/infer/rollout.py#L698) | `/infer`、`/update_flattened_params`、cache API |
| server worker 怎么收权重 | [WeightSyncWorkerExtension](../../swift/pipelines/infer/rollout.py#L105) | `update_flattened_params`、`process_weights_after_loading` |
| 多 server 请求如何分片 | [VLLMInferClient.infer](../../swift/rlhf_trainers/vllm_client.py#L110) | `ThreadPoolExecutor`、chunk order |
| server 结果如何回到各 rank | [_server_rollout](../../swift/megatron/trainers/grpo_trainer.py#L675) | gather lengths、last-rank broadcast、local slice |
| MoE router replay 怎么做 | `_maybe_compute_logps` 与 `_get_encoded_batch` | R2 RECORD/R3 REPLAY_FORWARD、routed_experts padding |
| GRPO 奖励/优势怎么聚合 | [_compute_advantages](../../swift/megatron/trainers/grpo_trainer.py#L797) | group/batch/GDPO normalization、zero-std |
| GRPO clip/KL/loss 怎么计算 | [Megatron GRPO loss](../../swift/megatron/trainers/grpo_trainer.py#L1228) | importance level、epsilon low/high、completion mask |

### 27.23 本案例的整体全貌

Qwen3.5-35B-A3B 把 GRPO 框架中几个原本可以在小模型上忽略的边界全部放大了：

~~~text
MoE 总参数 vs 激活参数
  → 计算省，权重/优化器/同步不按 3B 缩小

Megatron 训练并行 vs vLLM 推理并行
  → 通过 mcore-bridge 和 load_weights 边界重组，两侧 TP/PP/EP 不需一样

LoRA 训练状态 vs rollout 权重协议
  → LoRA 不必然意味着小同步；MoE native LoRA 不开时仍要发 merged full model

colocate vs external/server
  → 前者优化 GPU 分时复用，后者优化资源隔离和扩展，但不自动获得异步重叠

Full policy vs reference policy
  → 当前 Megatron full GRPO 会构造独立 ref，beta=0 只跳过计算，不当然跳过模型

HTTP control plane vs NCCL weight plane
  → external 请求走 HTTP，大权重 tensor 走 PyNCCL，两类端口/网络都要正确
~~~

对未来代码深挖，最有价值的四个专题依次是：

1. 以 `_move_model_to_vllm()` 为入口，完整跟踪 LoRA/full 的 bridge export 和 vLLM loader 名称映射；
2. 以 `_replace_data_iterator()` 为入口，画出 generation batch、G、DP、micro-batch 和 optimizer step 的精确索引；
3. 以 `VLLMClient`/`WeightSyncWorkerExtension` 为入口，实测 bucket、NCCL、repack 三段成本；
4. 以 rollout/old log-prob 和 Router Replay 为入口，分析 MoE 路由在 vLLM 与 Megatron 之间的数值一致性。

至此，四种场景可以用同一个核心问题来理解：**在每个新 policy version 到来时，训练态权重如何以正确、可容纳、可扩展的方式变成 rollout policy，然后如何把 G 条真实采样轨迹还原为 Megatron 可训练的 old/ref/current-policy 数据。** LoRA/full 决定权重和 reference 的成本，colocate/external 决定这个转换是通过 GPU 分时还是跨机协议完成。

---

## 28. Qwen3.5-35B-A3B GRPO：FSDP2 的 LoRA/Full × colocate/external 全路径分析

### 28.1 分析范围与先给结论

本节把第 27 节的四种场景保持不变，只把训练后端从 Megatron-SWIFT 换成 Transformers/TRL 主路径中的 FSDP2：

~~~text
训练入口：swift rlhf --rlhf_type grpo --fsdp fsdp2
训练循环：Transformers Trainer + Swift GRPOTrainer
训练分片：Accelerate FSDP2 + PyTorch fully_shard + DTensor
rollout：vLLM colocate 或独立 swift rollout server
~~~

四种组合仍是：

| 编号 | 训练方式 | rollout 拓扑 | 训练规模示例 | 本节定位 |
|---:|---|---|---|---|
| 1 | LoRA | 单机共卡 colocate | 1 节点 × 8 GPU | FSDP2 中最适合先验证的 35B 路径 |
| 2 | Full | 单机共卡 colocate | 1 节点 × 8 GPU | 高显存容量压力测，不承诺任意 8 卡可运行 |
| 3 | LoRA | 多机分离 external/server | 2×8 training GPU + 1×8 rollout GPU | 训练/生成资源隔离，默认仍可能整模同步 |
| 4 | Full | 多机分离 external/server | 2×8 training GPU + 1×8 rollout GPU | 架构简单但跨机整模同步和 FSDP 通信都很重 |

核心结论如下：

1. **FSDP2 是本仓库 rollout 训练支持的唯一 FSDP 版本。** [RolloutTrainerMixin](../../swift/rlhf_trainers/rollout_mixin.py#L94) 检测到 FSDP1 会直接抛出 `NotImplementedError`。
2. **FSDP2 不是 Megatron TP/PP/EP 的平替。** 它主要把参数、梯度和优化器状态沿 FSDP data-parallel mesh 分片；默认不会把矩阵乘法切成 TP、把层切成 PP，也不会把 256 个专家变成 Megatron EP。
3. **对 Qwen3.5 MoE，默认 auto-wrap 的基本语言单元是整个 `Qwen3_5MoeDecoderLayer`。** 一个 decoder layer 内含完整 MoE block；FSDP 前向仍要把该 FSDP unit 的参数 all-gather 完整后再计算。
4. **LoRA 只必然减少训练态梯度/优化器，不必然减少 rollout 同步。** `vllm_enable_lora=false` 时，FSDP2 会在普通 Tensor 层面把 LoRA delta 合并进 base，然后把 merged full weights 送给 vLLM。
5. **Full + `beta=0` 在标准 Transformers 路径可以真正不加载 reference model。** 这与第 27 节当前 Megatron full 实现形成明确差异。
6. **FSDP2 到 vLLM 的关键边界是 `DTensor.full_tensor()`。** 当前实现会让全部训练 rank 参加重建；external 模式虽只有 rank 0 发送，但其他 rank 仍必须完成相同 collective。
7. **`move_model_batches` 能限制一次保留多少组完整权重，但不能拆分一个单独参数。** Qwen3.5 的 fused expert tensor 本身就可能很大，因此它不是无限有效的 OOM 开关。
8. **35B-A3B 的 FSDP2 标准模板是源码推导，不是本机实测。** 当前 `.venv` 的 Transformers 为 4.57.6，实际没有 `transformers.models.qwen3_5_moe`；而仓库 Qwen3.5 loader 声明需要更新的 Transformers 版本，所以本节不伪造运行日志或性能数字。

本节的直接源码入口：

- [FSDP2 内置配置](../../swift/config/fsdp2.json)；
- [SftArguments._init_fsdp](../../swift/arguments/sft_args.py#L281)；
- [SwiftRLHF pipeline](../../swift/pipelines/train/rlhf.py)；
- [RLHFTrainerMixin reference 包装](../../swift/rlhf_trainers/rlhf_mixin.py#L17)；
- [prepare_fsdp](../../swift/rlhf_trainers/utils.py#L954)；
- [RolloutTrainerMixin](../../swift/rlhf_trainers/rollout_mixin.py#L87)；
- [GRPOTrainer rollout/缓存/loss](../../swift/rlhf_trainers/grpo_trainer.py#L87)；
- [Qwen3.5 loader](../../swift/model/models/qwen.py#L1412)；
- [上游 Qwen3.5 MoE 模型定义](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)；
- [Qwen3.5-35B-A3B config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)。

### 28.2 重要概念速查：先建立正确心智模型

#### 28.2.1 FSDP、FSDP2 与 FULL_SHARD

**FSDP（Fully Sharded Data Parallel）** 的目标，是让多个 data-parallel rank 不再各自常驻一份完整训练状态。FULL_SHARD 大体对应：

| 对象 | 稳态存储 | 使用前/后发生什么 |
|---|---|---|
| 参数 | 按 rank 分片 | 某 FSDP unit 前向前 all-gather，用完按策略 reshard |
| 梯度 | 按 rank 分片 | backward 后通过 reduce-scatter 聚合并分片 |
| optimizer state | 按 rank 分片 | 每个 rank 更新自己负责的参数 shard |

FSDP2 使用 PyTorch `fully_shard` API，把原参数转为 DTensor，并原地给 module 安装 FSDP 行为；它不像 FSDP1 那样主要依赖一个外层 `FullyShardedDataParallel` wrapper 和 FlatParameter。对本仓库最实际的差异是：

- `named_parameters()` 看到的是分片 DTensor；
- 要给 vLLM 一份普通完整权重，必须从 `state_dict()` 取值并执行 `full_tensor()`；
- auxiliary model 也必须走 FSDP2 包装，不能把 DTensor policy 和普通 Tensor reference 混用；
- rollout mixin 明确拒绝 FSDP1。

#### 28.2.2 DTensor、DeviceMesh、placement

**DTensor** 是“一个全局逻辑 tensor + 它在 device mesh 上如何放置”的表示。可以把它理解成：

~~~text
global tensor W
  + DeviceMesh(rank 0..N-1)
  + placement=Shard(dim=0) / Replicate / Partial
  = 每个 rank 上的 local shard + 全局形状/布局元数据
~~~

普通训练前向只操作 local shard 和 FSDP collective；但 vLLM 的 HF `load_weights` 接口需要普通完整 Tensor，所以当前 Swift 同步代码调用：

~~~python
param.full_tensor()
~~~

这不是无成本的“取属性”，而是一个需要全部 FSDP rank 参加的集合通信，并在每个调用 rank 上物化完整参数。

#### 28.2.3 FSDP unit 与 auto-wrap

**FSDP unit** 是一次参数 gather/reshard 的生命周期边界。unit 太大，单次完整参数峰值高；unit 太小，collective 次数、Python hook 和延迟增加。

内置配置使用：

~~~json
"auto_wrap_policy": "TRANSFORMER_BASED_WRAP"
~~~

Accelerate 会读取模型 `_no_split_modules`。当前上游 Qwen3.5 MoE 定义为：

~~~python
_no_split_modules = ["Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionBlock"]
~~~

因此语言主干按 decoder layer 包装，视觉塔按 vision block 包装；Accelerate 还会单独处理 input embedding、final norm/output embedding 和 root leftovers。

#### 28.2.4 all-gather、reduce-scatter、all-reduce

- **all-gather**：每个 rank 贡献自己的参数 shard，所有 rank 得到完整 unit 参数；FSDP 前向和 `full_tensor()` 都会用到类似集合过程。
- **reduce-scatter**：先对各 rank 梯度求和，再把结果分片给对应 rank；FULL_SHARD 的梯度落盘形态通常靠它形成。
- **all-reduce**：每个 rank 最终都得到相同完整归约结果；传统 DDP 常用于完整梯度同步。

对 35B MoE，真正要关注的不是“有没有通信”，而是：每层参数 all-gather 是否跨节点、专家 tensor 是否作为一个巨型参数参与、通信能否与计算重叠。

#### 28.2.5 reshard_after_forward

内置 FSDP2 配置设置：

~~~json
"reshard_after_forward": true
~~~

这表示 unit 前向完成后重新只保留 shard，降低驻留显存；backward 需要时再 gather。关闭它可避免部分重复 gather，但会让完整参数跨更长时间驻留，35B 容量压力显著上升。对本节容量优先的基线保持 true。

#### 28.2.6 Activation Checkpointing 与 Gradient Checkpointing

两者目的都是不长期保存全部 activation，backward 时重算前向；区别在于安装 checkpoint wrapper 的层和与 FSDP 生命周期的配合方式。

本仓库内置 FSDP2 配置启用：

~~~json
"activation_checkpointing": true
~~~

如果 CLI 同时打开 `gradient_checkpointing`，[`_check_fsdp2_compatibility()`](../../swift/arguments/sft_args.py#L321) 会告警并自动关闭后者。模板因此显式使用：

~~~bash
--gradient_checkpointing false
~~~

activation checkpointing 省的是 activation，不会省 policy 权重 shard、optimizer state、vLLM 权重或 KV cache；它还会增加重计算，且可能增加 FSDP unit 再 gather 的压力。

#### 28.2.7 Sharded State Dict 与 Full State Dict

- **SHARDED_STATE_DICT**：每个 rank 保存自己的 shard，保存峰值低、适合训练恢复；内置配置的默认选择。
- **FULL_STATE_DICT**：收集普通完整模型，便于直接分发/推理，但在 35B 上会增加 CPU/GPU 内存和集合通信。

训练 checkpoint 的 state-dict 类型与“每次给 vLLM 同步权重”是两个边界：即使 checkpoint 使用 SHARDED_STATE_DICT，rollout 仍会通过 `full_tensor()` 临时重建 vLLM 所需权重。

#### 28.2.8 FSDP 分片不是 MoE Expert Parallel

Qwen3.5-35B-A3B 的官方 config 包含 40 个语言层、256 个专家、每 token 选择 8 个专家。两种并行语义不同：

| 维度 | FSDP2 默认路径 | Megatron EP |
|---|---|---|
| 专家参数放置 | 每个参数沿 FSDP mesh 切 shard | 不同 expert 分配给不同 EP rank |
| 前向前 | gather 当前 FSDP unit 的参数 | token 按 router 结果 all-to-all 到 expert rank |
| 本 rank 计算 | 对本地 batch 执行完整层语义 | 主要计算本 rank 拥有的专家 |
| 是否利用只激活 8/256 专家减少参数 gather | 默认不能 | EP 设计目标之一 |

所以 FSDP2 能把 35B 总状态切开存储，却不能自动把 MoE 专家计算变成 Megatron 的 expert-parallel 执行。

#### 28.2.9 Policy、old policy、reference policy 与 rollout policy

这四个概念在 FSDP2 下不变：

| 对象 | 实体/时机 | 用途 |
|---|---|---|
| current policy | 当前 FSDP2 policy，有梯度 | 计算训练 loss |
| old policy | rollout 后、更新前由 policy 无梯度重算 | importance ratio 分母 |
| reference policy | LoRA 时禁用 adapter；full 时可为独立 frozen FSDP2 model | KL 约束 |
| rollout policy | vLLM 中最近一次同步的权重 | 真正采样 completion |

`steps_per_generation > 1` 时，一批 rollout 数据会跨多个 `training_step` 重用，因此后续 current policy 可能已经更新，而 old/rollout log-prob 仍对应生成时附近的版本。

### 28.3 `--fsdp fsdp2` 实际展开成什么

[内置 fsdp2.json](../../swift/config/fsdp2.json) 的有效部分是：

~~~json
{
  "fsdp": "full_shard auto_wrap",
  "fsdp_config": {
    "fsdp_version": 2,
    "reshard_after_forward": true,
    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "cpu_ram_efficient_loading": true,
    "state_dict_type": "SHARDED_STATE_DICT",
    "activation_checkpointing": true
  }
}
~~~

参数初始化时，[SftArguments._init_fsdp](../../swift/arguments/sft_args.py#L281) 还会：

1. 把字符串 `fsdp2` 映射到该 JSON；
2. 将 `self.fsdp` 展开为 `full_shard auto_wrap`；
3. 设置环境变量 `FSDP_VERSION=2`；
4. 若未显式设置，则写入 `TORCH_NCCL_AVOID_RECORD_STREAMS=1`；
5. 拒绝同时使用 DeepSpeed；
6. 拒绝由 `device_map` 形成的单进程模型并行；
7. 检查 `save_only_model + SHARDED_STATE_DICT` 冲突；
8. 处理 activation checkpointing 与 gradient checkpointing 冲突。

`cpu_ram_efficient_loading=true` 的意图是只有主 rank 读入完整权重，再把 state 分发给其他 rank，降低初始化时的主机 RAM 重复占用。它不是训练期间的 parameter CPU offload，也不是 rollout 的 `--offload_model`。

当前环境 Accelerate 的 `fsdp2_prepare_model()` 还有一个对 full 训练极重要的行为：当 mixed precision 不是 `no` 且模型不是 4-bit 时，**可训练的低精度参数会先上转 FP32 master weights**，再由 MixedPrecisionPolicy 控制计算 dtype。

由此得到：

| 场景 | base/frozen 参数 | 可训练参数 | rollout state_dict 的潜在 dtype |
|---|---|---|---|
| LoRA + BF16 base | 大部分 frozen base 保持 BF16 shard | LoRA 参数可能 FP32 shard | merged full 基本随 base dtype，adapter tensor 可能 FP32 |
| Full + BF16 compute | policy trainable master shard 可能 FP32 | 全部 policy | 当前同步代码未显式降为 BF16，需实测 state_dict dtype |
| Full reference | 包装前已 freeze | 通常保持加载 dtype | BF16 reference shard，不参与 vLLM 同步 |

这意味着 full FSDP2 的“参数 shard 内存”和“external 权重传输量”不能只按 35B×2 bytes 估算。当前同步实现没有在 `_process_state_dict_for_vllm()` 中显式把 full policy FP32 tensor cast 为 rollout BF16；应在目标版本打印 dtype 和 bucket 字节数，而不能假设一定只传约 70 GB。

### 28.4 Qwen3.5 MoE 的实际 wrap 粒度与单参数下界

上游 Qwen3.5 MoE 的一个 decoder layer 包含：

~~~text
Qwen3_5MoeDecoderLayer
  ├─ linear attention 或 full attention
  ├─ input/post-attention norm
  └─ Qwen3_5MoeSparseMoeBlock
       ├─ router/gate
       ├─ fused experts
       └─ shared expert
~~~

按当前 config 粗略观察 fused expert 权重：

~~~text
hidden_size = 2048
num_experts = 256
moe_intermediate_size = 512

gate_up_proj 参数量约 = 256 × 2048 × (2×512) ≈ 5.37e8
down_proj 参数量约    = 256 × 512 × 2048     ≈ 2.68e8
~~~

仅这两块每层就约 8.05e8 参数；忽略其他权重时，BF16 原始字节约 1.5 GiB，FP32 约 3.0 GiB。精确 shape 以所锁定 model revision 为准，但该估算揭示两个工程事实：

1. decoder-layer FSDP unit 在前向中需要容纳完整专家权重的瞬时 gather；
2. 即使 `move_model_batches=40` 让每个同步组接近一层，单个 fused `gate_up_proj` 仍不能被参数组逻辑继续切开。

因此 `move_model_batches` 的峰值模型应写成：

~~~text
peak ≳ max(
  一个同步组中所有 full_tensor 的总大小,
  最大单个参数 full_tensor 的大小
)
~~~

而不是简单理解为“设成 40 就把所有临时内存除以 40”。

### 28.5 共享的 Transformers/FSDP2 GRPO 调用链

四种场景共享下面的训练主干：

~~~mermaid
flowchart TD
    A["swift rlhf --rlhf_type grpo --fsdp fsdp2"] --> B["RLHFArguments/SftArguments 参数初始化"]
    B --> C["_init_fsdp: 载入 fsdp2.json 与兼容性检查"]
    C --> D["SwiftRLHF._prepare_model_tokenizer"]
    D --> E["加载 policy；按 beta/tuner 决定 ref_model"]
    E --> F["SwiftSft.run: dataset → tuner/LoRA → GRPOTrainer"]
    F --> G["Transformers Trainer 创建 Accelerator/FSDP plugin"]
    G --> I["aux ref/reward: prepare_fsdp"]
    G --> J["GRPOTrainer.prepare_rollout"]
    I --> J
    J --> K["_prepare_vllm + split_batches，记录包装前参数名"]
    K --> H["Trainer.train 开始时包装主 policy: fully_shard + DTensor"]
    H --> L["RepeatSampler 形成 G 个 completion 的 generation batch"]
    L --> M["_prepare_inputs: 新 rollout 或取 buffered_inputs"]
    M --> N["_fast_infer: 权重同步 + vLLM generation"]
    N --> O["reward → 后编码 → old/ref/rollout log-prob"]
    O --> P["group-relative advantage"]
    P --> Q["按 steps_per_generation 切训练缓冲"]
    Q --> R["FSDP2 current-policy forward/backward"]
    R --> S["reduce-scatter grad → optimizer shard update"]
    S --> M
~~~

关键文件与职责：

| 阶段 | 入口 | FSDP2 相关职责 |
|---|---|---|
| 参数 | [sft_args.py](../../swift/arguments/sft_args.py#L281) | 展开 preset、设置环境、拒绝冲突 |
| 模型/ref 决策 | [rlhf_args.py](../../swift/arguments/rlhf_args.py#L305) | `beta=0`、LoRA/full 决定 ref 是否加载 |
| pipeline | [pipelines/train/rlhf.py](../../swift/pipelines/train/rlhf.py) | 加载 policy/ref/reward，组装 Trainer 参数 |
| 主 policy 包装 | Transformers Trainer + Accelerate | `fully_shard` policy，建立 DTensor |
| auxiliary model 包装 | [prepare_fsdp](../../swift/rlhf_trainers/utils.py#L954) | 冻结后调用 `fsdp2_prepare_model` |
| rollout 初始化 | [RolloutTrainerMixin.prepare_rollout](../../swift/rlhf_trainers/rollout_mixin.py#L109) | FSDP2 校验、engine/client、参数组 |
| generation 缓存 | [GRPOTrainer._prepare_inputs](../../swift/rlhf_trainers/grpo_trainer.py#L196) | rollout 一次，切成多个 training step |
| 权重同步 | [_move_model_to_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L450) | full/adapter 决策，DTensor 重建 |
| FSDP2 state 收集 | [_collect_state_dict_for_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L715) | `state_dict()` + `full_tensor()` |
| current loss | [GRPOTrainer](../../swift/rlhf_trainers/grpo_trainer.py) | ratio、clip、KL、mask、反向 |

### 28.6 一个 FSDP unit 的前向/反向时序

~~~mermaid
sequenceDiagram
    participant R as "各 FSDP rank 的参数 shard"
    participant U as "当前 decoder-layer FSDP unit"
    participant C as "Qwen3.5 layer compute"
    participant O as "optimizer shard"

    R->>U: "pre-forward all-gather 完整 unit 参数"
    U->>C: "本 rank 的 micro-batch 前向"
    C-->>U: "activation / checkpoint 边界"
    U->>R: "reshard_after_forward=true，释放非本地 shard"
    R->>U: "backward 需要时重新 all-gather"
    U->>C: "重算 activation（若 checkpoint）+ 反向"
    C->>R: "reduce-scatter 梯度"
    R->>O: "本 rank optimizer state 更新本地参数 shard"
~~~

这里没有 TP 把单次 expert GEMM 切到多卡，也没有 PP 让不同 rank 只拥有部分层语义。每个 FSDP rank 处理不同数据，但按层共同提供权重 shard。

### 28.7 LoRA 与 Full 的 reference policy：FSDP2 路径的关键优势

[RLHFArguments.__post_init__](../../swift/arguments/rlhf_args.py#L305) 的条件顺序是：

~~~text
如果 GRPO 且 beta == 0
  → ref_model = None
否则如果任务需要 reference 且 tuner_type == full
  → ref_model 默认为同一 model id，pipeline 再加载独立模型
否则 LoRA
  → 不接受独立 ref_model，训练时 disable adapter 得到 reference
~~~

因此：

| 方式 | beta | reference 实体 | FSDP2 行为 | 额外 35B 权重 copy |
|---|---:|---|---|---:|
| LoRA | 0 | 无 | 不算 ref log-prob | 否 |
| LoRA | >0 | 同一 policy 禁用 adapter | `null_ref_context()`，base shard 不变 | 否 |
| Full | 0 | 无 | 参数阶段直接 `ref_model=None` | 否 |
| Full | >0 | 独立 frozen model | `prepare_fsdp(..., evaluation_mode=True)` | 是，但也按 FSDP2 分片 |

`prepare_fsdp()` 会先 `eval()`、把参数 `requires_grad_(False)`，再调用 Accelerate 的 `fsdp2_prepare_model()`。先冻结很重要：可避免 evaluation-only reference 被按 trainable mixed-precision 参数上转 FP32，节省显存。

与第 27 节对照：

~~~text
当前 Megatron full：beta=0 跳过 ref 前向，但 prepare_model 仍构造 ref
当前 Transformers/FSDP2 full：beta=0 在 pipeline 加载前就把 ref_model 设为 None
~~~

所以 full FSDP2 容量基线必须显式写 `--beta 0.0`。如果实验需要 KL reference，再把 reference shard 的 BF16 权重、前向 all-gather 和计算成本加入预算。

### 28.8 FSDP2 → vLLM 权重同步的精确分支

#### 28.8.1 统一决策树

~~~text
tuner_type == full
  → _move_full_model_to_vllm

tuner_type == lora 且满足任一条件：
  - base_sync_done 仍为 false
  - sleep_level == 2
  - rollout engine 未启用 native LoRA
  → _move_full_model_to_vllm

否则
  → _move_adapter_to_vllm
~~~

#### 28.8.2 Full 或 merged-LoRA state 收集

FSDP2 不能像普通模型那样直接把 `named_parameters()` 中的 `.data` 交给 vLLM，因为那只是 DTensor local shard。当前代码按每个 `parameter_group` 执行：

~~~text
model.state_dict()
  → 过滤本组 key
  → 对 DTensor 调用 full_tensor()
  → 清理 base_model/checkpoint wrapper/LoRA 名称
  → 必要时 tensor-level 合并 LoRA
  → colocate: inner_model.load_weights
     server: rank 0 打平 bucket 后 PyNCCL 发送
  → 删除该组普通 full Tensor
~~~

所有训练 rank 都必须执行到 `full_tensor()`，因为这是 FSDP collective；external 模式只有 `accelerator.is_main_process` 的 rank 0 执行网络发送，非主 rank 仍不能跳过前面的 collective。

#### 28.8.3 为什么 FSDP2 LoRA 不调用 PEFT merge/unmerge

[_merge_lora_into_state_dict](../../swift/rlhf_trainers/rollout_mixin.py#L652) 的注释明确说明：PEFT 的 `merge_adapter()/unmerge_adapter()` 对 DTensor 不能可靠恢复原始权重。因此 FSDP2 路径改成：

~~~text
先 full_tensor 得到普通 base、lora_A、lora_B
delta = (lora_B @ lora_A) × scaling
merged_weight = base_weight + delta.cast(base_dtype)
删除 LoRA keys
把 merged 普通 Tensor 送入 vLLM
~~~

这个操作只改变用于 rollout 同步的临时 state dict，不原地修改 FSDP2 policy shard。

#### 28.8.4 native vLLM LoRA 快速路径

当 server/colocate engine 确实启用 LoRA 时：

- 第一次先同步 base full weights；
- 再用 `get_peft_model_state_dict()` 提取 adapter；
- DTensor adapter 同样通过 `full_tensor()` 还原；
- colocate 构造 `TensorLoRARequest` 并 `add_lora`；
- server rank 0 用 flattened adapter 协议发送；
- 后续 rollout 可以只传 adapter。

但 Qwen3.5 同时是 MoE 和 multimodal 注册。代码会明确警告：vLLM 原生 LoRA 可能不支持专家层，ViT adapter 也可能存在兼容问题。因此本节四个基线都先使用 `vllm_enable_lora=false`；adapter-only 只作为经过模块白名单与版本验证后的优化分支。

#### 28.8.5 `move_model_batches` 到底控制什么

`split_batches()` 会找到语言模型的 `ModuleList`，将 decoder layer 参数分成 N 个组，然后再为 embedding/head 和非语言多模态参数追加组。

本节用 `--move_model_batches 40` 作为 Qwen3.5 的容量优先起点，含义接近“每个语言层一个同步组”，不是“每 40 层一个组”。代价是：

- `state_dict()` 过滤、`full_tensor()` 和 loader 调用次数增加；
- colocate 会有更多 vLLM load_weights 调用；
- external 有更多 bucket/control 同步边界；
- 单个 fused expert 参数的完整 tensor 峰值仍存在。

性能调优时可从 40 向 20、10、5 逐步合并，前提是同时记录 full-tensor 峰值，而不是直接套用小模型的组数。

### 28.9 generation batch、training step 与 optimizer global step

FSDP2 不改变 GRPO batch 语义。Transformers 主路径使用：

~~~text
global_micro_batch = per_device_train_batch_size × world_size
generation_batch_size = global_micro_batch × steps_per_generation
rollout_prompt_count = generation_batch_size / num_generations
~~~

需要同时满足：

1. `generation_batch_size` 能被 `global_micro_batch` 整除；
2. `generation_batch_size` 能被 `num_generations` 整除；
3. eval 开启时，global eval batch 也能被 `num_generations_eval` 整除。

最容易误判的是三种 step：

| 名称 | 在当前代码中的推进点 | 与 rollout 的关系 |
|---|---|---|
| `GRPOTrainer._step` | 每次 `training_step()` 后加 1 | `steps_per_generation` 直接按它计数 |
| gradient accumulation micro-step | Trainer 每次前反向 | 通常与 `_step` 同步推进 |
| `state.global_step` | optimizer 真正更新后加 1 | `_fast_infer` 用它避免同一 policy version 重复 load |

因此 `steps_per_generation` 文档中的“step”是 training micro-batch，不天然等于 optimizer global step。只有本节模板这样设置：

~~~bash
--gradient_accumulation_steps 1 \
--steps_per_generation 2
~~~

才会得到直观的三步时序：optimizer step 0 新 rollout、step 1 复用、step 2 再 rollout。如果把 gradient accumulation 改为 8 而保持 SPG=2，一个 optimizer update 内可能触发多次 rollout；必须重新画时序，不能沿用下文表格。

### 28.10 单机共卡的 FSDP2/vLLM 状态机

colocate 下，8 个进程同时是 FSDP2 rank 和 vLLM external-launcher worker。`vllm_tensor_parallel_size=8` 时形成一个 8-rank vLLM TP group。

训练与生成的并行视图是：

~~~text
训练时：8-rank FSDP mesh
  - 每个 rank 保存 policy 的 1/8 shard
  - 每个 rank 处理不同 GRPO micro-batch slice

生成时：1 个 vLLM TP=8 engine group
  - 每个 rank 保存 vLLM 的 1/8 推理权重
  - 8 rank 聚合本地 rollout requests 后共同生成
~~~

这两个 1/8 的含义不同：FSDP shard 是训练 DTensor placement；vLLM shard 是推理 tensor parallel placement。当前代码不是把 FSDP shard 原样交给对应 vLLM TP rank，而是先重建完整 HF tensor，再由每个 vLLM loader 选取自己的 TP shard。

colocate 新 rollout 的状态机：

~~~mermaid
sequenceDiagram
    participant P as "FSDP2 policy/optimizer"
    participant S as "DTensor sync"
    participant V as "vLLM TP=8"
    participant G as "GRPO postprocess/train"

    V->>V: "若 sleep>0，wake weights"
    P->>S: "state_dict + full_tensor，逐 parameter group"
    S->>V: "load_weights；全部组完成后 finish reload"
    V->>V: "reset prefix/encoder cache"
    P->>P: "offload_model/offload_optimizer → CPU"
    V->>V: "wake KV cache"
    V->>V: "生成 G 个 completion"
    V->>V: "sleep(level)"
    P->>P: "policy/optimizer 回迁 GPU"
    V->>G: "token_ids/logprobs/finish_reason"
    G->>P: "reward、old/ref logp、advantage、FSDP forward/backward"
~~~

`--offload_model` 在 FSDP2 分支实现为 `model.cpu()`，回迁为 `model.to(device)`；`--offload_optimizer` 遍历普通 optimizer state tensor 在 CPU/GPU 间移动。两者只在 colocate 将 `enable_offload` 设为 true 时进入 rollout `offload_context`。

注意权重同步发生在 `offload_context` 之前：

~~~text
wake weights → full_tensor/load_weights → 再 offload trainer → rollout
~~~

所以最大的同步峰值仍可能同时看到 FSDP policy shard、临时 full tensor 和 vLLM 权重；不能因为打开 offload 就假设同步阶段只驻留一侧。

### 28.11 场景一：FSDP2 LoRA + 单机 colocate

#### 28.11.1 三步验链模板

下面是容量优先的源码推导模板。它假设单机 8 张大显存 GPU，不适用于 8GB 单卡，也没有在当前环境执行：

~~~bash
SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --enable_thinking false \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --reward_funcs accuracy \
    --tuner_type lora \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32 \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --fsdp fsdp2 \
    --gradient_checkpointing false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size 8 \
    --vllm_gpu_memory_utilization 0.45 \
    --vllm_max_model_len 9192 \
    --vllm_max_num_seqs 16 \
    --vllm_enable_prefix_caching true \
    --vllm_enable_lora false \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --move_model_batches 40 \
    --max_length 1000 \
    --max_completion_length 8192 \
    --overlong_filter true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --steps_per_generation 2 \
    --num_generations 8 \
    --num_iterations 1 \
    --learning_rate 5e-5 \
    --beta 0.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --loss_type grpo \
    --temperature 1.0 \
    --split_dataset_ratio 0 \
    --max_steps 3 \
    --save_strategy no \
    --logging_steps 1 \
    --report_to none \
    --output_dir output/qwen3_5_35b_a3b_grpo_fsdp2_lora_colocate_3step
~~~

这里把 `max_completion_length=8192` 保留下来，是为了与第 27 节 Qwen3.5 数学场景对齐，并不表示它是容量测试的首个长度。首次验链应先降到 256/512，确认初始化、同步、rollout、backward 闭环，再逐级恢复 8192。

#### 28.11.2 batch 算术

~~~text
world_size = 8
per_device_train_batch_size = 1
global_micro_batch = 8 completions/training_step
steps_per_generation = 2
generation_batch_size = 16 completions
num_generations = 8
rollout_prompt_count = 2 prompts
每 rank 收到 2 个 completion 的 generation slice，之后切成 2 个 bs=1 训练缓冲
~~~

`vllm_tensor_parallel_size=8` 整除 world size，8 个 rank 只组成一个 rollout engine group。该组聚合 16 个全局 completion 请求的相应本地 slice并生成；具体请求 gather/slice 由 `_colocate_rollout()` 按 TP group 完成。

#### 28.11.3 三个 optimizer step 的时序

| optimizer step | `_step` | 新 rollout | FSDP2→vLLM 同步 | 训练数据 |
|---:|---:|---:|---|---|
| 0 | 0 | 是 | base+LoRA 在普通 Tensor 层合并后 full sync | rollout batch A 的第 1/2 缓冲 |
| 1 | 1 | 否 | 无 | rollout batch A 的第 2/2 缓冲 |
| 2 | 2 | 是 | 更新后的 LoRA 再次 merged full sync | rollout batch B 的第 1/2 缓冲 |

step 0/2 的完整路径：

~~~text
FSDP2 policy state_dict
  → 逐层 full_tensor(base BF16 + LoRA A/B)
  → delta=(B@A)×scale，临时合并到 base dtype
  → vLLM TP rank load_weights
  → finish_vllm_weight_reload + reset caches
  → FSDP shard/optimizer state offload CPU
  → vLLM rollout
  → vLLM sleep
  → trainer state 回迁 GPU
  → policy old log-prob
  → reward/advantage
  → FSDP2 current-policy forward/backward/update
~~~

因为 `beta=0`，不计算 reference。如果改为 `beta>0`，LoRA 不会多加载一份 35B model，而是在 old/ref 阶段通过 `disable_adapter()` 用同一 FSDP2 base 计算 reference log-prob。

#### 28.11.4 该路径最值得先验证的四点

1. `all-linear` 的实际 trainable names 是否包含 expert/vision 层，`freeze_vit` 是否按预期生效；
2. 每个 parameter group 中 full Tensor 的峰值，尤其 fused expert 参数；
3. `offload_model` 对当前 PyTorch/DTensor/FSDPModule 的 CPU 回迁是否稳定；
4. vLLM Qwen3.5 MoE loader 在分组 load 后的 `finish_vllm_weight_reload` 是否正确重建专家运行时格式。

#### 28.11.5 native LoRA 优化支线

若要把后续同步降为 adapter-only，必须同时改为：

~~~bash
--vllm_enable_lora true \
--vllm_max_lora_rank 8
~~~

并把 target_modules 缩到当前 vLLM 版本明确支持的非专家、非 ViT 线性层。不能只打开开关而继续假设 `all-linear` 安全。正确验收标准包括：首次 base+adapter、第二次仅 adapter、相同 prompt/seed 下 merged full 与 native adapter 输出/log-prob 误差可接受。

### 28.12 场景二：FSDP2 Full + 单机 colocate

#### 28.12.1 三步验链模板

~~~bash
SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --enable_thinking false \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --reward_funcs accuracy \
    --tuner_type full \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --fsdp fsdp2 \
    --gradient_checkpointing false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size 8 \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_max_model_len 9192 \
    --vllm_max_num_seqs 16 \
    --vllm_enable_prefix_caching true \
    --sleep_level 2 \
    --offload_model true \
    --offload_optimizer true \
    --move_model_batches 40 \
    --max_length 1000 \
    --max_completion_length 8192 \
    --overlong_filter true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --steps_per_generation 2 \
    --num_generations 8 \
    --num_iterations 1 \
    --learning_rate 1e-6 \
    --beta 0.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --loss_type grpo \
    --temperature 1.0 \
    --split_dataset_ratio 0 \
    --max_steps 3 \
    --save_strategy no \
    --logging_steps 1 \
    --report_to none \
    --output_dir output/qwen3_5_35b_a3b_grpo_fsdp2_full_colocate_3step
~~~

该脚本是**高显存容量验证模板，不是 8×80GB 必然可行承诺**。full trainable policy 在当前 Accelerate mixed-precision FSDP2 实现中可能使用 FP32 master shard；Adam 类 optimizer state、gradient shard、layer all-gather、activation 和 vLLM runtime 叠加后，训练阶段可能接近或超过单卡容量。

#### 28.12.2 `beta=0` 为什么是容量闸门

full 场景若使用默认 GRPO beta（0.04），pipeline 会再加载一份 frozen Qwen3.5 reference，并把它包装为 FSDP2。虽然 reference 也分片，仍要增加：

- 每 rank 的 reference BF16 shard；
- reference layer all-gather 临时峰值；
- 每个新 generation batch 的 reference log-prob 前向；
- colocate `offload_context` 中 reference CPU/GPU 搬运。

模板显式 `--beta 0.0`，使 ref 在模型加载前即为 None。这是当前 FSDP2 full 相比当前 Megatron full 路径的一个实际容量优势。

#### 28.12.3 三步时序

batch 算术与 LoRA colocate 相同：generation batch=16 completions、2 prompts、每个 rollout 覆盖两个 training step。

| optimizer step | 新 rollout | 权重同步 | reference | policy 更新 |
|---:|---:|---|---|---|
| 0 | 是 | full policy state | 无 | full FSDP2 update |
| 1 | 否 | 无 | 无 | 同 rollout A 的第二缓冲继续 full update |
| 2 | 是 | 更新后的 full policy state | 无 | rollout B 第一缓冲 full update |

与 LoRA 不同，full 路径不做 adapter merge；但当前 `model.state_dict()` 中 trainable policy tensor 可能是 FP32 master dtype。一次同步要观测：

~~~text
参数 logical dtype
  → DTensor local dtype
  → full_tensor dtype
  → flattened/server bucket dtype（若 external）
  → vLLM load 后实际 storage dtype
~~~

如果 full tensor 是 FP32，colocate 每个 rank 不仅会重建更大的参数，还把 FP32 tensor 交给 BF16 vLLM loader 再做转换，临时峰值会显著高于“BF16 整模”估算。

#### 28.12.4 六个容量观测点

1. FSDP2 policy + optimizer 建立后、vLLM 尚未初始化；
2. vLLM 初始化期间，policy 被临时 CPU offload 的前后；
3. 新 rollout 前 `full_tensor()` 正在重建一层专家参数；
4. vLLM 全部权重 load/finish，trainer 尚未进入 rollout offload context；
5. trainer state 在 CPU、vLLM KV cache 活跃并生成；
6. vLLM sleep、trainer 回迁后执行 full backward/optimizer step。

如果第 3/4 点 OOM，降低 generation batch 或 `vllm_max_num_seqs` 作用有限，因为主因是权重转换；应优先减小同步组、确认 export dtype、改 external、增加显存，或改 LoRA。若第 5 点 OOM，才优先缩短 completion/KV cache。

### 28.13 FSDP2 external/server 的系统结构

external 中 rollout server 与第 27.12 节完全相同，训练侧换成 FSDP2 ranks：

~~~text
Training node 0: FSDP rank 0..7
Training node 1: FSDP rank 8..15
Rollout node:    vLLM TP=8 server workers
~~~

训练 mesh 覆盖 16 个 rank。每个 decoder FSDP unit 的 all-gather/reduce-scatter 都可能跨节点；这与 Megatron 可以通过 TP/PP/EP 映射控制某些通信局部性不同。

#### 28.13.1 server 启动模板

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rollout \
    --model Qwen/Qwen3.5-35B-A3B \
    --host 0.0.0.0 \
    --port 8000 \
    --vllm_tensor_parallel_size 8 \
    --vllm_data_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.90 \
    --vllm_max_model_len 9192 \
    --vllm_max_num_seqs 64 \
    --vllm_enable_prefix_caching true \
    --vllm_enable_lora false
~~~

首次跑通仍关闭 native LoRA。trainer 必须使用 rollout 节点可路由 IP；8000 是 HTTP 控制/请求面，`vllm_server_group_port` 是权重 PyNCCL 数据面，两类端口都要可达。

#### 28.13.2 FSDP2 external 初始化

~~~mermaid
sequenceDiagram
    participant S as "rollout server/vLLM workers"
    participant R0 as "training global rank 0"
    participant RN as "其他 FSDP ranks"
    participant F as "16-rank FSDP mesh"

    S->>S: "加载 Qwen3.5 base，启动 HTTP"
    R0->>S: "health / close communicator / init communicator"
    R0->>S: "查询 engine type、LoRA/multi-turn 能力"
    R0->>RN: "broadcast server capabilities from process 0"
    R0->>F: "policy 进入 FSDP2 fully_shard"
    RN->>F: "加入同一 device mesh"
    F->>F: "准备 generation batch"
~~~

标准 Transformers 路径由 global rank 0 创建 `VLLMClient`，不是 Megatron 的全局最后 rank。`_prepare_vllm()` 也从 process 0 广播 server 能力。

#### 28.13.3 external 新 rollout 的权重与请求时序

~~~mermaid
sequenceDiagram
    participant FR as "全部 FSDP ranks"
    participant M as "rank 0 / VLLMClient"
    participant VW as "external vLLM workers"
    participant T as "GRPO training"

    FR->>FR: "每组 state_dict + DTensor.full_tensor collective"
    FR->>M: "rank 0 获得待发送普通 Tensor；其他 rank 完成 collective 后释放"
    M->>VW: "HTTP metadata + PyNCCL flattened buckets"
    M->>VW: "process_weights_after_loading（仅 rank 0）"
    M->>VW: "reset prefix/encoder cache"
    FR->>M: "gather_object 所有 rollout requests"
    M->>VW: "HTTP /infer"
    VW-->>M: "outputs/token ids/logprobs"
    M->>FR: "broadcast_object_list，按 rank slice"
    FR->>T: "reward/old-ref logp/advantage/FSDP update"
~~~

这里没有第 27 节审计到的 Megatron `process_weights_after_loading()` 非主 rank guard 缺口：标准 FSDP2/Transformers `_move_full_model_to_vllm()` 明确只在 `accelerator.is_main_process` 时调用 server client。

#### 28.13.4 external 不会使用 colocate rollout offload

`_prepare_vllm()` 只在 `vllm_mode=colocate` 时根据 `offload_model/offload_optimizer` 设置 `enable_offload`。server 模式下：

- `sleep_level` 在参数初始化中被强制为 0；
- trainer 不会为了 external generation 自动进入 GRPO `offload_context`；
- FSDP policy/optimizer 仍驻留训练 GPU；
- rollout GPU 完全独立，所以也不需要为 vLLM 腾出训练卡空间。

如果要让 FSDP 本身长期 CPU offload，应使用独立的 FSDP CPU-offload 配置并单独评估性能；它不是这里的 `--offload_model`。

### 28.14 场景三：FSDP2 LoRA + 多机 external/server

#### 28.14.1 两个训练节点的统一模板

rollout 节点先按 28.13.1 启动。两个训练节点运行相同命令，只把 `NODE_RANK` 分别设为 0、1：

~~~bash
SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

: "${NODE_RANK:?Set NODE_RANK to 0 or 1}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the training-master routable IP}"
: "${ROLLOUT_IP:?Set ROLLOUT_IP to the rollout-server routable IP}"
export NODE_RANK MASTER_ADDR
export NNODES=2
export NPROC_PER_NODE=8
export MASTER_PORT="${MASTER_PORT:-29500}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE=256
# 生产环境应按实际网络设置 NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME/NCCL_IB_HCA。

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --enable_thinking false \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --reward_funcs accuracy \
    --tuner_type lora \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32 \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --fsdp fsdp2 \
    --gradient_checkpointing false \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "$ROLLOUT_IP" \
    --vllm_server_port 8000 \
    --vllm_server_group_port 51216 \
    --vllm_server_timeout 600 \
    --enable_flattened_weight_sync true \
    --move_model_batches 40 \
    --max_length 1000 \
    --max_completion_length 8192 \
    --overlong_filter true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --steps_per_generation 2 \
    --num_generations 8 \
    --num_iterations 1 \
    --learning_rate 5e-5 \
    --beta 0.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --loss_type grpo \
    --temperature 1.0 \
    --split_dataset_ratio 0 \
    --max_steps 3 \
    --save_strategy no \
    --logging_steps 1 \
    --report_to none \
    --output_dir output/qwen3_5_35b_a3b_grpo_fsdp2_lora_external_3step
~~~

`NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT/NPROC_PER_NODE` 由 Swift CLI 的分布式启动逻辑使用。所有训练节点必须具有相同 model revision、dataset 配置和 FSDP 参数；`NODE_RANK` 必须唯一。

#### 28.14.2 batch 算术

~~~text
training world_size = 2×8 = 16
global_micro_batch = 1×16 = 16 completions
steps_per_generation = 2
generation_batch_size = 32 completions
G = 8
rollout_prompt_count = 4 prompts
每 rank generation slice = 2 completions
~~~

FSDP2 的 world 16 全部属于同一个分片 mesh，没有 Megatron 的 `DP=world/(TP×PP×CP)` 计算。对这里的 Trainer batch 语义，16 个进程都是 data-parallel data consumers，同时也是一个 FSDP 参数分片集合。

#### 28.14.3 保守同步和 native LoRA 快速同步

| 路径 | rollout server 配置 | step 0 | 后续新 rollout | 风险 |
|---|---|---|---|---|
| 保守基线 | `vllm_enable_lora=false` | tensor-merge 后 full sync | 每次 merged full sync | 网络量大，但 MoE 兼容边界更少 |
| 快速支线 | `vllm_enable_lora=true` | base full + adapter | adapter-only | expert/ViT target 兼容、数值一致性 |

保守路径虽然只训练 LoRA，但 external 权重数据量仍接近 BF16 整模。训练显存收益与网络同步收益是两件事。

快速支线中，server 还要配置：

~~~bash
--vllm_enable_lora true \
--vllm_max_lora_rank 8
~~~

训练侧不需要再显式写 `--vllm_enable_lora true` 来改变 server；`_prepare_vllm()` 会查询 server capability 并广播 `rollout_enable_lora`。但训练的 target_modules 必须与 server/vLLM 支持集合匹配。

#### 28.14.4 三步时序

| optimizer step | rollout | 保守同步 | native LoRA 同步 | FSDP 训练 |
|---:|---:|---|---|---|
| 0 | batch A | merged full BF16 为主 | base full + FP32/BF16 adapter | A 的第 1/2 缓冲 |
| 1 | 无 | 无 | 无 | A 的第 2/2 缓冲 |
| 2 | batch B | merged full | adapter-only | B 的第 1/2 缓冲 |

step 0/2 的保守路径：

~~~text
16 ranks 逐层执行 DTensor.full_tensor
  → 每个 rank 都暂时得到本组完整普通 Tensor
  → rank 0 按 256 MiB bucket 打平并发送
  → external vLLM workers load/repack
  → rank 0 reset caches
  → 16 ranks 的 requests gather 到 rank 0
  → HTTP infer → outputs broadcast
  → 各 rank reward/old-logp/advantage
  → 16-rank FSDP2 forward/backward/update
~~~

从网络角度有两类大通信串行叠加：

1. FSDP mesh 内为了 `full_tensor()` 重建完整权重的 collectives；
2. rank 0 到 external vLLM workers 的权重广播。

这就是 FSDP2 external 同步不能只按“trainer 到 server 的网卡速度”估算的原因。

### 28.15 场景四：FSDP2 Full + 多机 external/server

#### 28.15.1 两个训练节点的统一模板

~~~bash
SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

: "${NODE_RANK:?Set NODE_RANK to 0 or 1}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the training-master routable IP}"
: "${ROLLOUT_IP:?Set ROLLOUT_IP to the rollout-server routable IP}"
export NODE_RANK MASTER_ADDR
export NNODES=2
export NPROC_PER_NODE=8
export MASTER_PORT="${MASTER_PORT:-29500}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE=256

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-35B-A3B \
    --enable_thinking false \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --reward_funcs accuracy \
    --tuner_type full \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --fsdp fsdp2 \
    --gradient_checkpointing false \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "$ROLLOUT_IP" \
    --vllm_server_port 8000 \
    --vllm_server_group_port 51216 \
    --vllm_server_timeout 600 \
    --enable_flattened_weight_sync true \
    --move_model_batches 40 \
    --max_length 1000 \
    --max_completion_length 8192 \
    --overlong_filter true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --steps_per_generation 2 \
    --num_generations 8 \
    --num_iterations 1 \
    --learning_rate 1e-6 \
    --beta 0.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --loss_type grpo \
    --temperature 1.0 \
    --split_dataset_ratio 0 \
    --max_steps 3 \
    --save_strategy no \
    --logging_steps 1 \
    --report_to none \
    --output_dir output/qwen3_5_35b_a3b_grpo_fsdp2_full_external_3step
~~~

该模板与 LoRA external 的 batch 算术相同。最大的区别不是 generation，而是训练状态和同步 dtype。

#### 28.15.2 full 训练状态的粗略下界

不计实现细节，仅用 35B 做量级估算：

~~~text
FP32 policy master params / 16 ranks ≈ 35B×4/16 ≈ 8.75 GB/rank
FP32 grads / 16 ranks（若保持 FP32）≈ 8.75 GB/rank
Adam m,v / 16 ranks ≈ 35B×8/16 ≈ 17.5 GB/rank
~~~

再加：

- activation 和 checkpoint 重算工作集；
- 每层/每参数 full all-gather 临时 buffer；
- CUDA allocator/NCCL buffer；
- frozen vision/其他状态；
- rollout 同步时的普通 full Tensor；
- 若 beta>0，再加 reference BF16 shard和前向。

这是量级模型，不是 PyTorch 精确 memory accounting；实际 optimizer、dtype、padding、参数 tying 和版本会改变数字。但它解释了为何 world=16 的 external full 比 world=8 colocate 更有容量余地，同时仍不等于低显存可运行。

#### 28.15.3 full external 的 dtype 放大风险

LoRA 保守路径最终把 delta cast 到 base dtype，通常可接近 BF16 整模同步。Full 路径没有这一步。当前代码链是：

~~~text
FSDP2 trainable FP32 master DTensor
  → full_tensor() 可能得到 FP32
  → FlattenedTensorBucket 保留 tensor dtype/metadata
  → PyNCCL 发送
  → server/vLLM loader 转成目标 dtype
~~~

如果目标版本确认为 FP32，则一次原始 full sync 量级可能接近 140 GB，而非 BF16 的约 70 GB，下游还要承担 FP32 flat buffer/转换峰值。生产前必须记录实际 `sum(numel×element_size)`，不能只记录参数量。

如果希望在同步边界显式 cast 为 BF16，需要进行代码级改动和数值验证；当前 CLI 没有一个已核实的开关可保证这件事。本报告不把尚未实现的 cast 写成可用参数。

#### 28.15.4 三步时序与时间分解

| optimizer step | FSDP2 full sync | rollout | training |
|---:|---:|---:|---|
| 0 | 是，全部 policy 参数 | batch A | A 的第 1/2 缓冲 full update |
| 1 | 否 | 无 | A 的第 2/2 缓冲 full update |
| 2 | 是，更新后全部 policy 参数 | batch B | B 的第 1/2 缓冲 full update |

时间可分解为：

~~~text
T_new_rollout ≈
    T_FSDP_full_tensor_collectives
  + T_rank0_bucket_flatten
  + T_cross_cluster_weight_broadcast
  + T_vLLM_load_and_repack
  + T_request_gather_HTTP_rollout_output_broadcast
  + T_reward_encode_old_logp_advantage

T_training_step ≈
    T_layer_param_all_gather
  + T_forward
  + T_activation_recompute
  + T_backward_reduce_scatter
  + T_optimizer_shard_update
~~~

step 0/2 包含两部分，step 1 主要只有训练部分。增大 `steps_per_generation` 能摊薄第一部分，但也扩大一批 completion 被多个 policy update 重用的 off-policy/staleness 程度。

#### 28.15.5 什么时候不应继续强推 FSDP2 full external

出现下列组合时，应优先回到 Megatron/LoRA/更大资源的架构选择，而不是只调 bucket：

- FSDP layer all-gather 已经占训练 step 大头；
- `full_tensor()` 重建和 external send 都是 full-size，形成双重通信；
- full sync 实际为 FP32，rollout 同步超过生成时间很多；
- 256-expert decoder unit 的瞬时 gather 无法容纳；
- 需要 EP 来把专家计算/参数局部化；
- 多机网络无法提供稳定 RDMA/NCCL 带宽。

### 28.16 checkpoint、恢复与最终模型导出

#### 28.16.1 默认 checkpoint 语义

内置 preset 使用 `SHARDED_STATE_DICT`：

~~~text
checkpoint-k/
  ├─ 各 rank 模型 shard
  ├─ optimizer shard/state
  ├─ scheduler/trainer state
  └─ RNG/数据进度等恢复信息
~~~

它适合在相同或受支持的 FSDP 拓扑中恢复训练，避免保存时在 rank 0 聚合一份 35B full model。

#### 28.16.2 `save_only_model` 限制

当前参数检查明确拒绝：

~~~text
save_only_model=true
+ SHARDED_STATE_DICT
~~~

可选方向是保留完整训练 checkpoint，或使用自定义 FSDP config 改成 FULL_STATE_DICT；后者会增加保存内存和时间。不要为得到一个普通 HF 目录而在训练主循环中频繁 full gather 35B。

#### 28.16.3 LoRA 与 full 的最终交付

- LoRA：训练 checkpoint 的主要交付物是 adapter；如果需要独立 merged model，应在训练后专门 export/merge，并给足 CPU RAM/GPU 内存。
- Full：SHARDED checkpoint 先用于恢复；最终发布时再执行一次受控 consolidated export。
- rollout 临时 merged state：只用于 vLLM 同步，不等于已经产生可持久化 merged checkpoint。

#### 28.16.4 恢复时的检查

恢复三步短跑或生产 checkpoint 时，要同时核对：

1. FSDP2 state-dict 类型和 Accelerate/PyTorch 版本；
2. world size/device mesh 是否受当前 checkpoint loader 支持；
3. LoRA adapter 名称与 target_modules；
4. optimizer/scheduler/RNG 是否保存；
5. GRPO buffered rollout 不作为通用精确恢复对象时，恢复后的首次 step 是否重新 rollout；
6. external server 的权重必须重新从 policy 同步，不能假设旧 server 状态等同于 checkpoint policy。

### 28.17 FSDP2 与 Megatron 四场景逐项对照

| 维度 | FSDP2 | Megatron |
|---|---|---|
| 入口 | `swift rlhf --fsdp fsdp2` | `megatron rlhf` |
| 训练循环 | HF Trainer/Swift GRPOTrainer | Megatron pipeline schedule |
| 参数分片 | DTensor FULL_SHARD | TP/PP/EP/DP partition |
| MoE 专家并行 | 默认无 | 原生 EP |
| 层流水 | 无 PP | 原生 PP |
| tensor parallel | 训练侧默认无 | 原生 TP |
| activation memory | FSDP activation checkpointing | recompute + CP/SP 等 |
| LoRA reference | disable adapter | disable adapter |
| Full beta=0 reference | 不加载独立 ref | 当前路径仍构造 ref |
| vLLM full sync 来源 | `state_dict + DTensor.full_tensor` | mcore-bridge export iterator |
| LoRA 非原生同步 | 普通 Tensor 层 merge | Megatron adapter merge/export |
| external client rank | global rank 0 | global last rank |
| server finish guard | main-process guard 已存在 | 当前提交需核对第 27 节 guard 风险 |
| checkpoint | FSDP sharded state dict | Megatron distributed checkpoint |
| 上手/生态 | HF/PEFT/Trainer 路径直观 | 大模型并行配置复杂 |
| 35B MoE 扩展效率 | 简单但可能通信重 | 通常更适合 TP/PP/EP |

四场景选择建议：

| 目标 | 首选 | 原因 |
|---|---|---|
| 先验证 Qwen3.5 GRPO 功能与 reward | FSDP2 LoRA external 或 colocate | 主路径功能完整，beta=0 无 ref copy |
| 单机高显存、希望最少集群组件 | FSDP2 LoRA colocate | 一个进程组内完成训练/生成 |
| full 且必须 HF Trainer 生态 | FSDP2 full external，先短序列容量测 | rollout 显存隔离，world16 减少训练 shard |
| full 35B MoE 追求吞吐/专家并行 | Megatron external | TP/PP/EP 更贴合 MoE |
| external 网络敏感、可接受 LoRA target 限制 | 验证 native vLLM LoRA | 首次后 adapter-only |
| 8GB 单卡/8×8GB | 不使用本节 35B 模板 | 仅 BF16 base FSDP shard 下界就已不合理 |

### 28.18 FSDP2 的显存/内存不能混为一个数字

建议按状态分类：

| 状态 | 主要 GPU 对象 | 主要 CPU 对象 | 主峰值来源 |
|---|---|---|---|
| 初始化 | full/sharded model、FSDP load buffer | rank0 full state、其他 rank meta/empty | CPU RAM efficient load 的分发 |
| 训练 forward | policy shard + 当前 unit full params + activation | 少量数据对象 | decoder/MoE unit all-gather |
| backward | shard + gathered unit + gradients + recompute | — | activation 重算、reduce-scatter buffer |
| optimizer | policy/grad/optimizer shards | — | FP32 master + Adam states |
| colocate sync | policy shard + vLLM weights + full_tensor group | 已 offload与否取决于时点 | 普通 full Tensor 与 vLLM loader |
| colocate rollout | vLLM weights/KV | policy/optimizer offload copy | KV cache、CPU RAM/PCIe 带宽 |
| external sync | trainer shard + full_tensor group | server独立 | FSDP重建 + rank0 flat bucket |
| checkpoint | shard/save buffers | 分片文件页缓存 | state-dict 类型 |

“稳态每卡参数 shard 大小”只覆盖表中的一小部分。容量验收应至少记录 `torch.cuda.max_memory_allocated/reserved`、CPU RSS、bucket bytes 和各阶段时间戳。

### 28.19 重要参数的真实作用与常见误解

| 参数 | 真实作用 | 常见误解 |
|---|---|---|
| `--fsdp fsdp2` | FULL_SHARD + auto-wrap + sharded state + activation ckpt | 会自动开启 TP/EP |
| `reshard_after_forward=true` | unit 前向后释放非本地完整参数 | 会完全消除 gather 峰值 |
| `cpu_ram_efficient_loading=true` | 初始化时减少多 rank full CPU load | 训练期间持续 CPU offload |
| `activation_checkpointing=true` | 省 activation，backward 重算 | 省参数/optimizer/vLLM KV |
| `--offload_model` | colocate rollout 期间把 trainer model 搬 CPU | external 也会自动执行 |
| `--offload_optimizer` | colocate rollout 期间搬 optimizer state | optimizer 永久在 CPU |
| `--move_model_batches 40` | vLLM 同步时分组重建/加载 | 把单个 expert tensor 也切 40 份 |
| `--sleep_level 1/2` | colocate vLLM 释放不同级别资源 | server 模式也由 trainer 控制 |
| `--vllm_enable_lora true` | 允许 adapter-only 快速同步 | 所有 MoE/ViT LoRA target 都支持 |
| `--beta 0` | 跳过 ref KL；FSDP2 full 还不加载 ref | 不影响算法正则/行为 |
| `steps_per_generation` | 一批 rollout 覆盖几个 training micro-step | 永远等于 optimizer steps |
| `SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE` | external full sync 分块大小 | 能减少总传输字节 |

### 28.20 观测、失败模式与定位路径

#### 28.20.1 必须记录的指标

| 层 | 指标 | 回答的问题 |
|---|---|---|
| FSDP unit | 每层 all-gather/reduce-scatter 时间 | 训练慢在通信还是计算 |
| DTensor export | 每个 group `full_tensor()` 时间/峰值/dtype | rollout sync 是否被重建拖慢 |
| parameter floor | 最大单 tensor numel/bytes | `move_model_batches` 的不可再分峰值 |
| colocate load | `load_weights` 和 finish reload 时间 | vLLM loader/repack 成本 |
| external send | bucket 数、bytes、PyNCCL GB/s | 跨机权重数据面瓶颈 |
| lifecycle | model/optimizer CPU↔GPU bytes/time | offload 是否得不偿失 |
| rollout | TTFT、decode tok/s、KV 使用、finish_reason | generation 容量与吞吐 |
| policy drift | rollout/old/current log-PPL、KL、clip ratio | engine 差异与 staleness |
| GRPO | reward 分布、zero-std group、advantage、有效 token | 算法是否有学习信号 |
| host | 各 rank RSS、page fault、PCIe/NVLink/IB 带宽 | CPU RAM/offload/网络是否饱和 |

#### 28.20.2 典型失败模式

| 现象 | 首查 | 解释/方向 |
|---|---|---|
| `Qwen3_5Moe...` import 失败 | Transformers 版本 | 当前旧环境没有该 module；按 loader 依赖升级并锁版本 |
| FSDP1 NotImplementedError | fsdp config 中 version | rollout mixin 只支持 FSDP2 |
| FSDP 与 DeepSpeed 冲突 | CLI 同时传了两者 | 两条分片后端二选一 |
| device_map 冲突 | 单进程 MP/可见 GPU 与 world | FSDP2 要由多进程 mesh 管理，不用 device_map 切模型 |
| auto-wrap 找不到层 | `_no_split_modules`、模型 revision | 核对 `Qwen3_5MoeDecoderLayer/VisionBlock` 类名 |
| 第一次 forward OOM | decoder unit expert gather | 不是 batch 单独问题；检查 wrap 单元/资源/改 Megatron EP |
| 第一次 vLLM sync OOM | full_tensor group/单 tensor | 增大 move batches、看最大 fused expert 参数、检查 dtype |
| full external 流量翻倍 | state_dict tensor 为 FP32 | 记录 element_size；当前代码未显式 rollout cast |
| rank 0 server send 卡住 | 其他 FSDP rank 未进 full_tensor collective | external 也要求所有训练 rank 同步进入导出 |
| health 正常但权重同步失败 | group port/NCCL NIC | HTTP 控制面与 PyNCCL 数据面分开 |
| native LoRA add/load 报错 | expert/ViT target、rank | 回退 merged full；再做模块白名单 |
| sync 后仍像旧 policy | cache reset/finish reload | 检查 prefix/encoder cache 和 process_weights_after_loading |
| colocate offload 后回迁失败/OOM | FSDP2 `.cpu()`/`.to()`、CPU RAM | 分阶段验证，记录 host RAM 和碎片 |
| full beta=0 仍有 ref | args.json/实际 beta/显式 ref_model | FSDP2 正常应在 pipeline 阶段置 None |
| save_only_model 报错 | SHARDED_STATE_DICT | 保留完整 checkpoint 或改 full state 配置 |
| rollout 频率与预期不符 | grad accumulation、SPG、`_step` | SPG 按 training_step，不只看 global_step |

#### 28.20.3 推荐的短跑验收顺序

1. 先在兼容版本上只加载 Qwen3.5 config/model，打印 `_no_split_modules` 和 trainable names；
2. FSDP2 不接 vLLM，跑一个极短 forward/backward，确认 unit wrap、dtype、checkpoint；
3. 单独启动/验证 vLLM Qwen3.5 server 或 colocate engine；
4. 接通首次 full weight sync，先不追求 8192 completion；
5. 用 `max_completion_length=64/128` 完成一个 rollout/reward/old-logp/backward；
6. 跑本节 3 step，确认 rollout 在 step 0/2 发生、step 1 使用 buffer；
7. LoRA 再实验 native adapter-only，full 再观测 FP32/BF16 export；
8. 最后逐步放大 completion length、generation batch、rollout replicas 和保存频率。

### 28.21 后续深入代码导航

| 想研究的问题 | 第一入口 | 建议断点/输出 |
|---|---|---|
| `--fsdp fsdp2` 如何展开 | [SftArguments._init_fsdp](../../swift/arguments/sft_args.py#L281) | `self.fsdp/self.fsdp_config`、环境变量 |
| Qwen3.5 wrap 类名从哪来 | 上游 `_no_split_modules` | 打印实际 model class 与 decoder/vision block |
| policy 何时变成 DTensor | Accelerate `fsdp2_prepare_model` | fully_shard 前后 `type(param)`、placements、dtype |
| reference 为什么有/没有 | [RLHFArguments.__post_init__](../../swift/arguments/rlhf_args.py#L305) | beta/tuner_type/ref_model |
| frozen ref 如何 FSDP2 化 | [prepare_fsdp](../../swift/rlhf_trainers/utils.py#L954) | freeze 时点、FP32 upcast 是否发生 |
| rollout 参数如何分组 | [split_batches](../../swift/rlhf_trainers/rollout_mixin.py#L346) | 每组 names、numel、最大单 tensor |
| DTensor 如何变完整权重 | [_collect_state_dict_for_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L715) | `full_tensor()` 时间、dtype、bytes |
| FSDP2 LoRA 如何合并 | [_merge_lora_into_state_dict](../../swift/rlhf_trainers/rollout_mixin.py#L652) | A/B shape、scaling、base dtype |
| full/adapter 如何决策 | [_move_model_to_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L450) | base_sync_done、sleep、rollout_enable_lora |
| colocate 如何 load MoE | [_move_full_model_to_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L775) | MoE loader patch、finish reload |
| external bucket 怎么传 | [_load_state_dict_to_vllm](../../swift/rlhf_trainers/rollout_mixin.py#L549) | bucket metadata/size、main rank guard |
| server client 在哪个 rank | [_init_external_vllm](../../swift/arguments/rlhf_args.py#L447) | `is_master()`、communicator rank |
| rollout 请求如何聚合 | [_server_rollout](../../swift/rlhf_trainers/rollout_mixin.py#L1013) | gather lengths、rank slice、broadcast |
| rollout/offload 顺序 | [_fast_infer](../../swift/rlhf_trainers/rollout_mixin.py#L934) | wake、sync、offload context、sleep |
| SPG 为什么按 micro-step | [GRPOTrainer._prepare_inputs](../../swift/rlhf_trainers/grpo_trainer.py#L196) | `_step`、buffer index、global_step |
| old/ref log-prob 时点 | [_prepare_batch_inputs](../../swift/rlhf_trainers/grpo_trainer.py#L828) | beta、null_ref_context、completion mask |
| checkpoint state 类型 | [fsdp2.json](../../swift/config/fsdp2.json) | SHARDED/FULL、rank 文件与恢复 |

### 28.22 本节整体全貌

FSDP2 的 Qwen3.5 GRPO 可以用两条交错的生命周期理解：

~~~text
训练生命周期
  shard policy/reference/gradient/optimizer
  → 每层 gather 参数
  → current/old/ref forward
  → reduce-scatter gradient
  → optimizer shard update

rollout 生命周期
  FSDP DTensor state
  → full_tensor 普通权重
  → 可选 LoRA tensor merge
  → vLLM TP 权重布局
  → completion/token ids/logprobs
  → reward/advantage/buffer
~~~

colocate 让这两条生命周期在同一批 GPU 上交替，通过 sleep 和 CPU offload 争取容量；external 把 vLLM 显存移到独立机器，却仍需先在 FSDP mesh 中重建普通 full tensor，再从 rank 0 发给 server。

LoRA/full 决定训练状态和 reference 成本；native-LoRA 是否可用决定 rollout 同步是 adapter 量级还是整模量级；FSDP unit 和 DTensor 决定训练/导出的集合通信；colocate/external 决定 GPU 分时还是跨机传输。

对 Qwen3.5-35B-A3B 这个 256-expert MoE，最关键的架构判断不是“FSDP2 能不能把 35B 除以 GPU 数”，而是：

> **默认 decoder-layer FSDP unit 会重建整层专家参数，rollout 同步还会把 DTensor 还原成普通 full tensor；如果这两类重建/通信成为主成本，就应转向 Megatron TP/PP/EP，而不是继续把 FSDP2 当成专家并行。**

FSDP2 的价值在于提供一条与 HF Trainer、PEFT、Swift GRPO 功能紧密集成、reference 语义清晰的原生路径；Megatron 的价值在于更贴合 35B MoE 的训练并行与专家执行。第 27、28 节合起来，才构成 Qwen3.5 四种部署场景在两套后端上的完整选择图。
