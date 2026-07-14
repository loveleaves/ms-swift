# GRPO LoRA 训练 vLLM Rollout 层输出乱码/重复 Token 问题定位思路

> 适用前提：已通过前序排查确认 loss/reward 恒为 0 的根因位于**采样(rollout)层**——即开启 `--log_completions true` 后，观察到 vLLM 生成的 completion 表现为乱码（多语言混杂、无意义符号）或重复 token（同一 token/短语反复输出直至 `max_completion_length`）。本手册聚焦这一具体现象，给出比通用报告更深入、更可操作的定位路径。

---

## 0. 先明确一件事：生成退化 ≠ reward 函数的问题

一旦确认"生成内容本身就乱了"，就不要再往 reward 函数、advantage 计算方向排查——那些环节的输入（completions）已经是坏的，输出必然异常，是**症状**而非根因。真正的根因一定在"模型是怎么被加载/同步到 vLLM 并生成这段文字"的链路里。本手册把这条链路拆解为 6 类根因，并给出对应的判别手段。

---

## 1. 根因分类树：为什么 vLLM 会生成乱码/重复 Token

```
生成乱码/重复 token
├── A. 权重/参数问题
│   ├── A1 LoRA 权重未同步 / 同步失败（vLLM 侧仍是原始底座权重，或权重损坏）
│   ├── A2 LoRA 相关参数不一致（lora_rank ≠ vllm_max_lora_rank 等）导致 vLLM LoRA kernel 出错但未报错/被吞掉
│   ├── A3 权重精度/dtype 不匹配（训练侧 bf16，同步到 vLLM 后精度转换异常）
│   └── A4 量化模型（GPTQ/AWQ/BNB）与 LoRA 权重合并/加载路径不兼容
├── B. 模板/Tokenizer 问题
│   ├── B1 chat_template 与训练时不一致（例如 enable_thinking/preserve_thinking 配置不同步）
│   ├── B2 special token（eos_token_id、pad_token_id、bos_token_id）配置与模型不匹配
│   └── B3 误用 base（非对齐）模型或 model_type 指定错误，导致模板套用错误
├── C. 采样参数问题
│   ├── C1 temperature/top_p/top_k 设置不合理（尤其过高的 temperature 叠加长序列）
│   ├── C2 repetition_penalty 缺失或过低
│   └── C3 max_completion_length 与 stop 条件配置不当，导致本应提前停止的序列被迫续写成乱码
├── D. 版本/环境兼容性问题
│   ├── D1 vLLM 特定版本存在已知的重复生成 bug（有明确版本回归案例）
│   └── D2 ms-swift / transformers / vLLM / deepspeed 版本组合不在官方验证矩阵内
├── E. 多模态/特殊架构问题（若为 VL/Omni 模型）
│   ├── E1 ViT/aligner 层未随 LoRA 正确同步
│   └── E2 多模态混合数据（图像/音频/视频）预处理异常，污染了纯文本生成的上下文
└── F. 并行/分布式权重收集问题（TP>1 或多机场景）
    ├── F1 ZeRO-3 下权重 all-gather 不完整（大模型未分批收集导致部分权重丢失/错位）
    └── F2 Megatron 训练路径下权重导出到 HF 格式的转换存在偏差
```

**关键先验判断**（32B 级别 + LoRA + colocate 的场景下，根因出现频率从高到低大致为）：A1/A2（LoRA 同步问题）＞ D1/D2（版本兼容性）＞ F1（大模型 ZeRO-3 权重收集不完整，32B 属于典型的高发体量）＞ B1/B3（模板问题）＞ C（采样参数，通常只是"加重"而非"制造"乱码）。这个优先级用于指导下面第 2 节的排查顺序，但**不能替代实际验证**。

---

## 2. 核心判别实验：定位问题在"同步/加载"还是"模型/配置"本身

在深入具体根因之前，必须先做一个**分岔点实验**，它能把 A/D/F 一类问题（权重同步/环境相关）和 B/C 一类问题（模型/模板/采样参数本身）彻底分开，避免在错误的方向上反复排查。

### 实验步骤

1. 取当前 GRPO 训练所用的 **同一份权重**（LoRA adapter + 对应底座模型，训练开始前的初始版本即可）。
2. 完全脱离 GRPO 训练流程，用以下两种方式之一做纯推理：
   - `swift infer --model <base_model> --adapters <lora_adapter_path> --infer_backend vllm`
   - 或直接 `vllm serve` / `swift deploy` 加载同一底座+adapter，用与训练脚本**完全相同**的 `temperature`/`top_p`/`top_k`/`max_new_tokens`、以及**相同的 prompt（从训练数据集里原样取几条）**发起请求。
3. 观察这次独立推理是否也出现乱码/重复 token。

### 结果判读

| 独立推理结果 | 结论 | 后续排查方向 |
|---|---|---|
| **正常**（生成通顺，无乱码/重复） | 问题与模型本身、chat_template、tokenizer 无关，一定出在 **GRPO 训练流程中"训练权重 → vLLM 推理引擎"这条同步链路**，或训练脚本里传给 vLLM 的采样参数与独立推理不一致 | 直接跳到本手册第 3 节（权重同步类根因） |
| **同样异常** | 问题与 GRPO 训练流程本身无关，是**模型/权重/模板/tokenizer 配置**的问题，与是否用 GRPO 训练无关（哪怕不做强化学习，正常推理也会乱） | 跳到本手册第 4 节（模型与模板类根因） |

这一步是整份手册里性价比最高的一步操作——花几分钟做一次独立推理对比，就能把排查范围直接砍掉一半以上。**社区中一个非常典型的案例**（Qwen3-Omni-30B-A3B 用 Megatron+GRPO 训练）就是通过这个对比发现：同一 checkpoint 用 `swift infer` 和 `vllm serve` 直接推理完全正常，但走 GRPO 训练的 rollout 环节（无论 colocate 还是 server 模式）都输出乱码/重复/截断，从而把根因精确定位到了"训练框架内部的权重同步/传递环节"，而不是模型或数据本身。

---

## 3. 若判定为"权重同步/训练链路"问题（分岔结果为"独立推理正常"）

### 3.1 A1：LoRA 权重未同步或同步失败

**验证方法**：在训练若干个 step（建议至少能触发 1~2 次 policy 更新）前后，分别打印/记录 vLLM 对**同一个 prompt** 的生成结果。若训练前后生成内容完全一致（哪怕都是乱码），说明 vLLM 侧根本没有拿到更新后的权重，这是最需要优先排除的一种情况——因为它意味着"乱码"甚至可能只是初始未训练权重本身在特定采样参数下的表现，而非训练引入的新问题。

**排查动作**：
- 检查训练日志中是否有权重同步相关的报错、警告，或耗时异常长（同步失败有时会静默跳过而不报错，需要在源码层面确认同步是否真正发生）。
- Colocate 模式下确认 `--sleep_level`、`--offload_model` 等参数没有导致同步时机错乱（例如同步发生在模型被 offload 到 CPU 之后，导致同步的是空/旧张量）。
- 若为 ZeRO-3，确认权重收集（all-gather）过程完整（见本节 3.4）。

### 3.2 A2：LoRA 相关参数不一致，触发 vLLM LoRA kernel 层面的隐性错误

**核心检查项**：`--lora_rank`（训练脚本）与 `--vllm_max_lora_rank`（vLLM/rollout 侧，仅当使用"仅同步 LoRA adapter"优化、即 `vllm_enable_lora true` 时才涉及）必须严格一致。社区中已有明确复现：当二者不匹配或 vLLM 版本对 LoRA kernel 支持不完善时，会在 vLLM 的 LoRA 算子（如 `lora_shrink`）内部触发形如 `token_lora_mapping` 维度不匹配的断言错误；即使该错误没有直接让训练进程崩溃（比如被上层 try/except 吞掉），也足以导致该 batch 的 LoRA 权重没有正确应用到本次生成，从而让 completion 表现为"实际上是未经 LoRA 微调的底座模型在生成"，如果底座本身是尚未充分对齐的模型或采样参数不合适，就会退化为乱码/重复。

**排查动作**：
- 核对 `lora_rank` 与 `vllm_max_lora_rank` 一致；
- 检查是否有能支持当前 `lora_rank` 的 vLLM 版本（不同 vLLM 版本对 LoRA rank 支持的算子实现存在差异）；
- 若怀疑该问题，可临时关闭"仅同步 LoRA"优化（即不设置 `vllm_enable_lora`，走全量权重同步路径）做对照实验：如果关闭后乱码消失，基本可以确认是 LoRA 同步路径的问题。

### 3.3 A3/A4：精度不匹配、量化模型兼容性问题

- **精度问题**：确认 `--torch_dtype` 与 vLLM 侧加载精度一致（一般统一用 `bfloat16`）；ZeRO-3/FSDP 环境下，模型分片状态与同步给 vLLM 时的合并精度容易在混合精度配置不当时出现数值异常，从而在极端情况下表现为生成質量骤降甚至乱码。
- **量化模型**：若底座模型本身是 GPTQ/AWQ 等量化格式，需要额外注意——量化模型通常无法直接与训练出的 LoRA adapter 做常规 merge，若走"外部部署 vLLM server + merge 后模型"的路径，量化模型的 merge 本身就可能不受支持或产生错误权重，这种情况下应改用非量化底座做 GRPO 训练，或确认所用版本对"量化底座 + LoRA 训练"这一组合的支持状态。

### 3.4 F1：ZeRO-3 场景下权重收集(all-gather)不完整（32B 级别模型高发）

对于 32B 这个体量的模型，在 ZeRO-3 下把训练权重同步给 vLLM 需要先做一次跨设备的参数收集（gather），如果一次性收集全部参数，容易在显存/通信压力下出现收集不完整或超时被截断的情况，导致同步到 vLLM 的权重实际上是"残缺"的（部分层是新权重，部分层还是初始化或垃圾值），这种"半新半旧甚至半随机"的权重状态非常容易表现为生成乱码。

**排查/修复动作**：
- 确认使用了分批收集参数的机制（`--move_model_batches`），而不是一次性全量收集，尤其是 32B 级别模型；
- 检查权重同步过程的日志耗时是否与预期相符（异常短的同步耗时可能意味着提前退出/收集不完整）；
- 若怀疑收集不完整，可在同步后加入权重一致性校验（比如同步前后对某几个具体张量做 checksum/范数对比），确认目标权重被完整、正确地传输。

### 3.5 F2：Megatron 路径下权重导出转换问题

若训练路径是 Megatron-swift（而非常规 Transformers-based ms-swift），需要额外确认 Megatron 格式权重导出为 HF 格式、再同步给 vLLM 这一额外环节没有出错——这是 Megatron 路径独有的风险点，常规路径不涉及。社区已有的复现（如 Qwen3-Omni-30B-A3B 用 Megatron+GRPO 训练乱码，但 `swift infer`/`vllm serve` 直接加载同一权重却正常）明确指向了这一环节：说明权重本身没问题，是导出/同步过程中出现了偏差。排查时应重点核对导出脚本/接口版本，并对导出后的 HF 格式权重单独做一次离线推理验证（即再执行一次本手册第 2 节的判别实验，但这次针对"导出后的 HF 权重"而不是原始 Megatron checkpoint）。

---

## 4. 若判定为"模型/模板/配置本身"问题（分岔结果为"独立推理同样异常"）

### 4.1 B1：chat_template 与训练阶段不一致（Qwen3 系列的高发点）

Qwen3 系列模型有"思考模式"（`<think>...</think>`）的概念，ms-swift 提供了 `enable_thinking`、`preserve_thinking`、`chat_template_kwargs` 等参数来控制训练与推理时是否携带/保留思考过程。如果训练侧（SFT 阶段模板）与 rollout 侧（vLLM 应用的模板）对这些参数的处理不一致，容易导致 prompt 拼接错位（例如该由 vLLM 侧补全的思考前缀 `<think>` 被错误地放进了 prompt 还是 completion 里），进而让生成从错误的上下文状态开始续写，退化成乱码或重复。

**排查动作**：
- 核对训练与 rollout 阶段使用的 `model_type`/模板版本是否一致；
- 若使用多轮/思考模式，确认 `enable_thinking`/`preserve_thinking` 等参数在训练与 rollout 侧设置一致；
- 直接打印 vLLM 实际接收到的、模板渲染后的完整 prompt 文本（而不是原始 messages），肉眼核对格式是否符合预期（有无异常拼接、重复的特殊 token 等）。

### 4.2 B2：special token 配置错误

如果 `eos_token_id`/`pad_token_id` 等特殊 token 的配置与模型实际训练时使用的不一致，会导致模型本该在合适位置停止生成，却因为 vLLM 没有正确识别停止条件而被迫继续生成，越过正常内容后进入"无意义延伸"状态，表现为大量重复的换行符、标点或固定词汇填充至 `max_completion_length`——这也是本手册开头"多语言乱码混合 + 换行符重复填充至上限"这一类现象的典型成因之一。

**排查动作**：确认 tokenizer/生成配置中的 `eos_token_id` 与模型实际保存的配置一致（ms-swift 在加载模型时通常会打印"tokenizer 的 PAD/BOS/EOS token 与模型配置不一致，已自动对齐"这类提示，需要重点关注这类日志，确认自动对齐后的取值是你期望的）。

### 4.3 B3：误用 base 模型或 model_type 指定错误

若实际加载的是未经指令对齐的 base 模型（而非 instruct/chat 版本），或 `--model_type` 指定错误导致套用了不匹配的 chat template，模型本身就没有"服从模板续写"的能力，独立推理阶段就会表现出重复、跑题、乱码等退化现象，这与是否做 GRPO 训练完全无关。

**排查动作**：确认模型路径确实是 instruct/chat 版本；确认 `--model_type` 与模型架构、tokenizer 匹配（不要凭经验/复制别的脚本，逐一核对）。

### 4.4 C 类：采样参数问题（通常是"放大器"而非"根本原因"，但仍需核对）

即便模型和模板都正确，过高的 `temperature`（如 >1.2）叠加较长的 `max_completion_length`，也可能显著增加生成陷入重复循环的概率；`repetition_penalty` 缺省或设置过低同样会加重这一问题。

**排查动作**：对照官方 GRPO 最佳实践中的推荐采样参数区间（`temperature` 常见 0.7~1.0，配合适当的 `top_p`/`top_k`/`repetition_penalty`），如果当前配置明显偏离推荐区间，先做一次参数收敛的对照实验。**但要注意**：如果第 2 节的判别实验显示乱码在正常采样参数下的独立推理中依然出现，说明采样参数不是根本原因，只是让本已存在的问题更容易被观察到。

### 4.5 D 类：版本兼容性问题——存在明确的版本回归案例

社区中已有明确记录：在某个 vLLM 版本（0.14.0）下，使用相同环境和配置，模型会大量输出重复 token（如同一词语反复出现）直到达到最大长度；而将 vLLM 降级到前一个版本（0.13.0）后，同样的环境和配置即可恢复正常生成。这是一个**版本回归导致的已知 bug**，不是配置或代码逻辑问题。

**排查动作**：
- 记录当前 vLLM（以及 transformers、ms-swift、deepspeed）的精确版本号；
- 在 ms-swift 和 vLLM 的 GitHub issue 中按"版本号 + repetitive/重复 completion"等关键词搜索，确认当前版本是否已被社区报告存在类似问题；
- 若怀疑是版本回归，可采用"控制变量降级测试"：仅将 vLLM 降级到前一个稳定版本，其余环境不变，重新跑最小复现场景，观察乱码是否消失；
- 若确认是版本回归 bug，优先选择官方已验证或社区确认可用的版本组合，而不是花时间在业务代码层面找"假根因"。

### 4.6 E 类：多模态/特殊架构问题（仅当训练对象为 VL/Omni 等多模态模型时适用）

若训练对象是多模态模型且开启了 ViT/aligner 层的 LoRA（`freeze_vit false`/`freeze_aligner false`），需要确认这些模块的权重同步路径与语言模型主干是分开处理的，一旦遗漏，会导致视觉/音频编码部分仍是旧权重、语言模型部分是新权重，模态间不匹配可能间接导致语言输出端出现前后不连贯甚至乱码；此外，audio/video 等多模态数据的预处理异常（如像素/帧数超限、编解码失败）也可能污染送入语言模型的上下文，进而在纯文本输出侧表现为乱码。这类问题的判别方法是：用纯文本样本单独测试（不携带任何图像/音频/视频输入）是否仍然乱码——如果纯文本样本正常，说明问题出在多模态编码/对齐环节，而不是语言模型本身。

---

## 5. 完整定位决策流程图（文字版）

```
Step 0: 最小复现（少量数据、logging_steps=1、log_completions true）确认问题稳定复现
   │
Step 1: 用同一权重做"独立推理对比实验"（本手册第2节）
   │
   ├── 独立推理正常 → 问题在"权重同步/训练链路"
   │     │
   │     ├── 训练前后 vLLM 生成内容是否变化？
   │     │     ├── 不变化 → 优先怀疑 A1（LoRA 权重未同步/同步失败）
   │     │     └── 有变化但仍乱码 → 继续下面判断
   │     │
   │     ├── 是否使用 vllm_enable_lora + lora_rank/vllm_max_lora_rank 不一致？→ A2
   │     ├── 是否 32B 级别 + ZeRO-3 且未使用 move_model_batches 分批收集？→ F1
   │     ├── 是否量化底座 + LoRA？→ A4
   │     └── 是否 Megatron 训练路径？→ F2（重点检查权重导出转换环节）
   │
   └── 独立推理同样乱码 → 问题在"模型/模板/配置本身"，与 GRPO 训练框架无关
         │
         ├── 是否为 base（非 instruct）模型或 model_type 配置错误？→ B3
         ├── 是否 enable_thinking/preserve_thinking 训练与推理不一致？→ B1
         ├── 打印渲染后的完整 prompt，格式是否异常？→ B1/B2
         ├── eos/pad/bos token 配置是否与模型实际配置一致？→ B2
         ├── 采样参数（temperature/top_p/repetition_penalty）是否明显偏离推荐区间？→ C
         └── 是否为已知版本回归 bug（尤其近期升级过 vLLM/transformers）？→ D
```

---

## 6. 根因 → 修复动作对照表

| 根因编号 | 现象特征 | 修复动作 |
|---|---|---|
| A1 权重未同步 | 训练前后 vLLM 生成内容完全不变 | 核查同步时机与 offload/sleep 配置的先后顺序；检查同步是否有静默失败的报错/日志 |
| A2 LoRA 参数不一致 | 关闭"仅同步 LoRA"后乱码消失 | 统一 `lora_rank` 与 `vllm_max_lora_rank`；必要时升级/降级 vLLM 到支持当前 rank 的版本 |
| A3 精度不匹配 | 数值异常、生成质量骤降 | 统一 `torch_dtype`，核对 ZeRO-3/FSDP 合并精度配置 |
| A4 量化模型不兼容 | 量化底座 + LoRA 走 merge 部署路径报错/异常 | 改用非量化底座训练，或确认版本对该组合的支持状态 |
| F1 ZeRO-3 收集不完整 | 大模型（32B+）+ ZeRO-3，未配置分批收集 | 配置 `--move_model_batches`，对同步后权重做一致性校验 |
| F2 Megatron 导出偏差 | 同一权重 `swift infer`/`vllm serve` 正常，走 GRPO rollout 异常 | 核查 Megatron→HF 权重导出脚本/版本，导出后单独验证 |
| B1 模板不一致 | 渲染后 prompt 格式异常、思考标签位置错乱 | 统一训练/推理阶段 `enable_thinking`/`preserve_thinking`/`model_type` 配置 |
| B2 special token 错误 | 输出被无意义内容填充至上限 | 核对 eos/pad/bos token 与模型实际配置一致，关注自动对齐日志 |
| B3 base 模型误用 | 独立推理即退化，与训练框架无关 | 确认加载的是 instruct/chat 版本，`model_type` 与模型架构匹配 |
| C 采样参数不当 | 参数越极端乱码越明显，但并非唯一诱因 | 收敛 `temperature`/`top_p`/`top_k`/`repetition_penalty` 到推荐区间，作对照实验 |
| D 版本回归 bug | 近期升级过 vLLM/transformers，且社区有相同版本组合的复现报告 | 回退到已知稳定的版本组合，跟踪官方 issue 的修复进度 |
| E 多模态编码问题 | 纯文本样本正常，携带图文/音视频输入才异常 | 检查 ViT/aligner 权重同步、多模态预处理参数（分辨率/帧数等） |

---

## 7. 验证修复效果的标准

无论采用了哪一条修复动作，都应满足以下全部条件才能判定问题解决，而不是"看起来好一点就算了"：

1. 在与第 0 步相同的最小复现场景下，连续多个 step（建议 ≥10 step）的 `log_completions` 输出均为通顺、无重复退化的文本；
2. `reward`、`reward_std` 出现非零且有波动的数值；
3. `loss` 不再恒为 0（可正可负，但不应长期钉在 0）；
4. `grad_norm` 大于 0；
5. 训练前后同一 prompt 的生成内容确实随训练发生变化（验证权重同步持续生效，而非仅第一次生效）；
6. 将复现规模逐步恢复到完整训练配置（batch size、num_generations、数据量）后，上述现象依然稳定，没有在大规模下重新出现。

---

## 8. 预防性配置清单（面向 32B + LoRA + vLLM colocate/server 场景）

- [ ] 已做过"独立推理对比实验"作为训练前的例行检查，而非等到出问题才想起来做
- [ ] `lora_rank` 与 `vllm_max_lora_rank`（若使用 `vllm_enable_lora`）严格一致，且已确认当前 vLLM 版本对该 rank 的 LoRA kernel 支持完善
- [ ] 32B 级别模型在 ZeRO-3 下已配置 `--move_model_batches` 做分批权重收集
- [ ] 已记录当前 vLLM/transformers/ms-swift/deepspeed 的精确版本号，并核对过官方推荐版本矩阵与近期是否有相关 issue
- [ ] `enable_thinking`/`preserve_thinking`/`model_type` 等模板相关参数在训练与 rollout 阶段保持一致
- [ ] 已核对 eos/pad/bos 等 special token 的自动对齐日志，确认取值符合预期
- [ ] 采样参数（`temperature`/`top_p`/`top_k`/`repetition_penalty`）在官方推荐区间内
- [ ] 若为多模态模型，已确认 ViT/aligner 权重同步策略与 LoRA 同步方式（全量 vs 仅 adapter）匹配
- [ ] 若为 Megatron 训练路径，已单独验证过权重导出（Megatron→HF）后的模型可离线正常推理

---

## 9. 参考资料

- ms-swift 官方文档《GRPO / GetStarted》：LoRA 权重同步机制（`vllm_enable_lora`/`vllm_max_lora_rank`）、ZeRO-3 权重收集分批策略（`move_model_batches`）等参数说明
- ms-swift GitHub Issues（modelscope/ms-swift）中的相关复现案例，包括：vLLM 特定版本导致重复生成的回归问题、`vllm_enable_lora` 参数不一致触发 LoRA kernel 断言错误、Megatron+GRPO 训练乱码但离线推理正常、多轮/思考模式下训练与推理模板不一致导致的输出异常、GPTQ 量化模型与 LoRA 训练的兼容性讨论
- ms-swift Release Notes：`preserve_thinking`、`chat_template_kwargs`（`enable_thinking` 等）参数的引入与说明
- HuggingFace TRL / vLLM 官方文档：colocate 与 server 两种模式下权重同步机制的通用说明

> 说明：以上 GitHub issue 均为公开案例，可在 modelscope/ms-swift 仓库 Issues 页面按关键词（如"repetitive completion""乱码""vllm_enable_lora""move_model_batches""Megatron GRPO 乱码"）检索定位，结合自身的模型规模、并行模式、vLLM 版本逐条比对后再套用对应修复动作。

# 定位过程
按 Tokenizer → Sampling → KV Cache → Model Weight → XPU Kernel 这一层层排查，因为 rollout 只是调用 vLLM generate()。

## 第一层：确认是不是模型本身生成错误
### 方法1：不用RL，直接调用vLLM generate
```python
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/model/Qwen3-32B/",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.30,
        max_model_len=1024,
        dtype="bfloat16",
    )

    prompts = ["请解答：小明有5个苹果，吃了2个，还剩几个？"]
    sampling_params = SamplingParams(
        temperature=0.9,
        max_tokens=200,
    )

    outputs = llm.generate(prompts, sampling_params)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
```

### 方法2：HF generate
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/model/Qwen3-32B/"


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    prompt = "请解答：小明有5个苹果，吃了2个，还剩几个？"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.9,
            max_new_tokens=50,
            return_dict_in_generate=True,
        )

    sequences = outputs.sequences

    new_tokens = sequences[:, input_len:]

    print("======== Generated Token IDs ========")
    print(new_tokens[0].tolist())

    print()

    print("======== Per Token ========")
    for tid in new_tokens[0]:
        token = tokenizer.decode([tid])
        print(f"{tid.item():8d}  {repr(token)}")

    print()

    print("======== Final Text ========")
    print(tokenizer.decode(sequences[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()  
```

### 分析
结果分三种：
HF和vLLM都正常，进入下一步。

#### HF正常+vLLM乱码
说明问题100%在vLLM或者XPU backend。

例如
- logits错误
- sampler错误
- KV cache错误
- attention错误

#### HF也乱码
说明模型就坏了。
例如
- checkpoint损坏
- tokenizer不一致
- weight转换错误

## 第二层：确认token是不是已经错了
查看completion的原始token ids，而不是输出文本。
- 如果token ids正常，decode出来乱码。说明Tokenizer有问题。
- 如果token ids也是重复token，进入下一步

## 第三层：检查logits
进入sampling代码，打印logits信息，比如
```python
topk = torch.topk(logits, 10)

print(topk.indices)
print(topk.values)
```
如果一直是同一个tokenlogit最高，说明模型输出已经坏了。问题不是Sampler。

## 第四层：关闭Sampling
很多时候不是模型坏，而是Sampling坏。
例如改成Greedy，
原来
```python
temperature=1

top_p=0.95
```
改成
```python
temperature=0

top_p=1

top_k=-1
```

如果Greedy正常，Sampling乱码，说明Sampler实现有问题。

## 第五层：检查Prefill和Decode
重复token最容易发生在Decode阶段，有可能KV Cache更新失败，建议打印
```
        Prefill输出
        ↓
        第一个Decode
        ↓
        第二个Decode
        ↓
        ...
```

## 第六层：检查KV Cache
如果怀疑KV Cache，可以直接：
- 关闭KV Cache。

例如vLLM支持
```
enforce_eager=True
或者
--enforce-eager
```
或者关闭CudaGraph。
如果关闭以后恢复正常，说明：KV Cache或者PagedAttention有问题。

可以进一步：
- 调小 max_num_batched_tokens
- 调小 block_size
如果恢复正常，就是KV Cache。

## 第七层：比较Eager和Graph
直接做四组实验：

| eager | graph | 结果 |
| ----- | ----- | -- |
| on    | off   |    |
| off   | off   |    |
| off   | on    |    |
| on    | on    |    |

如果只有Graph错，就是Graph捕获问题。

## 第八层：打印每一步token概率

例如

```
Step1
    你好
    ↓
    token
    123
    prob=0.92
Step2
    token
    123
    prob=0.99999
Step3
    123
    0.999999
```

这种就是典型

```
logits爆炸
```

一般来自：

* RMSNorm
* RoPE
* Attention
* KV Cache

## 第九层：检查Attention
重复token还有一种原因：
- Attention一直只能看到最后一个token。

例如
- Attention Mask错了。
- RoPE offset错了。

都会导致：
```
    hello
    ↓
    hello hello hello hello
```
因为模型认为历史不存在。

## 建议的排查优先级（按投入产出排序）

1. **脱离 RL，直接运行 vLLM `generate()`**，确认问题是否可复现。
2. **同一模型运行 Hugging Face `generate()`**，确定是模型还是 vLLM/XPU 后端。
3. **打印每一步生成的 `token_ids`、Top-K logits 和对应概率**，判断是 decode 问题还是 logits 已异常。
4. **切换为 Greedy（`temperature=0`）**，排除采样器实现问题。
5. **关闭 CUDA Graph、使用 Eager 模式**，观察是否恢复正常。
6. **重点验证 Decode 阶段的 KV Cache 更新**（这是重复 token 最常见的根因），包括比较 Prefill 后第一步 Decode 与后续 Decode 的行为。
7. **对比 Attention/RoPE/KV Cache 相关实现**，尤其是在国产 XPU 自定义 Kernel 或 Flash Attention 实现上。