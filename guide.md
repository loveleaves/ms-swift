# ms-swift 新人学习与贡献指导手册

**调研日期**：2026-06-13
**调研对象**：[modelscope/ms-swift](https://github.com/modelscope/ms-swift)（基于 main 分支 v4.4.0.dev0，最新提交 2026-06-12，并结合官方文档与社区资料交叉验证）
**调研意图**：为新人提供一条「从零基础使用 → 理解源码 → 参与特性开发」的完整学习路径
**更新记录**：2026-06-13 二次调研，新增第七~十章（架构总览深入版、核心概念与依赖库、数据流与调度策略、实现指南）
**适用读者**：有 Python 基础、了解深度学习基本概念、希望进入大模型训练工程领域并参与开源贡献的开发者

---

## 一、执行摘要

1. **ms-swift 是魔搭（ModelScope）社区官方的大模型训练全链路框架**，全称 Scalable lightWeight Infrastructure for Fine-Tuning，由阿里通义/魔搭团队主导，覆盖预训练、微调、人类对齐（RLHF）、推理、评测、量化、部署全流程，支持 600+ 纯文本大模型与 400+ 多模态大模型，论文被 AAAI 2025 接收（arXiv: 2408.05517）。
2. **项目处于高速迭代期**：约 1.4 万 stars、1300+ forks、125+ releases、发布频率约每周一次；2026 年 3 月发布的 v4.0 完成了大规模架构重构，模块全部平铺在 `swift/` 一级目录下，**对新贡献者显著更友好**——几乎每个模块都有标准化的「注册（mapping）机制」作为扩展点。
3. **新人最佳贡献切入点高度明确**：官方在 CONTRIBUTING 中点名欢迎「新模型/新数据集支持、新训练技术集成、教程文档」三类贡献，且为活跃贡献者提供**免费 A10 GPU 算力**、电子证书和周边激励，这在开源训练框架中相当少见。
4. **学习曲线呈阶梯状**：CLI 使用门槛极低（一条命令即可微调），但源码涉及 transformers、PEFT、vLLM、Megatron、强化学习算法等多领域知识。本手册据此设计了 **0→4 五个阶段、约 2~3 个月**的渐进路线，每个阶段都有可验收的产出物。
5. **核心结论**：对想进入大模型训练工程领域的新人，ms-swift 是当前中文社区里"使用文档最全 + 贡献通道最畅通"的训练框架之一；建议从 LoRA 微调实战入手，以 `template`（对话模板）与 `model`（模型注册）两个模块作为源码突破口，以"注册一个新模型/新数据集"作为第一个 PR 目标。

---

## 二、项目概述

### 2.1 项目定位与背景

ms-swift 诞生于 2023 年 8 月，最初定位是 ModelScope 生态的轻量微调库，三年间逐步演化为**训练全链路基础设施**。它在生态中的位置可以这样理解：

- 对上：与魔搭/HuggingFace 模型与数据集 Hub 无缝衔接（默认走 ModelScope 下载，`--use_hf true` 切换 HuggingFace），训练完可一键 push 回 Hub；
- 对内：训练后端同时支持 transformers 路线（PEFT/全参）与 **Megatron 路线**（TP/PP/CP/EP 并行，面向 MoE 与超大模型）；
- 对下：推理/部署集成 vLLM、SGLang、LMDeploy 三大引擎，量化支持 GPTQ/AWQ/BNB/FP8，评测对接 EvalScope。

一句话定位：**「数据准备好之后，从训练到上线的所有事情都让一个 `swift` 命令解决」**。

### 2.2 关键数据（截至 2026-06）

| 维度 | 数据 | 说明 |
|---|---|---|
| Stars / Forks | ≈13.8K / ≈1.4K | 同类中文框架中第一梯队（与 LLaMA-Factory 同档） |
| 开放 Issue | ≈1000+ | 活跃度高，也意味着可参与空间大 |
| Release | 125+，约每周一发 | v4.0 大版本：2026-03-03；main 分支为 4.4.0.dev0 |
| 模型支持 | 600+ LLM、400+ MLLM | Qwen3/3.5、DeepSeek-R1/V4、GLM、InternLM/InternVL、Llama4 等，热门模型 Day-0 支持 |
| 内置数据集 | 150+ | 覆盖预训练/微调/对齐/多模态 |
| 训练算法 | 全参/LoRA/QLoRA/DoRA/ReFT/GaLore/LISA 等；RLHF 含 DPO/KTO/RM/PPO/CPO/SimPO/ORPO；GRPO 算法族含 GRPO/DAPO/GSPO/SAPO/CISPO/RLOO/Reinforce++ | 强化学习方向更新最快 |
| 许可证 | Apache-2.0 | 商用友好 |
| 学术背书 | AAAI 2025 | 论文 arXiv: 2408.05517 |

### 2.3 与同类框架的差异（帮助你理解"学它值不值"）

- 相比 **LLaMA-Factory**：两者使用层体验接近，但 ms-swift 的多模态模型覆盖、Megatron 并行整合（Mcore-Bridge 让 Megatron 训练像 transformers 一样易用）和 RL 算法族更全；
- 相比 **TRL/PEFT 裸用**：ms-swift 把数据模板、模型注册、分布式、推理部署整条链路工程化了，省去大量胶水代码；
- 相比 **Megatron-LM 原生**：Megatron-SWIFT 子系统大幅降低了 MoE/超大模型并行训练门槛。

对学习者的含义：**学透 ms-swift ≈ 同时建立 transformers 生态、PEFT、RLHF、Megatron 并行、推理引擎五个领域的工程化认知**，这是它作为学习载体的最大价值。

---

## 三、核心架构与代码地图（v4.x）

> v4.0 重构后不再有旧版 `swift/llm` 的嵌套结构，所有模块平铺于 `swift/` 一级目录。下表基于仓库实际结构与官方 `docs/source/Customization/Architecture.md` 整理，**标 ⭐ 的是官方明确开放、带注册机制的扩展点，也是贡献 PR 的主要落点**。

### 3.1 源码目录地图（`swift/`）

| 目录 | 职责 | 新人关注度 |
|---|---|---|
| `cli/` | 命令行入口。`swift sft ...` 等价于 `python swift/cli/main.py sft`，再路由到 `sft.py` | ★★★ 源码阅读的起点 |
| `arguments/` | 全部命令行参数定义（`SftArguments`、`RLHFArguments` 等 dataclass） | ★★★ 查任何参数含义先来这里 |
| `pipelines/` | `sft_main / rlhf_main / infer_main` 等主流程编排 | ★★★ 理解训练全流程的主线 |
| `template/` ⭐ | 对话模板：把 messages 转成 input_ids/labels 的核心逻辑 + data_collator；`register_template` 注册 | ★★★ **理解整个框架的钥匙** |
| `model/` ⭐ | 模型加载与注册：`register_model(ModelMeta)`、`model_arch.py`、各模型实现在 `models/` | ★★★ 新模型支持类 PR 的主战场 |
| `dataset/` ⭐ | 数据集注册、预处理、packing、流式数据 | ★★★ 新数据集 PR 落点 |
| `trainers/` | 预训练/SFT/Embedding/Reranker/序列分类 Trainer（基于 transformers Trainer 扩展） | ★★ |
| `rlhf_trainers/` | GRPO/GKD/DPO/KTO/RM 等对齐算法 Trainer | ★★（RL 方向必读） |
| `rollout/` | RL 中 rollout 采样实现 | ★★ |
| `rewards/` ⭐ | 奖励函数：ORM（结果奖励）/PRM（过程奖励）/RM 插件，支持自定义 | ★★★ GRPO 玩法的核心扩展点 |
| `tuners/` | LoRA、LLaMA-Pro、LongLoRA、ReFT、SCEdit 等轻量训练方法实现 | ★★ |
| `tuner_plugin/` ⭐ | 自定义 tuner 插件注册 | ★★ |
| `loss/` ⭐、`loss_scale/` ⭐ | 自定义损失函数与 token 级 loss 权重（Agent 训练常用），mapping 注册 | ★★ |
| `callbacks/` ⭐ | 训练回调（接口与 transformers `TrainerCallback` 一致），mapping 注册 | ★★ 最简单的扩展点之一 |
| `metrics/` ⭐、`optimizers/` ⭐ | 自定义评估指标与优化器，mapping 注册 | ★★ |
| `agent_template/` ⭐ | Agent 工具调用数据格式模板（继承 `BaseAgentTemplate` 实现 4 个方法） | ★★ |
| `infer_engine/` | transformers/vLLM/SGLang/LMDeploy 四种推理后端的统一封装 | ★★ |
| `megatron/` | Megatron-SWIFT 子系统（含 Mcore-Bridge） | ★（进阶后再看） |
| `sequence_parallel/` | Ulysses + ring-attention 序列并行 | ★ |
| `dataloader/` | shard/dispatcher 两种数据加载方式 | ★ |
| `ray/`、`ray_utils/` | Ray 分布式支持 | ★ |
| `ui/` | `swift web-ui` 的 Gradio 界面 | ★ |
| `hub/`、`config/`、`utils/` | Hub 上传下载、deepspeed/fsdp2 配置、通用工具 | ★ |

### 3.2 仓库顶层结构

- `examples/`：**最重要的学习资产**。`train/` 下有 30+ 子目录（lora_sft、qlora、grpo、rlhf、moe、multi-gpu、multi-node、sequence_parallel、packing、agent、embedding、reranker……），每个都是可直接运行的 shell 脚本；`custom/` 给出了自定义模型/数据集注册的完整范例（`model.py`、`dataset.py` + `--external_plugins` 用法）；`notebook/` 有自我认知微调、VL grounding、OCR 三个端到端 Jupyter 教程。
- `docs/source/`：官方文档源文件，四大板块——GetStarted（快速开始/安装/Web-UI）、Instruction（命令行参数、预训练与微调、RLHF、GRPO、推理部署、评测、Agent 等）、BestPractices（Qwen3/Qwen3-VL/DeepSeek-V4 最佳实践、GRPO 代码训练、多模态注册、NPU/AMD 支持等）、Customization（**架构介绍、自定义模型、自定义数据集——贡献者必读三篇**）。在线版即 swift.readthedocs.io。
- `tests/`：按模块组织（train/tuners/infer/megatron/test_align 等）。`test_align/test_template/` 用于校验模板编码对齐，是模板类 PR 必须附带的测试位置。
- `CONTRIBUTING_CN.md`、`Makefile`（`make linter` / `make test` / `make docs`）、`requirements/`（按 megatron/eval/swanlab/ray 等场景分组的依赖）。

### 3.3 一次 `swift sft` 的生命周期（建立全局心智模型）

```
swift sft --model ... --dataset ... --tuner_type lora
  └─ cli/main.py 解析子命令 → cli/sft.py
      └─ pipelines/train: sft_main()
          ├─ arguments/: 解析+校验 SftArguments
          ├─ model/: 据 model_id 后缀+config.json 自动匹配 model_type → ModelLoader 加载模型与 tokenizer
          ├─ template/: 据 model_type 取默认 template → encode(messages) → input_ids/labels
          ├─ dataset/: 数据集注册表加载 → 预处理/packing/streaming
          ├─ tuners/: 按 tuner_type 包装模型（LoRA 等）
          └─ trainers/: 构建 Trainer（loss/loss_scale/callbacks/metrics/optimizers 均可经 mapping 注入）→ train() → 保存 checkpoint（含 args.json）
```

后续 `swift infer --adapters <ckpt>` 会自动读取 `args.json` 还原配置，`swift export` 负责 merge-lora/量化/推 Hub。**这条主线贯穿了 80% 的源码，建议在阶段 3 用断点完整走一遍。**

---

## 四、分阶段学习路线（0 → 贡献者，约 2~3 个月）

> 设计原则：每个阶段有明确的**验收产出**；先当用户、再读源码、最后写 PR；所有实验优先用 0.5B~4B 小模型（单张消费级显卡即可，魔搭社区也提供免费 GPU Notebook 额度）。

### 阶段 0：前置知识自检（0~1 周，按需补齐）

| 必备 | 程度 |
|---|---|
| Python（dataclass、装饰器、继承） | 熟练——注册机制大量使用这三者 |
| PyTorch 基础（Module/Dataset/训练循环） | 会写最小训练循环 |
| transformers 库（AutoModel/AutoTokenizer/Trainer 概念） | 用过即可 |
| LLM 基本概念（token、chat template、SFT/LoRA 是什么） | 概念清楚 |
| Git/GitHub（fork、branch、rebase、PR） | 基本操作 |

加分项（可边学边补）：PEFT 原理、DPO/GRPO 论文、vLLM 使用经验、DeepSpeed/Megatron 并行概念。

### 阶段 1：用户视角入门（第 1~2 周）

目标：**不看源码，把框架当产品用熟**。

1. 安装：`pip install ms-swift -U`（体验用）。读官方文档 GetStarted 三篇。
2. 跑通第一个微调（官方 Quick Start，10 分钟级）：

   ```bash
   CUDA_VISIBLE_DEVICES=0 swift sft \
     --model Qwen/Qwen3-4B-Instruct-2507 \
     --dataset AI-ModelScope/alpaca-gpt4-data-zh \
     --tuner_type lora \
     --output_dir output
   ```

3. 完成闭环：`swift infer --adapters output/vx-xxx/checkpoint-xxx` 推理 → `swift export --push_to_hub true` 推送模型。理解 `args.json` 自动还原参数的机制。
4. 跑 `examples/notebook/qwen2_5-self-cognition` 自我认知微调 notebook——这是理解"数据如何改变模型行为"的最直观实验。
5. 体验 `swift web-ui`，对照界面反推 CLI 参数。
6. 通读《命令行参数》文档一遍（不求记住，求知道"有什么、去哪查"）。

**验收**：能独立用自己的 jsonl 数据（messages 格式）微调一个小模型并完成推理部署；说得清 sft/infer/export/eval/deploy 五个子命令各做什么。

### 阶段 2：进阶使用（第 3~5 周）

目标：**覆盖框架的主要能力面，为读源码积累"使用直觉"**。

1. **自定义数据集**：精读 `docs/source/Customization/Custom-dataset.md`，分别用"本地 jsonl 直接喂"和"注册数据集"两种方式各做一次。
2. **多卡训练**：跑 `examples/train/multi-gpu`（DDP + deepspeed zero2/zero3），观察显存与吞吐差异。
3. **RLHF 初体验**：先 DPO（`examples/train/rlhf`），再 GRPO（`examples/train/grpo`，配合 `--use_vllm` rollout 加速）。读 BestPractices 里的 GRPO 完整文档。
4. **多模态**：跑一个 Qwen3-VL LoRA 微调（`examples/train/multimodal`），理解多模态数据格式（images 字段）。
5. **量化与部署**：GPTQ/AWQ 量化导出 + `swift deploy` 起 OpenAI 兼容服务。
6. 横向浏览 `examples/train/` 全部子目录的脚本——每个脚本就是一个特性的"官方用法答案"。

**验收**：能针对一个真实小项目（如客服问答微调）独立完成「数据构造 → LoRA/全参选型 → 训练 → 评测（EvalScope）→ 量化部署」全流程，并写出一篇实验记录。

### 阶段 3：源码阅读（第 6~9 周）

目标：**建立"参数 → 代码路径"的映射能力**。推荐开发模式安装：

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift && pip install -e .
```

推荐阅读顺序（由主线到分支）：

1. **主线**：`cli/main.py → cli/sft.py → pipelines/train`，对照 3.3 节的生命周期图，用 debugger 或 `print` 走一遍最小 LoRA 训练。
2. **template/（重点突破口）**：这是全框架信息密度最高的模块。用官方提供的 debug 片段直接观察编码结果：

   ```python
   from swift import get_processor, get_template
   template = get_template(get_processor('Qwen/Qwen3-8B'))
   template.set_mode('train')
   encoded = template.encode({"messages": [...]})
   print(template.safe_decode(encoded['input_ids']))
   print(template.safe_decode(encoded['labels']))   # 观察哪些 token 参与 loss
   ```

   搞懂 `TemplateMeta` 的 prefix/prompt/chat_sep/suffix/system_prefix 五要素，以及多模态模板要重写的 `_encode/_post_encode/_data_collator`。
3. **model/**：读 `register.py` 与 `models/qwen.py`，理解 `ModelMeta`（model_type、model_groups、architectures 自动匹配、model_arch 对多模态 llm/vit/aligner 前缀的意义）。
4. **dataset/**：注册表（`dataset/dataset`、`dataset/data`）+ 预处理器 + packing 实现。
5. **arguments/ + trainers/**：看参数如何一路传到 Trainer；对照 transformers Trainer 看 swift 改了哪些点（loss 注入、loss_scale、callbacks mapping）。
6. **按兴趣选一条深入线**：RL 线（rlhf_trainers/grpo + rollout + rewards）、多模态线（template 多模态子类 + model_arch）、性能线（sequence_parallel + packing + megatron）。
7. 配合读 AAAI 2025 论文（arXiv: 2408.05517）补设计动机。

**验收**：① 任选一个命令行参数，能从 arguments 一路追到生效的代码行；② 能徒手画出 sft 数据流图；③ 用 `--external_plugins` 在**不改框架源码**的情况下注册一个自定义模型/数据集/loss 并跑通（`examples/custom/` 是模板）。

### 阶段 4：参与贡献（第 10 周起，长期）

详见第五章。第一个 PR 建议从"新模型注册 / 新数据集 / 文档修复 / 测试补充"四类中选，难度低、官方明确欢迎、review 周期短。

---

## 五、贡献流程与切入点

### 5.1 官方欢迎的贡献类型（来自 CONTRIBUTING_CN.md）

1. **新技术和新模型**：支持更多开源模型、数据集，或集成官方尚未关注的新训练技术——这是代码类贡献的主航道；
2. **技术布道**：撰写教程文档/视频并提交链接；
3. **社区供稿**：技术文章经审核后发布在魔搭官方账号（知乎/公众号）并署名。

**激励机制**：电子贡献证书、魔搭周边、以及**开发期间免费 A10 算力**（邮件 contact@modelscope.cn 或加官方微信群申请）。没有本地 GPU 不是参与障碍。

### 5.2 标准 PR 流程

1. **Fork** modelscope/ms-swift → clone 到本地 → 拉新分支开发（开发中常点 Sync Fork 同步 main 防冲突）；
2. 本地开发 + 自测；
3. **提交前必做 Code Lint**：

   ```bash
   pip install pre-commit
   pre-commit run --all-files   # 修到全绿
   ```

4. 推送分支 → 在 GitHub 发起 PR（目标 `modelscope/ms-swift:main`），写清 feature 描述；
5. CI 会跑 Lint + 冒烟/单元测试；**新功能需自带测试用例保护**（如模板类改动要在 `tests/test_align/test_template/` 加对齐测试），Reviewer 会检查这一点；
6. Review 讨论 → 合入。注意官方提醒：review 意见针对代码而非个人。

代码规范要点：变量蛇形命名、类名大驼峰、4 空格缩进、优先选用知名开源库、避免重复造轮子。

### 5.3 按难度排序的切入点路线图

| 难度 | 贡献类型 | 涉及模块 | 备注 |
|---|---|---|---|
| ⭐ | 文档修复/翻译、FAQ 补充、示例脚本修正 | docs/、examples/ | 熟悉 PR 流程的最佳起手 |
| ⭐ | 给已有功能补测试用例 | tests/ | Reviewer 好感度高 |
| ⭐⭐ | **注册新模型**（新发布的开源模型 Day-N 支持） | model/models/ + template/templates/ + 文档表格 | 套路固定：写 ModelMeta + TemplateMeta + 跑 `scripts/utils/run_model_info.py` 更新支持列表 + 对齐测试；多模态参考 BestPractices/MLLM-Registration.md |
| ⭐⭐ | 注册新数据集 | dataset/ | 同样是模板化工作 |
| ⭐⭐ | 自定义 callback/metric/optimizer/loss 沉淀为内置 | 对应 mapping 模块 | 从自己的 external_plugins 升级而来最自然 |
| ⭐⭐⭐ | 新 RL 算法/奖励函数/loss_scale 策略 | rlhf_trainers/、rewards/、loss_scale/ | RL 算法族是当前迭代最快、需求最旺的方向 |
| ⭐⭐⭐ | Bug 修复（从 1000+ open issues 里认领可复现问题） | 任意 | 修 bug 是理解深层机制的捷径 |
| ⭐⭐⭐⭐ | 推理引擎适配、Megatron/并行特性、新硬件（NPU/AMD/沐曦）支持 | infer_engine/、megatron/ 等 | 需要资源与经验，后期再碰 |

### 5.4 跟踪社区的姿势

- Watch 仓库 releases（约每周一发，README News 区记录每个重要特性的落地时间）；
- 浏览 open issues 中的 feature request 与他人 PR 的 review 过程——**读 PR 是学习"什么样的代码能被合入"成本最低的方式**；
- 加入官方微信/Discord 群（README 中有入口），新模型支持需求常在群里先出现。

---

## 六、综合分析：学习建议与避坑

**优势（为什么值得投入）**：扩展点全部标准化为 mapping 注册机制，贡献模式可复制；examples 即文档，几乎每个特性都有可运行脚本；中文文档一手且完整；官方提供算力支持，贡献回报路径清晰（证书/署名/简历价值）。

**挑战与对策**：

1. **迭代太快**：约每周一个 release，博客教程极易过期（网上大量教程仍基于 3.x 甚至 2.x 的 `swift/llm` 旧结构）。对策：**一切以仓库内 docs/ 与 examples/ 为准**，社区文章只当思路参考；学习期锁定一个 release tag。
2. **参数海量**：Command-line-parameters 文档极长。对策：不背参数，记住「examples 找用法、arguments/ 源码找定义」两条检索路径。
3. **环境与依赖**：torch/transformers/vllm/megatron 版本矩阵复杂。对策：用官方推荐镜像或 `requirements/install_all.sh`；Megatron 相关实验放到最后。
4. **多模态模板是难点**：`_encode/_post_encode/_data_collator` 三函数 + model_arch 前缀机制需要耐心调试。对策：先吃透纯文本模板，再用官方 debug 片段逐 token 观察多模态编码。
5. **新人易犯的 PR 错误**：没跑 pre-commit、没带测试、PR 描述空泛、一个 PR 塞多个改动。对策：严格遵循 5.2 清单，首个 PR 保持最小改动面。

---

## 七、架构总览（深入版）

> 本章及后续三章基于对 main 分支（2026-06-12，v4.4.0.dev0）源码的逐文件分析，是第三章"代码地图"的纵深展开。

### 7.1 五层分层架构

```
┌─────────────────── 入口层 ───────────────────┐
│  swift CLI (cli/main.py 子命令路由)  │  swift web-ui  │  Python API (sft_main 等)  │
├─────────────────── 参数层 ───────────────────┤
│  arguments/：BaseArguments → SftArguments / RLHFArguments / InferArguments ...     │
│  （dataclass 继承树；args.json 序列化，保证 train→infer→export 配置自动续传）        │
├─────────────────── 管线层 ───────────────────┤
│  pipelines/：SwiftPipeline 抽象基类 → SwiftSft / SwiftRLHF / SwiftInfer ...        │
├─────────────────── 能力层 ───────────────────┤
│  model(加载/注册)  template(编码)  dataset(数据)  tuners(PEFT)                      │
│  trainers / rlhf_trainers(训练)  rewards / rollout(RL)  infer_engine(推理)         │
│  + 八大 mapping 扩展点：loss/loss_scale/callbacks/metrics/optimizers/...           │
├─────────────────── 后端层 ───────────────────┤
│  transformers + PEFT + TRL + DeepSpeed/FSDP2   │   Megatron-Core (Mcore-Bridge)   │
│  vLLM / SGLang / LMDeploy / transformers（推理四后端）                              │
└──────────────────────────────────────────────┘
```

两条关键设计主线值得新人特别留意：

**主线一：模板方法模式的管线基类。** `pipelines/base.py` 中的 `SwiftPipeline` 只做四件事——解析参数（`args_class` 类属性 + `_parse_args`，对无法识别的参数默认直接报错而非静默忽略）、设种子（`seed + rank`，保证多卡可复现且各 rank 不同源）、记录起止时间、调用抽象方法 `run()`。所有子命令（sft/rlhf/infer/eval/export/app...）都是它的子类，只需声明自己的 `args_class` 并实现 `run()`。读懂这一个 60 行的基类，所有 `swift xxx` 命令的骨架就都通了。

**主线二："元信息 dataclass + 注册函数 + mapping 字典"三件套。** 框架的开放性全部建立在同一个模式上：

```
ModelMeta    + register_model()    → MODEL_MAPPING
TemplateMeta + register_template() → TEMPLATE_MAPPING
DatasetMeta  + register_dataset()  → DATASET_MAPPING
（loss / loss_scale / callbacks / metrics / optimizers / tuner_plugin /
  agent_template / rewards 同理，各模块均有 mapping.py 或 register.py）
```

运行时通过命令行参数（`--model_type`、`--loss_type`、`--callbacks` 等）查表取实现；用户侧则可用 `--external_plugins my_file.py` 把自己的注册代码注入进程，**完全不改框架源码就能扩展**。理解了这个模式，你就理解了"在 ms-swift 中开发一个特性"的标准形态——大多数 PR 本质上就是"写一个 Meta + 一个实现类 + 一行注册"。

### 7.2 `SwiftSft` 的训练编排（管线层实例）

`pipelines/train/sft.py` 中 `SwiftSft(SwiftPipeline, TunerMixin)` 的方法序列就是一张训练流程图：`_prepare_model_tokenizer`（查 MODEL_MAPPING 加载模型）→ `_prepare_template`（查 TEMPLATE_MAPPING 并 `set_mode('train')`）→ `_get_dataset / _prepare_dataset / _post_process_datasets`（数据三段式）→ `_encode_dataset`（编码或包惰性数据集）→ TunerMixin 注入 LoRA 等 → 构建 Trainer → `train()` → `_save_trainer_state`。RLHF 管线（`rlhf.py`）继承复用了其中绝大部分步骤，只替换 Trainer 与数据格式——这就是为什么读懂 sft 主线后看 GRPO 代码会非常快。

### 7.3 双训练后端与四推理后端

- **训练后端**：默认走 transformers Trainer 体系（PEFT/全参 + DeepSpeed/FSDP2）；超大模型与 MoE 走 `megatron/` 子系统，依赖 `mcore-bridge>=1.4` + `megatron-core>=0.15`，Mcore-Bridge 负责 HF 权重 ↔ Megatron 权重的双向桥接，使 `megatron sft` 的使用体验接近 `swift sft`。
- **推理后端**：`infer_engine/` 把 transformers / vLLM / SGLang / LMDeploy 统一封装为相同的 `infer(InferRequest, RequestConfig)` 接口（OpenAI 风格 protocol），训练评测、`swift deploy`、GRPO rollout（专用 `grpo_vllm_engine.py`）共用这一层。**Template 是连接训练与推理的枢纽**：同一个模板对象通过 `set_mode()` 切换 7 种模式（train/rlhf/kto/transformers/vllm/lmdeploy/sglang），保证"训练时怎么编码，推理时就怎么编码"，从根上消除 train-infer 模板不一致这一大类 bug。

---

## 八、核心概念、关键技术与依赖库

### 8.1 必须吃透的核心概念

| 概念 | 一句话定义 | 所在模块 |
|---|---|---|
| `model_type` | 模型结构的唯一 ID；按 model_id 后缀 + config.json 的 architectures 自动匹配 | model/ |
| `ModelMeta` / `ModelGroup` | 模型元信息：魔搭/HF 双 id、默认 template、model_arch、是否多模态 | model/ |
| `Template` / `TemplateMeta` | messages → input_ids/labels 的全部逻辑；五要素 prefix/prompt/chat_sep/suffix/system_prefix | template/ |
| `model_arch` | 多模态模型中 llm/vit/aligner 各部分的参数前缀映射，决定 `--freeze_vit` 等开关冻结谁 | model/model_arch.py |
| `agent_template` | 统一 Agent 数据格式到各家工具调用格式的适配层（4 个抽象方法） | agent_template/ |
| `tuner` | 训练参数化策略：lora/qlora/dora/reft/全参…，经 TunerMixin 注入 | tuners/ |
| `loss_scale` | token 级 loss 权重：支持精确匹配/正则 json 配置，Agent 训练核心技巧 | loss_scale/ |
| ORM / PRM / RMPlugin | 三类奖励：结果奖励函数（含 Async 版）、过程奖励、奖励模型插件（含生成式 GenRM） | rewards/ |
| rollout | RL 中由推理引擎生成候选回复的采样过程；支持 multi-turn、gym_env、agent_loop | rollout/ |
| packing / padding_free | 多条短样本装箱进同一序列 / 不 padding 的变长 batch，两种吞吐优化 | dataset/packing.py |
| `lazy_tokenize` | 训练时才编码样本，省内存且自动跳过坏样本 | dataset/utils.py |
| `args.json` | checkpoint 内的参数快照，infer/export 自动读取续传配置 | arguments/ |

### 8.2 三项有代表性的关键技术实现

**① 容错式惰性编码（LazyLLMDataset）**：`__getitem__` 时才调用 `template.encode`，失败则随机重试至多 `n_try_fetch=10` 条其他样本（strict 模式则直接抛错）。工程意义：百万级真实数据里总有超长/脏样本，框架默认"跳过并告警"而非训练中途崩溃。

**② 装箱式 packing**：依赖第三方库 `binpacking` 做近似最优装箱（`calculate_matched_group` 把样本按长度装入 `packing_length` 容量的箱子），提供 map 式 `PackingDataset`（预先建好 packed 索引）与流式 `IterablePackingDataset`（多进程 + 队列边读边装）两种实现，配合 flash-attention 的变长注意力保证样本间互不可见。

**③ 序列并行**：`sequence_parallel/` 实现 Ulysses（按 head 切）+ zigzag ring-attention（按序列环形切，含 NPU 版），二者可组合，使 `--sequence_parallel_size N` 不再受注意力头数整除限制——这是 2025.09 后长文本训练的主力方案。

### 8.3 依赖库分层解析（来自 requirements/ 与 setup.py 实测）

| 层 | 依赖 | 在框架中的角色 |
|---|---|---|
| 训练核心 | `transformers>=4.33,<5.11`、`peft>=0.11,<0.20`、`trl>=0.15,<1.0`、`accelerate`、`datasets>=3.0,<4.8.5` | Trainer/PEFT/RLHF 算法基座；**注意全部钉了版本上界**，跟进新版本依赖适配本身就是常见 PR 类型 |
| Hub 生态 | `modelscope>=1.23`、`oss2`、`safetensors`、`sentencepiece`、`tiktoken` | 模型/数据下载上传与分词 |
| 服务与 UI | `fastapi`+`uvicorn`+`openai`（deploy 的 OpenAI 兼容协议）、`gradio>=3.40,<6.0`（web-ui） | 部署与零代码界面 |
| 数据与工具 | `binpacking`（packing 装箱）、`json_repair`（容错解析模型输出的工具调用 JSON）、`dacite`/`addict`（dataclass 构造）、`tensorboard`、`rouge`/`nltk`（指标） | 细节但关键的轮子 |
| 可选 extras | `megatron`（mcore-bridge、megatron-core）、`eval`（evalscope 及 opencompass/vlmeval 后端）、`ray`、`swanlab` | `pip install ms-swift[all]` 全装 |
| 推荐性能栈 | `requirements/install_all.sh`：Python 3.10/3.11 + CUDA 12.x、vLLM、`flash-attn==2.8.3`、`deepspeed<0.19`、`liger_kernel`、qwen_vl_utils 等多模态工具；Megatron 线另需 TransformerEngine、DeepGEMM、flash-linear-attention（均源码编译） | 环境踩坑高发区，**新人务必照抄此脚本而非自由组合版本** |

一个实用推论：当某个依赖（如 transformers 大版本）发布后 ms-swift 出现兼容问题，相关 issue 会迅速出现——盯住这些"版本适配"issue 是低门槛、高确定性的贡献机会。

---

## 九、数据流与调度策略

### 9.1 训练数据流全链路（七步）

```
--dataset 'ms_id#2000' 'path.jsonl' ...
  ①dataset_syntax.py  解析数据集表达式（采样数、子集、混合比例）
  ②loader.py          ModelScope/HF/本地 三源下载与加载（--use_hf 切换）
  ③preprocessor/      各数据集的 Preprocessor 把原始字段标准化为 messages 格式
  ④拆分与混洗          train/val 切分（--split_dataset_ratio）、多数据集 concat
  ⑤template.encode    messages → input_ids/labels/loss_scale（三种时机，见 9.2）
  ⑥packing/padding    --packing 装箱 或 --padding_free 变长 batch（可选）
  ⑦data_collator      template._data_collator 组 batch（多模态另走 _data_collator_mm_data）
         ↓
  dataloader/ 分发到各 rank（见 9.3）→ Trainer 训练循环
```

记忆要点：**③前是"原始格式"，③后是统一的 messages 标准格式，⑤后是张量**。自定义数据集本质上只是提供一个第③步的 Preprocessor。

### 9.2 编码时机的三种调度策略

| 策略 | 参数 | 机制 | 适用 |
|---|---|---|---|
| 预编码 | 默认（非多模态） | 训练前 map 全量编码，可配 `swift export --to_cached_dataset` 预编码落盘复用 | 中小数据集，追求训练期零编码开销 |
| 惰性编码 | `--lazy_tokenize true`（多模态默认） | LazyLLMDataset 训练时逐条编码 + 坏样本自动跳过 | 多模态/大数据集（图像解码内存不可预编码） |
| 流式 | `--streaming true`（须设 `--max_steps`） | IterableDataset 边下边训，不落盘 | 超大规模/TB 级预训练数据 |

### 9.3 分布式数据分发的两种调度器（dataloader/）

- **DataLoaderShard（默认）**：经典分片——每个 rank 用 `BatchSamplerShard` 各自读取属于自己的那份索引，无通信开销；要求数据集可随机访问。
- **DataLoaderDispatcher**：rank0 统一读取，再用 `dist.scatter_object_list` 把 batch 分发给各 rank。代价是 rank0 成为读取瓶颈，但换来对"不可分片数据源"（流式、动态产生的数据，典型如 GRPO rollout 结果）的支持。源码仅百余行，是理解"分布式数据调度"概念绝佳的入门读物。

### 9.4 GRPO 的训推混合调度（框架内最复杂、也最值得学的调度设计）

GRPO 每个 step 需要"先用当前策略采样（推理），再用采样结果训练"，训练与推理争抢 GPU 是核心矛盾。ms-swift 提供两种官方调度模式：

**Colocate（共卡）模式**（`--use_vllm true --vllm_mode colocate`）：在 Trainer 进程内启动 vLLM，训推分时复用同一组卡。配套一整套"显存腾挪"时序参数：`--sleep_level 1`（训练阶段让 vLLM 释放显存休眠）、`--offload_model/--offload_optimizer true`（推理阶段把训练模型与优化器状态卸到 CPU）、`--vllm_gpu_memory_utilization` 调低水位、`--vllm_tensor_parallel_size` 推理侧 TP、zero3 下 `--move_model_batches` 分批 gather 权重同步给 vLLM。适合卡少的中小规模实验。

**Server（分离）模式**（`--vllm_mode server`）：先用 `swift rollout --model ... --vllm_tensor_parallel_size T --vllm_data_parallel_size D` 在独立机器/卡上起 vLLM 采样服务，训练侧通过 `--vllm_server_host/port` 连接（`vllm_client.py`），并可 `--async_generate` 异步采样让训练与采样流水线重叠。适合多机集群与大规模 RL。LoRA 训练还可只同步 adapter 权重（而非全量权重）来加速每步的权重更新同步。

这套设计浓缩了当前 RL infra 的典型权衡（共卡省资源 vs 分离高吞吐），`rlhf_trainers/rollout_mixin.py` + `infer_engine/grpo_vllm_engine.py` 是对应源码。更复杂的多角色编排（训练/rollout/reward 各占资源组）由 Ray 集成承担（`examples/ray`、docs Instruction/Ray.md）。

---

## 十、实现指南：四个由浅入深的实战案例

> 以下代码均节选自仓库 `examples/custom/` 与 `swift/rewards/` 的真实实现，可直接作为你的开发模板。通用工作流：先用 `--external_plugins` 在框架外跑通 → 再决定是否沉淀为内置实现提 PR。

### 案例 A：注册一个新模型 + 对话模板（最经典的第一个 PR）

```python
# my_plugin.py —— 摘自 examples/custom/model.py
from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.template import TemplateMeta, register_template

register_template(TemplateMeta(
    template_type='custom',
    prefix=['<extra_id_0>System\n{{SYSTEM}}\n'],
    prompt=['<extra_id_1>User\n{{QUERY}}\n<extra_id_1>Assistant\n'],
    chat_sep=['\n']))

register_model(ModelMeta(
    model_type='custom',
    model_groups=[ModelGroup([Model('AI-ModelScope/Nemotron-Mini-4B-Instruct',
                                    'nvidia/Nemotron-Mini-4B-Instruct')])],
    template='custom'))
```

外置验证：`swift sft --model ... --external_plugins my_plugin.py ...`。模板正确性校验技巧（官方示例同款）：用 swift 模板与 tokenizer 自带 jinja 模板各推理一次，断言输出一致（`engine.template.template_backend = 'jinja'`）。

升级为 PR 的清单：① 实现放入 `swift/model/models/<family>.py` 与 `swift/template/templates/<family>.py`；② 多模态模型补 `model_arch`（参考 BestPractices/MLLM-Registration.md，需自定义 `_encode/_post_encode/_data_collator`）；③ 运行 `scripts/utils/run_model_info.py` 自动更新支持模型文档；④ 在 `tests/test_align/test_template/` 添加编码对齐测试；⑤ `pre-commit run --all-files` 全绿后提 PR。

### 案例 B：注册自定义数据集

```python
# 摘自 examples/custom/dataset.py
from swift.dataset import DatasetMeta, ResponsePreprocessor, register_dataset

class CustomPreprocessor(ResponsePreprocessor):
    prompt = "Task: ...\nSentence 1: {text1}\nSentence 2: {text2}\nSimilarity score: "
    def preprocess(self, row):
        return super().preprocess({
            'query': self.prompt.format(text1=row['text1'], text2=row['text2']),
            'response': f"{row['label']:.1f}"})

register_dataset(DatasetMeta(
    ms_dataset_id='swift/stsb', hf_dataset_id='SetFit/stsb',
    preprocess_func=CustomPreprocessor()))
```

核心心法呼应 9.1：**Preprocessor 的唯一职责是把任意原始字段映射成标准 messages/query-response 格式**，之后的编码、packing、collate 全部由框架接管。内置数据集 PR 的落点是 `swift/dataset/dataset/` 与 `swift/dataset/data/`。

### 案例 C：自定义 GRPO 奖励函数（RL 方向的标准切入点）

`swift/rewards/orm.py` 定义的接口极简：

```python
from swift.rewards import ORM  # 异步外部服务可用 AsyncORM

class MyReward(ORM):
    def __call__(self, completions, **kwargs) -> list[float]:
        # completions: 模型生成的回复列表；kwargs 携带数据集中其他列（如 solution）
        return [1.0 if check(c, s) else 0.0
                for c, s in zip(completions, kwargs['solution'])]
```

注册进 mapping 后用 `--reward_funcs my_reward format` 组合多个奖励。内置的 `MathAccuracy`、`Format`、`ReActFormat` 是最好的参考实现；需要奖励模型打分时改用 `rm_plugin.py` 的 `DefaultRMPlugin`/生成式 `GenRMPlugin`。完整工程范例见 `examples/train/grpo/plugin/`（含外部奖励函数、外部奖励模型、外部调度器三个运行脚本及 deepeyes/treepo 两个复刻案例）。

### 案例 D：自定义 Callback / Loss（最轻量的代码贡献）

Callback 与 transformers `TrainerCallback` 接口完全一致（覆写 `on_train_begin/on_save/...`），在 `swift/callbacks/mapping.py` 注册后以 `--callbacks <name>` 启用；自定义 Loss 继承 `BaseLoss` 实现 `__call__(outputs, labels, **kwargs) -> Tensor`，注册后以 `--loss_type <name>` 启用（当前支持 sft/pretrain/reranker/embedding 任务）。这两类改动面小、接口清晰，非常适合作为熟悉 mapping 机制与 CI 流程的练手 PR。

### 通用调试工具箱

1. **模板逐 token 检查**（一切训练问题先看 labels）：`template.encode(data)` 后 `safe_decode(encoded['labels'])`，确认哪些 token 参与 loss；
2. **最小复现**：0.5B 模型 + `--dataset 'xxx#100'`（取 100 条）+ 单卡，几分钟内验证全流程；
3. **坏数据排查**：临时关闭 `--lazy_tokenize` 的容错（strict）或观察 LazyLLMDataset 的 traceback 告警；
4. **提交前三连**：`pre-commit run --all-files` → 相关单测（`tests/` 对应目录）→ 跑一条 `examples/` 里最接近你改动的脚本做冒烟。

---

## 十一、调研局限性与待补充方向

1. GitHub API 当日限流，stars/forks/issue 数取自第三方镜像站点（2026 年 4~6 月快照），与实时值可能有小幅出入；
2. 未对训练性能做实测 benchmark，与 LLaMA-Factory 等框架的对比基于公开资料与功能面分析，非量化评测；
3. "good first issue" 标签的实时数量未能确认，建议读者直接在 issue 页按标签筛选；
4. 项目迭代极快，本手册的目录结构与 API 以 2026-06-12 的 main 分支为准，若干月后请以仓库内 `docs/source/Customization/Architecture.md` 为权威更新源。

---

## 参考来源

1. 官方仓库：https://github.com/modelscope/ms-swift （main 分支源码、README_CN.md、CONTRIBUTING_CN.md，本地 clone 实地分析）
2. 官方文档：https://swift.readthedocs.io/zh-cn/latest/ （Quick-start、Command-line-parameters、Customization/Architecture 等）
3. SWIFT 论文（AAAI 2025）：https://arxiv.org/pdf/2408.05517
4. 仓库内文档：docs/source/Customization/{Architecture,Custom-model,Custom-dataset}.md、docs/source/BestPractices/MLLM-Registration.md
5. 项目统计：Ecosyste.ms / ReleaseAlert 仓库快照（2026-04~06）
6. 社区教程（参考性质，注意版本时效）：CSDN《AI大模型ms-swift框架实战指南》系列、博客园《ms-swift 深度解析》、SwanLab 官方集成文档