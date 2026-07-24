# 大模型对齐（Alignment）技术深度研究报告
## ——基于 ms-swift 框架的原理剖析、主流技术综述与工程实现研究

**报告版本**：v1.0
**编写日期**：2026 年 7 月
**研究对象**：ms-swift（modelscope/ms-swift, https://github.com/modelscope/ms-swift.git）中 Alignment（模型对齐）相关模块
**报告范围**：对齐技术基本原理、主流对齐算法（RLHF/PPO、DPO 系列、GRPO 系列及其衍生算法）、ms-swift 框架中的工程实现、以及相关学术论文与技术报告的梳理与总结

---

## 摘要

大语言模型（LLM）的能力已经从"预测下一个 token"跃迁为"理解并执行复杂指令""进行多步推理""与人类价值观保持一致"。这一跃迁背后，"对齐"（Alignment）技术起到了决定性的作用。从 OpenAI 的 InstructGPT 首次系统性提出"预训练—监督微调（SFT）—基于人类反馈的强化学习（RLHF）"三段式范式开始，对齐技术经历了从复杂的 PPO 强化学习流程，到以 DPO 为代表的直接偏好优化范式的简化，再到以 GRPO、DAPO、GSPO 为代表的、面向可验证奖励（Reinforcement Learning with Verifiable Rewards, RLVR）的强化学习范式的又一次跃迁。每一次范式转移，都伴随着训练稳定性、工程复杂度、样本效率与最终对齐效果之间权衡关系的重新调整。

ms-swift（ModelScope SWIFT）是阿里巴巴通义实验室 / ModelScope 社区开源的大模型与多模态大模型训练部署框架（论文被 AAAI 2025 接收），目前已经支持 600+ 纯文本大模型与 300+ 多模态大模型的全参数/PEFT 训练，并将"人类偏好对齐"作为其训练能力矩阵中最为重要、迭代最为频繁的模块之一。截至本报告编写时，ms-swift 通过统一的 `swift rlhf` 命令行入口，支持 DPO、ORPO、SimPO、CPO、KTO、RM、PPO、GRPO、GKD 等九大类主流对齐算法，并在 GRPO 之上进一步集成了 DAPO、GSPO、CISPO、SAPO、RLOO、REINFORCE++、CHORD、TreePO、熵掩码（entropy mask）、训练-推理不一致校正（Training-Inference-Mismatch correction）等十余项 2024—2026 年间学术界最新的强化学习细粒度改进方案，形成了目前开源社区中对齐算法覆盖面最广、工程实现最为系统化的框架之一。

本报告的目标是：(1) 系统梳理"对齐"这一概念的技术内涵、发展脉络与理论基础；(2) 对当前主流的对齐算法——涵盖基于人类反馈的强化学习（RLHF/PPO）、直接偏好优化家族（DPO/IPO/KTO/ORPO/CPO/SimPO 等）、以及基于可验证奖励的强化学习家族（GRPO 及其十余种工程变体）——进行数学原理层面的深入推导与相互比较；(3) 深入 ms-swift 源码与官方文档，剖析其对齐模块的整体架构、命令行参数体系、数据格式规范、分布式训练支撑（DeepSpeed、FSDP、Megatron、Ray、vLLM 协同）、奖励函数与奖励模型体系、多轮/多任务/Agent 场景支持等工程实现细节；(4) 结合实际训练脚本，给出可复现的最佳实践指南；(5) 对 ms-swift 对齐能力进行横向比较，并展望对齐技术的未来发展方向。

本报告的写作原则是"论文原理—工程实现—实践细节"三位一体，尽量避免空泛的概念堆砌，而是尽可能给出可追溯的公式推导、可对照的源码/文档依据与可执行的命令行示例，希望能够成为算法研究者理解对齐技术全貌、工程人员基于 ms-swift 落地对齐训练的一份系统性参考资料。

---

## 目录

**第一部分：总论**
- 第一章 引言：为什么需要模型对齐
- 第二章 对齐问题的定义、分类与评价维度
- 第三章 对齐技术发展脉络回顾

**第二部分：基础理论**
- 第四章 从预训练到对齐：三段式训练流程总览
- 第五章 人类偏好建模：Bradley-Terry 模型与奖励模型训练
- 第六章 强化学习基础回顾：MDP、策略梯度与 Actor-Critic
- 第七章 RLHF 的完整数学推导：KL 约束下的策略优化

**第三部分：主流对齐算法技术原理详解**
- 第八章 PPO 用于 RLHF：InstructGPT 范式与四模型架构
- 第九章 直接偏好优化 DPO：数学推导与隐式奖励模型
- 第十章 DPO 家族的演化：IPO、KTO、ORPO、CPO、SimPO 及混合损失
- 第十一章 奖励模型（RM）训练细节与 Reward Hacking 应对
- 第十二章 可验证奖励强化学习（RLVR）与 GRPO
- 第十三章 GRPO 家族的工程演化：DAPO、GSPO、CISPO、SAPO、RLOO、REINFORCE++、CHORD、Dr.GRPO
- 第十四章 训练-推理不一致问题与重要性采样校正
- 第十五章 知识蒸馏对齐 GKD 与其他前沿方向
- 第十六章 多模态对齐与 Constitutional AI / RLAIF 学术脉络

**第四部分：ms-swift 框架工程实现深度剖析**
- 第十七章 ms-swift 项目总览与生态定位
- 第十八章 整体架构：命令行体系、参数系统与 Plugin 机制
- 第十九章 对齐训练数据格式规范详解
- 第二十章 分布式与并行技术支撑体系
- 第二十一章 逐算法命令行与关键超参数详解
- 第二十二章 GRPO 工程实现深潜：vLLM Colocate/Server 双模式与显存优化
- 第二十三章 奖励函数与奖励模型体系
- 第二十四章 多轮训练、多任务训练与 Agent/GYM 环境
- 第二十五章 Megatron-SWIFT：大规模对齐训练
- 第二十六章 前沿算法集成全景：从熵掩码到训练-推理一致性
- 第二十七章 多模态大模型对齐支持
- 第二十八章 实战案例：从 SFT 到 GRPO 的完整训练流程

**第五部分：综合分析与展望**
- 第二十九章 ms-swift 对齐能力横向对比分析
- 第三十章 工程实践建议与常见问题排查
- 第三十一章 未来发展方向展望
- 第三十二章 总结

**附录**：参考文献与延伸阅读

---

# 第一部分 总论

## 第一章 引言：为什么需要模型对齐

### 1.1 从"能干"到"愿意好好干"

在 GPT-3 时代，研究者已经发现，仅仅依靠海量文本的自回归预训练（next-token prediction），语言模型确实可以习得极为丰富的世界知识与语言能力，但这种能力的"涌现"并不等价于模型会按照人类期望的方式去使用这些能力。一个预训练模型面对"如何制作炸弹"这样的问题，可能会因为训练语料中存在相关讨论而给出实质性帮助；面对"帮我写一封邮件"这样的指令，也可能因为预训练目标只是"续写"而不是"服从指令"，从而给出续写问题本身而非回答问题的文本。这就是所谓的"能力—意图错位"（capability-intention mismatch）：模型具备某种能力，但其行为分布并不必然与人类的真实意图、偏好和价值观对齐。

OpenAI 在 InstructGPT 论文（Ouyang et al., 2022, *Training language models to follow instructions with human feedback*，[arXiv:2203.02155](https://arxiv.org/abs/2203.02155)）中，将这一问题系统化地定义为：语言模型的训练目标（预测下一个 token）与我们希望模型具备的行为（有帮助地、诚实地、无害地遵循用户指令）之间存在着不对齐（misalignment）。他们提出使用"人类反馈强化学习"（RLHF）来弥合这一鸿沟，并在 1.3B 参数的 InstructGPT 上验证了：经过 RLHF 微调的小模型，在人类评价者眼中比未经对齐的 175B GPT-3 更受欢迎。这一结果第一次以严谨的实验证明：对齐比单纯扩大模型规模更能直接提升"人类可感知的有用性"。

### 1.2 对齐的三个核心目标：3H 原则

学术界和工业界通常用"3H"来概括对齐的核心目标，这一提法最早由 Anthropic 在其 HH-RLHF（Helpful and Harmless RLHF）数据集与相关论文中系统化提出：

1. **Helpful（有帮助）**：模型应当尽可能准确、完整、简洁地帮助用户完成任务，理解用户的真实意图而非仅仅字面意思。
2. **Honest（诚实）**：模型应当如实反映自己知道与不知道的内容，不应编造虚假信息（即"幻觉"，hallucination），也不应误导用户。
3. **Harmless（无害）**：模型应当拒绝协助违法、危险或不道德的请求，避免生成歧视性、暴力性或其他有害内容。

这三个目标之间往往存在张力：过度强调 Harmless 可能导致模型对无害请求也变得"过度拒绝"（over-refusal），从而牺牲 Helpful；过度强调 Helpful 又可能导致模型在被诱导（jailbreak）时突破安全边界。因此，对齐从来不是单目标优化问题，而是一个多目标、需要精细权衡（trade-off）的系统工程问题，这也是奖励建模、偏好数据构建、强化学习超参数选择等环节均需要精心设计的根本原因。

### 1.3 从"指令遵循对齐"到"推理能力对齐"

2024 年以来，随着 OpenAI o1、DeepSeek-R1 等推理模型的出现，"对齐"这一概念的外延被进一步拓展。除了传统意义上的"价值观对齐""指令遵循对齐"之外，业界开始大规模使用强化学习来对齐模型的**推理行为**：即通过在数学、代码等具有可验证正确性的任务上使用强化学习（RLVR, Reinforcement Learning with Verifiable Rewards），让模型自发地学会"思考更长时间""自我反思""回溯纠错"等复杂推理模式。DeepSeek-R1 论文（*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025）证明了：仅通过强化学习（甚至无需监督微调冷启动，即 DeepSeek-R1-Zero），模型就可以自发涌现出长链推理（long Chain-of-Thought）能力。这使得 GRPO 这一原本用于数学推理场景的算法，迅速取代 PPO 成为 2024—2026 年间强化学习对齐的主流范式，也直接推动了 ms-swift 等训练框架将 GRPO 及其一系列工程变体作为对齐能力建设的重心。

因此，本报告所讨论的"对齐"，既包含传统的"三段式" RLHF（SFT→RM→RL）价值观/指令对齐，也包含新兴的"可验证奖励强化学习"能力对齐，两者在 ms-swift 框架中通过统一的 `swift rlhf` 命令行入口与统一的 Trainer 抽象被有机地组织在一起。

### 1.4 本报告的研究方法与信息来源

本报告的撰写基于以下信息来源的交叉验证：(1) ms-swift GitHub 仓库（modelscope/ms-swift）的 README、examples/train/rlhf 与 examples/train/grpo 目录下的训练脚本；(2) swift.readthedocs.io 官方文档中的 RLHF、GRPO（含 Get Started / Developer Guide / Advanced Research 三个子模块）等章节；(3) Qwen 官方文档中关于 ms-swift 的训练最佳实践；(4) DPO、PPO、GRPO、KTO、ORPO、CPO、SimPO、GSPO、DAPO 等算法的原始论文；(5) InstructGPT、Anthropic HH-RLHF、Constitutional AI、DeepSeek-R1、MM-RLHF 等对齐领域的代表性论文与技术报告。在下文的算法原理讲解中，公式推导尽量与 ms-swift 官方文档中给出的实现公式保持一致，以确保"原理讲解"与"工程实现"章节能够相互印证、前后呼应。


## 第二章 对齐问题的定义、分类与评价维度

### 2.1 对齐的形式化定义

在技术层面，我们可以将"对齐"形式化地理解为一个优化问题：给定一个预训练语言模型所诱导的策略分布 $\pi_{\text{pt}}(y|x)$，我们希望找到一个新的策略 $\pi_\theta(y|x)$，使得该策略在人类偏好意义下的期望收益最大化，同时不过度偏离原始模型的行为分布（以保留其语言能力、知识储备和多样性）。这一目标可以写成如下带 KL 正则的优化问题：

$$
\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \big[ r(x, y) \big] - \beta \, \mathbb{D}_{\text{KL}}\big[\pi_\theta(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x)\big]
$$

其中 $r(x,y)$ 是刻画"人类偏好"或"任务正确性"的奖励函数，$\pi_{\text{ref}}$ 通常是 SFT 之后的参考模型，$\beta$ 是控制偏离程度的超参数。几乎所有本报告将要介绍的对齐算法——无论是 PPO、DPO 还是 GRPO——本质上都是在求解这一带约束优化问题的不同近似方式，区别仅在于：奖励从何而来（显式奖励模型 vs. 隐式重参数化 vs. 规则化可验证奖励）、如何估计梯度（Actor-Critic vs. 蒙特卡洛组内相对基线 vs. 闭式解）、以及如何在工程上高效地实现采样与更新。

### 2.2 对齐问题的分类维度

站在工程实现的角度，可以从以下几个维度对当前的对齐技术进行分类：

**(1) 按奖励来源分类：**
- **基于人类标注的偏好数据**：人类标注员对同一 prompt 下的多个候选回答进行两两比较或打分排序，这是 RLHF、DPO 及其绝大多数变体的数据基础。
- **基于 AI 反馈（RLAIF）**：使用一个更强的模型（或同一模型自我评价）来替代人类标注，代表性工作是 Anthropic 的 Constitutional AI（CAI）与 Google 的 RLAIF 论文。
- **基于规则/可验证信号（RLVR）**：在数学、代码、逻辑推理等具有确定性正确答案或可执行验证机制的任务中，直接用规则（如答案匹配、单元测试通过率）作为奖励，代表性工作是 DeepSeek-R1、Kimi k1.5 等推理模型的训练方案，也是 GRPO 家族算法最主要的应用场景。

**(2) 按算法范式分类：**
- **在线强化学习范式（Online RL）**：如 PPO、GRPO、RLOO、REINFORCE++，需要在训练过程中不断从当前策略采样新数据（rollout），并使用采样数据更新策略，通常需要引入价值函数或组内基线来降低方差。
- **离线/直接优化范式（Offline / Direct Optimization）**：如 DPO、IPO、KTO、ORPO、CPO、SimPO，这类方法不需要在训练过程中进行在线采样，而是直接在已经收集好的偏好数据集上，通过重参数化技巧将强化学习目标转化为一个可以用监督学习方式优化的损失函数。
- **混合范式**：如 ms-swift 中支持的 MPO（Mixed Preference Optimization），将多种损失函数（如 DPO 损失 + BC 损失 + 生成损失）加权混合；又如 RPO（在 DPO 损失中混入 SFT 损失）。

**(3) 按训练阶段与目标分类：**
- **格式与指令遵循对齐**：让模型学会遵循特定的输出格式、语气与指令结构。
- **价值观与安全对齐**：让模型的行为符合社会伦理规范，拒绝有害请求。
- **能力/推理对齐（RLVR）**：让模型在数学、代码等任务上学会更长、更准确的推理链条。
- **多模态对齐**：让视觉-语言等多模态模型在跨模态理解、幻觉抑制、多模态安全等维度上与人类偏好对齐。

**(4) 按是否需要显式奖励模型分类：**
- **需要显式 RM**：PPO、GRPO（当使用 reward model 作为打分器时）都可以配合一个通过 Bradley-Terry 损失训练出的标量奖励模型。
- **不需要显式 RM（隐式奖励）**：DPO 及其变体通过数学推导，将最优策略与奖励函数之间建立了闭式关系，从而可以直接在偏好数据上优化策略网络本身，不需要单独训练一个奖励模型。
- **规则奖励（无需训练 RM）**：GRPO 在 RLVR 场景下往往直接使用规则函数（如数学答案校验器 `math_verify`、代码执行沙箱）作为奖励来源。

### 2.3 对齐效果的评价维度

对齐效果的评估通常包含以下几个维度，这些维度在 ms-swift 的训练日志（如 `reward/{reward_func_name}/mean`、`kl`、`completions/clipped_ratio` 等指标）与最佳实践文档中均有直接体现：

- **胜率（Win Rate）**：在成对比较（pairwise comparison）评测中，对齐后模型的回答相对于基线模型或参考模型被人类/裁判模型（LLM-as-a-Judge）判定为更优的比例，代表性评测集包括 AlpacaEval、MT-Bench、Arena-Hard 等。
- **任务准确率**：在 RLVR 场景下，通常直接使用数学题（如 GSM8K、MATH、AIME）、代码题（如 HumanEval、LiveCodeBench）等基准的准确率来衡量推理能力对齐效果。
- **安全性/无害性**：通过红队测试（red-teaming）、有害请求拒绝率、越狱攻击成功率等指标衡量。
- **过度对齐/过度拒绝（Over-alignment / Over-refusal）**：衡量模型是否对无害请求也表现出过度谨慎的拒绝倾向，这是安全对齐中容易出现的"副作用"。
- **训练稳定性指标**：如策略与参考模型的 KL 散度、奖励标准差、梯度裁剪比例（clip ratio）、训练崩溃（collapse）风险等，这些是 ms-swift GRPO 模块日志中重点监控的工程指标。
- **多样性与创造力保留**：对齐训练容易导致模型输出分布"坍缩"（mode collapse），生成内容趋同、缺乏多样性，因此困惑度（perplexity）、n-gram 多样性等指标也常被用作辅助评价。

## 第三章 对齐技术发展脉络回顾

![大模型对齐技术发展脉络](./assets/01_alignment_timeline.png)

### 3.1 萌芽期（2017—2021）：从 RL 用于文本生成到 Reward Modeling 的雏形

将强化学习引入自然语言生成任务的探索可以追溯到 2017 年前后。Christiano et al. (2017) 在 *Deep reinforcement learning from human preferences* 中首次系统性地提出了"用人类成对比较来训练奖励模型，再用奖励模型指导策略优化"的框架，尽管当时的应用场景主要是 Atari 游戏与机器人控制，但其核心思想——Bradley-Terry 偏好模型 + 策略梯度优化——直接奠定了后续 RLHF 的理论基础。2019—2020 年间，OpenAI 团队（Ziegler et al., *Fine-Tuning Language Models from Human Preferences*；Stiennon et al., *Learning to summarize from human feedback*）率先将这一框架系统性地应用于文本摘要等 NLP 任务，验证了"人类偏好奖励模型 + PPO 微调语言模型"这一范式在文本生成任务上的有效性。

### 3.2 奠基期（2022）：InstructGPT 与 RLHF 三段式范式的确立

2022 年 3 月，OpenAI 发布 InstructGPT 论文，正式确立了"预训练（Pretraining）→有监督微调（SFT）→奖励建模（RM）→基于 PPO 的强化学习（RL）"的四步流程，这一流程后来被简称为"RLHF 三段式"（因为预训练通常被视为独立的第零阶段）。同年年底，ChatGPT 的发布让这一整套技术方案第一次进入大众视野，"RLHF"迅速成为大模型对齐的代名词。同期，Anthropic 发布了 HH-RLHF 数据集与相关论文，将"有帮助且无害"作为奖励建模的核心目标，并开源了大规模人类偏好比较数据，为后续学术界复现 RLHF 提供了重要的数据基础。

### 3.3 简化期（2023）：DPO 与直接偏好优化范式的兴起

PPO 版本的 RLHF 尽管效果显著，但其工程复杂度极高：需要同时维护策略模型、参考模型、奖励模型、价值模型四个模型，训练过程中还涉及在线采样（rollout）、优势函数估计（GAE）、价值函数拟合等一系列容易出现不稳定性的环节，对超参数极为敏感，复现门槛很高。2023 年 5 月，斯坦福大学 Rafailov et al. 发表了具有里程碑意义的论文 *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*（DPO，[arXiv:2305.18290](https://arxiv.org/abs/2305.18290)），从数学上证明了：在 KL 正则化的强化学习目标下，最优策略与奖励函数之间存在解析闭式关系，因此可以通过对偏好数据的最大似然估计，直接优化策略模型本身，而无需显式训练奖励模型，也无需在线采样。DPO 的出现极大地降低了 RLHF 的工程门槛，几乎在一年之内成为开源社区进行偏好对齐的事实标准，并催生了 IPO、KTO、ORPO、CPO、SimPO 等一系列后续变体，2023—2024 年间对齐研究的主战场很大程度上聚集在"如何改进 DPO 的损失函数设计"这一方向上。

### 3.4 强化期（2024—2025）：可验证奖励强化学习与 GRPO 的崛起

2024 年 2 月，DeepSeekMath 论文（[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)）提出了 GRPO（Group Relative Policy Optimization）算法，其核心创新是使用"组内相对优势"替代 PPO 中需要额外训练的价值模型，从而在数学推理任务上大幅降低了强化学习的工程复杂度与显存开销。2025 年 1 月，DeepSeek-R1 论文（[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)）进一步证明：仅通过大规模 GRPO 强化学习（配合可验证的规则奖励），模型即可在数学、代码等任务上自发涌现出长链思维（long CoT）、自我反思、回溯纠错等复杂推理行为，且效果可以媲美甚至超越同期最强的闭源推理模型。这一结果彻底改变了对齐技术的研究重心：以"可验证奖励强化学习"（RLVR）为核心的 GRPO 系列算法迅速取代 PPO 与 DPO，成为 2024 年下半年至 2026 年间大模型能力对齐（尤其是推理能力对齐）的主流范式。与此同时，围绕 GRPO 训练稳定性、样本效率、训练-推理一致性等工程细节的改进百花齐放：字节跳动的 DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization，[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)）针对 GRPO 在长链推理场景下的梯度消失、长度偏差等问题提出了"Clip-Higher""动态采样""超长惩罚"等一系列工程改进；阿里巴巴 Qwen 团队提出的 GSPO（Group Sequence Policy Optimization，[arXiv:2507.18071](https://arxiv.org/abs/2507.18071)）指出 token 级别重要性采样在长序列生成中会引入高方差噪声，转而在序列级别计算重要性采样比率；此外 CISPO（[arXiv:2506.13585](https://arxiv.org/abs/2506.13585)，MiniMax-M1 技术报告）、SAPO、RLOO（[arXiv:2402.14740](https://arxiv.org/abs/2402.14740)）、REINFORCE++（[arXiv:2501.03262](https://arxiv.org/abs/2501.03262)）、CHORD、TreePO 等一系列改进方案相继涌现，共同构成了当前 GRPO"大家族"。

### 3.5 工程化与生态化期（2024—2026）：训练框架的百花齐放

伴随着算法层面的快速迭代，一批面向大规模强化学习/偏好对齐训练的开源框架相继出现并快速成熟，包括 HuggingFace TRL（trl 库中的 DPOTrainer / PPOTrainer / GRPOTrainer 等）、OpenRLHF、veRL（字节跳动开源，原名 HybridFlow）、以及本报告重点研究的 ms-swift。这些框架的共同特征是：通过与高性能推理引擎（如 vLLM、SGLang）的深度集成来加速在线强化学习中的采样（rollout）阶段，通过与 DeepSpeed、FSDP、Megatron-LM 等分布式训练框架的集成来支持超大规模模型的全参数强化学习，并通过将各类前沿算法改进（如 GSPO、DAPO 的具体 trick）封装为可插拔的超参数选项，使得算法研究人员可以在同一套工程基础设施上快速复现、组合、消融各类学术论文中的改进点。ms-swift 正是这一生态化趋势中的代表性框架之一，其"多算法统一入口 + 丰富前沿研究集成 + 多模态支持 + 多种并行技术兼容"的设计理念，也构成了本报告第四部分重点剖析的对象。

# 第二部分 基础理论

## 第四章 从预训练到对齐：三段式训练流程总览

![三段式RLHF训练流程](./assets/02_rlhf_pipeline.png)

### 4.1 阶段零：预训练（Pretraining）

预训练阶段的目标是通过在海量无标注文本上进行自回归语言建模（预测下一个 token），让模型习得广泛的世界知识、语言规律与初步的推理模式。这一阶段的损失函数是标准的交叉熵损失：

$$
\mathcal{L}_{\text{PT}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}_{\text{pt}}} \left[ \sum_{t=1}^{T} \log \pi_\theta(x_t \mid x_{<t}) \right]
$$

预训练模型 $\pi_{\text{pt}}$ 尽管具备强大的语言能力，但其行为模式是"续写"而非"服从指令"，因此不能直接用于对话类应用，需要经过后续的对齐阶段。

### 4.2 阶段一：有监督微调（SFT，Supervised Fine-Tuning）

SFT 阶段使用人工编写或筛选的高质量"指令-回答"配对数据，对预训练模型进行微调，使其初步具备遵循指令、进行多轮对话的能力。SFT 的损失函数形式上与预训练相同，但仅在数据集中"回答"部分的 token 上计算：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{sft}}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \right]
$$

SFT 之后得到的模型 $\pi_{\text{sft}}$ 通常被用作后续 RM 训练与 RL 微调阶段的初始化模型（policy 的初始值）与参考模型 $\pi_{\text{ref}}$。ms-swift 官方文档中特别强调：在进行 DPO 等偏好优化训练之前，"建议先在偏好数据集中的 chosen（偏好）回答上进行一轮 SFT 训练，这有助于让数据分布更好地匹配 DPO 算法的前提假设"，这一实践建议本质上是为了缓解偏好优化阶段模型初始策略与数据分布之间的分布偏移（distribution shift）问题。

### 4.3 阶段二：奖励建模（RM，Reward Modeling）

奖励建模阶段的目标是训练一个标量奖励函数 $r_\phi(x,y)$，使其能够刻画人类对回答 $y$ 相对于问题 $x$ 的偏好程度。训练数据通常是三元组 $(x, y_w, y_l)$，其中 $y_w$（winner）是人类标注员认为更优的回答，$y_l$（loser）是相对较差的回答。奖励模型通常复用 SFT 模型的骨干网络，并在最后一层替换为一个输出标量的线性头（value head），使用 Bradley-Terry 偏好模型对成对比较数据进行最大似然估计。这一阶段的详细损失函数与工程注意事项将在第五章、第十一章中展开。

### 4.4 阶段三：强化学习微调（RL）

在得到奖励模型之后，第三阶段使用强化学习算法（如 PPO）对策略模型 $\pi_\theta$（通常初始化为 $\pi_{\text{sft}}$）进行微调，使其在奖励模型 $r_\phi$ 给出的评分意义下的期望回报最大化，同时通过 KL 正则项约束策略不过度偏离参考模型 $\pi_{\text{ref}}$（通常也是 $\pi_{\text{sft}}$）：

$$
\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

这一阶段是整个 RLHF 流程中工程复杂度最高、也最容易出现训练不稳定性的环节，第六、七、八章将对其数学原理与工程实现进行深入剖析。

### 4.5 三段式流程在 ms-swift 中的映射

ms-swift 将上述四个阶段（预训训练+SFT+RM+RL）分别映射为四个相对独立又可以无缝衔接的命令行入口：`swift pt`（继续预训练，CPT）、`swift sft`（有监督微调）、以及统一的 `swift rlhf --rlhf_type {dpo,orpo,simpo,kto,cpo,rm,ppo,grpo,gkd}`（覆盖奖励建模与各类强化学习/偏好优化算法）。这种"分阶段独立命令 + 统一 checkpoint 格式"的设计，使得用户可以非常方便地将上一阶段训练产出的模型（无论是全参数权重还是 LoRA adapter）无缝作为下一阶段的初始化模型或参考模型，这是 ms-swift 工程设计中的一个重要理念，将在第十七、十八章中详细展开。

## 第五章 人类偏好建模：Bradley-Terry 模型与奖励模型训练

### 5.1 Bradley-Terry 模型

人类对两个回答的偏好比较，本质上是一个二元比较问题。Bradley-Terry（BT）模型是刻画这种成对比较数据的经典统计模型，其基本假设是：每个待比较对象 $i$ 都有一个潜在的"实力值" $s_i$，两个对象 $i, j$ 比较时，$i$ 胜出的概率为：

$$
P(i \succ j) = \frac{\exp(s_i)}{\exp(s_i) + \exp(s_j)} = \sigma(s_i - s_j)
$$

其中 $\sigma(\cdot)$ 是 sigmoid 函数。在 RLHF 语境下，我们将"实力值"替换为奖励函数的输出 $r_\phi(x,y)$，从而得到人类偏好 $y_w \succ y_l$（给定问题 $x$）的概率模型：

$$
P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)
$$

### 5.2 奖励模型的最大似然估计与损失函数

给定标注数据集 $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$，奖励模型的训练目标是最大化在 Bradley-Terry 假设下观测到的标注数据的似然，等价于最小化负对数似然损失：

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \Big[ \log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big) \Big]
$$

ms-swift 官方文档中给出的 RM 训练损失函数在此基础上增加了两个工程实践中十分重要的正则项：

$$
\text{loss} = -\log \sigma\big(r^{(c)} - r^{(r)} - m\big) + \lambda \big(r^{(c)} + r^{(r)}\big)^2
$$

其中 $r^{(c)}$、$r^{(r)}$ 分别是模型对 chosen（偏好）与 rejected（拒绝）回答给出的评分；$m$ 是"margin"（间隔）项，用于让模型在难度不同的样本对上学到不同程度的区分度，该值需要数据集额外提供 `margin` 列，默认取 0；$\lambda$ 是由参数 `center_rewards_coefficient` 控制的 L2 正则系数，用于约束奖励模型输出值不过度偏离零点，从而使得奖励分数的绝对数值具有可解释性，这一设计源自 Llama 2 论文（*Llama 2: Open Foundation and Fine-Tuned Chat Models*）中提出的 reward centering 技巧。

### 5.3 奖励模型的架构设计：Value Head

奖励模型通常以 SFT 后的模型（或 base 模型）为骨干，在最后一个 Transformer 层的隐藏状态之上，新增一个输出维度为 1 的线性层（value head），将序列最后一个有效 token 位置的隐藏状态映射为一个标量分数。这一新增的 value head 权重通常与骨干网络的其余参数一同参与训练（或仅训练 value head，视 LoRA/全参数训练策略而定），并单独保存为 `value_head.safetensors` 或 `value_head.bin` 文件，这一细节在 ms-swift 的 RM 训练文档中被明确提及，也是后续 PPO 训练阶段加载奖励模型时需要特别处理的工程细节。

### 5.4 Reward Hacking 问题及其应对

由于奖励模型本身也是一个通过有限数据训练出来的神经网络，其对"好回答"的刻画必然存在偏差与过拟合风险。当策略模型在强化学习过程中被过度优化以最大化奖励模型评分时，容易出现"Reward Hacking"（奖励黑客/奖励欺骗）现象：策略模型学会利用奖励模型的某些系统性偏差（如偏好更长的回答、偏好特定的措辞模式、偏好带有列表格式的回答）来"骗取"高分，而实际回答质量并未真正提升，甚至可能下降。应对 Reward Hacking 的常见工程手段包括：(1) 使用 KL 正则项约束策略不过度偏离参考模型；(2) 使用集成奖励模型（reward model ensemble）或分歧惩罚来降低对单一奖励模型系统性偏差的依赖；(3) 在 RLVR 场景中直接使用规则/可验证奖励（如答案正确性校验），从根本上避免了奖励模型本身可能存在偏差的问题，这也是 GRPO 在数学、代码等任务上广受欢迎的重要原因之一；(4) 使用长度惩罚等专门针对已知偏差模式设计的辅助奖励函数，如 ms-swift 内置的 `cosine`、`repetition`、`soft_overlong` 等奖励函数，均是为了缓解特定类型的 Reward Hacking 而设计（详见第二十三章）。

## 第六章 强化学习基础回顾：MDP、策略梯度与 Actor-Critic

### 6.1 语言生成任务的 MDP 建模

将语言模型的文本生成过程建模为马尔可夫决策过程（Markov Decision Process, MDP）是理解 RLHF 强化学习阶段的基础。在这一建模下：
- **状态（State）** $s_t$：当前的输入 prompt $x$ 与已生成的前缀 $y_{<t}$ 的拼接。
- **动作（Action）** $a_t$：在词表上选择下一个 token $y_t$。
- **策略（Policy）** $\pi_\theta(a_t|s_t)$：即语言模型本身，给定当前上下文，输出下一个 token 的概率分布。
- **奖励（Reward）**：在大多数 RLHF 实现中，仅在生成完整回答的最后一个 token 上给予奖励模型评分 $r_\phi(x,y)$，中间 token 的即时奖励为 0（在 PPO 实现中，通常还会在每一步加入一个 KL 惩罚项作为稠密的中间奖励，详见第八章）。
- **回合终止**：当模型生成结束符（EOS）或达到最大生成长度时，回合（episode）终止。

这种"整个回答只在末尾获得一次奖励"的稀疏奖励特性，是 RLHF 强化学习区别于传统游戏/机器人控制强化学习任务的一个重要特点，也是为什么需要精心设计优势函数估计方法（如 GAE、组内相对基线）来降低训练方差的原因。

### 6.2 策略梯度定理

策略梯度方法直接对策略参数 $\theta$ 关于期望回报的梯度进行估计与优化。策略梯度定理给出了如下梯度表达式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t \right]
$$

其中 $\hat{A}_t$ 是优势函数（advantage function）的估计，衡量在状态 $s_t$ 下采取动作 $a_t$ 相对于该状态下平均水平的"额外收益"。使用优势函数而非原始回报，是策略梯度方法中降低方差的核心技巧（baseline subtraction）。

### 6.3 Actor-Critic 架构与广义优势估计（GAE）

在 Actor-Critic 架构中，除了策略网络（Actor）$\pi_\theta$ 之外，还引入一个价值网络（Critic）$V_\psi(s_t)$ 来估计状态的期望回报，从而计算优势函数。广义优势估计（Generalized Advantage Estimation, GAE，Schulman et al., 2016，[arXiv:1506.02438](https://arxiv.org/abs/1506.02438)）通过引入折扣因子 $\gamma$ 与平滑系数 $\lambda$，在偏差与方差之间进行权衡：

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

这正是 ms-swift PPO 文档中提到的超参数 `gamma`（折扣因子，默认 1.0）与 `lam`（GAE 的 $\lambda$ 系数，默认 0.95）的数学来源。在语言模型 RLHF 场景下，由于每个回答通常只有一步"有效"的稀疏奖励（加上逐 token 的 KL 惩罚作为稠密奖励），$\gamma$ 常被设为 1.0（不进行折扣）。

### 6.4 On-Policy 与 Off-Policy 的辨析

策略梯度方法在理论上是"on-policy"（同策略）的：用于计算梯度的采样数据，必须来自当前正在更新的策略本身。但在实际工程实现中，出于采样效率的考虑，通常会先用当前策略 $\pi_{\theta_{\text{old}}}$ 采样一批数据，然后对这批数据进行多轮（epoch）梯度更新，这就使得策略在更新过程中逐渐偏离采样时的策略，产生了一定程度的"离策略性"（off-policyness）。PPO 通过重要性采样比率与截断机制来修正这种偏差（详见第八章），而 GRPO 及其变体在使用 vLLM 等外部推理引擎加速采样时，还会面临"训练模型与推理引擎数值不一致"这一新的 off-policy 来源，这正是第十四章要重点讨论的"训练-推理不一致"（Training-Inference-Mismatch）问题的根源。

## 第七章 RLHF 的完整数学推导：KL 约束下的策略优化

### 7.1 优化目标的建立

综合第四、五、六章的内容，RLHF 强化学习阶段的完整优化目标可以写为：

$$
\max_{\pi_\theta} \; \mathbb{E}_{x\sim\mathcal{D}} \, \mathbb{E}_{y\sim\pi_\theta(\cdot|x)} \left[ r_\phi(x,y) \right] - \beta \, \mathbb{E}_{x\sim\mathcal{D}} \, \mathbb{D}_{\text{KL}}\big[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\big]
$$

这一目标函数的第一项鼓励策略生成高奖励的回答，第二项则通过 KL 散度惩罚，防止策略过度偏离参考模型，从而避免出现两个问题：(1) 过拟合到奖励模型的系统性偏差（Reward Hacking）；(2) 策略退化为只能生成极少数高奖励模式的回答，丧失语言的流畅性与多样性。

### 7.2 最优策略的解析解

将上述目标函数展开，并利用 KL 散度的定义，可以证明该优化问题存在闭式解。将目标函数改写为：

$$
\max_{\pi_\theta} \; \mathbb{E}_{x} \left[ \mathbb{E}_{y\sim\pi_\theta} \left[ r_\phi(x,y) - \beta \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right] \right]
$$

对于固定的 $x$，这是一个关于 $\pi_\theta(\cdot|x)$ 的泛函优化问题。利用变分法（或直接构造拉格朗日函数并对概率分布的归一化约束使用拉格朗日乘子），可以求得最优策略的解析形式为：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y|x) \, \exp\left(\frac{1}{\beta} r_\phi(x,y)\right)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta}r_\phi(x,y)\right)$ 是配分函数（partition function），用于保证概率分布归一化。这一"指数倾斜"（exponential tilting）形式的解析解，直观地说明了 RLHF 的本质：最优对齐后的策略，是在参考模型的基础上，按照奖励函数的指数权重对概率质量进行重新分配——奖励越高的回答，其概率相对于参考模型被放大得越多；$\beta$ 越大，这种放大效应越弱（越接近参考模型）。

### 7.3 从解析解到 DPO：奖励函数的反向重参数化

上述解析解的一个重要推论是：给定策略 $\pi_\theta$ 与参考模型 $\pi_{\text{ref}}$，我们可以反过来将隐含的奖励函数表示为：

$$
r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

这一"奖励的重参数化"正是 DPO 算法的理论基石：由于配分函数 $Z(x)$ 只与 $x$ 有关、与 $y$ 无关，在将该式代入 Bradley-Terry 偏好模型 $P(y_w \succ y_l|x) = \sigma(r(x,y_w) - r(x,y_l))$ 时，$\log Z(x)$ 项会相互抵消，从而使得我们可以完全绕过显式奖励模型与在线采样，直接在偏好数据上对策略网络进行最大似然估计。这一推导将在第九章中完整展开。

### 7.4 为什么需要在线强化学习：解析解无法直接求解的原因

尽管上一节给出了最优策略的解析形式，但在实践中我们**无法**直接根据这一公式采样或计算策略，原因在于配分函数 $Z(x)$ 需要对指数级大小的输出空间 $y$ 进行求和，在语言模型场景下（词表大小 $|V| \sim 10^5$，序列长度可达数千）完全不可行。因此，在不诉诸于第九章 DPO 式重参数化技巧的前提下，我们必须借助策略梯度类的在线强化学习算法（如 PPO、GRPO），通过采样-评估-更新的迭代过程，以数值优化的方式去逼近这一解析解所刻画的最优策略。这正是"在线强化学习范式"（PPO/GRPO 家族）与"离线直接优化范式"（DPO 家族）两条技术路线的分野所在，也是本报告第三部分将要重点展开的核心内容。

# 第三部分 主流对齐算法技术原理详解

## 第八章 PPO 用于 RLHF：InstructGPT 范式与四模型架构

![PPO用于RLHF的四模型架构](./assets/03_ppo_architecture.png)

### 8.1 PPO 算法回顾

近端策略优化（Proximal Policy Optimization, PPO，Schulman et al., 2017，[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)）是策略梯度方法中最具工程实用性的代表性算法之一，其核心思想是通过"截断重要性采样比率"（clipped surrogate objective）来限制每次策略更新的幅度，从而在保证训练稳定性的同时提升样本利用效率（允许对同一批采样数据进行多轮梯度更新）。PPO 的核心目标函数为：

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\Big( \rho_t(\theta) \hat{A}_t, \; \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\, \hat{A}_t \Big) \right]
$$

其中 $\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新旧策略的重要性采样比率，$\epsilon$（通常取 0.1—0.3）是截断范围超参数。当优势 $\hat{A}_t > 0$ 时，目标函数会限制比率不超过 $1+\epsilon$，避免"过度自信"地增大该动作的概率；当 $\hat{A}_t < 0$ 时，则限制比率不低于 $1-\epsilon$，避免过度削减该动作的概率。这种"取最小值"的设计使得目标函数成为真实目标的悲观（pessimistic）下界，从而在工程上实现了保守但稳健的策略更新。

### 8.2 RLHF 场景下的四模型架构

将 PPO 应用于 RLHF 时，需要同时维护四个模型，这也是 ms-swift 官方文档中明确列出的 PPO 训练所需要的四个组件：

1. **策略模型（model / policy model）**：即待训练的语言模型，通常初始化自 SFT 后的模型或 base 模型。
2. **参考模型（ref_model）**：默认与策略模型的初始状态相同，在训练过程中参数冻结，用于计算 KL 惩罚项，防止策略过度偏离。
3. **奖励模型（reward_model）**：由第五章、第十一章所述的 RM 训练阶段产出的、带 value head 的打分模型，训练过程中参数冻结，仅用于对策略模型生成的回答进行打分。
4. **价值模型（value_model）**：用于估计状态价值 $V_\psi(s_t)$ 的 Critic 网络，通常由奖励模型的权重初始化，并在 PPO 训练过程中与策略模型同步更新。

四个模型同时驻留显存，是 PPO 版 RLHF 工程复杂度和显存开销远高于 DPO、GRPO 的直接原因——尤其是当模型规模达到数十亿甚至上百亿参数时，四份权重（加上各自的优化器状态，若参与训练）对显存和多机分布式调度都提出了很高的要求。

### 8.3 逐 token KL 惩罚与最终奖励的合成

在实践中，PPO 版 RLHF 通常并不是仅在回答末尾加入一次性的 KL 惩罚，而是将 KL 惩罚作为逐 token 的稠密奖励信号叠加到奖励模型给出的稀疏奖励之上：

$$
r_t = \underbrace{\mathbb{1}[t=T] \cdot r_\phi(x,y)}_{\text{仅末�token的RM评分}} \; - \; \beta \, \big(\log \pi_\theta(y_t|x,y_{<t}) - \log \pi_{\text{ref}}(y_t|x,y_{<t})\big)
$$

这种设计使得 KL 惩罚在训练早期就能对每一步生成产生即时约束效果，而不必等到整个回答生成完毕才能感知偏离程度，有助于提升训练稳定性。这一合成奖励随后被送入 GAE 计算优势函数 $\hat{A}_t$，进而代入 PPO 的截断目标函数进行优化。

### 8.4 PPO-RLHF 的关键超参数

结合 ms-swift 官方文档中列出的 PPO 超参数，我们可以将其归纳如下：

| 超参数 | 含义 | 默认值 |
|---|---|---|
| `local_rollout_forward_batch_size` | 采样阶段单次前向的 batch 大小 | 64 |
| `whiten_rewards` | 是否对奖励做归一化（白化） | False |
| `kl_coef` | KL 惩罚系数 $\beta$ | 0.05 |
| `cliprange` | 策略损失的截断范围 $\epsilon$ | 0.2 |
| `vf_coef` | 价值函数损失的权重系数 | 0.1 |
| `cliprange_value` | 价值损失的截断范围 | 0.2 |
| `gamma` | 折扣因子 | 1.0 |
| `lam` | GAE 的 $\lambda$ 系数 | 0.95 |
| `num_sample_generations` | 训练过程中用于调试展示的采样样本数 | 10 |

价值函数的损失同样采用截断机制，以防止价值网络单步更新幅度过大：

$$
\mathcal{L}^{\text{VF}}(\psi) = \mathbb{E}_t \left[ \max\Big( (V_\psi(s_t) - V^{\text{target}}_t)^2, \; (\text{clip}(V_\psi(s_t), V_{\text{old}}-\delta, V_{\text{old}}+\delta) - V^{\text{target}}_t)^2 \Big) \right]
$$

总损失为策略损失与价值损失（乘以 `vf_coef` 权重）之和，工程上还常常加入一个熵正则项以鼓励探索，防止策略过早收敛到确定性分布。

### 8.5 PPO-RLHF 的工程痛点

PPO 版 RLHF 在实践中常见的工程痛点包括：(1) 四模型同时驻留显存导致的资源消耗巨大，尤其在全参数训练场景下需要精细的显存管理（如模型卸载 offload、ZeRO 分片等）；(2) 训练过程对超参数（尤其是 `kl_coef`、学习率、`cliprange`）极为敏感，容易出现奖励崩溃或 KL 散度爆炸等不稳定现象；(3) 采样（rollout）阶段是主要的时间瓶颈，需要借助高性能推理引擎加速；(4) 价值函数的准确拟合本身是一个有难度的子问题，价值函数估计偏差会直接传导到优势函数估计，进而影响策略更新方向。这些痛点正是促使学术界探索"去 Critic 化"（Critic-free）强化学习算法（如 GRPO、RLOO、REINFORCE++）以及"离线直接优化"算法（如 DPO 家族）的直接动因。

## 第九章 直接偏好优化 DPO：数学推导与隐式奖励模型

![PPO与DPO范式对比](./assets/04_dpo_vs_ppo.png)

### 9.1 从奖励重参数化到 DPO 损失

延续第七章 7.3 节的推导，我们已经得到：给定最优策略 $\pi^*$ 与参考模型 $\pi_{\text{ref}}$，隐含的奖励函数可以表示为 $r(x,y) = \beta \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$。将这一表达式代入 Bradley-Terry 偏好概率模型：

$$
P(y_w \succ y_l \mid x) = \sigma\Big( r(x,y_w) - r(x,y_l) \Big) = \sigma\left( \beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)
$$

由于两项中的 $\log Z(x)$ 恰好相互抵消，我们得到了一个仅依赖于策略模型 $\pi_\theta$ 与参考模型 $\pi_{\text{ref}}$（不依赖任何显式奖励模型或配分函数）的偏好概率表达式。对这一表达式在偏好数据集上做最大似然估计，即得到 DPO 的损失函数：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma\left( \beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

这一损失函数的直观含义是：拉大策略模型相对于参考模型，对 chosen 回答的对数概率提升幅度与对 rejected 回答的对数概率提升幅度之间的差距，使得 chosen 回答相对更受偏好。DPO 论文将其形象地总结为"你的语言模型秘密地就是一个奖励模型"（Your Language Model is Secretly a Reward Model）。

### 9.2 DPO 梯度的直观解释

对 DPO 损失关于 $\theta$ 求梯度，可以得到：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \mathbb{E}_{(x,y_w,y_l)} \left[ \sigma\big(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w)\big) \Big( \nabla_\theta \log\pi_\theta(y_w|x) - \nabla_\theta \log\pi_\theta(y_l|x) \Big) \right]
$$

其中 $\hat{r}_\theta(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是隐式奖励。该梯度公式揭示了一个重要性质：梯度的权重系数 $\sigma(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w))$ 恰好是"隐式奖励模型判断错误"（认为 rejected 回答比 chosen 回答更好）的概率——当隐式奖励模型已经能够正确区分两者时（$\hat{r}_\theta(x,y_w) \gg \hat{r}_\theta(x,y_l)$），该权重趋近于 0，梯度自然减弱；反之当隐式奖励模型判断错误或含糊时，梯度权重更大，这与直觉上"重点纠正判断错误的样本"是一致的，也是 DPO 训练相对稳定的原因之一。

### 9.3 DPO 的实践建议与超参数

ms-swift 官方文档中列出的 DPO 关键超参数包括：

- **beta**：KL 正则系数，数值越大对参考模型的偏离惩罚越强，默认 0.1。
- **loss_type**：DPO 损失函数的变体选择（默认 `sigmoid`，即标准 DPO 损失），此外还支持 `hinge`、`ipo`、`kto_pair`、`robust` 等多种变体（详见第十章）。
- **label_smoothing**：DPO 标签平滑，默认 0，用于处理偏好标注中存在噪声（人类标注错误）的情形，源自 cDPO（conservative DPO）思想。
- **rpo_alpha**：在 DPO 损失基础上混入一定比例的 SFT（NLL）损失，用于提升训练稳定性，这一技巧源自 RPO（Regularized Preference Optimization）思想，因为纯粹的 DPO 损失有时会导致 chosen 回答的绝对概率也随之下降（尽管相对 rejected 回答的概率差距在扩大），混入 SFT 损失可以缓解这一现象。
- **ld_alpha**：源自 LD-DPO 论文（*LD-DPO: Local Difference Preference Optimization*），对 chosen 与 rejected 回答共享前缀之外的 token 对数概率乘以一个小于 1 的权重 $\alpha$，从而缓解 DPO 训练中的长度偏差（DPO 天然倾向于让模型生成更长的回答，因为更长的回答在对数概率求和意义下更容易被"拉开差距"）。
- **discopop_tau**：源自 DiscoPOP 论文（通过 LLM 自动发现新损失函数的工作），是在 sigmoid 调制前对数比率进行缩放的温度参数，默认 0.05。

此外，官方文档建议：在开始 DPO 训练之前，最好先在偏好数据集的 chosen 回答上进行一轮 SFT，以使模型初始策略更贴近偏好数据的分布，从而提升 DPO 训练的稳定性与最终效果。

## 第十章 DPO 家族的演化：IPO、KTO、ORPO、CPO、SimPO 及混合损失

![DPO家族演化关系图](./assets/05_dpo_family_tree.png)

DPO 提出之后，学术界围绕其潜在缺陷（如对训练数据过拟合、长度偏差、无法建模非成对偏好数据等）提出了大量改进方案，形成了一个庞大的"DPO 家族"。ms-swift 对这一家族中的主要成员均提供了原生支持。

### 10.1 IPO（Identity Preference Optimization）

IPO（Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences*，[arXiv:2310.12036](https://arxiv.org/abs/2310.12036)）指出：DPO 隐含地假设了 Bradley-Terry 模型是对人类偏好的准确刻画，但真实的人类偏好可能是非传递的、带噪声的，直接对 DPO 损失中的 sigmoid 项做最大似然估计，容易在偏好差异被数据完全可分（可确定性区分）时导致隐式奖励差距无限增大，进而使模型对训练集过拟合、KL 惩罚失效。IPO 通过将损失函数替换为一个均方误差（而非对数几率）形式来解决这一问题：

$$
\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E}_{(x,y_w,y_l)} \left[ \left( \beta^{-1}\big(\hat{r}_\theta(x,y_w) - \hat{r}_\theta(x,y_l)\big) - \frac{1}{2} \right)^2 \right]
$$

这一损失函数的最优解会将隐式奖励差距限制在一个有限值附近，而非无限增大，从而在理论上具有更好的正则化特性。在 ms-swift 中，IPO 损失可以通过设置 `--loss_type ipo` 直接启用。

### 10.2 KTO（Kahneman-Tversky Optimization）

KTO（Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*，[arXiv:2402.01306](https://arxiv.org/abs/2402.01306)）的核心创新在于：它不需要成对比较数据 $(x, y_w, y_l)$，而只需要形如 $(x, y, \text{label})$ 的数据——即针对单个回答标注"好/坏"二元标签即可，这大幅降低了偏好数据的标注成本（成对比较需要标注员同时看到两个候选回答并排序，而 KTO 只需对单个回答做出二元判断）。KTO 的理论基础是行为经济学中的"前景理论"（Prospect Theory，Kahneman & Tversky 提出，也是 2002 年诺贝尔经济学奖工作），该理论认为人类对"损失"比对等量的"收益"更为敏感（损失厌恶）。KTO 损失函数为：

$$
\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{(x,y)} \Big[ \lambda_{y} \big(1 - v(x,y)\big) \Big]
$$

其中当 $y$ 是 desirable（偏好）样本时，$v(x,y) = \sigma\big(\beta \hat{r}_\theta(x,y) - z_0\big)$，权重为 $\lambda_D$（默认 1.0）；当 $y$ 是 undesirable（拒绝）样本时，$v(x,y) = \sigma\big(z_0 - \beta \hat{r}_\theta(x,y)\big)$，权重为 $\lambda_U$（默认 1.0）；$z_0$ 是参考点（reference point），通常由批次内其他样本的 KL 散度估计得到。ms-swift 官方文档特别指出一个重要的超参数设置准则：设 $n_D$、$n_U$ 分别为数据集中偏好样本与拒绝样本的数量，论文建议设置 $\lambda_D, \lambda_U$ 使得 $\frac{\lambda_D n_D}{\lambda_U n_U} \in [1, \frac{4}{3}]$，以平衡两类样本对训练的贡献度，避免因数据不平衡导致模型对某一类样本过拟合。

### 10.3 CPO（Contrastive Preference Optimization）

CPO（Xu et al., *Contrastive Preference Optimization*，[arXiv:2401.08417](https://arxiv.org/abs/2401.08417)，最初应用于机器翻译场景）的动机是解决 DPO 依赖参考模型带来的两个问题：(1) 训练时需要额外维护并进行前向传播的参考模型，增加了显存与计算开销；(2) DPO 的优化目标只关心相对偏好，不直接约束 chosen 回答的绝对生成质量，可能导致 chosen 回答本身的似然也在下降。CPO 使用 chosen 回答本身的分布（而非依赖参考模型）作为隐式的"参照基准"，损失函数中不再包含参考模型的对数概率项，同时额外混入一项 NLL（negative log-likelihood，即 SFT 式）损失，鼓励模型直接拟合 chosen 回答：

$$
\mathcal{L}_{\text{CPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma\Big(\beta \log \pi_\theta(y_w|x) - \beta \log \pi_\theta(y_l|x)\Big) \right] + \alpha_{\text{cpo}} \cdot \mathcal{L}_{\text{NLL}}(y_w)
$$

ms-swift 中对应超参数为 `beta`（隐式奖励系数，默认 0.1）与 `cpo_alpha`（NLL 损失权重，默认 1.0）。由于不需要参考模型，CPO 相比 DPO 在训练时可以节省近一半的显存与前向计算开销。

### 10.4 ORPO（Odds Ratio Preference Optimization）

ORPO（Hong et al., *Reference-free Monolithic Odds Ratio Preference Optimization*，[arXiv:2403.07691](https://arxiv.org/abs/2403.07691)）进一步将偏好优化与 SFT 阶段直接融合为一个单一阶段（monolithic），完全不需要预先进行独立的 SFT 训练，也不需要参考模型。其核心思想是在标准的 SFT（NLL）损失基础上，叠加一个基于"几率比"（odds ratio）的对比惩罚项，直接在同一个训练过程中，既让模型学会生成 chosen 回答（通过 NLL 损失），又让模型学会相对抑制 rejected 回答（通过几率比对比项）。定义某个回答 $y$ 相对输入 $x$ 的生成"几率"（odds）为：

$$
\text{odds}_\theta(y|x) = \frac{\pi_\theta(y|x)}{1-\pi_\theta(y|x)}
$$

ORPO 的对比损失为：

$$
\mathcal{L}_{\text{OR}} = -\log\sigma\left( \log\frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right)
$$

总损失为 $\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{NLL}}(y_w) + \lambda \cdot \mathcal{L}_{\text{OR}}$，其中 $\lambda$（在 ms-swift 中通过复用 `--beta` 参数传递）控制对比项的权重。由于 ORPO 不需要参考模型、也不需要独立的 SFT 阶段，其工程效率与显存效率是 DPO 家族中最高的之一，非常适合资源受限场景下的快速偏好对齐。

### 10.5 SimPO（Simple Preference Optimization）

SimPO（Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*，[arXiv:2405.14734](https://arxiv.org/abs/2405.14734)）同样是一种无需参考模型的方法，但与 ORPO 在损失形式上有所不同。SimPO 提出使用序列的**平均对数概率**（而非总对数概率）作为隐式奖励，从根本上消除了 DPO 中隐式奖励与回答长度的耦合（DPO 的隐式奖励是对数概率之和，天然随长度增长；而平均对数概率则不然），同时引入一个显式的目标间隔（margin）$\gamma$：

$$
\hat{r}_{\text{SimPO}}(x,y) = \frac{\beta}{|y|} \log \pi_\theta(y|x), \qquad \mathcal{L}_{\text{SimPO}} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log\sigma\Big( \hat{r}_{\text{SimPO}}(x,y_w) - \hat{r}_{\text{SimPO}}(x,y_l) - \gamma \right) \right]
$$

ms-swift 中对应超参数为 `beta`（默认 2.0，注意 SimPO 的 beta 取值通常显著大于 DPO）、`simpo_gamma`（奖励间隔项，默认 1.0）以及 `cpo_alpha`（用于混合 CPO 的 NLL 损失以提升稳定性，默认 1.0；设为 0 则还原为论文原始的 SimPO 算法）。SimPO 论文报告其在 AlpacaEval 2、Arena-Hard 等评测集上相对 DPO 有稳定的性能提升，同时因不需要参考模型而具有更高的训练效率。

### 10.6 混合损失与 MPO

在实际工程中，单一损失函数往往难以同时兼顾"偏好区分度""生成质量""长度鲁棒性"等多个目标，因此 ms-swift 支持通过设置多个 `loss_type` 值并配合 `loss_weights` 参数进行加权组合，这一机制被称为混合偏好优化（Mixed Preference Optimization, MPO，源自 *Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization* 论文，最初应用于多模态模型的推理能力对齐）。典型的 MPO 组合方式是同时使用 DPO 的成对偏好损失、BCO（Binary Classifier Optimization）式损失以及生成式 NLL 损失的加权和，从而在保证偏好区分能力的同时，兼顾生成质量与训练稳定性。这一机制使得 ms-swift 的 DPO 训练模块具备了高度的可组合性和可扩展性，用户可以像"搭积木"一样组合出适合特定场景的定制化损失函数。

## 第十一章 奖励模型（RM）训练细节与 Reward Hacking 应对（工程延伸）

本章在第五章理论基础上，进一步展开奖励模型训练在工程实践中需要关注的细节问题，为第四部分中 ms-swift 的 RM 实现分析做铺垫。

### 11.1 奖励模型的初始化策略

奖励模型的初始化对最终效果影响显著。常见策略包括：(1) 直接复用目标策略模型的 SFT checkpoint 作为奖励模型的初始化，这样可以保证奖励模型与策略模型享有相近的知识背景与语言理解能力，是最常见的做法；(2) 使用比策略模型更大规模的模型作为奖励模型骨干，以期获得更强的评判能力（但会增加训练与推理阶段的资源开销）；(3) 在 PPO 阶段，价值模型（value model）通常直接由奖励模型的权重初始化，因为二者的输出空间（标量分数）是一致的，这也是 ms-swift PPO 文档中提到"value_model 由 reward_model 初始化"的原因。

### 11.2 奖励模型的过拟合与早停

由于偏好标注数据集规模通常远小于 SFT 数据集（人工标注成对比较的成本远高于生成指令-回答对），奖励模型极易在训练数据上过拟合。工程实践中常见的应对措施包括：限制训练 epoch 数（通常 1 个 epoch 即可，多个 epoch 容易导致过拟合）、使用验证集上的成对比较准确率作为早停依据、以及在 PPO/GRPO 强化学习阶段密切监控策略相对参考模型的 KL 散度增长速度，一旦发现奖励快速上升但 KL 散度也同步快速发散，往往意味着策略正在利用奖励模型的过拟合弱点进行"刷分"。

### 11.3 奖励模型与规则奖励的互补关系

在 RLVR 场景下（详见第十二章），规则奖励（如数学答案校验）虽然消除了奖励模型本身可能存在的偏差问题，但规则奖励通常只能覆盖"结果正确性"这一单一维度，无法评价推理过程的合理性、简洁性、或是否存在"蒙对答案"的侥幸情形。因此，实践中越来越多的工作开始探索"规则奖励 + 生成式奖励模型（Generative Reward Model, GenRM）"的混合奖励方案：GenRM 让一个大模型对候选回答生成自然语言评语（critique）后再给出分数，兼具可解释性与判别准确性，MM-RLHF 论文中提出的"Critique-Based Reward Model"正是这一思路在多模态场景下的代表性实现（详见第十六章）。ms-swift 的 GRPO 模块中同样支持将奖励模型（包括 GenRM）作为可插拔的奖励来源之一，与规则奖励函数共同组成加权奖励，这一机制将在第二十三章详细展开。

## 第十二章 可验证奖励强化学习（RLVR）与 GRPO

![GRPO组内相对优势估计机制](./assets/06_grpo_mechanism.png)

### 12.1 RLVR 的基本思想

可验证奖励强化学习（Reinforcement Learning with Verifiable Rewards, RLVR）的核心思想是：在数学、代码、逻辑推理等具备"客观正确答案"或"可执行验证机制"的任务领域中，完全绕开需要单独训练的奖励模型，直接使用确定性的规则函数作为奖励信号。例如，对于数学题，可以将模型输出的最终答案与标准答案进行符号匹配（如使用 `math_verify` 库解析数学表达式并判断等价性）；对于代码生成任务，可以直接执行模型生成的代码并运行单元测试，以测试通过率作为奖励。这种奖励机制的优势是：(1) 完全没有 Reward Hacking 风险（规则本身不存在"可被利用的偏差"）；(2) 不需要额外训练和维护一个奖励模型，节省大量标注与算力成本；(3) 奖励信号是"零噪声"的客观信号，训练过程通常更加稳定。RLVR 的局限性在于：仅适用于具备客观验证机制的任务领域，对于开放式问答、创意写作、价值观对齐等任务并不适用，因此在实践中 RLVR 与传统基于人类偏好的 RM/RLHF 往往是互补而非替代关系。

### 12.2 GRPO 算法原理

GRPO（Group Relative Policy Optimization，出自 DeepSeekMath 论文，*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*，2024）的核心创新在于：抛弃了 PPO 中需要额外训练的价值网络（Critic），转而对同一个 prompt 采样一组（group）候选回答，使用组内奖励的均值作为基线（baseline），以组内标准化后的相对优势替代 GAE 估计的优势函数。GRPO 的目标函数为：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q\sim P(Q),\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)} \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \Bigg\{ \min\Big[ \rho_{i,t}\hat{A}_{i,t}, \, \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,t} \Big] - \beta\, \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \Bigg\}
$$

其中组内相对优势定义为对该组（同一 prompt 采样出的 $G$ 个回答）奖励的标准化：

$$
\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}
$$

需要特别指出的是，GRPO 中的 KL 惩罚项并非像 PPO 那样作为逐 token 的奖励叠加到优势函数计算中，而是直接作为一个独立的正则项加入最终损失函数，通常使用无偏、低方差的 K3 估计量来近似计算：

$$
\mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \approx \exp\big(\log\pi_{\text{ref}} - \log\pi_\theta\big) - \big(\log\pi_{\text{ref}} - \log\pi_\theta\big) - 1
$$

该估计量恒为非负，且在 $\pi_\theta = \pi_{\text{ref}}$ 时取值为 0，方差显著低于朴素的对数比率差估计，是 John Schulman 在其博客 *Approximating KL Divergence* 中提出的"K3 估计量"，已成为当前 GRPO 类算法的标准实现。

### 12.3 GRPO 相较 PPO 的核心优势

GRPO 相较 PPO 的核心工程优势在于：(1) 无需训练和维护价值网络，节省了近 1/4 到 1/2 的显存与计算开销（视模型规模而定）；(2) 无需 GAE 这一相对复杂且对超参数敏感的优势估计过程，用简单的组内标准化替代；(3) 由于同一 prompt 下采样多个回答并进行组内比较，天然适合结果可验证的场景——只要组内存在正确与错误的回答，组内标准化后就能产生非零的优势信号，指导模型学习；(4) 算法实现相对简单，更容易与高性能推理引擎（vLLM/SGLang）集成实现大规模并行采样加速。这些优势使得 GRPO 成为 DeepSeek-R1 等最新推理模型训练的核心算法，也是 ms-swift 目前投入研发资源最多的对齐算法模块。

### 12.4 GRPO 训练的完整流程

结合 ms-swift 官方文档给出的伪代码，GRPO 的训练流程可以概括为四个阶段：

**阶段一：Rollout 生成**——对于每个 prompt，使用当前策略模型（或加速推理引擎）采样 $G$（`num_generations`）个候选回答，通常设置较高的采样温度（`temperature`，如 1.0）以保证组内多样性。

**阶段二：奖励计算**——对每个候选回答，调用一个或多个奖励函数（规则函数、奖励模型或二者组合）计算标量奖励，并将多个奖励函数的加权和作为最终奖励；随后在组内对奖励做均值-标准差标准化，得到每个回答的优势值。

**阶段三：策略优化**——分别计算当前策略 $\pi_\theta$、旧策略 $\pi_{\theta_{\text{old}}}$（即采样时使用的策略快照）、参考策略 $\pi_{\text{ref}}$ 在候选回答上的逐 token 对数概率，构造 PPO 式截断目标函数与 KL 正则项，得到总损失。

**阶段四：参数更新**——对总损失执行反向传播与优化器更新，并根据 `num_iterations`（同一批采样数据的重复利用次数）判断是否需要重新进行下一轮 rollout。

### 12.5 GRPO 训练效果的典型案例：DeepSeek-R1

DeepSeek-R1 论文提供了 GRPO 用于 RLVR 场景最具说服力的实证证据：DeepSeek-R1-Zero 完全跳过了 SFT 冷启动阶段，直接在 base 模型上使用 GRPO 结合规则奖励（答案正确性 + 格式合规性）进行强化学习，模型在训练过程中自发涌现出了逐渐变长的思维链、自我验证、回溯纠错等复杂推理行为模式，在 AIME 2024 等高难度数学竞赛基准上的准确率随训练步数的增加持续稳定提升。这一结果证明了：只要奖励信号足够客观可靠（可验证），纯强化学习（甚至无需人工标注的思维链数据）就足以激发大模型的深层推理能力，这也是 RLVR 范式在 2025 年之后被广泛采纳、并直接推动 GRPO 系列算法成为对齐领域研究热点的根本原因。

## 第十三章 GRPO 家族的工程演化：DAPO、GSPO、CISPO、SAPO、RLOO、REINFORCE++、CHORD、Dr.GRPO

![GRPO家族工程演化全景](./assets/07_grpo_family.png)

原始 GRPO 算法在被广泛应用于大规模长链推理训练的过程中，逐渐暴露出若干工程与理论层面的问题（如长度偏差、训练不稳定、样本利用效率低下等），学术界与工业界围绕这些问题提出了大量改进方案。ms-swift 通过 `--loss_type`、`--importance_sampling_level` 等一系列可插拔超参数，将下述十余种改进方案全部纳入统一的 GRPO Trainer 实现之中，形成了当前开源社区中最为全面的"GRPO 家族"工程集成。

### 13.1 归一化维度的辨析：GRPO / BNPO / Dr.GRPO

原始 GRPO 损失在归一化时，采用"先在每个样本内部对 token 损失取平均，再对样本取平均"的两级平均方式（第一章 12.2 节目标函数中的 $\frac{1}{|o_i|}\sum_t$）。这种归一化方式存在一个潜在问题：较短回答中每个 token 的损失权重相对较大，较长回答中每个 token 的损失权重相对较小，这会在客观上鼓励模型生成更短的回答（因为通过样本内平均，长回答中单个高优势 token 的贡献被"稀释"）。针对这一问题，学术界与工程实践中提出了若干种替代归一化方式：

- **BNPO（Batch Normalized Policy Optimization）**：将全部样本的全部 token 损失直接求和，再除以该批次的总 token 数，即在 token 维度上进行归一化，而非样本维度：
$$
\mathcal{L}_{\text{BNPO}} = \frac{\sum_{i=1}^{N}\sum_{t=1}^{T_i} \mathcal{L}_{i,t}}{\sum_{i=1}^N T_i}
$$
- **Dr.GRPO（Dr-GRPO，"Done Right" GRPO，出自论文 *Understanding R1-Zero-Like Training: A Critical Perspective*，[arXiv:2503.20783](https://arxiv.org/abs/2503.20783)）**：进一步指出，即使是 BNPO 式的 token 级归一化，其分母（当前批次的实际 token 总数）仍然是一个随训练动态变化的量，会引入额外的方差；因此 Dr.GRPO 使用一个固定的分母（批次大小乘以预设的最大生成长度 $N \times L_{\max}$）进行归一化：
$$
\mathcal{L}_{\text{DR-GRPO}} = \frac{\sum_{i=1}^{N}\sum_{t=1}^{T_i} \mathcal{L}_{i,t}}{N \times L_{\max}}
$$
这种固定分母的设计从理论上完全消除了归一化维度本身随训练动态变化所带来的偏差，使得优化目标更加"公正"、不偏向鼓励或惩罚特定长度的回答。
- **DAPO 归一化**：与 BNPO 归一化方式相似（token 级），但进一步将归一化分母扩展为**全部并行进程（多机多卡）的全局 token 总数**，而非仅当前进程内的 token 总数，从而使得损失的尺度不受数据并行切分方式的影响：
$$
\mathcal{L}_{\text{DAPO}} = \frac{\sum_{i=1}^N\sum_{t=1}^{T_i}\mathcal{L}_{i,t}}{\sum_{\text{all processes}}\sum_{i=1}^{N_p} T_{p,i}}
$$

在 ms-swift 中，用户可以通过 `--loss_type {grpo,bnpo,dr_grpo,dapo,cispo,sapo}` 参数在上述几种归一化方式之间自由切换，进行消融实验。

### 13.2 DAPO：Clip-Higher、动态采样与超长惩罚

DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization，字节跳动 Seed 团队，*DAPO: An Open-Source LLM Reinforcement Learning System at Scale*，2025）是针对原始 GRPO 在超大规模长链推理训练中暴露出的若干具体问题提出的系统性改进方案，除了上一节介绍的 token 级全局归一化之外，DAPO 论文还提出了以下几项关键技术，其中多项已在 ms-swift 中以独立的可插拔组件形式实现：

- **Clip-Higher（非对称截断）**：原始 PPO/GRPO 使用对称的截断区间 $[1-\epsilon, 1+\epsilon]$，DAPO 论文指出这一设计会导致"熵坍缩"（entropy collapse）问题——由于低概率 token 的重要性采样比率上升空间被压缩得很小（因为 $\rho_t \le 1+\epsilon$），模型很难通过强化学习去提升那些初始概率很低、但可能带来正确答案的"探索性"token 的概率，从而使得策略的探索能力随训练很快枯竭。DAPO 提出将上界截断参数放宽（例如 $\epsilon_{\text{high}} > \epsilon_{\text{low}}$，如 $\epsilon_{\text{high}}=0.28$），从而给予低概率但有潜力的 token 更大的提升空间。ms-swift 中通过 `--epsilon`（下界）与 `--epsilon_high`（上界）两个独立参数支持这一非对称截断设计。
- **动态采样（Dynamic Sampling）**：在原始 GRPO 中，如果某个 prompt 采样出的一组回答全部正确或全部错误，组内奖励标准差为零，会导致优势值全部为零，这些样本对训练"无贡献"却仍然占用了采样与计算资源。DAPO 提出过滤掉这类"全对或全错"的 prompt，转而持续采样直到收集到足够数量的、组内存在有效优势差异的样本，从而提升样本利用效率。
- **Token 级策略梯度损失**：即前述的 token 级全局归一化方式，避免长回答中每个 token 梯度被过度稀释。
- **超长回答惩罚（Overlong Reward Shaping）**：针对训练中生成长度经常触及最大长度上限被截断的回答，直接给予懲罚而非简单丢弃截断样本的梯度信息不完整的问题，DAPO 提出在一个长度区间内施加线性惩罚，即 ms-swift 内置奖励函数中的 `soft_overlong` 惩罚（详见第二十三章），其参数 `soft_max_length`（对应论文中的 $L_{\max}$）与 `soft_cache_length`（对应论文中的 $L_{\text{cache}}$）共同定义了惩罚区间 $[L_{\max}-L_{\text{cache}}, L_{\max}]$，在该区间内线性施加 $[-1,0]$ 的惩罚。

### 13.3 GSPO：序列级重要性采样

GSPO（Group Sequence Policy Optimization，阿里巴巴 Qwen 团队，*Group Sequence Policy Optimization*，2025，[arXiv:2507.18071](https://arxiv.org/abs/2507.18071)）针对的是原始 GRPO（以及 PPO）中"token 级重要性采样"这一设计的理论合理性问题。GSPO 论文指出：由于强化学习中的奖励通常是在**序列级别**（整个回答）给出的，而非逐 token 给出的，理论上更合理的做法应当是在序列级别计算重要性采样比率，而非对每个 token 单独计算——因为每个 token 在采样过程中只被采样了一次，逐 token 计算的重要性采样比率无法真正实现"分布修正"的统计学意义，反而会在长序列生成中引入高方差噪声，容易导致训练不稳定甚至崩溃。GSPO 提出的序列级重要性采样比率为：

$$
w_i^{\text{GSPO}} = \left[\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}\right]^{\frac{1}{|y_i|}} = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\right)
$$

即取序列级别似然比的几何平均（等价于逐 token 对数比率的算术平均后再取指数）。此外，论文还提出了 GSPO-token 变体，通过"停止梯度"（stop-gradient）技巧，将序列级别的重要性权重与 token 级别的梯度传播解耦，使得在未来支持细粒度（token 级）优势函数时具备更好的兼容性：

$$
w_{i,t}^{\text{GSPO-token}} = \text{sg}\big[w_i^{\text{GSPO}}\big] \cdot \frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\text{sg}\big[\pi_\theta(y_{i,t}|x,y_{i,<t})\big]}
$$

在当前"每个 token 优势相同（组内共享）"的设定下，可以证明 GSPO-token 与 GSPO 在梯度上是理论等价的。ms-swift 通过 `--importance_sampling_level {token, sequence, sequence_token}` 参数在 GRPO（token 级，默认）、GSPO（序列级）、GSPO-token 三种模式之间自由切换，并在文档中给出了论文推荐的配套超参数（如 `--epsilon 3e-4`、`--epsilon_high 4e-4`、`--beta 0` 即关闭 KL 正则、`--steps_per_generation 32` 等），便于用户直接复现论文结果。

### 13.4 CISPO：截断重要性采样策略优化

CISPO（Clipped Importance Sampling Policy Optimization，出自 MiniMax 团队 *MiniMax-M1* 等相关技术报告）提出了一种与 PPO/GRPO 截然不同的截断思路：不再截断"梯度更新的幅度"（即不对 $\rho_{i,t}\hat{A}_{i,t}$ 这一整体乘积做截断），而是仅对重要性采样权重本身进行截断（且截断操作被 detach，不参与梯度计算），同时将 log 似然本身保留在梯度路径中：

$$
\mathcal{L}_{i,t}^{\text{CISPO}} = -\,\text{detach}\big(\min(\rho_{i,t}, \epsilon_{\text{high}})\big) \cdot A_{i,t} \cdot \log\pi_\theta(y_{i,t}|y_{i,<t})
$$

这一设计的动机是：PPO/GRPO 式的截断在 $\rho_{i,t}$ 超出截断区间时，会将该 token 的**梯度直接置零**（因为 `min` 操作选择了梯度为零的截断分支），意味着一旁超出截断范围的 token 完全无法继续从该次更新中学习，即使其优势信号非常有价值。而 CISPO 通过将截断操作从梯度路径中分离（detach），使得截断只影响该 token 对总损失的"贡献权重"，而不会完全切断该 token 的梯度流动，从而保留了对所有 token（包括那些重要性采样比率较大的 token）的学习能力，这对充分利用长链推理场景中稀疏但关键的探索性 token 尤为重要。

### 13.5 SAPO：软自适应策略优化

SAPO（Soft Adaptive Policy Optimization）针对 PPO/GRPO 中"硬截断"（hard clipping）机制的一个固有缺陷——截断边界处梯度不连续（分段函数在截断点处的导数发生跳变）——提出使用一个连续可导的温度调制 sigmoid 软门控函数替代硬截断：

$$
\mathcal{L}_{i,t}^{\text{SAPO}} = -g_{i,t}\cdot A_{i,t}, \qquad g_{i,t} = \sigma\big(\tau\cdot(\rho_{i,t}-1)\big)
$$

其中 $\tau$ 是控制门控函数陡峭程度的温度参数。当 $\tau \to \infty$ 时，该软门控函数退化为硬截断的阶梯函数；当 $\tau$ 较小时，门控函数变化更加平滑，从而在整个重要性采样比率的取值范围内都能提供连续、非零的梯度信号，有助于缓解硬截断导致的梯度"死区"问题，理论上有助于提升训练的平滑性与稳定性。

### 13.6 RLOO：REINFORCE Leave-One-Out

RLOO（*Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback*）重新审视了经典的 REINFORCE 算法，并结合"留一法"（leave-one-out）基线估计技术，提出了一种比 PPO 更简单、比原始 GRPO 更少偏差的优势估计方式。具体而言，对同一 prompt 采样出的 $K$ 个回答，第 $i$ 个回答的基线值使用**除自身之外其余 $K-1$ 个回答**奖励的均值来估计，而非使用包含自身在内的全体均值（GRPO 的做法）：

$$
b_i = \frac{1}{K-1}\sum_{j\ne i} R_j, \qquad \hat{A}_i^{\text{RLOO}} = R_i - b_i
$$

这一"留一法"设计使得基线估计与当前样本本身独立（不存在自相关偏差），是统计学中降低方差估计偏差的经典技巧，理论上比 GRPO 直接使用包含自身的组内均值作为基线具有更小的估计偏差，尤其在组内样本数 $K$ 较小时这一差异更为显著。此外 RLOO 不使用 PPO 式的截断重要性采样机制，而是直接使用完整的 REINFORCE 策略梯度，进一步简化了算法复杂度。

### 13.7 REINFORCE++

REINFORCE++（*REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models*）在经典 REINFORCE 算法基础上，融合了 PPO 中若干被证明行之有效的稳定性技巧（如逐 token KL 惩罚、优势归一化、梯度裁剪等），但去掉了 PPO 需要额外训练价值网络这一环节，旨在提供一种"无需 Critic、但保留 PPO 大部分稳定性收益"的轻量级强化学习算法，论文报告其在训练效率与鲁棒性（对不同质量的 prompt 与奖励模型的敏感度更低）方面均优于原始 REINFORCE 与部分 PPO 配置。

### 13.8 CHORD：在线策略与离线专家的动态加权融合

CHORD（*On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL via Dynamic Weighting*）关注的是一个更宏观的训练流程设计问题：在强化学习训练过程中，如何动态地融合"在线策略自我探索得到的经验（on-policy RL 数据）"与"来自更强专家模型的离线示范数据（off-policy SFT 式数据，例如更强模型生成的高质量解答）"。CHORD 提出使用一个随训练进度动态调整的加权系数，在训练初期更多地依赖专家示范数据（类似于 SFT 冷启动，帮助模型快速掌握基本的任务范式），随着训练推进逐渐加大在线强化学习信号的权重（让模型基于自身探索进一步精炼能力），从而兼顾了收敛速度与最终性能上限，是"SFT 冷启动 + 纯 RL 微调"这一传统两阶段流程的一种更加平滑、自适应的统一形式。

### 13.9 TreePO 与其他前沿方向

TreePO（*TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-based Modeling*）借鉴了树搜索（如蒙特卡洛树搜索）的思想，将原本"每个 prompt 独立采样 $G$ 个完全独立的回答"的朴素 rollout 方式，改造为具备共享前缀的树形展开结构——即多个候选回答在生成的前若干步共享相同的路径，仅在后续步骤分叉产生多样性，这样可以显著减少重复计算（共享前缀部分的 KV 缓存可以复用），在保持采样多样性的同时提升推理侧的计算效率，尤其适合于长链推理场景下 rollout 阶段计算开销极高的问题。除此之外，ms-swift 的 GRPO Advanced Research 模块还持续跟踪并集成了包括熵掩码（Entropy Mask，仅对高熵的"关键决策 token"计算策略梯度损失，源自论文 *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning*，该论文发现仅约 20% 的高熵"分岔点" token 对强化学习的最终效果起主导作用）、FIPO（Future-KL Influenced Policy Optimization）、DeepEyes（面向"图像中思考"多模态推理场景的强化学习方案）、REAL（将 RLVR 重新表述为分类问题的视角）、Router Replay（面向 MoE 模型专家路由一致性的强化学习修正技术）等一系列前沿研究工作，充分体现了 ms-swift 团队对学术界最新进展的快速跟踪与工程复现能力。

## 第十四章 训练-推理不一致问题与重要性采样校正

### 14.1 问题的根源

GRPO 类算法为了加速采样（rollout）阶段，普遍引入了 vLLM、SGLang 等高性能推理引擎来替代训练框架自带的（通常较慢的）生成逻辑。GRPO 算法的理论基础要求采样所用的策略与被更新的策略是同一个策略（on-policy 假设），理想情况下，通过权重同步机制，vLLM 中的模型权重应当与训练框架中的策略模型权重完全一致（$\pi_{\text{vLLM}} \equiv \pi_\theta$）。然而在实践中，即便权重已经完全同步，由于 vLLM 与训练框架（如 PyTorch/DeepSpeed/Megatron）在算子实现（如注意力机制的具体 CUDA kernel、数值精度处理、批处理与 padding 策略等）上存在细微差异，两者对同一输入计算出的下一 token 概率分布仍然会存在不可忽略的数值偏差，即 $\pi_{\text{vLLM}}(y|x) \ne \pi_\theta(y|x)$。这就导致实际参与梯度计算的"重要性采样比率"$\rho_t(\theta)=\pi_\theta/\pi_{\theta_{\text{old}}}$ 中，分母本应是采样时的策略概率，实际却被替换为了推理引擎给出的概率，使得 on-policy 假设被违反，从而引入额外的、来源于训练-推理数值差异的偏差，可能导致训练不稳定甚至性能崩溃。

### 14.2 重要性采样校正机制

ms-swift 借鉴了业界最新的相关研究（包括 DeepSeek-V3.2 技术报告中提出的方案），为这一问题提供了系统性的重要性采样（Importance Sampling, IS）校正机制。校正后的损失函数引入一个额外的权重项 $w(x,y)$，用以修正 $\pi_{\text{vLLM}}$ 与 $\pi_\theta$ 之间的分布差异：

$$
\mathcal{L}_{\text{corrected}} = -\mathbb{E}_{y\sim\pi_{\text{vLLM}}}\left[ w(x,y)\cdot \min\big(\rho_t(\theta)\hat{A}_t, \text{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\big) \right]
$$

该权重可以在 **token 级别**（$w_{i,t}^{\text{token}} = \pi_\theta(y_{i,t}|\cdot)/\pi_{\text{vLLM}}(y_{i,t}|\cdot)$）或**序列级别**（对 token 级对数比率取平均后再指数化，与 GSPO 的序列级重要性采样思路一致）计算。为了防止个别异常样本的重要性权重过大导致梯度爆炸，ms-swift 提供了两种权重控制策略：**截断（Truncate）**——将权重限制在 $[0,\tau]$ 区间内（$w_{\text{truncate}}=\min(w,\tau)$），保留所有样本但限制其影响；**掩码（Mask）**——直接丢弃权重超过阈值 $\tau$ 的 token/序列（$w_{\text{mask}} = w \cdot \mathbb{1}[w\le\tau]$）。二者与两种粒度（token/sequence）组合，形成了 `--rollout_importance_sampling_mode {token_truncate, token_mask, sequence_truncate, sequence_mask}` 四种可选模式，配合 `--rollout_importance_sampling_threshold`（默认 2）控制截断/掩码阈值。

### 14.3 一致性监控指标体系

即便不启用主动的重要性采样校正，ms-swift 也支持通过 `--log_rollout_offpolicy_metrics true` 开启一套完整的"训练-推理一致性"诊断指标体系（日志前缀为 `rollout_correction/`），用于监控训练过程中两个策略分布的偏离程度，这些指标包括：(1) **KL 散度**（直接估计量 `kl` 与低方差的 K3 估计量 `k3_kl`）；(2) **困惑度（PPL）比较**：分别计算训练策略与推理策略（rollout policy）在生成序列上的困惑度（`training_ppl`/`rollout_ppl`），并计算二者对数困惑度之差 `log_ppl_diff`（反映分布漂移的方向与程度）及比值 `ppl_ratio`；(3) **卡方散度（χ² Divergence）**：分别在 token 级别（`chi2_token`）与序列级别（`chi2_seq`，基于几何平均）度量重要性采样权重的方差大小，数值越大说明训练-推理不一致越严重、训练稳定性风险越高；(4) **有效样本量（Effective Sample Size, ESS）**：$\text{ESS} = 1/\mathbb{E}[(w/\mathbb{E}[w])^2]$，反映经过重要性采样加权后实际"有效利用"的样本比例，ESS 越接近 1 说明权重分布越均匀（越接近真正的 on-policy 状态）；(5) **IS 权重统计量**：如平均权重 `is_weight_mean`（理想值应接近 1.0）与被截断/掩码样本占比 `clipped_frac`。

### 14.4 离策略序列掩码（Off-Policy Sequence Masking）

除了上述基于重要性采样的连续型校正机制外，ms-swift 还实现了源自 DeepSeek-V3.2 技术报告的一种更为激进的离散型校正策略——离策略序列掩码。其核心思想是：当某个序列的当前策略与旧策略（rollout/behavior policy）之间的平均对数概率偏差 $\delta_i$ 超过阈值 $\tau$，**且**该序列的优势值为负（$\hat{A}_i<0$）时，直接将该序列从损失计算中完全剔除（掩码）。这一设计的直觉是：负优势样本（模型被要求"降低"其生成概率的样本）本身在策略偏移较大时的梯度方向最不可靠、最容易引发训练不稳定，因此优先对这类"高风险"样本进行保守处理，而对正优势样本（无论偏移程度如何）予以保留，从而在风险控制与样本利用效率之间取得平衡。该机制通过 `--off_policy_sequence_mask_delta`（默认 None，即禁用）参数控制阈值 $\tau$。

## 第十五章 知识蒸馏对齐 GKD 与其他前沿方向

### 15.1 GKD：广义知识蒸馏

GKD（Generalized Knowledge Distillation，源自 Google DeepMind 论文 *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*，[arXiv:2306.13649](https://arxiv.org/abs/2306.13649)）是一种介于传统监督式知识蒸馏与强化学习之间的对齐/压缩范式，其目标是让一个较小的学生模型（student）模仿一个更大的教师模型（teacher）的行为分布。与传统蒸馏（在固定数据集上最小化教师与学生输出分布的 KL 散度）不同，GKD 强调"On-Policy"式的蒸馏：让学生模型自己生成回答（而非仅在教师生成的固定数据上训练），再用教师模型对学生自己生成的序列打分（计算教师在该序列上的对数概率分布），从而使学生模型学习"如何从自己实际会犯的错误中纠正"，而非仅仅模仿教师在其"完美轨迹"上的行为——这一思想与强化学习中"从自身探索的经验中学习"高度一致，也是为何 GKD 常被归入广义对齐算法家族、并与 GRPO 等强化学习算法共享许多工程实现细节（如需要教师模型进行推理评分、可复用 vLLM 采样加速等）的原因。ms-swift 通过统一的 `--rlhf_type gkd` 入口提供支持，其 KL 正则系数 `beta` 默认取值为 0.5，显著高于 DPO/GRPO 等算法的默认值，这与 GKD 更强调"贴近教师分布"而非"仅追求相对偏好"的设计目标是一致的。

### 15.2 强化微调（RFT）与 Reject Sampling Fine-Tuning

强化微调（Reinforced Fine-Tuning, RFT）是另一种介于 SFT 与 RL 之间的轻量级对齐范式，其基本思路是：对每个训练样本使用当前模型采样多个候选回答，通过规则或奖励模型筛选出其中正确（或高质量）的回答，再将这些筛选出的"自生成正确样本"重新用作 SFT 数据对模型进行监督微调，如此迭代多轮。这一范式也被称为 "Reject Sampling Fine-Tuning"（拒绝采样微调）或 "STaR"（Self-Taught Reasoner，出自同名论文），其相比 GRPO 等在线强化学习方法工程实现更为简单（本质上退化为多轮 SFT），但通常样本效率和最终性能上限低于完整的强化学习方法。ms-swift 在其"Reinforced Fine-Tuning"文档模块中对这一范式的适用场景、实现方式与实验结果进行了专门介绍，将其定位为强化学习之外的一种更轻量的对齐补充手段，尤其适合计算资源受限、或希望快速验证奖励设计合理性的场景。

## 第十六章 多模态对齐与 Constitutional AI / RLAIF 学术脉络

### 16.1 多模态大模型对齐的特殊挑战

将上述对齐算法应用于多模态大模型（Multimodal LLM, MLLM，涵盖图文、视频、音频等多种模态）时，会面临若干纯文本场景所不具备的特殊挑战：(1) **模态幻觉（Multimodal Hallucination）**：模型可能描述图像中并不存在的物体、属性或关系，这是多模态对齐要着重解决的核心问题之一；(2) **跨模态对齐质量的度量困难**：人类标注员在对比两个候选回答的偏好时，需要同时考虑文本表达质量与其对视觉/听觉信息描述的准确性，标注一致性与标注成本均高于纯文本场景；(3) **模态特定的安全风险**：如图像中隐含的不安全内容、图文结合的越狱攻击等，需要专门设计的安全对齐数据与评测体系。MM-RLHF 论文（*MM-RLHF: The Next Step Forward in Multimodal LLM Alignment*，2025，[arXiv:2502.10391](https://arxiv.org/abs/2502.10391)，项目主页 [mm-rlhf.github.io](https://mm-rlhf.github.io/)）针对性地构建了一个包含 12 万条精细化人类标注偏好比较对的多模态偏好数据集，覆盖对话能力、感知准确性、安全性等 10 个评价维度、27 个基准测试，并提出了两项关键技术创新：其一是"基于评述的奖励模型"（Critique-Based Reward Model），即让奖励模型在给出标量分数之前，先生成一段对候选回答的自然语言评述（critique），这一"先评述、后打分"的设计相比传统仅输出标量分数的奖励模型具有更强的可解释性，且论文实验表明其打分准确性也显著优于传统方式；其二是"动态奖励缩放"（Dynamic Reward Scaling），根据奖励信号的置信度动态调整每个样本在损失函数中的权重，从而更充分地利用高质量的比较数据、降低低置信度比较数据对训练造成的噪声干扰。该论文报告，将 MM-RLHF 数据集与上述对齐算法应用于 LLaVA-ov-7B 模型后，对话能力提升了 19.5%，安全性提升幅度高达 60%，充分证明了系统性多模态偏好对齐工作的实际价值。ms-swift 对多模态大模型的 RLHF 训练提供了原生支持（详见第二十七章），其 DPO、GRPO 等训练流程均可直接应用于视觉-语言、全模态（Omni）等多模态模型，且在 GRPO Advanced Research 模块中还专门集成了 DeepEyes 等面向"图像中思考"（Thinking with Images）这一新兴多模态推理场景的强化学习方案。

### 16.2 Constitutional AI（CAI）与 RLAIF

Anthropic 在 2022 年提出的 Constitutional AI（*Constitutional AI: Harmlessness from AI Feedback*，[arXiv:2212.08073](https://arxiv.org/abs/2212.08073)）代表了对齐技术中"减少对人类标注依赖"这一方向的重要探索。CAI 的核心思想分为两个阶段：(1) **监督式自我批评与修订阶段**：给定一份人类预先撰写的"宪法"（constitution，即一组高层次的行为原则），模型首先针对可能存在问题的初始回答，依据宪法条款自我生成批评意见（critique），再依据批评意见对回答进行自我修订（revision），并使用修订后的回答对模型自身进行 SFT；(2) **基于 AI 反馈的强化学习阶段（RLAIF）**：使用同一个（或另一个）模型依据宪法对成对回答进行偏好比较，替代人类标注员生成偏好数据，再基于这些"AI 反馈"训练奖励模型并进行标准的 RLHF 流程。CAI 的意义在于证明了：借助模型自身的语言理解与推理能力，并配合一份显式、透明、可审查的行为准则文档，可以在显著减少人工标注成本的同时，实现与传统人工标注 RLHF 相当甚至更好的安全对齐效果，同时"宪法"文本本身相较于隐式的人类偏好标注更具透明度与可解释性，这一思路也被后续大量工作（包括 Google 的 RLAIF 论文，*RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*，[arXiv:2309.00267](https://arxiv.org/abs/2309.00267)）进一步验证和推广。尽管 ms-swift 本身作为训练框架并不内置某个特定的"宪法"文本或专门的 CAI 训练流程，但 CAI/RLAIF 这一"用 AI 模型自身或另一强模型替代人类标注员"的思想，与 ms-swift 中广泛支持的"生成式奖励模型"（GenRM）、"自定义奖励函数可调用外部大模型 API 进行打分"（第二十三章将展开介绍的异步奖励函数机制）等工程能力高度契合，用户完全可以基于 ms-swift 现有的 Reward Function/Reward Model 插件机制，自行实现一套完整的 CAI/RLAIF 式对齐训练流程。

# 第四部分 ms-swift 框架工程实现深度剖析

## 第十七章 ms-swift 项目总览与生态定位

### 17.1 项目基本信息

ms-swift（ModelScope SWIFT，全称 Scalable lightWeight Infrastructure for Fine-Tuning）是阿里巴巴通义实验室 / ModelScope 社区开源的大模型与多模态大模型训练、推理、评测与部署一体化框架，代码仓库地址为 `https://github.com/modelscope/ms-swift`。根据仓库 README 的介绍，该项目支持使用 PEFT（参数高效微调，如 LoRA、QLoRA、DoRA 等）或全参数训练方式，对 600+ 纯文本大语言模型（涵盖 Qwen 系列、DeepSeek 系列、GLM 系列、InternLM 系列、Llama 系列等主流开源模型家族）与 300+ 多模态大模型（涵盖 Qwen-VL/Omni 系列、InternVL 系列、Ovis 系列、GLM-V 系列、Gemma 多模态版本、LLaVA、Phi 多模态版本等）进行持续预训练（CPT）、有监督微调（SFT）、直接偏好优化（DPO）、组相对策略优化（GRPO）等各类训练任务。该项目对应的学术论文《SWIFT: A Scalable lightWeight Infrastructure for Fine-Tuning》已被 AAAI 2025 接收（[arXiv:2408.05517](https://arxiv.org/abs/2408.05517)），是学术界正式认可的、具有完整方法论支撑的开源训练框架；框架官方文档见 [swift.readthedocs.io](https://swift.readthedocs.io/)，代码仓库见 [github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift)。

### 17.2 版本演进与发布节奏

从公开的更新日志可以观察到，ms-swift 项目保持着极高频率的功能迭代节奏，几乎每周都有新特性发布，其中与"对齐"直接相关的重要更新包括（节选自 README 更新记录）：支持 Reranker（重排序）模型微调；支持 GKD（广义知识蒸馏）训练（同时覆盖纯文本与多模态模型）；支持使用 Megatron 并行技术进行 RLHF 训练；支持在预训练/SFT/DPO/GRPO 全流程中使用序列并行（Sequence Parallel）技术；GRPO 支持针对奖励模型的自定义处理逻辑（GenRM 示例）；支持多轮 GRPO 训练以适配多轮对话与 Agent 工具调用场景等。这种高频迭代的特点，使得 ms-swift 能够几乎与学术界前沿论文的发布保持同步（如本报告第十三章介绍的 DAPO、GSPO、CISPO 等算法均在论文发表后的较短时间内即被集成进框架），这也是本报告选择 ms-swift 作为对齐技术工程实现研究对象的重要原因之一。

### 17.3 生态定位：训练全流程一体化平台

从生态定位角度看，ms-swift 并非一个仅聚焦于单一训练范式（如仅支持 LoRA 微调，或仅支持强化学习）的专用工具，而是致力于成为覆盖"预训练—微调—对齐—评测—量化—部署"全生命周期的一体化平台。其核心能力矩阵包括：(1) **多样化的模型类型支持**：纯文本 LLM 与多模态 MLLM 统一支持，是国内开源社区中模型覆盖面最广的训练框架之一；(2) **多样化的硬件后端支持**：兼容 CPU、消费级 RTX 系列显卡、T4/V100、A10/A100/H100 等数据中心级 GPU、华为昇腾（Ascend）NPU、Metax（沐曦）、AMD GPU 以及 Apple MPS 等多种硬件后端；(3) **多样化的并行训练技术**：集成 DDP、Device Map、DeepSpeed ZeRO-2/ZeRO-3、FSDP 等数据并行/模型并行技术，并进一步整合 Megatron-LM 的张量并行、流水线并行、序列并行与专家并行技术（用于 MoE 模型的高效训练）；(4) **完整的对齐训练能力矩阵**：即本报告重点研究的 DPO/ORPO/SimPO/CPO/KTO/RM/PPO/GRPO/GKD 九大类算法，以及围绕 GRPO 展开的十余种前沿工程变体；(5) **推理部署一体化**：与 vLLM、SGLang、TGI 等推理引擎深度集成，训练产出的模型可以直接用于高性能推理部署，尤其是在 GRPO 等在线强化学习场景下，这种"训练-推理一体化"设计直接决定了采样效率，是框架工程实现中的核心难点之一（详见第二十二章）。

### 17.4 与同类框架的定位比较（预览）

在正式展开 ms-swift 对齐模块的架构剖析之前，有必要先简要说明其在整个开源生态中的定位（详细的横向对比分析见第二十九章）：HuggingFace TRL 是最早、也是目前使用最广泛的偏好对齐训练库之一，其优势在于与 HuggingFace 生态（Transformers、PEFT、Accelerate 等）的无缝集成与简洁易用的 API 设计，但在超大规模分布式训练（尤其是 Megatron 级别的模型并行）与前沿 GRPO 变体的覆盖广度上通常滞后于 ms-swift；OpenRLHF 与 veRL（HybridFlow）是专注于大规模强化学习训练效率的框架，二者在 rollout 与训练解耦、异步流水线设计等方面有独到的工程优化，但在模型覆盖面（尤其是国产模型与多模态模型的适配）与开箱即用的算法丰富度上，与 ms-swift 相比各有侧重。ms-swift 的差异化优势在于：其模型覆盖面来自 ModelScope 社区对国产大模型生态的深度耕耘，其对齐算法覆盖面则得益于团队对学术前沿的高频跟踪与快速工程复现能力，二者叠加使其成为国内开发者进行大模型对齐训练时的主流选择之一。

## 第十八章 整体架构：命令行体系、参数系统与 Plugin 机制

![ms-swift对齐训练模块整体架构](./assets/08_msswift_architecture.png)

### 18.1 统一命令行入口设计

ms-swift 采用了"动词式子命令"的命令行设计范式，核心训练相关命令包括：`swift pt`（继续预训练）、`swift sft`（有监督微调）、`swift rlhf`（统一的人类偏好对齐入口，通过 `--rlhf_type` 参数区分具体算法）、`swift rollout`（专用于 GRPO 场景下部署 vLLM 推理服务以加速采样）、`swift infer`（推理）、`swift export`（模型导出，包括 LoRA 合并、量化、推送至模型仓库等）、`swift eval`（模型评测）、以及面向 Megatron 并行训练的 `megatron sft` / `megatron rlhf` 系列命令。这种设计的核心优势在于：所有训练阶段（预训练、SFT、九大类对齐算法）共享同一套底层参数体系与数据处理管线，用户只需切换 `--rlhf_type` 参数值，即可在不同对齐算法之间自由切换，而无需学习各自独立的 API 或配置文件格式，这与 HuggingFace TRL 中每种算法对应独立 Trainer 类（DPOTrainer、PPOTrainer、GRPOTrainer 等，各自 API 存在细节差异）的设计形成鲜明对比。

### 18.2 `--rlhf_type` 参数与算法分发机制

`swift rlhf` 命令的核心参数 `rlhf_type` 支持九个取值：`dpo`（默认值）、`orpo`、`simpo`、`kto`、`cpo`、`rm`、`ppo`、`grpo`、`gkd`。框架内部依据这一参数值，将训练任务分发至对应的 Trainer 实现（各 Trainer 内部又与 HuggingFace TRL 库中的同名 Trainer 存在密切的复用与继承关系，同时针对多模态输入、Megatron 并行、vLLM 集成等场景进行了大量适配性改造）。与 `rlhf_type` 密切相关的另一个重要参数是 `ref_model`：对于 DPO、KTO、PPO、GRPO 等需要参考模型的算法，当采用全参数训练（`--train_type full`）时，必须显式指定 `--ref_model`（否则无法区分策略模型与参考模型，因为二者共享同一份权重存储路径）；而当采用 LoRA 等 PEFT 训练方式时，由于原始模型权重被冻结，仅有新增的 adapter 参数参与训练，因此可以通过"临时禁用 adapter"的方式直接复用同一份底层权重作为参考模型，无需额外加载一份完整的模型权重，这是 LoRA 训练相比全参数训练在 RLHF 场景下能够显著节省显存的重要原因之一。

### 18.3 Plugin 机制与可扩展性设计

为了在保持核心代码简洁的同时支持高度定制化的训练需求（尤其是 GRPO 场景下多样化的奖励函数、多轮对话状态管理逻辑等），ms-swift 设计了一套 Plugin（插件）机制：用户可以在独立的 Python 文件（如 `examples/train/grpo/plugin/plugin.py`）中定义自定义的奖励函数类（继承自框架提供的 `ORM` 或 `AsyncORM` 基类）、自定义的多轮对话环境类等，并通过 `--external_plugins <plugin_file_path>` 参数将该文件注册到训练进程中，随后即可在 `--reward_funcs` 等参数中直接引用插件中定义的自定义组件名称。这一设计使得框架核心代码与用户自定义逻辑实现了良好的解耦：框架维护者无需为每一种可能的自定义奖励函数或环境逻辑修改核心代码，用户也无需 fork 整个代码仓库即可实现高度定制化的训练流程，这是 ms-swift 能够快速响应学术界层出不穷的新型奖励设计、并保持代码库整体简洁可维护的关键工程设计。

### 18.4 与 Megatron 并行体系的桥接：Bridge 机制

对于超大规模模型（如百亿、千亿参数级别）的全参数 RLHF 训练，仅依赖 DeepSpeed ZeRO 或 FSDP 等数据并行式分片技术往往难以满足显存与通信效率的要求，此时需要引入 Megatron-LM 风格的张量并行（Tensor Parallelism）、流水线并行（Pipeline Parallelism）等模型并行技术。然而，Megatron-LM 原生的模型权重格式（mcore 格式）与 HuggingFace Transformers 生态的权重格式存在结构性差异，二者之间的转换在处理多模态模型、MoE 模型等复杂结构时尤为繁琐。为此，ms-swift 提供了一套"Mcore Bridge"桥接机制，负责在 HuggingFace 格式权重与 Megatron mcore 格式权重之间进行双向、自动化的转换，并支持将转换后的 mcore 格式权重直接以内存（而非磁�入）方式传递给 vLLM 等推理引擎用于 GRPO 场景下的权重同步，官方文档中提到的 `--offload_bridge true` 参数（将桥接转换过程中产生的 HF 格式权重存储在 CPU 主存而非 GPU 显存中，以降低 GRPO 训练时权重同步阶段的显存峰值）正是这一桥接机制在工程实现上的一个具体优化点，将在第二十五章详细展开。

## 第十九章 对齐训练数据格式规范详解

### 19.1 三大类数据格式总览

根据 ms-swift 官方 RLHF 文档，九大类对齐算法根据其理论基础的不同，可以归纳为三种不同的数据格式需求：

**格式一：仅需模型输入（Prompt-Only）**——适用于 PPO 与 GRPO 算法。数据集仅需包含系统提示（system，可选）与用户查询（query/messages），不需要预先提供任何"标准答案"式的回复内容（回复由模型在训练过程中在线生成）。若使用 GRPO 中依赖标准答案的奖励函数（如 `accuracy` 准确率奖励、`cosine` 长度感知奖励），则数据集还需额外包含一列 `solution`（标准答案）作为奖励计算的参考依据；数据集中的其他自定义列，则会作为关键字参数（`**kwargs`）原样传递给自定义奖励函数，供其按需使用。典型的 GRPO 数据格式如下（以 JSONL 形式组织）：
```json
{"messages": [{"role": "user", "content": "Tell me tomorrow's weather"}]}
{"messages": [{"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}]}
```
需要特别注意的是，即便是多轮对话数据，最后一轮的 `assistant` 回复内容也不应包含在训练数据中——因为这正是模型需要在线生成、并被奖励函数评估的部分；而中间轮次的历史对话（如上例第二条数据中第一轮的 user/assistant 配对）则作为上下文正常保留。

**格式二：成对偏好数据（Pairwise Preference）**——适用于 RM、DPO 及 DPO 家族中的 ORPO、CPO、SimPO 等算法。数据格式为三元组 $(x, y_w, y_l)$，其中 $x$ 是模型输入（可包含系统提示与多轮历史），$y_w$（chosen）是符合人类偏好的回复，$y_l$（rejected）是不符合人类偏好的回复。

**格式三：带二元标签的单回复数据（Single Response with Binary Label）**——专用于 KTO 算法。数据格式为三元组 $(x, y, \text{label})$，其中 $\text{label}$ 是一个布尔值，标识该回复 $y$ 相对输入 $x$ 是"值得偏好"（true）还是"应当被拒绝"（false）。这一数据格式的最大优势在于其标注方式更接近人类的自然判断习惯（对单个回答做"好/坏"二元判断，而非同时比较两个回答并排序），因此标注成本显著低于成对比较数据，这也是 KTO 论文的核心卖点之一（详见第十章 10.2 节）。

### 19.2 多模态数据格式扩展

对于多模态大模型的对齐训练（详见第二十七章），上述数据格式在保持整体结构不变的基础上，会在 `messages` 字段的用户输入部分嵌入图像、视频、音频等多模态占位符（如 `<image>`、`<video>`），并在数据条目中额外提供对应的媒体文件路径（`images`、`videos`、`audios` 等字段）。ms-swift 的多模态数据处理管线会自动完成媒体文件的加载、预处理（如图像的分辨率归一化、视频的关键帧采样）以及占位符与实际多模态特征的对齐替换，这一整套处理逻辑与纯文本 SFT/DPO/GRPO 训练共享同一套底层数据处理框架，用户在切换纯文本模型与多模态模型进行对齐训练时，训练脚本的整体结构几乎不需要改动，仅需替换 `--model` 参数指向对应的多模态模型即可。

### 19.3 数据集来源与格式转换工具

ms-swift 默认从 ModelScope 社区下载模型与数据集（若需切换至 HuggingFace 社区，需额外指定 `--use_hf true`），同时也完全支持用户提供本地数据集文件（JSON、JSONL、CSV 等格式）或直接指向本地文件路径。对于社区中广泛存在的、格式与 ms-swift 原生格式不完全一致的开源偏好数据集（如各类以 `chosen`/`rejected` 字段命名的通用偏好数据集），ms-swift 的数据处理管线内置了较为宽松的字段名自动识别与转换逻辑，同时官方文档也提供了详细的自定义数据集接入指南，允许用户通过编写简单的数据预处理函数（Preprocessor）将任意格式的原始数据转换为框架内部统一的 `messages` 格式，这一设计充分保证了框架在面对社区海量、格式各异的开源数据集时的兼容性与易用性。

## 第二十章 分布式与并行技术支撑体系

### 20.1 数据并行技术：DDP、DeepSpeed ZeRO 与 FSDP

在中小规模模型（数十亿参数以内）的对齐训练场景下，ms-swift 主要依赖三种数据并行技术：(1) **DDP（Distributed Data Parallel）**：PyTorch 原生的数据并行方案，每张 GPU 保有完整的模型副本，仅对梯度进行 all-reduce 同步，适用于模型能够完整放入单卡显存的场景；(2) **DeepSpeed ZeRO（Zero Redundancy Optimizer）**：通过对优化器状态（ZeRO-1）、梯度（ZeRO-2）、模型参数（ZeRO-3）在数据并行维度上进行分片存储，显著降低单卡显存占用，是 ms-swift 中全参数训练（尤其是需要同时加载多个模型的 PPO/DPO/GRPO 场景）最常用的显存优化手段，`--deepspeed zero2`/`--deepspeed zero3` 是训练脚本中最常见的配置项之一；(3) **FSDP（Fully Sharded Data Parallel）**：PyTorch 原生的全分片数据并行方案，功能上与 DeepSpeed ZeRO-3 类似，是近年来 PyTorch 生态中日益重要的替代方案，尤其在与 PyTorch 原生编译优化（torch.compile）等特性协同时具备一定优势。

### 20.2 序列并行（Sequence Parallel）

针对长序列训练场景（如长链推理数据、长文档摘要等，序列长度可能达到数万甚至数十万 token），单纯的数据并行/模型并行难以有效降低单卡显存中"激活值"（activation）随序列长度线性增长带来的显存压力，因此需要引入序列并行技术：将同一个序列的不同片段分配到不同的 GPU 上进行并行计算，通过额外的通信开销换取更长序列的可训练性。ms-swift 在 2025 年 5 月的更新中，将序列并行支持范围从最初的预训练/SFT 扩展到了 DPO 与 GRPO 训练场景，这对于近年来长链推理模型训练中普遍存在的超长回复（数千至上万 token）场景尤为重要。

### 20.3 Megatron 并行技术体系

对于百亿参数级别以上的超大规模模型，ms-swift 通过 Megatron-SWIFT 子模块集成了 Megatron-LM 的完整并行技术体系，包括张量并行（Tensor Model Parallelism，将单层内部的矩阵运算切分到多张 GPU 上）、流水线并行（Pipeline Model Parallelism，将模型的不同层分配到不同 GPU 上，通过流水线调度重叠计算与通信）、专家并行（Expert Parallelism，专用于 MoE 混合专家模型，将不同的专家网络分配到不同 GPU 上）以及序列并行等技术的组合使用。根据官方文档给出的 Qwen3-30B-A3B（MoE 模型）全参数微调性能对比数据，Megatron-LM 方案相比 DeepSpeed-ZeRO3 方案，训练速度从 91.2 秒/迭代提升至 9.6 秒/迭代（近 10 倍加速），显存占用也从 16 卡 × 80GiB（接近显存上限）降低至 16 卡 × 60GiB，这一显著的性能优势也是 ms-swift 将 Megatron 并行技术进一步扩展支持至 RLHF/GRPO 训练场景（详见第二十五章）的重要动因。

### 20.4 Ray 分布式调度支持

对于涉及多角色（如 GRPO 场景下的训练进程、vLLM 推理进程、奖励模型服务进程）协同工作的复杂强化学习训练流程，ms-swift 还提供了基于 Ray 的分布式调度支持（Ray Support 模块），允许用户将训练任务的不同角色（Megatron Ray、Swift Ray 等）灵活地调度到异构的计算资源池中，这对于超大规模、多机多卡的强化学习训练集群管理具有重要的工程价值，也是框架向更大规模生产级部署场景演进的重要基础设施。

### 20.5 vLLM/SGLang 推理引擎集成

在所有对齐算法中，GRPO（以及其他需要在线采样的算法如 PPO）对采样效率最为敏感——因为每一轮训练迭代都需要先用当前策略生成一批候选回答，这一"生成"过程如果使用训练框架自带的（通常未经过特别优化的）自回归解码逻辑，会成为整个训练流程的严重瓶颈。为此，ms-swift 将 vLLM（以及后续逐步扩展的 SGLang 等）高性能推理引擎深度集成到 GRPO/PPO 的训练流程中，形成了"Colocate（内部协同）模式"与"Server（外部异步）模式"两种部署形态，这是 ms-swift GRPO 工程实现中最为核心、也最为复杂的部分，将在下一章（第二十二章）中进行深入的专题剖析。

## 第二十一章 逐算法命令行与关键超参数详解

本章结合 ms-swift 官方训练脚本（`examples/train/rlhf/` 目录）与 RLHF 文档，对九大类对齐算法在 `swift rlhf` 命令行中的具体调用方式与关键超参数进行逐一梳理，力求做到"原理—参数—脚本"三者一一对应，便于读者直接对照实践。

### 21.1 DPO 训练脚本与参数解读

DPO 是 `--rlhf_type` 的默认取值，其典型 LoRA 训练脚本结构如下（节选自官方示例并适当精简）：

```bash
swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tuner_type lora \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --beta 0.1 \
    --loss_type sigmoid \
    --deepspeed zero2
```

其中 `--beta 0.1` 对应第九章推导中的 KL 正则系数 $\beta$；`--loss_type sigmoid` 是标准 DPO 损失（第十章介绍的 IPO/KTO-pair/robust 等变体均通过修改该参数启用）；由于采用 LoRA 训练，框架会自动通过"禁用 adapter"的方式复用同一份基座权重作为参考模型，无需显式指定 `--ref_model`；若采用全参数训练（`--tuner_type full`），则必须显式指定 `--ref_model`（通常指向与 `--model` 相同的路径）。此外，DPO 还支持 MPO（混合偏好优化）训练模式，通过同时设置多个 `--loss_type` 取值并配合 `--loss_weights` 参数指定各损失的加权系数来实现，具体脚本可参考官方仓库中的 `examples/train/rlhf/mpo.sh`。

### 21.2 ORPO 训练脚本与参数解读

```bash
swift rlhf \
    --rlhf_type orpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tuner_type lora \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 --lora_alpha 32 \
    --target_modules all-linear \
    --deepspeed zero2
```

需要特别注意：ORPO 的超参数 `lambda`（几率比损失权重，对应第十章 10.4 节公式中的 $\lambda$）在 ms-swift 中**复用了 `--beta` 参数名**进行传递（若未显式指定则使用框架默认值），这是官方文档中特别标注的一个"参数复用"细节，用户在从 DPO 脚本迁移到 ORPO 脚本时需要注意这一参数含义的切换，避免因参数名称相同而产生误解。由于 ORPO 不需要参考模型，其脚本中也无需（且不支持）配置 `--ref_model`。

### 21.3 SimPO 训练脚本与参数解读

```bash
swift rlhf \
    --rlhf_type simpo \
    --model Qwen/Qwen2.5-3B-Instruct \
    --tuner_type full \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --deepspeed zero2
```

SimPO 关键超参数：`--beta`（默认 2.0，注意其数值范畴与 DPO 的 0.1 量级显著不同，这是因为 SimPO 的隐式奖励基于**平均**对数概率而非**总**对数概率，数值尺度天然更小，需要更大的 $\beta$ 才能达到相当的正则化强度）、`--simpo_gamma`（奖励间隔项，默认 1.0）、`--cpo_alpha`（混合 CPO 式 NLL 损失以提升训练稳定性，默认 1.0；设为 0.0 则严格还原论文原始算法）。

### 21.4 CPO 训练脚本与参数解读

```bash
swift rlhf \
    --rlhf_type cpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tuner_type lora \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --learning_rate 1e-4 \
    --lora_rank 8 --lora_alpha 32
```

CPO 关键超参数：`--beta`（隐式奖励系数，默认 0.1）、`--cpo_alpha`（NLL 损失权重，默认 1.0）。与 ORPO 类似，CPO 同样不需要参考模型。

### 21.5 KTO 训练脚本与参数解读

KTO 使用 $(x, y, \text{label})$ 格式的数据（详见第十九章 19.1 节格式三），关键超参数包括 `--beta`（KL 正则系数，默认 0.1）、`--desirable_weight`（偏好样本损失权重 $\lambda_D$，默认 1.0）、`--undesirable_weight`（拒绝样本损失权重 $\lambda_U$，默认 1.0）。官方文档特别强调需要根据数据集中偏好/拒绝样本的数量比例来调整这两个权重，以满足第十章 10.2 节所述的 $\frac{\lambda_D n_D}{\lambda_U n_U} \in [1, \frac{4}{3}]$ 这一推荐范围。

### 21.6 RM 训练脚本与参数解读

```bash
swift rlhf \
    --rlhf_type rm \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tuner_type lora \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --learning_rate 1e-4 \
    --lora_rank 8 --lora_alpha 32
```

RM 训练使用与 DPO 相同的成对偏好数据格式，训练完成后会在输出目录中额外产出 `value_head.safetensors`（或 `.bin`）文件，保存新增的标量打分头权重，供后续 PPO 训练阶段加载使用。关键超参数 `center_rewards_coefficient`（对应第五章 5.2 节公式中的 $\lambda$，用于约束奖励绝对值趋近于零，默认 0）以及数据集中可选的 `margin` 列（用于实现难度自适应的间隔项）。

### 21.7 PPO 训练脚本与参数解读

```bash
swift rlhf \
    --rlhf_type ppo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_model <path-to-trained-reward-model> \
    --tuner_type full \
    --dataset <dataset> \
    --learning_rate 1e-6 \
    --kl_coef 0.05 \
    --cliprange 0.2 \
    --vf_coef 0.1 \
    --deepspeed zero3
```

PPO 训练需要额外指定 `--reward_model`（指向前一阶段 RM 训练产出的模型路径），框架会自动依据其权重初始化价值模型（value_model）。官方文档特别提醒：若训练对象是未经 SFT 的 base 模型，建议先进行 SFT 再进入 RLHF 阶段，并需要正确指定 chat template（对话模板），同时推荐使用全参数（`full`）训练方式而非 LoRA，这是因为 PPO 训练本身已经涉及四个模型的协同，若再叠加 LoRA 的适配器切换逻辑，会显著增加工程实现的复杂度与出错概率。

### 21.8 GRPO 训练脚本与参数解读

GRPO 的完整训练脚本与参数体系将在第二十二章进行专题深入剖析，此处仅给出核心命令行结构作为本章的收尾：

```bash
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --reward_funcs accuracy \
    --num_generations 16 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --temperature 1.0 \
    --top_p 0.85 \
    --deepspeed zero3
```

## 第二十二章 GRPO 工程实现深潜：vLLM Colocate/Server 双模式与显存优化

![vLLM Colocate与Server两种加速模式对比](./assets/09_vllm_colocate_server.png)

### 22.1 为什么 GRPO 训练需要高性能推理引擎

如第十二章所述，GRPO 训练的每一轮迭代都需要先对当前批次的 prompt 进行"组内多次采样"（例如 `num_generations=16` 意味着每个 prompt 需要生成 16 个候选回答），这一采样阶段本质上是一个纯粹的自回归解码（decode-only）过程，与训练阶段需要保存中间激活值以支持反向传播的前向计算存在本质区别。若直接复用训练框架（如 PyTorch/DeepSpeed）自身的生成逻辑进行采样，由于缺乏针对解码阶段的专门优化（如 KV Cache 的高效管理、PagedAttention 式显存复用、批处理请求的动态调度等），采样效率会远低于专门为推理场景设计的高性能引擎，成为整个训练流程的严重瓶颈。因此，ms-swift 选择将 vLLM（业界主流的高性能 LLM 推理引擎，以 PagedAttention 技术著称）深度集成到 GRPO 训练流程中，专门负责 rollout 阶段的高效采样。

### 22.2 Colocate（内部协同）模式

Colocate 模式下，训练进程与 vLLM 推理服务共享同一组 GPU 资源，vLLM 推理引擎在 Trainer 内部以进程内（in-process）方式启动，通过 `--use_vllm true --vllm_mode colocate` 启用。这种模式的优势是部署简单（无需额外启动独立的推理服务进程与网络通信），劣势是训练与推理需要共享有限的 GPU 显存资源，容易出现显存溢出（OOM）问题。为此，ms-swift 提供了一整套显存优化组合拳：

1. **降低 vLLM 显存占用比例**：通过 `--vllm_gpu_memory_utilization`（如设为 0.4）限制 vLLM 引擎可使用的显存上限，为训练过程预留足够显存空间。
2. **训练阶段释放 vLLM 显存（Sleep 机制）**：通过 `--sleep_level 1`，在进行策略梯度更新的训练阶段，主动将 vLLM 占用的显存"休眠"释放，待下一轮采样阶段再重新唤醒占用，从而实现训练与推理两个阶段在时间维度上对同一块显存的复用（而非同时占用）。
3. **模型/优化器状态卸载（Offload）**：通过 `--offload_model true`、`--offload_optimizer true`，在 vLLM 推理阶段将训练侧的模型权重与优化器状态临时卸载至 CPU 主存，进一步为 vLLM 采样腾出显存空间；配合 `--gc_collect_after_offload true` 主动触发垃圾回收，确保显存及时释放。
4. **vLLM 侧张量并行**：通过 `--vllm_tensor_parallel_size` 参数，使 vLLM 推理引擎本身也可以跨多张 GPU 进行张量并行，适用于单张 GPU 无法容纳完整模型推理所需显存的大模型场景。
5. **批量权重同步优化**：在 ZeRO-3 等参数分片场景下，训练侧更新完成后需要将最新权重同步给 vLLM 推理引擎，若一次性聚合全部参数分片再同步会造成显存峰值过高，`--move_model_batches` 参数支持将权重同步过程拆分为多个批次逐步进行，以时间换空间。
6. **Megatron 场景下的权重同步优化**：`--offload_bridge true`，将 Megatron mcore 格式权重转换为 HuggingFace 格式（供 vLLM 加载）这一桥接过程中产生的中间权重存储在 CPU 主存而非 GPU 显存中。

### 22.3 Async（外部异步）模式

Async（Server）模式下，训练进程与 vLLM 推理服务作为两个独立的进程（甚至可以部署在不同的物理机器上）运行，训练进程通过网络请求（HTTP/RPC）与远端 vLLM 服务通信以获取采样结果。部署方式为：首先使用专用的 `swift rollout` 命令单独启动一个（或多个，用于数据并行 `--vllm_data_parallel_size`）vLLM 推理服务：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
swift rollout \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --vllm_tensor_parallel_size 2 \
  --vllm_data_parallel_size 1
```

随后在训练脚本中通过如下参数指向该外部服务：

```bash
--use_vllm true \
--vllm_mode server \
--vllm_server_host <server_IP> \
--vllm_server_port <service_port> \
--vllm_server_timeout <timeout>
```

Async 模式的优势在于训练资源与推理资源实现了物理层面的解耦，可以针对两类负载的不同特性（训练侧对显存/算力要求高但吞吐相对稳定，推理侧对延迟与并发调度更敏感）分别进行独立的资源配置与弹性伸缩，尤其适合超大规模、多机多卡的生产级训练集群；劣势在于引入了额外的网络通信开销与部署复杂度（需要单独管理推理服务的生命周期、权重同步的一致性等）。官方文档中特别指出：当仅启用数据并行（DP）而未同时启用张量并行（TP）时，vLLM 异步引擎（`vllm_use_async_engine`）可能触发已知的上游 bug（关联 vLLM 项目 issue #18567），建议同时开启 TP+DP 或升级 vLLM 版本以规避该问题，这一细节体现了 ms-swift 文档在工程实践层面的细致程度。

### 22.4 LoRA 场景下的权重同步加速

对于采用 LoRA 训练的 GRPO 场景，若每一轮权重同步都传输完整的模型权重（LoRA 训练场景下基座权重本身并不更新，仅 adapter 权重更新），会造成巨大的通信浪费。为此，ms-swift 支持仅同步 LoRA adapter 权重的优化方案：在 `swift rollout`（Server 模式）侧设置 `--vllm_enable_lora true --vllm_max_lora_rank <rank>`（需与训练侧 `lora_rank` 保持一致），在训练侧（Colocate 或 Megatron GRPO 场景）设置 `--vllm_enable_lora true`，即可仅同步远小于完整模型权重的 adapter 参数，显著提升权重同步效率，官方文档同时提醒该方案会对 vLLM 推理速度带来轻微影响（因为 vLLM 需要在推理时动态应用 LoRA adapter，而非直接使用融合后的稠密权重）。对于多模态模型场景，若训练时视觉编码器（ViT）部分也启用了 LoRA（即 `--freeze_vit false`），还需要在 vLLM 侧通过 `--vllm_engine_kwargs '{"enable_tower_connector_lora": true}'` 额外开启对视觉塔（tower）与连接器（connector）模块 LoRA 的支持，官方文档标注这是一项目前仍处于实验阶段的 vLLM 特性，当前主要在 Qwen2.5-VL、Qwen3-VL 等模型上得到验证。

### 22.5 GRPO 训练监控指标体系

ms-swift 为 GRPO 训练设计了一套相当完善的训练动态监控指标体系，涵盖：(1) **生成长度相关**：`completions/mean_length`、`min_length`、`max_length`、`clipped_ratio`（因长度限制被截断的回复比例）；(2) **奖励相关**：按奖励函数名称分别记录的 `reward/{name}/mean`、`{name}/std`，以及经过奖励权重加权后的总体 `reward`、批内标准差 `reward_std`，还包括反映组内奖励多样性的 `frac_reward_zero_std`（组内标准差为零、意味着该 prompt 采样出的所有回复奖励完全相同，此类样本对训练梯度无贡献）；(3) **KL 散度与截断比例**：`kl`（仅在 `beta` 非零时记录）、`clip_ratio/region_mean`、`low_mean`/`low_min`、`high_mean`/`high_max`（分别统计上下界截断的触发比例）；(4) **熵相关（需设置 `log_entropy`）**：`entropy/mean`、`max`、`min`，若设置 `top_entropy_quantile<1.0` 还会记录熵阈值 `entropy/threshold`（用于熵掩码机制，详见第十三章 13.9 节）；(5) **训练-推理一致性相关**（第十四章已详述）：以 `rollout_correction/` 为前缀的一系列 KL、PPL、χ²、ESS、IS 权重统计指标。此外，设置 `--log_completions true` 后，框架会将每一步的训练动态（包括具体的 prompt、模型生成的完整回复、各奖励函数打分、token 熵等）保存至输出目录，并可配合 `--report_to wandb/swanlab` 将这些训练动态以可视化表格形式同步至实验跟踪平台，极大方便了研究人员对强化学习训练过程的细粒度调试与分析。

## 第二十三章 奖励函数与奖励模型体系

### 23.1 自定义奖励函数的开发范式

ms-swift 的奖励函数遵循统一的调用接口：接收模型生成的候选回复列表 `completions`，以及数据集中除 `messages` 之外的其他列（作为关键字参数）与训练状态（`trainer_state`，包含当前步数、总步数等信息），返回一个与 `completions` 等长的浮点数奖励列表。一个最简单的自定义奖励函数（基于长度的示例）实现如下：

```python
from swift.rewards import ORM, orms

class DummyLengthRewardFunction(ORM):
    def __call__(self, completions, **kwargs):
        return [1.0 if len(completion) > 1024 else 0.0 for completion in completions]

orms['dummy'] = DummyLengthRewardFunction
```

若奖励函数需要读取数据集中的额外列（如标准答案 `solution`），既可以在 `__call__` 方法的形参列表中显式声明该列名，也可以通过 `kwargs.get('solution')` 的方式动态读取；训练状态对象 `trainer_state` 同样可以类似方式获取，从而实现诸如"根据训练进度动态调整奖励权重"这类高级定制逻辑（这正是 CHORD、动态奖励缩放等前沿算法在工程落地时所依赖的底层机制）。自定义奖励函数编写完成后，需要将其放置于一个独立的插件文件中，并通过 `--external_plugins <plugin_file_path>` 参数注册，随后即可通过 `--reward_funcs dummy` 直接调用。

### 23.2 异步奖励函数

对于涉及网络 I/O（如调用外部大模型 API 进行打分、查询远程数据库等）的奖励函数，同步实现方式会因等待网络响应而严重拖慢训练进度。ms-swift 为此提供了异步奖励函数支持，用户只需继承 `AsyncORM` 基类并将 `__call__` 方法实现为协程（`async def`），框架会自动识别并使用 `asyncio.gather` 并行执行所有异步奖励函数的调用，同步与异步奖励函数还可以在同一次训练中混合使用（框架自动分别调度）。官方插件示例中提供了一个 `async_genrm`（异步生成式奖励模型）实现，展示了如何在奖励函数内部异步调用 `swift deploy` 部署的模型服务，为使用大模型本身作为"裁判"（LLM-as-a-Judge）进行奖励打分提供了直接可用的工程范式，这与第十六章讨论的 RLAIF/CAI 思想在工程实现上高度契合。

### 23.3 内置规则奖励函数详解

ms-swift 内置了五种基于规则的开箱即用奖励函数（源码位于 `swift/rewards/orm.py`），均直接对应特定学术论文中提出的奖励设计：

- **accuracy（准确率奖励）**：源自 DeepSeek-R1 论文，将模型生成内容中的答案部分使用 `math_verify` 库解析后与数据集 `solution` 列进行数学等价性比较，匹配则奖励 1.0，否则 0.0，主要适用于数学类任务。
- **format（格式奖励）**：同样源自 DeepSeek-R1 论文，检验模型输出是否严格遵循 `<think>推理过程</think><answer>答案</answer>` 这一预设格式模板，符合则奖励 1.0，否则 0.0，用于约束模型养成规范的"先思考后回答"输出习惯。
- **cosine（余弦长度感知奖励）**：源自论文 *Demystifying Long Chain-of-Thought Reasoning in LLMs*，通过余弦函数平滑地根据回复长度调节奖励值：对于答案正确的回复，长度越短奖励越高（鼓励简洁）；对于答案错误的回复，长度越长奖励越高（鼓励模型在得不到正确答案时进行更充分的探索性思考，而非过早放弃）。该奖励函数由四个边界值参数（`cosine_min/max_len_value_wrong/correct`）与最大长度参数 `cosine_max_len` 共同定义。
- **repetition（重复惩罚）**：同样源自上述论文，通过统计生成文本中 n-gram（默认 3-gram，由 `repetition_n_grams` 控制）的重复率来施加惩罚，重复率越高惩罚越大（惩罚上限由 `repetition_max_penalty` 控制，默认 -1.0），用于抑制强化学习训练中常见的"复读机"现象。
- **soft_overlong（软性超长惩罚）**：源自 DAPO 论文，在 `[soft_max_length - soft_cache_length, soft_max_length]` 这一长度区间内对超长回复施加线性递增的惩罚（范围 $[-1,0]$），相比直接截断丢弃超长样本的做法，能够为模型提供更平滑、更具信息量的长度控制信号。

### 23.4 奖励模型作为奖励来源

除规则函数外，GRPO 也支持直接使用一个（或多个）训练好的奖励模型作为奖励打分器，与规则奖励函数一样可以通过 `--reward_funcs`（对应奖励模型的调用被统一抽象为一种特殊的"奖励函数"）配置，并可以与规则奖励函数混合使用（通过 `--reward_weights` 参数为不同奖励来源分配加权系数）。GRPO 官方文档中的"奖励模型"（Reward Model）子模块，还专门支持对奖励模型的打分逻辑进行自定义处理（Custom Processing Logic for Reward Models），这一特性于 2025 年 5 月更新中引入，官方给出的 GenRM（生成式奖励模型）示例展示了如何让奖励模型先针对候选回复生成一段自然语言评述，再从评述文本中解析、提取出最终的标量分数，这一流程与 MM-RLHF 论文中"基于评述的奖励模型"（第十六章 16.1 节）在设计理念上一脉相承。

### 23.5 显存管理注意事项

官方文档特别提醒：若奖励函数内部需要加载一个独立的模型（如上述 GenRM 场景，或需要调用一个专门的分类器模型），在 DeepSpeed ZeRO-3 训练环境下，由于训练主循环已经激活了 DeepSpeed 的分布式插件（Accelerate 集成逻辑会自动将后续加载的任何模型也纳入 ZeRO-3 的分片管理），这可能导致奖励函数内部临时加载的模型无法正常执行独立的推理（因为其权重被错误地按照训练模型的分片规则切分）。针对这一问题，官方文档提供了绕过 DeepSpeed 环境初始化的具体解决方案（对应 GitHub issue #4580），是使用 ms-swift 进行 GRPO 二次开发时需要特别关注的一个工程细节。

## 第二十四章 多轮训练、多任务训练与 Agent/GYM 环境

### 24.1 多轮 GRPO 训练

自 2025 年 3 月起，ms-swift 支持多轮 GRPO 训练（Multi-turn Training），用于适配多轮对话、Agent 工具调用等场景下的强化学习需求。在这类场景下，模型的一次"决策"往往并非一次性生成完整回复，而是需要在生成过程中动态地与外部环境（如工具调用结果、多轮用户反馈）进行交互，模型需要根据环境返回的中间结果调整后续生成内容，最终的奖励也可能只在整个多轮交互的末尾才能确定（如判断整个 Agent 任务是否成功完成）。为支持这一场景，ms-swift 的 GRPO 模块提供了可扩展的多轮环境接口，允许用户自定义"环境响应逻辑"（即给定模型当前生成的内容，环境应当返回什么样的中间观测/工具调用结果，以及何时判定该轮交互结束），训练框架负责将这一多轮交互过程中产生的全部 token 序列正确地组织为强化学习所需的状态-动作-奖励轨迹，并处理好多轮场景下 KV 缓存复用、损失掩码（哪些 token 属于模型生成、哪些属于环境注入，只有前者才参与策略梯度损失计算）等一系列工程细节。

### 24.2 多任务训练

多任务训练（Multi-Task Training）模块允许用户在同一次 GRPO 训练中混合多个不同类型的任务（如数学推理任务与代码生成任务），每种任务可以配置各自独立的奖励函数组合与权重，框架负责按照用户指定的任务采样比例，在训练过程中动态混合来自不同任务的数据批次，这对于希望训练出"通才"式推理模型（而非仅擅长单一垂直领域）的场景具有重要价值，也是当前多个头部推理模型（如覆盖数学、代码、逻辑推理等多个领域的通用推理模型）训练方案中普遍采用的策略。

### 24.3 Agent 支持与工具调用

ms-swift 的 Agent Support 模块提供了完整的 Agent 训练数据格式规范（Tools Format）与配套的 `loss_scale` 使用指南，支持模型学习按照特定协议（如 OpenAI 兼容的 function calling 格式）生成工具调用请求、解析工具返回结果，并在此基础上进行下一步决策，这一能力可以与 SFT、DPO、GRPO 等各阶段训练自然衔接——例如先通过 SFT 让模型掌握基本的工具调用语法，再通过 GRPO 结合"任务是否成功完成"这一可验证的最终奖励信号，进一步强化模型的工具使用策略与多步规划能力。

### 24.4 GYM 环境训练

GYM Environment Training 模块借鉴了强化学习领域经典的 OpenAI Gym 环境接口设计范式，为 GRPO 训练提供了一套标准化的"环境"抽象接口，使得复杂的、有状态的交互式任务（如需要多步操作的软件工程任务、需要与模拟器交互的具身智能任务等）可以被统一地封装为符合该接口规范的环境类，并直接接入 GRPO 的训练流程，这一设计体现了 ms-swift 团队在支撑更加复杂、更加接近真实应用场景的强化学习训练需求方面所做的前瞻性工程布局。

## 第二十五章 Megatron-SWIFT：大规模对齐训练

### 25.1 Megatron GRPO 的独特价值

对于千亿参数级别的超大规模模型，即便使用 DeepSpeed ZeRO-3 等数据并行分片技术，单纯依靠数据并行维度的显存分摊也往往难以满足训练需求（尤其是叠加了 GRPO 场景下同时需要策略模型、参考模型、以及内部协同的 vLLM 推理引擎多重显存占用之后）。ms-swift 于 2025 年 6 月支持了基于 Megatron 并行技术的 RLHF 训练，将张量并行、流水线并行、专家并行等模型并行技术引入到 GRPO 等对齐算法的训练流程中，通过 `megatron rlhf --rlhf_type grpo` 命令行入口提供支持，这使得原本仅在预训练/SFT 阶段才能享受到的 Megatron 并行加速收益（如第二十章 20.3 节提到的近 10 倍训练速度提升），也能够被应用于对齐训练阶段，这对于业界头部机构在千亿参数模型上进行 RLVR 强化学习训练具有重要的工程意义。

### 25.2 Megatron 与 vLLM 的权重同步桥接

Megatron GRPO 训练中最核心的工程难点在于：训练侧使用 Megatron mcore 格式的模型权重（按照张量并行/流水线并行规则进行了复杂的切分），而负责采样加速的 vLLM 推理引擎需要的是标准的 HuggingFace 格式权重，二者之间的双向转换（尤其是需要在每一轮训练迭代之后，将刚刚更新过的 Megatron 格式权重实时转换并同步给 vLLM）构成了显著的工程挑战。ms-swift 通过 Mcore Bridge 机制解决这一问题：在训练循环内部，每次策略更新完成后，Bridge 组件自动将分片存储的 Megatron 权重重新聚合、转换为 HuggingFace 格式，并直接以内存传输（而非落盘再加载）的方式同步给 vLLM 引擎，这一转换过程本身也会消耗额外的临时显存，因此官方文档建议在显存紧张的场景下配合 `--offload_bridge true` 参数，将转换过程中的临时权重存储于 CPU 主存而非 GPU 显存中，以降低训练迭代过程中的显存峰值。

### 25.3 Megatron GKD 支持

除 GRPO 之外，ms-swift 的 Megatron-SWIFT 子模块也提供了对 GKD（广义知识蒸馏，第十五章 15.1 节）训练的原生支持，使得在教师模型（通常参数规模远大于学生模型）与学生模型均可能达到百亿甚至千亿参数级别的大规模蒸馏场景下，同样可以借助 Megatron 的并行技术体系高效完成训练，这对于当前业界"从超大规模旗舰模型蒸馏出高性价比的中小型部署模型"这一常见工程实践路径提供了直接的基础设施支撑。

## 第二十六章 前沿算法集成全景：从熵掩码到训练-推理一致性

为便于读者形成整体图景，本章以表格形式对 ms-swift GRPO 模块中集成的全部前沿研究（Advanced Research）方案进行汇总回顾，并标注各方案在 ms-swift 中对应的核心配置参数：

| 前沿方案 | 核心思想（详见对应章节） | ms-swift 核心配置参数 |
|---|---|---|
| 熵掩码 Entropy Mask | 仅对高熵"关键决策"token 计算策略梯度损失，聚焦约 20% 的分岔点 token | `--top_entropy_quantile` |
| CISPO | 截断重要性采样权重而非截断梯度整体，保留低概率高价值 token 的学习能力 | `--loss_type cispo` |
| DAPO | Clip-Higher 非对称截断、动态采样、token 级全局归一化、超长惩罚 | `--epsilon`/`--epsilon_high`、`--loss_type dapo`、`--reward_funcs soft_overlong` |
| GSPO | 序列级重要性采样比率替代 token 级，缓解长序列高方差问题 | `--importance_sampling_level sequence`/`sequence_token` |
| SAPO | 用连续可导的温度调制软门控替代硬截断 | `--loss_type sapo` |
| RLOO | 留一法基线估计，降低组内基线的自相关偏差 | 对应算法族内独立可选实现 |
| REINFORCE++ | 融合 PPO 稳定性技巧的轻量级无 Critic 强化学习 | 对应算法族内独立可选实现 |
| CHORD | 在线 RL 数据与离线专家数据的动态加权融合 | 相关训练流程配置 |
| TreePO | 树形共享前缀采样，兼顾多样性与推理效率 | 对应专题训练脚本 |
| Dr.GRPO | 固定分母（批次大小×最大长度）归一化，彻底消除长度偏差 | `--loss_type dr_grpo` |
| BNPO | Token 维度归一化（当前进程内） | `--loss_type bnpo` |
| 训练-推理一致性校正 | 基于重要性采样的 Truncate/Mask 校正机制，四种粒度组合模式 | `--rollout_importance_sampling_mode`、`--rollout_importance_sampling_threshold` |
| 离策略序列掩码 | 剔除策略偏移大且优势为负的高风险序列 | `--off_policy_sequence_mask_delta` |

这一表格集中体现了 ms-swift 团队在 GRPO 核心引擎之上构建的高度模块化、可插拔的算法研究基础设施——绝大多数前沿改进方案都被设计为独立的、可与其他方案自由组合的超参数选项，而非彼此互斥的整体训练模式切换，这使得研究人员可以在同一套代码基础上，便捷地开展诸如"DAPO 的动态采样 + GSPO 的序列级重要性采样 + 熵掩码"这类多方案组合消融实验，这种设计理念本身就是 ms-swift 区别于许多"论文配套开源代码"式实现（通常仅能复现单一论文的单一配置）的重要工程价值所在。

## 第二十七章 多模态大模型对齐支持

### 27.1 多模态 RLHF 的整体架构复用

ms-swift 官方仓库中特别设有 `examples/train/multimodal/rlhf` 目录，明确表明"多模态模型的 RLHF 训练同样得到完整支持"。从架构设计上看，多模态对齐训练与纯文本对齐训练共享几乎完全相同的 Trainer 实现与命令行参数体系，核心差异仅体现在：(1) 数据处理管线需要额外处理图像/视频/音频等多模态输入的加载、预处理与特征编码；(2) 模型前向传播过程中需要正确处理视觉编码器（ViT）、跨模态连接器（Connector/Projector）与语言模型骨干三部分的协同计算；(3) 在涉及 LoRA 微调时，需要分别控制语言模型骨干、视觉编码器、连接器三部分是否分别启用/冻结 LoRA（通过 `--freeze_vit`、`--freeze_aligner` 等参数精细控制）。

### 27.2 多模态 GRPO 训练与 vLLM 协同的特殊考量

对于多模态模型的 GRPO 训练，vLLM 权重同步环节需要额外处理视觉编码器部分的权重同步（而不仅仅是语言模型骨干），这在第二十二章 22.4 节中已经提及：若训练时视觉塔部分也启用了 LoRA（`--freeze_vit false`），则需要在 vLLM 侧通过 `vllm_engine_kwargs` 显式开启对应的实验性特性支持。此外，多模态奖励函数的设计也需要考虑跨模态因素，例如面向"图像中思考"（Thinking with Images）场景的 DeepEyes 方案（第十三章 13.9 节），其奖励设计需要同时评估模型对图像内容的定位准确性与最终文本回答的正确性，这类复合型多模态奖励函数的实现同样遵循第二十三章介绍的自定义 `ORM` 接口规范。

### 27.3 多模态对齐的数据与评价体系

如第十六章 16.1 节所述，高质量的多模态偏好数据集（如 MM-RLHF）是多模态对齐训练效果的重要保障。ms-swift 官方文档与最佳实践（Best Practices）模块中的"完整多模态 GRPO 实验流程"（GRPO-Multi-Modal-Training）章节，为用户提供了从数据准备、奖励函数设计到训练脚本配置的端到端指南，充分体现了框架团队对多模态对齐这一日益重要的应用场景的持续投入。

## 第二十八章 实战案例：从 SFT 到 GRPO 的完整训练流程

本章以 Qwen3-8B 模型为例，结合官方最佳实践文档给出的训练脚本，串联展示一次完整的"预训练模型→指令微调→强化学习对齐"全流程，帮助读者建立对 ms-swift 对齐训练实际落地过程的整体认知。

### 28.1 第一步：环境准备

```bash
pip install ms-swift -U
pip install transformers -U
pip install deepspeed          # 多卡训练
pip install liger-kernel       # 节省显存
pip install flash-attn --no-build-isolation
pip install "math_verify==0.5.2"   # GRPO 数学答案校验
pip install vllm               # GRPO 采样加速
```

### 28.2 第二步：有监督微调（SFT）

```bash
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type lora \
    --dataset 'swift/Qwen3-SFT-Mixin#2000' 'swift/self-cognition:qwen3#600' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
    --output_dir output
```

该阶段产出一个具备基本指令遵循能力（并可选地注入自我认知信息）的 LoRA adapter，训练完成后可通过 `swift export --adapters output/checkpoint-xxx --merge_lora true` 合并为完整权重，作为后续 GRPO 阶段的起点。

### 28.3 第三步：GRPO 强化学习训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --max_completion_length 4096 \
    --vllm_max_model_len 8192 \
    --reward_funcs accuracy \
    --num_generations 16 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.4 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --gc_collect_after_offload true \
    --deepspeed zero3 \
    --num_infer_workers 8 \
    --temperature 1.0 --top_p 0.85 \
    --log_completions true \
    --overlong_filter true \
    --output_dir output
```

该脚本使用 8 卡 A100/H100（约 70GB 显存/卡）配置，在数学推理数据集 NuminaMath-TIR 上以 `accuracy` 规则奖励进行 GRPO 训练，每个 prompt 采样 16 个候选回复（`num_generations`），配合 Colocate 模式的多项显存优化参数（sleep、offload、gc）确保 8 卡资源既能承载训练又能承载 vLLM 采样。

### 28.4 第四步：训练效果验证与推理部署

训练完成后，可通过 `swift infer --adapters/--model <checkpoint_path> --stream true` 快速验证模型的实际生成效果，并进一步通过 `swift eval` 在标准化评测集（如 AIME、MATH 等）上量化评估强化学习训练带来的推理能力提升幅度；确认效果满意后，可使用 `swift export --push_to_hub true` 将最终模型推送至 ModelScope 或 HuggingFace 模型仓库，或直接使用 `swift deploy` 将其部署为兼容 OpenAI API 格式的推理服务，供下游应用直接调用。

# 第五部分 综合分析与展望

## 第二十九章 ms-swift 对齐能力横向对比分析

### 29.1 与 HuggingFace TRL 的比较

TRL（Transformer Reinforcement Learning）是 HuggingFace 官方维护的对齐训练库，其最大优势在于与 Transformers/PEFT/Accelerate 生态系统的原生无缝集成，API 设计简洁、文档社区资源丰富，是学术界发表论文时最常用于复现基线实验的库之一。相较而言，ms-swift 在算法覆盖广度上更具优势（尤其是 GRPO 家族十余种前沿变体的集成速度与全面程度），在国产模型与多模态模型的适配深度上也更为领先；同时 ms-swift 提供了从命令行到 Web UI（Swift App/Gradio 界面）再到一体化推理部署的更完整工具链闭环，而 TRL 更专注于训练本身、生态定位相对纯粹。对于希望快速复现单一学术论文核心实验、且深度依赖 HuggingFace 生态其他工具的研究者，TRL 通常是更轻量的选择；对于需要覆盖多模型、多算法、多并行技术组合、且追求前沿算法快速落地的工程化训练需求，ms-swift 的综合能力更具优势。

### 29.2 与 OpenRLHF、veRL 的比较

OpenRLHF 与 veRL（字节跳动开源，原名 HybridFlow）均是专注于大规模强化学习训练效率的框架，二者在"训练与采样解耦的异步流水线设计"（即让 rollout 采样与策略参数更新在时间上尽可能重叠、减少相互等待）方面有着更为激进和精细的工程优化，这使得二者在超大规模（数百至数千卡）强化学习训练场景下的端到端吞吐效率上可能具备一定优势。ms-swift 目前的 Colocate/Server 双模式设计，虽然也提供了训练与推理资源解耦的能力（Server 模式），但在异步流水线的深度优化（如 rollout 与训练的细粒度重叠调度）方面，相较于专门为超大规模强化学习系统设计的 veRL、OpenRLHF，仍有一定的追赶空间；ms-swift 的核心差异化优势则更多体现在其"多模型覆盖 + 多算法覆盖 + 多模态原生支持 + 与 ModelScope 生态的深度绑定"这一综合定位上，二者并非完全的替代关系，而是在不同应用场景与团队技术栈偏好下各有其适用性。

### 29.3 综合评价

总体而言，ms-swift 在"对齐算法覆盖的全面性与前沿性"这一维度上，是目前开源社区中最为突出的框架之一——从传统的 PPO/DPO 到 GRPO 及其十余种 2024—2026 年间最新学术改进方案均已纳入统一框架，且保持了极高的迭代速度；在"多模型、多模态、多并行技术兼容性"这一维度上同样表现优异；相对而言，在专门针对超大规模（千卡级）强化学习系统的极致工程效率优化方面，与部分专注该细分方向的框架相比仍有一定的持续投入空间，但这一差距随着 Megatron GRPO、Ray 分布式调度等能力的持续引入正在逐步缩小。

## 第三十章 工程实践建议与常见问题排查

### 30.1 算法选型建议

结合本报告第三部分的原理分析，给出以下算法选型的一般性建议（仅供参考，具体效果需结合实际任务与数据进行验证）：(1) 若任务领域具备清晰的客观正确性验证机制（数学、代码、结构化信息抽取等），优先考虑 GRPO 及其变体（DAPO/GSPO 等），并配合规则奖励，通常能取得最佳的推理能力提升效果；(2) 若任务是开放式的、依赖主观偏好判断的场景（如对话风格、创意写作偏好），且已具备高质量成对比较标注数据，DPO 及其变体（SimPO/ORPO 等，视对训练效率与显存开销的敏感程度而定）是较为成熟稳妥的选择；(3) 若标注资源有限、只能获取单个回答的二元"好/坏"标签（而非成对比较），KTO 是更适合的选择；(4) 若追求训练全流程的极致简化（不需要参考模型、不需要独立 SFT 冷启动），ORPO 值得优先尝试；(5) 若目标是从一个更强的大模型压缩/蒸馏出一个能力相近的小模型，GKD 是更贴合该目标的算法选择；(6) 传统 PPO 版 RLHF 由于工程复杂度高、超参数敏感，目前在新项目中的应用场景已大幅收窄，通常只在需要精细化价值函数建模、或需要与既有 PPO 基础设施保持兼容的场景下才会被优先考虑。

### 30.2 显存优化实践清单

针对 GRPO/PPO 等在线强化学习训练场景，本报告归纳出以下显存优化实践清单，供工程落地时参考：优先使用 LoRA 而非全参数训练（尤其是在参考模型无需独立加载这一优势下）；启用 `--use_vllm true` 并合理设置 `--vllm_gpu_memory_utilization`；在 Colocate 模式下启用 `--sleep_level`、`--offload_model`、`--offload_optimizer` 组合；对超大规模模型优先考虑 Megatron 并行方案并配合 `--offload_bridge`；对长序列场景启用序列并行；使用 `liger-kernel`、`flash-attn` 等高效算子库降低激活值显存开销；合理设置 `--per_device_train_batch_size` 与梯度累积步数（`gradient_accumulation_steps`）的组合，在显存约束下尽量保证有效批大小满足训练稳定性需求。

### 30.3 常见问题与排查思路

结合本报告梳理的官方文档细节，总结以下几类常见问题的排查思路：(1) **训练不稳定/奖励崩溃**：检查 KL 正则系数（`beta`）是否设置过小、学习率是否偏高、是否存在大量"全对或全错"导致优势为零的无效样本（可考虑启用 DAPO 式动态采样过滤）、以及是否存在训练-推理不一致问题（可通过 `--log_rollout_offpolicy_metrics true` 开启诊断指标，视严重程度考虑启用重要性采样校正）；(2) **显存溢出（OOM）**：优先检查 vLLM 显存占用比例设置、是否启用了必要的 offload/sleep 机制、以及数据并行/模型并行策略是否与模型规模匹配；(3) **奖励函数内部加载模型异常**：在 DeepSpeed ZeRO-3 环境下检查是否需要绕过 Accelerate 对新加载模型的自动分片管理逻辑（参考第二十三章 23.5 节）；(4) **多轮/Agent 场景下损失计算异常**：检查自定义环境实现中损失掩码（loss mask）是否正确区分了"模型生成 token"与"环境注入 token"；(5) **vLLM 相关报错**：结合 vLLM 版本与官方文档中标注的已知 issue（如异步引擎在纯 DP 无 TP 场景下的兼容性问题）进行针对性排查，必要时升级/降级 vLLM 版本。

## 第三十一章 未来发展方向展望

### 31.1 算法层面：从"人类偏好对齐"到"多目标、多阶段自适应对齐"

未来对齐算法的发展预计将进一步走向精细化与自适应化：一方面，如 CHORD 所代表的"动态权重融合"思路，未来可能进一步扩展为在训练过程中自动感知模型当前能力状态、自动调整不同奖励来源（规则奖励、偏好奖励模型、专家示范数据）之间权重配比的自适应训练范式；另一方面，随着模型推理能力与工具使用能力的不断增强，对齐训练的目标也将进一步从"生成正确答案"扩展为"生成正确答案的同时保持推理过程可解释、可信、可审计"，这对奖励设计与训练算法都提出了更高的要求。

### 31.2 工程层面：训练-推理一体化与更极致的资源效率

如第十四章所讨论的训练-推理不一致问题，预计未来会有更多框架层面的系统性解决方案出现，包括训练与推理引擎在数值计算路径上的进一步统一（如共享同一套算子内核实现）、更精细的异步流水线调度以进一步压缩 rollout 阶段的时间开销、以及更高效的超大规模模型权重同步机制。ms-swift 通过持续跟踪并快速集成学术界在这一方向的最新进展（如本报告第十四章介绍的重要性采样校正机制），有望在这一领域继续保持其工程实现的前沿性。

### 31.3 应用层面：多模态、具身智能与 Agent 场景的深度对齐

随着大模型应用场景从纯文本对话逐步扩展到多模态理解、具身智能（Embodied AI）、复杂 Agent 任务执行等更广阔的领域，对齐技术也需要相应地扩展其适用边界——如何为具备物理世界交互能力的具身智能体设计合理的、可验证的奖励信号，如何为需要执行数十步甚至数百步复杂操作的 Agent 任务设计有效的信用分配（credit assignment）机制，都是当前学术界与工业界正在积极探索的前沿方向。ms-swift 通过 GYM 环境训练、多轮训练、Agent 支持等模块的持续建设，已经为这一方向的探索预留了良好的工程基础设施接口，预计将成为未来支撑这类前沿研究快速落地的重要平台之一。

### 31.4 安全与治理层面：可解释性与可控性的进一步增强

随着对齐技术能力的不断增强，如何确保对齐过程本身的透明度、可解释性与可控性也日益成为学术界与监管机构关注的焦点。Constitutional AI 所倡导的"显式、透明、可审查的行为准则文档"这一思路，以及基于评述的奖励模型（Critique-Based Reward Model）所体现的"先解释后打分"设计理念，预计将在未来的对齐技术发展中扮演越来越重要的角色，这也与更广泛的 AI 安全与治理讨论（如模型行为的可审计性、对齐训练过程的可复现性与可追溯性）紧密相关。

## 第三十二章 总结

本报告系统梳理了大语言模型对齐技术从萌芽、奠基、简化到强化、工程化的完整发展脉络，深入推导了 RLHF 的理论基础——KL 约束下的策略优化问题及其解析解，并在此基础上详细剖析了 PPO、DPO 及其庞大的家族变体（IPO、KTO、ORPO、CPO、SimPO）、以及以 GRPO 为核心的可验证奖励强化学习家族（DAPO、GSPO、CISPO、SAPO、RLOO、REINFORCE++、CHORD、Dr.GRPO 等十余种前沿工程变体）的数学原理与相互关系。在此理论基础之上，本报告深入研究了 ms-swift 框架中对齐模块的整体架构设计、命令行参数体系、数据格式规范、分布式并行技术支撑、九大类对齐算法的具体命令行实现、GRPO 场景下 vLLM Colocate/Server 双模式的工程实现细节、奖励函数与奖励模型体系、多轮/多任务/Agent 场景支持、Megatron 大规模并行训练支持，以及多模态对齐能力等工程实现的方方面面，并给出了可直接参考的实战训练脚本。

总体而言，ms-swift 作为一个高度工程化、高度模块化、且与学术前沿保持紧密同步的开源训练框架，为研究人员与工程从业者提供了一套系统性、可组合、可扩展的对齐训练基础设施。无论是希望复现某一篇最新论文的核心改进点，还是希望在生产环境中针对特定业务场景设计定制化的对齐训练流程，ms-swift 所提供的统一命令行入口、丰富的算法与超参数选项、以及可插拔的 Plugin 扩展机制，都能够提供相当程度的支持。展望未来，随着对齐技术本身在算法精细化、工程效率、应用场景广度与安全治理等多个维度的持续演进，ms-swift 这类兼具广度与前沿性的开源框架，预计将继续在推动大模型对齐技术的普及化、工程化落地进程中扮演重要角色。

---

# 附录：参考文献与延伸阅读

> 以下链接均指向论文的 arXiv 摘要页（可直接点击跳转查看摘要、PDF 与引用信息）或对应的官方文档/仓库地址，供读者进一步查阅原文、核实细节。部分 2025—2026 年间的技术报告类工作更新较快，建议以官方发布渠道的最新版本为准。

**基础理论与 RLHF 奠基性工作**
1. Christiano, P. F., et al. (2017). *Deep Reinforcement Learning from Human Preferences*. NeurIPS. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
2. Ziegler, D. M., et al. (2019). *Fine-Tuning Language Models from Human Preferences*. [arXiv:1909.08593](https://arxiv.org/abs/1909.08593)
3. Stiennon, N., et al. (2020). *Learning to Summarize from Human Feedback*. NeurIPS. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
4. Ouyang, L., et al. (2022). *Training Language Models to Follow Instructions with Human Feedback*（InstructGPT）. NeurIPS. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
5. Bai, Y., et al. (2022). *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*（HH-RLHF）. Anthropic. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
6. Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. Anthropic. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
7. Lee, H., et al. (2023). *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*. Google. [arXiv:2309.00267](https://arxiv.org/abs/2309.00267)
8. Touvron, H., et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. Meta AI. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)

**强化学习基础算法**
9. Schulman, J., et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*（GAE）. ICLR. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
10. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*（PPO）. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
11. Schulman, J. (2020). *Approximating KL Divergence*. 个人技术博客. [网页链接](http://joschu.net/blog/kl-approx.html)

**DPO 及其家族**
12. Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*（DPO）. NeurIPS. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
13. Azar, M. G., et al. (2023). *A General Theoretical Paradigm to Understand Learning from Human Preferences*（IPO）. AISTATS. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
14. Ethayarajh, K., et al. (2024). *KTO: Model Alignment as Prospect Theoretic Optimization*. ICML. [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)
15. Xu, H., et al. (2024). *Contrastive Preference Optimization: Pushing the Boundary of LLM Performance in Machine Translation*（CPO）. ICML. [arXiv:2401.08417](https://arxiv.org/abs/2401.08417)
16. Hong, J., et al. (2024). *ORPO: Monolithic Preference Optimization without Reference Model*. EMNLP. [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)
17. Meng, Y., et al. (2024). *SimPO: Simple Preference Optimization with a Reference-Free Reward*. NeurIPS. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
18. Park, R., Rafailov, R., et al. (2024). *Disentangling Length from Quality in Direct Preference Optimization*（长度去偏相关工作，LD-DPO 思路的理论背景）. [arXiv:2403.19159](https://arxiv.org/abs/2403.19159)
19. Lu, J., et al. (2024). *Discovering Preference Optimization Algorithms with and for Large Language Models*（DiscoPOP）. [arXiv:2406.08414](https://arxiv.org/abs/2406.08414)
20. Wang, C., et al. (2024). *Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization*（MPO）. [arXiv:2411.10442](https://arxiv.org/abs/2411.10442)
21. Saeidi, A., et al. (2024). *Insights into Alignment: Evaluating DPO and Its Variants Across Multiple Tasks*（DPO 家族综述型实证研究）. [arXiv:2404.14723](https://arxiv.org/abs/2404.14723)
22. Xiong, W., et al. (2024). *RainbowPO: A Unified Framework for Combining Improvements in Preference Optimization*（DPO 各类改进的统一框架，综述性质）. [arXiv:2410.04203](https://arxiv.org/abs/2410.04203)

**GRPO 及其家族（核心：RLVR 与强化学习工程改进）**
23. Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（GRPO 提出论文）. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
24. DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
25. Yu, Q., et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. 字节跳动 Seed 团队. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
26. Zheng, C., et al. (2025). *Group Sequence Policy Optimization*（GSPO）. 阿里巴巴通义千问团队. [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)（另见 [Qwen 官方博客解读](https://qwenlm.github.io/blog/gspo/)）
27. MiniMax. (2025). *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*（CISPO 提出论文）. [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)
28. Ahmadian, A., et al. (2024). *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback*（RLOO）. [arXiv:2402.14740](https://arxiv.org/abs/2402.14740)
29. Hu, J., et al. (2025). *REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models*. [arXiv:2501.03262](https://arxiv.org/abs/2501.03262)
30. Liu, Z., et al. (2025). *Understanding R1-Zero-Like Training: A Critical Perspective*（Dr.GRPO）. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
31. Yuan, Y., et al. (2025). *What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret*（PPO 在长链推理场景失效原因分析，可与 GRPO 去 Critic 化设计相互印证）. [arXiv:2503.01491](https://arxiv.org/abs/2503.01491)
32. 相关团队. (2025). *On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL via Dynamic Weighting*（CHORD）. 建议以 arXiv 最新检索结果核实版本号。
33. 相关团队. (2025). *TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-based Modeling*。建议以 arXiv 最新检索结果核实版本号。
34. 相关团队. (2025). *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning*（熵掩码机制来源）。建议以 arXiv 最新检索结果核实版本号。
35. Yuan, Z., et al. (2025). *Demystifying Long Chain-of-Thought Reasoning in LLMs*（`cosine`/`repetition` 等奖励函数设计来源）. 建议以 arXiv 最新检索结果核实版本号。
36. DeepSeek-AI. (2025). *DeepSeek-V3.2 技术报告*（训练-推理不一致重要性采样校正方案来源）. 参见 [DeepSeek 官方发布](https://github.com/deepseek-ai)。
37. Wang, Z., et al. (2024). *A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More*（对齐技术综述，覆盖本报告绝大多数算法脉络，适合作为延伸阅读起点）. [arXiv:2407.16216](https://arxiv.org/abs/2407.16216)

**知识蒸馏与其他前沿方向**
38. Agarwal, R., et al. (2024). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*（GKD）. Google DeepMind. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
39. Zelikman, E., et al. (2022). *STaR: Bootstrapping Reasoning With Reasoning*. [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)

**多模态对齐**
40. Zhang, Y.-F., et al. (2025). *MM-RLHF: The Next Step Forward in Multimodal LLM Alignment*. ICML 2025. [arXiv:2502.10391](https://arxiv.org/abs/2502.10391)（项目主页：[mm-rlhf.github.io](https://mm-rlhf.github.io/)，官方说明该数据集与训练脚本已适配 ms-swift）

**ms-swift 框架相关**
41. Zhao, Y., et al. (2024/2025). *SWIFT: A Scalable Lightweight Infrastructure for Fine-Tuning*. AAAI 2025. [arXiv:2408.05517](https://arxiv.org/abs/2408.05517)
42. ModelScope 团队. *ms-swift 官方文档*：[https://swift.readthedocs.io/](https://swift.readthedocs.io/)，其中 RLHF 专题：[Instruction/RLHF.html](https://swift.readthedocs.io/en/latest/Instruction/RLHF.html)，GRPO 专题：[Instruction/GRPO/index.html](https://swift.readthedocs.io/en/latest/Instruction/GRPO/index.html)
43. ModelScope 团队. *ms-swift GitHub 仓库*：[https://github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift)

**综述与延伸阅读推荐（建议作为进一步系统学习的起点）**
44. Wang, Z., et al. (2024). *A Comprehensive Survey of LLM Alignment Techniques*（同第 37 条，综述性质，建议优先阅读）. [arXiv:2407.16216](https://arxiv.org/abs/2407.16216)
45. Winata, G. I., et al. (2024). *Preference Tuning with Human Feedback on Language, Speech, and Vision Tasks: A Survey*（跨模态偏好对齐综述）. [arXiv:2409.11564](https://arxiv.org/abs/2409.11564)
46. Kaufmann, T., et al. (2023). *A Survey of Reinforcement Learning from Human Feedback*（RLHF 综述）. [arXiv:2312.14925](https://arxiv.org/abs/2312.14925)

---

**（全文完）**

*说明：本报告在算法原理与论文引用部分力求准确、可追溯，但由于对齐技术领域论文数量庞大、部分前沿工作（尤其是 2025—2026 年间的技术报告类工作）的作者信息、发表venue等细节可能随时间推移发生变更或存在多个版本，建议读者在进行严谨学术引用时，以论文/技术报告官方发布渠道（如 arXiv、对应机构官网）的最新版本信息为准。本报告中涉及的 ms-swift 具体命令行参数、默认值与代码实现细节，均以撰写时点（2026 年 7 月）能够访问到的官方文档与仓库信息为依据，读者在实际使用时建议以 ms-swift 最新版本的官方文档为准，因为该框架保持着非常高频的迭代更新节奏。*
