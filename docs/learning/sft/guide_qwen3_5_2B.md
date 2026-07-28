## 一、self-cognition 数据集说明

ms-swift 内置了 `swift/self-cognition` 数据集，专门用于让模型学会回答"你是谁""你是谁开发的"这类身份问题。数据集里的回答用占位符表示模型名字和开发者，需要通过 `--model_name` / `--model_author` 参数（各传中文、英文两个值）来替换model_name 是模型的中文名和英文名，model_author 是模型开发者的中文名和英文名。

⚠️ **重要**：单独用 self-cognition 数据集训练，很容易让模型"过拟合到只会回答身份问题"，通用对话能力会下降（灾难性遗忘）。官方最佳实践是**混入通用指令数据**一起训练CUDA_VISIBLE_DEVICES=0 swift sft --model Qwen/Qwen3-4B-Instruct-2507 --tuner_type lora --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' 'AI-ModelScope/alpaca-gpt4-data-en#500' 'swift/self-cognition#500'，所以下面的方案也采用了 alpaca 通用数据 + self-cognition 混合训练。

## 二、完整训练脚本（8GB 显存优化版）

```bash
#!/bin/bash
# ===== 8GB 显存优化：QLoRA + 小max_length + 小batch =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

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
    --max_steps 5 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --group_by_length true \
    --max_length 1024 \
    --gradient_checkpointing true \
    --dataloader_num_workers 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --output_dir output/Qwen3.5-2B-self-cognition
```

## 三、完整流程

### 1. 环境准备

```bash
pip install ms-swift -U
pip install bitsandbytes -U   # QLoRA 4bit量化依赖
pip install deepspeed         # 可选，本方案单卡不需要
```

### 2. 执行训练

```bash
bash sft.sh 2>&1 | tee train.log
```

训练完成后，权重（LoRA adapter）会保存在类似 `output/Qwen3.5-2B-self-cognition/vx-xxx/checkpoint-xxx` 的目录下。

### 3. 训练后快速推理测试（不合并权重，直接用 adapter）

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/Qwen3.5-2B-self-cognition/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 512
```

这会进入交互式命令行，可以直接输入问题测试，例如：

```
<<< 你是谁？
<<< 你是谁开发的？
<<< 介绍一下你自己
<<< What's your name and who created you?
<<< 帮我写一段快速排序的代码   # 验证通用能力是否保留
```

### 4. 合并 LoRA 权重（可选，便于部署）

```bash
swift export \
    --adapters output/Qwen3.5-2B-self-cognition/vx-xxx/checkpoint-xxx \
    --merge_lora true \
    --output_dir output/Qwen3.5-2B-merged
```

合并后用完整模型推理：

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model output/Qwen3.5-2B-merged \
    --stream true \
    --temperature 0 \
    --max_new_tokens 512
```

### 5. 批量效果验证脚本（Python，检查身份认知是否学会 + 通用能力是否保留）

```python
# eval_self_cognition.py
from swift.llm import PtEngine, InferRequest, RequestConfig

engine = PtEngine('output/Qwen3.5-2B-merged')  # 或用 adapters 参数加载 LoRA
request_config = RequestConfig(temperature=0, max_tokens=512)

test_cases = [
    "你是谁？",
    "你是谁开发的？",
    "请自我介绍一下",
    "Who are you and who created you?",
    "你叫什么名字",
    # 通用能力对照组，确认没有明显退化
    "用Python写一个斐波那契数列函数",
    "帮我总结一下三体这本书的主要内容",
]

infer_requests = [InferRequest(messages=[{'role': 'user', 'content': q}]) for q in test_cases]
resp_list = engine.infer(infer_requests, request_config)

for q, resp in zip(test_cases, resp_list):
    print(f"Q: {q}")
    print(f"A: {resp.choices[0].message.content}")
    print("-" * 50)
```

运行：

```bash
python eval_self_cognition.py
```

**判断标准**：
- 身份类问题（前 5 个）应准确报出你设置的 `model_name` / `model_author`；
- 通用能力类问题（后 2 个）回答质量应与未微调前基本一致，若明显变差说明混入的通用数据比例太低或 epoch 太多，可以调低 `num_train_epochs` 或增大 alpaca 数据条数。

## 四、 ms-swift `swift sft` 执行链路深度解读

### 1. 总结

ms-swift 的核心设计是**"注册表 + 组合式 dataclass + 对标准库做最小侵入式继承"**。它没有重新发明训练循环（复用 `transformers.Trainer`），没有重新发明 LoRA 实现（复用 `peft`），自己写的代码集中在"胶水层"：模型/数据集/模板的**统一抽象与注册机制**，这也是它能横向支撑 600+ 模型、150+ 数据集、十几种训练范式（SFT/DPO/GRPO/KTO/...）而不需要每种组合都写一遍训练脚本的原因。

### 2. 系统分层架构图

```mermaid
graph TB
    subgraph L0["Layer 0 · 入口层"]
        CLI["swift/cli/main.py<br/>命令分发器"]
    end

    subgraph L1["Layer 1 · 配置层"]
        ARG["swift/llm/argument/*<br/>Arguments dataclass 组合"]
    end

    subgraph L2["Layer 2 · 数据层"]
        DS["swift/llm/dataset/*<br/>下载 / 注册表 / 格式统一 / 混合切分"]
    end

    subgraph L3["Layer 3 · 模型层"]
        MODEL["swift/llm/model/*<br/>模型注册表 / 加载 / 量化"]
    end

    subgraph L4["Layer 4 · 模板层（核心汇合点）"]
        TPL["swift/llm/template/*<br/>messages → input_ids/labels"]
    end

    subgraph L5["Layer 5 · 微调策略层"]
        TUNER["swift/tuners + peft<br/>LoRA/QLoRA/DoRA 注入"]
    end

    subgraph L6["Layer 6 · 训练执行层"]
        TRAIN["swift/trainers/*<br/>继承 HF Trainer"]
        HF["transformers.Trainer<br/>标准训练循环（复用，不重写）"]
    end

    subgraph L7["Layer 7 · 服务层"]
        INFER["swift/llm/infer/*<br/>PtEngine / vLLM / SGLang"]
        EXPORT["swift/llm/export/*<br/>merge_lora / 量化导出"]
    end

    CLI --> ARG
    ARG --> DS
    ARG --> MODEL
    DS --> TPL
    MODEL --> TPL
    TPL --> TUNER
    MODEL --> TUNER
    TUNER --> TRAIN
    TRAIN --> HF
    HF -->|checkpoint| INFER
    HF -->|checkpoint| EXPORT
    TPL -.复用同一套编码逻辑.-> INFER

    style L4 fill:#fff3cd,stroke:#856404
    style TPL fill:#fff3cd,stroke:#856404
```

图中 **Layer 4（Template）被标黄**，是因为它是全框架唯一同时被"训练路径"和"推理路径"复用的模块（图中虚线）。这个设计保证了训练时喂给模型的文本格式和推理时完全一致，从架构上消除了"训练/推理模板不一致导致效果异常"这一类微调项目最常见的坑。

---

### 3. 完整执行时序图

这是本文档的核心图——把 `swift sft` 从敲下命令到 checkpoint 落盘的**全部关键调用**按时间顺序画出来。

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户/Shell
    participant CLI as CLI 分发器<br/>(cli/main.py)
    participant SFT as 训练总控<br/>(llm/train/sft.py::SwiftSft)
    participant ARG as 参数系统<br/>(llm/argument/*)
    participant MDL as 模型模块<br/>(llm/model/*)
    participant TPL as Template<br/>(llm/template/*)
    participant DS as 数据集模块<br/>(llm/dataset/*)
    participant TUN as Tuner/PEFT<br/>(tuners/* + peft)
    participant TR as Seq2SeqTrainer<br/>(trainers/*)
    participant HFT as HF Trainer 循环
    participant FS as 磁盘

    U->>CLI: swift sft --model ... --dataset ...
    CLI->>CLI: 识别子命令 "sft"，路由到对应模块
    CLI->>SFT: 调用 sft_main(argv)

    rect rgb(245, 245, 245)
    Note over SFT,ARG: 阶段①：参数解析
    SFT->>ARG: 解析全部 --xxx 参数
    ARG-->>SFT: 返回结构化 TrainArguments 对象
    end

    rect rgb(255, 243, 205)
    Note over SFT,MDL: 阶段②：模型加载 + 量化（8GB 显存优化核心）
    SFT->>MDL: get_model_tokenizer(model_path, quant_bits=4)
    MDL->>MDL: 按 config.json 匹配模型注册表（Qwen 系列）
    MDL->>MDL: 构造 BitsAndBytesConfig(4bit, nf4, double_quant)
    MDL->>MDL: from_pretrained(..., quantization_config=...)
    MDL-->>SFT: 返回 (4bit量化模型, tokenizer)
    end

    SFT->>TPL: get_template(tokenizer, template_type, max_length)
    TPL-->>SFT: 返回 Template 实例（含 self-cognition 分支处理）

    rect rgb(255, 243, 205)
    Note over SFT,DS: 阶段③：数据加载与编码
    SFT->>DS: load_dataset(['swift/self-cognition#600', 'alpaca-zh#500'])
    DS->>DS: 下载 / 统一为 {"messages": [...]} 格式
    DS->>DS: self-cognition 占位符替换（注入 model_name/model_author）
    DS->>DS: 多数据源混合 + split_dataset_ratio 切分 train/val
    DS-->>SFT: 返回 raw dataset

    SFT->>TPL: EncodePreprocessor(template)(raw_dataset, num_proc)
    loop 对每条样本
        TPL->>TPL: encode(): messages → 拼接chat template文本
        TPL->>TPL: tokenize → input_ids
        TPL->>TPL: 仅assistant部分计算loss，其余置 labels=-100
    end
    TPL-->>SFT: 返回 encoded train/val dataset
    end

    rect rgb(255, 243, 205)
    Note over SFT,TUN: 阶段④：LoRA 注入（QLoRA 另一支柱）
    SFT->>TUN: LoraConfig(rank=8, alpha=32, target_modules=all-linear)
    TUN->>TUN: 自动遍历模型，找出所有 nn.Linear 层名
    TUN->>TUN: get_peft_model(model, config)：插入LoRA矩阵A/B
    TUN->>TUN: 冻结基座权重，仅LoRA参数 requires_grad=True
    TUN-->>SFT: 返回 PeftModel
    end

    SFT->>TR: 构造 Seq2SeqTrainer(model, args, template, train_ds, eval_ds)

    rect rgb(217, 237, 247)
    Note over SFT,FS: 阶段⑤：训练循环（复用 HF Trainer，不重写）
    SFT->>TR: trainer.train()
    TR->>HFT: 进入标准训练循环
    loop 每个 optimizer step（累积16个micro-batch）
        HFT->>TPL: data_collator 动态 padding（group_by_length 分桶降低padding量）
        HFT->>TUN: forward()
        Note right of TUN: gradient_checkpointing=true<br/>反向传播时重算激活值而非缓存，省显存
        HFT->>TUN: backward()，梯度累积
        HFT->>HFT: 累积满16步 → optimizer.step() + scheduler.step()
        HFT->>HFT: logging_steps 到点 → 打印 loss/lr/显存
        alt 到达 eval_steps
            HFT->>TPL: 用 val_dataset 跑一次评估
        end
        alt 到达 save_steps
            HFT->>FS: 写 checkpoint-xxx/adapter_model.safetensors
            HFT->>FS: 写 checkpoint-xxx/args.json（含完整训练参数）
        end
    end
    HFT-->>TR: 训练完成
    TR-->>SFT: 返回 last_model_checkpoint 路径
    end

    SFT-->>U: 打印最终 checkpoint 路径，进程退出
```

注意阶段⑤里 `TR`（ms-swift 的 `Seq2SeqTrainer`）只是**转手**把控制权交给了 `HFT`（`transformers.Trainer` 的标准循环），`Seq2SeqTrainer` 真正 override 的方法很少（典型是 `compute_loss`、`prediction_step`、以及把 `data_collator` 换成 Template 提供的版本），这是一种**"最小化改造第三方库"**的工程策略：升级 `transformers` 版本时，ms-swift 需要适配的面很窄。

---

## 4. 关键子流程时序图

### 4.1 Template.encode() 内部细节（最值得精读的一步）

```mermaid
sequenceDiagram
    autonumber
    participant DS as 数据集样本
    participant T as Template.encode()
    participant Tok as Tokenizer

    DS->>T: {"messages": [{"role":"system",...}, {"role":"user","content":"你是谁"}, {"role":"assistant","content":"我是小助..."}]}
    T->>T: 按模型 chat_template 拼接特殊 token<br/>(system/user/assistant 分隔符、Qwen3.5 的 non_thinking 前缀)
    T->>Tok: tokenizer.encode(拼接后的完整文本)
    Tok-->>T: 返回完整 input_ids
    T->>T: 定位 assistant 回复片段在 input_ids 中的位置区间
    T->>T: labels = input_ids 的拷贝<br/>非 assistant 区间全部置为 -100
    alt input_ids 长度 > max_length
        T->>T: 按 truncation_strategy 截断或整条丢弃
    end
    T-->>DS: 返回 {input_ids, attention_mask, labels}
```

`labels=-100` 是 PyTorch `CrossEntropyLoss` 的约定——loss 计算时会自动忽略值为 `-100` 的位置。这就是为什么模型只会"学着说 assistant 该说的话"，而不会被要求预测 system/user 说的内容。

`swift/llm/template/base.py` 的 `Template.encode()`；批量调用入口是 `swift/llm/dataset/preprocessor/core.py` 里的 `EncodePreprocessor`（对 dataset 做 `.map()`）。

### 4.2 QLoRA 显存布局：量化与 LoRA 如何叠加

```mermaid
sequenceDiagram
    autonumber
    participant M as 基座权重 (2B参数)
    participant Q as BitsAndBytes 4bit
    participant L as LoRA 矩阵 A/B (rank=8)
    participant O as Optimizer (AdamW)

    Note over M,Q: 加载阶段
    M->>Q: 权重以 nf4 格式压缩存储（约1.2GB，而非bf16的~4GB）
    Q->>Q: requires_grad = False（全程冻结，不参与反向传播）

    Note over L,O: 前向传播
    Q->>Q: 计算时反量化为 bfloat16（--torch_dtype bfloat16）
    L->>L: LoRA 分支：h = Wx + (alpha/rank)·B(Ax)
    Note right of L: A/B 参数量极小（约几百万），bf16 全精度存储

    Note over O: 反向传播 + 优化器状态
    O->>L: 仅对 A/B 计算梯度并更新
    Note over O: AdamW 一阶/二阶动量只需为 LoRA 参数分配<br/>显存占用与基座参数量无关，这是LoRA省显存的第二重来源
```

8GB 卡能训 2B 模型的显存账本大致是：**4bit 基座权重（~1.2GB）+ LoRA 参数及其优化器状态（几十MB量级）+ gradient_checkpointing 后的激活值（随 max_length/batch_size 变化，是主要的可调项）**。这也是为什么显存紧张时，`--max_length` 和 `--gradient_accumulation_steps`（间接控制有效 batch）比调 `--lora_rank` 更有效——LoRA 参数本身占比很小，激活值才是大头。

### 4.3 `swift infer` 推理时序（复用 Template，验证一致性）

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant CLI as swift infer CLI
    participant IE as PtEngine<br/>(llm/infer/infer_engine/pt.py)
    participant ARGJ as checkpoint/args.json
    participant TPL as Template（同训练时的实现）
    participant M as 基座模型 + PeftModel(LoRA)

    U->>CLI: swift infer --adapters checkpoint-xxx --stream true
    CLI->>ARGJ: 读取训练时保存的参数（model路径/template类型等）
    Note right of ARGJ: 这就是为什么不用再传 --model<br/>（--load_args false 可关闭此行为）
    CLI->>IE: 初始化引擎，加载基座模型 + LoRA adapter
    U->>IE: InferRequest(messages=[{"role":"user","content":"你是谁"}])
    IE->>TPL: 用与训练时相同的 Template.encode() 编码 messages
    TPL-->>IE: 返回 input_ids
    IE->>M: model.generate(input_ids, ...)
    Note right of M: PeftModel 在forward时动态叠加LoRA增量<br/>W' = W + (alpha/rank)·BA，无需提前合并
    M-->>IE: 逐token流式返回
    IE-->>U: 流式打印回答
```

### 4.4 `swift export --merge_lora` 权重合并时序

```mermaid
sequenceDiagram
    autonumber
    participant CLI as swift export CLI
    participant EXP as export.py
    participant PM as PeftModel
    participant FS as 磁盘

    CLI->>EXP: --adapters checkpoint-xxx --merge_lora true
    EXP->>PM: 加载基座模型 + LoRA adapter
    alt 基座为4bit量化模型
        EXP->>PM: 先反量化回 bf16/fp16（4bit无法直接与LoRA做矩阵加法）
    end
    PM->>PM: merge_and_unload()：W' = W + (alpha/rank)·BA
    PM-->>EXP: 返回完整权重的标准 nn.Module（不再依赖peft）
    EXP->>FS: 保存为标准 safetensors + config.json
    Note over FS: 产出目录可被 transformers / vLLM 原生直接加载
```

---

## 5. 关键类关系

```mermaid
classDiagram
    class BaseArguments {
        +model: str
        +dataset: List[str]
        +torch_dtype: str
        +use_hf: bool
    }
    class QuantizeArguments {
        +quant_method: str
        +quant_bits: int
    }
    class TemplateArguments {
        +max_length: int
        +truncation_strategy: str
        +add_non_thinking_prefix: bool
    }
    class TunerArguments {
        +tuner_type: str
        +lora_rank: int
        +lora_alpha: int
        +target_modules: str
    }
    class Seq2SeqTrainingArguments {
        <<继承自HF TrainingArguments>>
        +learning_rate: float
        +gradient_checkpointing: bool
        +gradient_accumulation_steps: int
    }
    class TrainArguments {
        <<最终组合，sft_main真正接收的对象>>
    }
    BaseArguments <|-- TrainArguments
    QuantizeArguments <|-- TrainArguments
    TemplateArguments <|-- TrainArguments
    TunerArguments <|-- TrainArguments
    Seq2SeqTrainingArguments <|-- TrainArguments

    class Template {
        +encode(messages) input_ids, labels
        +data_collator(batch)
    }
    class QwenTemplate {
        <<针对Qwen系列的chat template实现>>
    }
    Template <|-- QwenTemplate

    class HFTrainer {
        <<transformers.Trainer>>
        +train()
        +training_step()
    }
    class Seq2SeqTrainer {
        +compute_loss() 重写
        +prediction_step() 重写
    }
    HFTrainer <|-- Seq2SeqTrainer

    TrainArguments ..> Seq2SeqTrainer : 配置注入
    Template ..> Seq2SeqTrainer : 提供data_collator
```

这张图的关键信息是**组合优于继承**（Arguments 层）+ **对第三方库最小侵入继承**（Template/Trainer 层）两种模式并存：
- Arguments 用多重继承把十几个独立关注点（模型/数据/量化/模板/训练超参）拼成一个大对象，**每个子 dataclass 可以单独在别处复用**（比如 `QuantizeArguments` 在 `swift export --quant_method` 场景也会用到）。
- Template/Trainer 走的是"继承但少改"路线，最大化复用生态标准库的正确性和维护成本分摊。

这两种模式的取舍标准很清晰：**框架自己独有的概念（模型/数据集/模板怎么注册和匹配）用组合式设计追求灵活扩展；能复用生态标准库的部分（LoRA实现、训练循环）绝不重复造轮子，只做最小适配。**

---

## 6. 设计模式全景

| 设计模式 | 应用位置 | 解决的问题 |
|---|---|---|
| **注册表模式（Registry）** | `register_model()` / `register_dataset()` / `register_template()` | 新增一个模型/数据集/模板不需要改动框架核心代码，只需在对应文件里注册一条元信息，天然支持"600+模型、150+数据集"的横向扩展 |
| **策略模式（Strategy）** | `--tuner_type lora/full/dora/...`、`--infer_backend pt/vllm/sglang/lmdeploy` | 同一套上层流程，底层实现可插拔切换，互不感知彼此实现细节 |
| **模板方法模式（Template Method）** | `Seq2SeqTrainer` 继承 `transformers.Trainer` 只重写少数 hook 方法 | 复用父类固定的算法骨架（训练循环），只定制变化的步骤（loss计算方式） |
| **组合式配置（Composition over Inheritance，dataclass mixin）** | `TrainArguments` 由多个独立 Arguments 类组合而成 | 避免出现一个几百字段的巨型配置类，关注点分离，便于不同命令（sft/export/infer）复用同一批基础配置类 |
| **适配器模式（Adapter）** | 各数据集的预处理函数把原始字段转换为统一 `messages` 格式 | 让下游 Template/Trainer 只需要认识一种输入格式，不关心数据来源 |
| **对象池/动态代理（PEFT PeftModel）** | LoRA 注入后模型 forward 时动态叠加增量，而非物理修改权重 | 支持训练/推理时不合并权重也能工作，且可以随时切换/叠加多个 adapter |

---

## 7. 参数 → 阶段 → 时序节点 对照表

| 脚本参数 | 所属阶段 | 对应时序图节点 | 一句话作用 |
|---|---|---|---|
| `--model` | 阶段② | `SFT->>MDL: get_model_tokenizer` | 指定基座模型路径 |
| `--dataset` | 阶段③ | `SFT->>DS: load_dataset` | 指定并混合多个数据源 |
| `--model_name` / `--model_author` | 阶段③ | `DS->>DS: 占位符替换` | 写入 self-cognition 数据里的身份信息 |
| `--quant_method` / `--quant_bits` | 阶段② | `MDL->>MDL: 构造BitsAndBytesConfig` | 决定基座权重存储精度（QLoRA支柱之一） |
| `--torch_dtype` | 阶段②④2.2 | 反量化计算dtype | 4bit权重实际计算时用的精度 |
| `--tuner_type/--lora_rank/--lora_alpha/--target_modules` | 阶段④ | `SFT->>TUN: LoraConfig(...)` | 控制LoRA注入位置与秩（QLoRA支柱之二） |
| `--max_length` / `--add_non_thinking_prefix` | 阶段③ | `T->>T: 拼接chat template` | 控制编码长度与Qwen3.5思考模式分支 |
| `--group_by_length` | 阶段③→⑤ | `HFT->>TPL: data_collator` | 减少同批次padding造成的显存浪费 |
| `--per_device_train_batch_size/--gradient_accumulation_steps` | 阶段⑤ | `loop 每个optimizer step` | 控制单步显存占用与有效batch size |
| `--gradient_checkpointing` | 阶段⑤ | `Note right of TUN: 反向传播重算激活值` | 用计算换显存的核心开关 |
| `--learning_rate/--num_train_epochs/--warmup_ratio` | 阶段⑤ | 透传给HF Trainer | 标准训练超参 |
| `--eval_steps/--save_steps` | 阶段⑤ | `alt 到达save_steps` | 定期评估与落盘的时机 |
| `--output_dir` | 阶段⑤末尾 | `HFT->>FS: 写checkpoint` | 产物存放位置 |

---

## 8. 8GB 显存优化在链路中的落点

结合第 3、4.2 节的时序图，本次训练针对 8GB 卡做的每一个改动，都能精确定位到某个阶段：

```mermaid
graph LR
    A["8GB显存挑战"] --> B["阶段②<br/>quant_bits=4"]
    A --> C["阶段⑤<br/>gradient_checkpointing"]
    A --> D["阶段③<br/>max_length=1024"]
    A --> E["阶段⑤<br/>batch_size=1 + 梯度累积"]
    A --> F["阶段④<br/>lora_rank=8（占比很小）"]

    B --> B1["基座权重 4GB→1.2GB"]
    C --> C1["激活值显存换成重算时间"]
    D --> D1["单条样本峰值激活值上限可控"]
    E --> E1["单步真实占用=1条样本的显存<br/>但保持有效batch=16的训练稳定性"]
    F --> F1["LoRA参数量本身占比极小<br/>调它对省显存边际收益有限"]

    style B fill:#d4edda
    style C fill:#d4edda
    style D fill:#d4edda
    style E fill:#d4edda
    style F fill:#f8d7da
```

如果之后还是 OOM，按这张图从左到右排查优先级：先看 `max_length` 能不能再降、`gradient_accumulation_steps` 能不能提高（保持有效batch不变的前提下降低单步真实batch），最后才是调 `lora_rank`——因为它对显存的影响本来就最小。

---

## 9. 建议的源码阅读路径（按依赖顺序）

1. `swift/cli/main.py` —— 建立"一切从这里开始"的整体感，5分钟看完分发逻辑。
2. `swift/llm/train/sft.py` 的 `SwiftSft` —— 对照本文档第3章的主时序图，逐行确认调用顺序与图上一致。
3. `swift/llm/argument/train_args.py` —— 对照第5章类图，看 `TrainArguments` 的多重继承关系。
4. `swift/llm/dataset/loader.py` + self-cognition 的预处理函数 —— 对照第3章阶段③，理解 `#600` 截取与占位符替换的实现。
5. `swift/llm/template/base.py` 的 `Template.encode()` —— 对照第4.1节子时序图，这是全框架最值得逐行精读的方法。
6. `swift/trainers/trainers.py` 的 `Seq2SeqTrainer` —— 搜索它 override 了哪些方法，对照第6章"模板方法模式"的说法验证。
7. `swift/llm/infer/infer_engine/pt.py` —— 对照第4.3节，验证推理是否复用了同一套 Template 逻辑。
8. `swift/llm/export/export.py` —— 对照第4.4节，看 `merge_and_unload()` 前后的量化处理分支。

📍 定位本地安装路径：`python -c "import swift, os; print(os.path.dirname(swift.__file__))"`

---

## 10. 一句话收尾

🔰 理解这条链路后再看任何新的 ms-swift 训练脚本（哪怕换成 DPO/GRPO/多模态），你会发现变化的只是**阶段③（数据集格式与预处理）、阶段④（tuner类型）、阶段⑤（用哪个 Trainer 子类）**，而阶段①②③的骨架完全复用。

🏗️ 这正是"注册表 + 组合式配置 + 最小侵入式继承"这套设计的价值所在：新增一种训练范式，本质是新增一个 Trainer 子类和一套数据预处理逻辑，而不是重写一遍从CLI到训练循环的全部代码。