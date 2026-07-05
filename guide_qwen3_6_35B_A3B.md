# ms-swift 实战教程：以 Qwen3.6-35B-A3B 为例，边用边读源码

> **文档定位**：本文以 `Qwen/Qwen3.6-35B-A3B` 这一具体模型为线索，串起 ms-swift 的"使用 → 概念 → 源码"三条线。
> 你将在训练/推理一个真实模型的过程中，理解框架的每个核心组件分别在做什么、对应代码在哪里。
>
> **编写依据**：基于仓库 `main` 分支（v4.4.0.dev0）实际源码逐文件核对，配套 `examples/models/qwen3_5/` 官方脚本。
>
> **前置阅读**：建议先读同目录 `guide.md`（框架总览与学习路线），本文是它的"单模型纵深实战版"。
>
> **适合谁**：有 Python / PyTorch / transformers 基础，想通过一个完整案例掌握 ms-swift 的新手。

---

## 〇、为什么选 Qwen3.6-35B-A3B 作为学习载体

先把这个模型名字拆开看，每个部分都对应 ms-swift 的一个知识点：

| 名字片段 | 含义 | 对应要学的框架知识 |
|---|---|---|
| `Qwen3.6` | 通义千问 3.6 代 | 模型注册机制（`model_type` 自动匹配） |
| `35B` | 总参数量 350 亿 | 大模型训练的显存/并行策略 |
| `A3B` | **A**ctivated **3B**：MoE 架构，每次前向只激活约 30 亿参数 | MoE 专属参数（`experts_impl`、`router_aux_loss_coef`、专家并行 EP） |
| （隐藏属性） | 在 ms-swift 中它是**多模态模型**（支持 vision/video） | 多模态模板 `_encode`、`model_arch` 冻结控制 |
| （隐藏属性） | 它是**混合思考模型**（hybrid thinking） | `thinking_prefix` / `non_thinking_prefix` / `loss_scale` |

> ⚠️ **第一个反直觉点**：很多人以为 35B-A3B 是纯文本大模型，但在 ms-swift 的注册表里，它属于 `MLLMModelType.qwen3_5`（M=Multimodal），
> 架构是 `Qwen3_5MoeForConditionalGeneration`，带 `tags=['vision', 'video']`。
> 也就是说它是一个**既能 MoE、又能多模态、还能混合思考**的"全都要"模型——
> 正好让我们一次性把 ms-swift 最有代表性的几个机制都覆盖到。

源码出处（`swift/model/models/qwen.py:1421`）：

```python
register_model(
    ModelMeta(
        MLLMModelType.qwen3_5,                       # model_type：这一类模型的唯一 ID
        [
            ModelGroup([...Qwen3.5 系列...], TemplateType.qwen3_5),
            ModelGroup([                              # 我们关注的 Qwen3.6 在这个 group
                Model('Qwen/Qwen3.6-35B-A3B', 'Qwen/Qwen3.6-35B-A3B'),       # (魔搭id, HF id)
                Model('Qwen/Qwen3.6-35B-A3B-FP8', 'Qwen/Qwen3.6-35B-A3B-FP8'),
            ], TemplateType.qwen3_5),
        ],
        Qwen3_5MoeLoader,                             # 自定义加载器（处理 MoE + 线性注意力）
        model_arch=ModelArch.qwen2_vl,               # 多模态参数前缀映射（控制 freeze_vit 等）
        architectures=['Qwen3_5MoeForConditionalGeneration'],  # 自动匹配 model_type 的第二依据
        requires=['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))
```

**读这段代码你应该问自己 3 个问题**（答案在后文）：
1. 我命令行只写了 `--model Qwen/Qwen3.6-35B-A3B`，框架怎么知道要用 `qwen3_5` 这个 model_type？→ 见第三章
2. `Qwen3_5MoeLoader` 比默认 `ModelLoader` 多做了什么？→ 见第四章
3. `TemplateType.qwen3_5` 这个模板长什么样、`thinking_prefix` 干嘛用？→ 见第五章

---

## 一、环境准备

```bash
# 开发模式安装（要读源码、改代码，必须用 -e 可编辑安装）
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .

# 该模型对依赖版本有要求（来自上面 ModelMeta 的 requires 字段）
pip install 'transformers>=5.2.0' 'qwen_vl_utils>=0.0.14' decord -U
# MoE 高效训练还需要（grouped_mm 专家算子、flash-attn）：
pip install flash-attn --no-build-isolation
```

> **学习提示**：`requires` 字段不是摆设。当你 `swift sft --model Qwen/Qwen3.6-35B-A3B` 时，
> `ModelMeta.check_requires()`（`swift/model/model_meta.py:106`）会逐项检查版本，缺了会打印 `pip install ...` 提示。
> 这是 ms-swift "把环境踩坑提前暴露"的设计。

**显存预算**（来自官方脚本注释）：
- LoRA 微调（transformers 后端）：4 × 30GiB，即 4 张 40G/80G 卡
- 全参微调（Megatron 后端）：8 × 80GiB

如果你没有这么多卡，**学习阶段强烈建议先用同系列小模型** `Qwen/Qwen3.5-4B` 把全流程跑通——
它和 35B-A3B 走的是几乎相同的代码路径（同一个 `TemplateType.qwen3_5` 模板，只是 `Qwen3_5Loader` 而非 MoE 版），
单卡即可，命令里把 `--model` 一换就行。本文所有命令都可以这样降配验证。

---

## 二、五分钟跑通：先当用户，建立全局体感

ms-swift 的核心理念是"一个 `swift` 命令解决从训练到部署的所有事"。先完整走一遍闭环：

### 2.1 训练（LoRA 微调）

直接用官方脚本 `examples/models/qwen3_5/transformers.sh`，这里逐行加注释：

```bash
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \  # 缓解显存碎片
NPROC_PER_NODE=4 \                  # 4 进程（4 卡）。这个环境变量决定是否启用 torchrun，见第三章
IMAGE_MAX_TOKEN_NUM=1024 \          # 单张图最多占多少 token（多模态模型才有意义）
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \                         # 子命令 sft = supervised fine-tuning
    --model Qwen/Qwen3.6-35B-A3B \  # 只给名字，model_type/template 全自动推断
    --tuner_type lora \             # 用 LoRA（而非 full 全参），只训练少量新增参数
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \  # 数据集:子集#采样2000条
    --add_non_thinking_prefix true \   # 混合思考模型专属，见第五章
    --split_dataset_ratio 0.01 \    # 切 1% 作为验证集
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \                 # LoRA 的秩，越大可训练参数越多
    --lora_alpha 32 \
    --target_modules all-linear \   # 给所有线性层都挂 LoRA
    --experts_impl grouped_mm \     # ★ MoE 专属：用分组矩阵乘高效计算专家，见第四章
    --router_aux_loss_coef 1e-3 \   # ★ MoE 专属：路由负载均衡辅助损失权重
    --group_by_length true \        # 按长度分组，减少 padding 浪费
    --max_length 2048 \
    --output_dir output/Qwen3.6-35B-A3B \
    --deepspeed zero3               # 用 DeepSpeed ZeRO-3 切分优化器/梯度/参数
```

运行后，`output/Qwen3.6-35B-A3B/vx-xxx/` 下会生成：
- `checkpoint-xxx/`：LoRA adapter 权重
- `args.json`：**本次训练的全部参数快照**（关键！下一步推理会自动读它还原配置）
- `logging.jsonl`：训练指标日志

### 2.2 推理（加载 LoRA）

```bash
IMAGE_MAX_TOKEN_NUM=1024 VIDEO_MAX_TOKEN_NUM=128 FPS_MAX_FRAMES=12 \
swift infer \
    --adapters output/Qwen3.6-35B-A3B/vx-xxx/checkpoint-xxx \  # 只需指向 adapter 目录
    --stream true \
    --experts_impl grouped_mm \
    --enable_thinking false \       # 关闭思考模式（直接出答案，不输出 <think>）
    --load_data_args true           # 从 args.json 还原训练时的数据参数
```

> **第二个反直觉点**：`swift infer` 只给了 `--adapters` 路径，**没写 `--model`**，框架却知道基座模型是谁。
> 因为它读取了 `adapters/args.json` 里记录的 `model`。这就是 ms-swift 的 `args.json` 自动续传机制
> （实现见 `swift/template/register.py:_read_args_json_template_type` 和 `model_meta.py:_read_args_json_model_type`）。
> 这个机制贯穿 train→infer→export 全流程，是它"配置不丢失"的核心设计。

### 2.3 导出（合并 LoRA / 量化 / 推送）

```bash
swift export \
    --adapters output/Qwen3.6-35B-A3B/vx-xxx/checkpoint-xxx \
    --merge_lora true \             # 把 LoRA 权重合并回基座，得到完整模型
    --output_dir output/merged
# 之后可继续 --quant_method gptq 量化，或 --push_to_hub true 推回魔搭
```

至此你已经体验完 **sft → infer → export** 主链路。下面进入源码，看每一步背后发生了什么。

---

## 三、源码主线（一）：从命令行到训练流程

### 3.1 `swift sft` 这条命令是怎么被执行的

入口是 `swift/cli/main.py`（已加注释）。核心路由表：

```python
ROUTE_MAPPING = {
    'sft': 'swift.cli.sft',   # swift sft → 执行 swift/cli/sft.py
    'infer': 'swift.cli.infer',
    'export': 'swift.cli.export',
    ...
}
```

`cli_main()` 做三件事：
1. 取子命令名 `sft`，查表得到入口文件 `swift/cli/sft.py`；
2. 检查 `NPROC_PER_NODE`/`NNODES` 环境变量（`use_torchrun()`）——我们设了 `NPROC_PER_NODE=4`，
   所以它会用 `python -m torch.distributed.run --nproc_per_node 4 ...` 拉起 **4 个训练进程**（多卡分布式）；
3. 用子进程方式执行入口文件。

> **学习提示**：理解"为什么用环境变量控制分布式"很重要。ms-swift 不在 Python 里写 `mp.spawn`，
> 而是让 torchrun 在外层拉起多个完整进程，每个进程跑同一份 `sft.py`。这是 PyTorch 分布式的标准做法，
> 也是为什么训练脚本里满是 `NPROC_PER_NODE=4 \` 这种前缀。

`swift/cli/sft.py` 内部只有一行实质代码：调用 `sft_main()`。

### 3.2 `sft_main` → `SwiftSft`：训练编排的主类

`swift/pipelines/train/sft.py` 的 `SwiftSft` 是整个框架信息密度最高的类之一。它继承自 `SwiftPipeline`（基类，模板方法模式）。

**`SwiftPipeline` 基类**（`swift/pipelines/base.py`）做的是所有子命令的公共骨架：

```
__init__: 解析参数(_parse_args) → 设随机种子(_set_seed, seed+rank) → ...
main():   记录开始时间 → run()(子类实现) → 记录结束时间
```

**`SwiftSft` 的生命周期**（按调用顺序，对应注释）：

```
__init__:
  super().__init__(args)          # 基类：解析 SftArguments + 设种子
  _prepare_model_tokenizer()      # ① 加载模型与 processor（查 MODEL_MAPPING）
  _prepare_template()             # ② 取对话模板并 set_mode('train')

run():
  _prepare_dataset()              # ③ 加载+编码数据（messages → input_ids/labels）
  args.save_args()                # ④ 写出 args.json 快照
  prepare_model(...)              # ⑤ TunerMixin 注入 LoRA
  TrainerFactory.get_trainer_cls  # ⑥ 选 Trainer 子类
  trainer.train()                 # ⑦ 进入 transformers 训练循环
  _save_trainer_state()           # ⑧ 保存 checkpoint 信息到 logging.jsonl
```

这 8 步就是一次 SFT 的全部。**建议你打开 `sft.py` 对照注释，用调试器在每一步打个断点走一遍**（小模型即可）。
更妙的是：RLHF 管线（`pipelines/train/rlhf.py`）继承了 `SwiftSft` 并复用其中绝大部分步骤，
只替换 Trainer 和数据格式——所以读懂 SFT 主线，GRPO 等对齐算法的代码会看得非常快。

### 3.3 参数从哪来：`SftArguments`

命令行里的每个 `--xxx` 都对应 `swift/arguments/` 下某个 dataclass 的字段。查参数含义的最快路径：
- **想知道某参数怎么用** → 在 `examples/` 里搜脚本；
- **想知道某参数的定义/默认值/作用** → 在 `swift/arguments/` 里搜字段名。

比如 `--experts_impl`、`--router_aux_loss_coef` 这些 MoE 参数，就能在 arguments 里找到定义和注释。

---

## 四、源码主线（二）：模型加载，重点看 MoE

### 4.1 model_type 是怎么自动推断出来的

我们命令行只写了 `--model Qwen/Qwen3.6-35B-A3B`，没写 `--model_type`。推断发生在
`swift/model/model_meta.py:get_model_info_meta()`（已加注释），分两级：

1. **按模型名后缀匹配**（`get_matched_model_meta`）：拿 `Qwen3.6-35B-A3B` 去和注册表中所有 `ModelGroup` 的
   模型 id 后缀比对，命中 → 得到 `model_type = qwen3_5`；
2. **兜底按 architectures 匹配**：如果名字没匹配上（比如你用了本地改名的模型目录），就读 `config.json` 的
   `architectures` 字段（这里是 `Qwen3_5MoeForConditionalGeneration`）去匹配。

匹配出 model_type 后，框架就拿到了这一类模型的全部"配方"：用哪个 loader、哪个 template、哪个 model_arch。

> **`ModelMeta` vs `ModelInfo` 的区别**（新手易混）：
> - `ModelMeta` = **注册时**的静态信息（一类模型共享一份，写死在 `qwen.py` 里）；
> - `ModelInfo` = **运行时**针对你这个具体模型目录解析出的信息（torch_dtype、是否量化、是否 MoE、最大长度等）。

### 4.2 `Qwen3_5MoeLoader`：MoE 模型的特殊加载

普通模型用默认 `ModelLoader`（`swift/model/register.py`，已加注释，本质是包装 `AutoModel.from_pretrained`）。
但 35B-A3B 用的是定制的 `Qwen3_5MoeLoader`（`qwen.py:1412`）：

```python
class Qwen3_5MoeLoader(Qwen3VLLoader):
    def get_model(self, model_dir, config, processor, model_kwargs):
        from transformers import Qwen3_5MoeForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Qwen3_5MoeForConditionalGeneration
        _patch_qwen3_5_linear_attention_sequence_parallel()  # 给"线性注意力"打序列并行补丁
        return Qwen2VLLoader.get_model(self, model_dir, config, processor, model_kwargs)
```

它比默认加载器多做两件事：
1. **指定专用的 AutoModel 类** `Qwen3_5MoeForConditionalGeneration`（因为这是 MoE + 多模态结构）；
2. **打线性注意力的序列并行补丁**：Qwen3.5/3.6 用了 Gated Delta Net 这类线性注意力，
   要让它支持长文本序列并行，需要对 `forward` 做 monkey-patch。

> **学习提示**：这就是 ms-swift "为每个模型族写一个 Loader 子类"的扩展模式。
> 你贡献新模型支持时，90% 情况只需在 `ModelMeta` 里填字段（用默认 Loader）；
> 只有当模型有特殊结构（MoE、线性注意力、需要 patch）时才写自定义 Loader。

### 4.3 MoE 训练的关键参数（结合 A3B 理解）

A3B 表示"激活 3B"，MoE 的本质是**很多专家（expert），每个 token 只路由到少数几个专家**。这带来几个专属参数：

| 参数 | 作用 | 在哪生效 |
|---|---|---|
| `--experts_impl grouped_mm` | 专家计算的底层实现。`grouped_mm`（分组矩阵乘）比逐专家循环快得多，需要 transformers≥5.0 | `ModelLoader.get_model` 把它塞进 `model_kwargs['experts_implementation']`（`register.py:272`） |
| `--router_aux_loss_coef 1e-3` | 路由"负载均衡"辅助损失的权重。防止所有 token 都挤向少数专家 | 训练时加到总 loss 上 |
| `--deepspeed zero3` + MoE | ZeRO-3 下 MoE 的专家模块要设为 `z3_leaf_modules`，否则 DeepSpeed 会错误切分专家 | `register.py:_deepspeed_set_z3_leaf_modules`（已能看到针对各种 MoE 架构的处理分支） |

> **第三个反直觉点**：MoE 模型在 ZeRO-3 下需要特殊处理。`register.py` 里有一长串 `if hf_model_type == 'qwen3_moe'` 之类的分支，
> 就是在告诉 DeepSpeed "专家这个模块是叶子，别往里切"。这是 MoE 训练最容易踩的坑之一，ms-swift 帮你内置处理了。

---

## 五、源码主线（三）：对话模板，理解整个框架的钥匙

模板（Template）是 ms-swift 里**信息密度最高、也最该吃透**的模块。它负责把人类可读的 `messages` 转成模型要的 `input_ids` / `labels`。

### 5.1 Qwen3.6-35B-A3B 用的模板长什么样

它用 `TemplateType.qwen3_5`，注册在 `swift/template/templates/qwen.py:629`：

```python
register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_5,
        template_cls=Qwen3_5Template,           # 自定义模板类（处理多模态 + 思考规整）
        default_system=None,
        thinking_prefix='<think>\n',            # 思考模式的前缀
        non_thinking_prefix='<think>\n\n</think>\n\n',  # 非思考模式自动插入的空思考块
        agent_template='qwen3_5',               # Agent 工具调用的数据格式
        is_thinking=True))                      # 这是个混合思考模型
```

`QwenTemplateMeta` 继承自 `ChatmlTemplateMeta`，也就是熟悉的 ChatML 格式（`<|im_start|>role\n...<|im_end|>`）。
回顾 `template_meta.py` 里讲的**五要素**（已加注释）：

```
prefix:        对话开头
prompt:        <|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n   # 每轮 user 的包装
chat_sep:      <|im_end|>\n                  # 多轮之间的分隔
suffix:        <|im_end|>                     # 最后一轮结尾（标记生成结束）
system_prefix: <|im_start|>system\n{{SYSTEM}}<|im_end|>\n
```

### 5.2 混合思考（hybrid thinking）：thinking_prefix / non_thinking_prefix

Qwen3.6 是"混合思考"模型——同一个模型既能输出 `<think>...思考过程...</think>` 再回答，也能直接回答。
这给训练带来一个问题：**训练数据里如果没有思考过程，模型可能学坏**。ms-swift 的处理：

- `--add_non_thinking_prefix true`（我们训练命令里有）：当某条数据没有思考内容时，
  自动补一个空思考块 `<think>\n\n</think>\n\n`（即 `non_thinking_prefix`），告诉模型"这题不用想，直接答"。
- 实现见 `Template._add_non_thinking_prefix`（`base.py`，被 `_swift_encode` 调用）。

> **学习提示**：再看 `mcore.sh` 里还有个 `--loss_scale ignore_empty_think`，
> 它配合上面机制，让"空思考块"那部分 token **不参与 loss**（loss 权重设 0）。
> 这就是 `loss_scale` 机制——token 级别的 loss 权重控制，Agent/思考模型训练的核心技巧。

### 5.3 多模态编码：`Qwen3_5Template._encode`

因为这是多模态模型，模板要处理图像/视频。`Qwen3_5Template` 继承链是
`Qwen3_5Template → Qwen3VLTemplate → Qwen2VLTemplate → Template`。重写的核心是 `_encode`（`qwen.py:536`）：

```python
def _encode(self, inputs):
    encoded = Template._encode(self, inputs)   # 先走父类：messages → input_ids/labels
    input_ids = encoded['input_ids']
    for media_type in ['images', 'videos']:    # 再处理图像/视频
        mm_data = getattr(inputs, media_type)
        if mm_data:
            # 用 processor 把图片编码成视觉特征，并算出它占多少个 token
            media_inputs = processor.image_processor(images=mm_data, ...)
            media_grid_thw = media_inputs['image_grid_thw']
            # 把 input_ids 里的占位符 token（<image>）展开成 N 个真实视觉 token
            idx_list = findall(input_ids, media_token)
            input_ids, labels, ... = self._extend_tokens(input_ids, labels, ..., _get_new_tokens, ...)
            encoded.update(media_inputs)        # 把视觉特征（pixel_values 等）一起带上
    ...
```

理解这段的关键概念——**占位符展开**：文本里先放一个 `<image>` 占位符，编码时把它替换成
"这张图实际占用的 N 个视觉 token"（N 由图片分辨率经 `image_grid_thw` 算出）。
纯文本模型没有这步，所以纯文本模板的 `_encode` 简单得多。

> **学习路径建议**：先吃透**纯文本**模板（拿 `Qwen/Qwen3.5-4B` 调试），用下面的 debug 片段逐 token 观察；
> 再来看多模态的 `_encode` 就不会懵。

### 5.4 亲手观察编码结果（强烈推荐的练习）

```python
from swift import get_template, get_processor

# 用小模型，单卡秒级加载
processor = get_processor('Qwen/Qwen3.5-4B')
template = get_template(processor)
template.set_mode('train')              # 训练模式才会生成 labels

encoded = template.encode({'messages': [
    {'role': 'user', 'content': '你好'},
    {'role': 'assistant', 'content': '你好！有什么可以帮你的？'},
]})

print(template.safe_decode(encoded['input_ids']))   # 看完整输入
print('---')
print(template.safe_decode(encoded['labels']))      # 看哪些 token 参与 loss
```

你会发现：`labels` 里 **user 部分全是 `-100`（不算 loss），只有 assistant 回复部分是真实 token**。
这正是 SFT "只对回答计算损失"的本质，对应 `base.py:_encode` 里那句 `encoded['labels'][0] = -100` 和
`loss_scale` 的逻辑。**搞懂这一点，你就理解了 SFT 训练的核心。**

### 5.5 set_mode：一套模板服务训练与推理

`template.set_mode(...)`（`base.py:1585`，已加注释）能在 7 种模式间切换：
`train / rlhf / kto`（训练）+ `transformers / vllm / lmdeploy / sglang`（四种推理后端）。

> **第四个反直觉点也是最重要的设计**：训练和推理**共用同一个模板对象**，只是 mode 不同。
> 这从根本上保证了"训练时怎么编码，推理时就怎么编码"，消灭了一大类"训练好好的、一推理就胡说"的诡异 bug。
> 这也是为什么 `infer` / `deploy` / GRPO rollout 都复用 template——Template 是连接训练与推理的枢纽。

---

## 六、源码主线（四）：数据流，从 `--dataset` 到张量

我们训练用的是 `--dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#2000'`。这个表达式经历了什么？

### 6.1 数据集表达式的解析

`swift/dataset/loader.py:load_dataset()`（已加注释）逐个解析数据集字符串：

```
'AI-ModelScope/LaTeX_OCR : human_handwrite # 2000'
       数据集 id           子集名         采样 2000 条
```

- `DatasetSyntax.parse` 拆出 id / 子集 / 采样数；
- 查 `DATASET_MAPPING`（数据集注册表）拿到 `DatasetMeta`；
- `loader.load` 从魔搭/HF/本地下载，并立即调用 `DatasetMeta.preprocess_func` 把原始字段**标准化为 messages 格式**；
- `post_process` 做采样和 train/val 切分。

### 6.2 七步数据流全景

```
--dataset 'id:subset#2000'
  ① dataset_syntax  解析表达式（采样/子集）
  ② loader          下载（魔搭/HF/本地）
  ③ preprocess_func 原始字段 → 统一 messages 格式      ← 自定义数据集只需写这一步
  ④ split/concat    切 train/val、多数据集合并
  ⑤ template.encode messages → input_ids/labels        ← 第五章讲的模板
  ⑥ packing/padding 装箱或变长 batch（可选优化）
  ⑦ data_collator   组 batch
        ↓
  dataloader → Trainer 训练循环
```

记忆口诀：**③之前是各家原始格式，③之后是统一 messages，⑤之后是张量。**

### 6.3 编码时机：预编码 vs 惰性编码

`SwiftSft._post_process_datasets`（`sft.py`，已加注释）里有三种策略：

| 策略 | 触发 | 机制 |
|---|---|---|
| 预编码 | 默认（纯文本） | 训练前用多进程 map 全量编码 |
| **惰性编码** | `--lazy_tokenize true`（**多模态默认**） | `LazyLLMDataset` 训练时才编码，坏样本自动跳过 |
| 流式 | `--streaming true` | 边下边训，不落盘 |

35B-A3B 是多模态模型，默认走**惰性编码**。看 `LazyLLMDataset`（`swift/dataset/utils.py`，已加注释）：

```python
def __getitem__(self, idx):
    for i in range(self.n_try_fetch):       # 最多重试 10 次
        data = self.dataset[idx]
        try:
            return self.encode_func(data, return_length=True)   # 这里才真正编码
        except MaxLengthError:
            continue                          # 超长样本：换一条，不报错
        ...
```

> **第五个反直觉点**：百万级真实数据里总有超长/损坏样本。ms-swift 默认策略是
> **"跳过坏样本并随机换一条重试"** 而非让训练崩溃（`strict=False`）。
> 多模态图片解码内存开销大，也是它默认惰性编码的原因——不可能把所有图片预先解码进内存。

---

## 七、源码主线（五）：Tuner（LoRA）注入与 Trainer

### 7.1 LoRA 是怎么"挂"到模型上的

回到 `SwiftSft.run()` 里的 `self.prepare_model(...)`，它来自 `TunerMixin`（`pipelines/train/tuner.py`）。
我们传了 `--tuner_type lora --target_modules all-linear`，于是框架：
1. 给模型所有线性层旁边加上低秩矩阵 A、B（LoRA）；
2. 冻结原始权重，只让 A、B 可训练；
3. 打印可训练参数量——你会看到类似 `trainable params: 0.3%` 的输出（`get_model_parameter_info`）。

这就是为什么 LoRA 微调 35B 模型，显存需求远小于全参训练：**350 亿参数绝大部分被冻结，只训练新增的零点几个百分点。**

### 7.2 Trainer：选型与训练循环

`TrainerFactory.get_trainer_cls(args)`（`sft.py:run`）按任务类型选 Trainer 子类。
ms-swift 的 Trainer 都基于 transformers `Trainer` 扩展，注入了：
- 自定义 `data_collator`（来自 template，负责组 batch + padding）；
- `loss_scale`（token 级 loss 权重）；
- `callbacks` / `metrics`（可通过 mapping 注册扩展）。

真正的前向/反向/优化器步进仍在 transformers `Trainer.train()` 内部。
所以**如果你熟悉 transformers Trainer，只需关注 ms-swift 改了哪几个注入点**。

---

## 八、进阶：Megatron 后端（35B 全参训练的正确姿势）

LoRA 用 transformers 后端就够了。但如果要**全参微调 35B-A3B**，单靠 DeepSpeed 会很吃力，
官方推荐 **Megatron 后端**（`examples/models/qwen3_5/mcore.sh`）。命令从 `swift sft` 变成 `megatron sft`：

```bash
megatron sft \
    --model Qwen/Qwen3.6-35B-A3B \
    --tuner_type full \                     # 全参
    --tensor_model_parallel_size 4 \        # 张量并行 TP=4
    --expert_model_parallel_size 8 \        # ★ 专家并行 EP=8（MoE 专属并行维度）
    --moe_grouped_gemm true \               # MoE 分组 GEMM 加速
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --packing true \                        # 装箱提升吞吐
    --recompute_granularity full \          # 激活重算省显存
    --sequence_parallel true \              # 序列并行
    --mtp_num_layers 1 \                    # Multi-Token Prediction
    ...
```

关键新概念：
- **EP（专家并行）**：MoE 特有的并行维度，把不同专家分到不同卡上。`TP × EP` 共同决定模型如何切分。
- **Mcore-Bridge**：Megatron 权重和 HF 权重格式不同，ms-swift 用 Mcore-Bridge 做双向桥接，
  让 `megatron sft` 的体验接近 `swift sft`（输入 HF 模型，输出也能转回 HF）。
- 训练后用 `swift export --to_hf` 或脚本里的 `--save_safetensors true` 转回 HF 格式再推理。

> **学习建议**：Megatron 线放到最后学。先用 transformers 后端把 LoRA + 推理 + 导出全链路吃透，
> 建立"参数 → 代码路径"的映射能力后，再碰并行训练。

---

## 九、把整条链路连起来：一张全景图

```
swift sft --model Qwen/Qwen3.6-35B-A3B --tuner_type lora --dataset '...' --experts_impl grouped_mm
│
├─ cli/main.py        识别子命令 sft，NPROC_PER_NODE=4 → torchrun 拉起 4 进程
│                     └→ cli/sft.py → sft_main() → SwiftSft
│
├─ arguments/         解析为 SftArguments（--experts_impl 等都在这里定义）
│
├─ model/  ②          model_meta.py: 名字后缀匹配 → model_type=qwen3_5
│                     register.py:  Qwen3_5MoeLoader 加载 MoE 模型
│                                   （指定 grouped_mm、打线性注意力补丁、ZeRO-3 叶子模块）
│
├─ template/  ⑤       qwen.py: Qwen3_5Template（ChatML + 思考前缀 + 多模态占位符展开）
│                     set_mode('train') → messages 编码出 input_ids/labels
│                     （user 部分 -100 不算 loss，空思考块 loss_scale=0）
│
├─ dataset/  ①③④     loader.py: 解析 'id:subset#2000' → 下载 → preprocess → messages
│                     utils.py:  LazyLLMDataset 惰性编码（多模态默认，坏样本自动跳过）
│
├─ tuners/  ⑤         tuner.py: 给所有线性层挂 LoRA，冻结基座，可训练参数 ~0.3%
│
├─ trainers/  ⑥⑦      TrainerFactory 选 Trainer → trainer.train()（transformers 训练循环）
│                     注入 data_collator / loss_scale / router_aux_loss
│
└─ 产出              output/.../checkpoint-xxx（adapter）+ args.json（配置快照）+ logging.jsonl
                     ↓
   swift infer --adapters ...    读 args.json 自动还原 model/template → 复用同一套 template 推理
   swift export --merge_lora     合并/量化/推 Hub
```

带圈数字对应第六章 6.2 节的"七步数据流"和各章节，方便你交叉定位。

---

## 十、给新手的学习路径与练习清单

按这个顺序动手，每步都有可验收产出：

1. **降配跑通**：用 `Qwen/Qwen3.5-4B`（单卡）跑完 2.1→2.2→2.3 三条命令，理解 sft/infer/export 闭环。
   ✅ 产出：一个能推理的 LoRA 模型。

2. **观察编码**：跑 5.4 的 debug 片段，亲眼看到 `labels` 里 user 部分是 `-100`。
   ✅ 产出：能说清"SFT 为什么只对 assistant 回答算 loss"。

3. **追一个参数**：选 `--experts_impl`，从命令行 → `SftArguments` 定义 → `ModelLoader.get_model`
   把它塞进 `model_kwargs` 的全过程追一遍。
   ✅ 产出：能徒手画出"一个参数如何一路传到生效代码行"。

4. **读 Loader 差异**：对比默认 `ModelLoader` 和 `Qwen3_5MoeLoader`，说清 MoE 模型多做了哪两件事。
   ✅ 产出：理解"为什么有的模型要写自定义 Loader"。

5. **改数据集**：用自己的一份 jsonl（messages 格式）替换官方数据集训练一次；
   再试试 `examples/custom/dataset.py` 的方式注册一个数据集。
   ✅ 产出：理解 `preprocess_func` 的唯一职责（原始字段 → messages）。

6. **（进阶）上 MoE 全参**：有多卡的话，跑 `mcore.sh`，理解 TP/EP 并行和 Mcore-Bridge。

### 配套源码地图（都已加中文注释）

| 模块 | 文件 | 看什么 |
|---|---|---|
| 入口 | `swift/cli/main.py` | 子命令路由、torchrun 启动 |
| 管线基类 | `swift/pipelines/base.py` | 所有命令的公共骨架 |
| **训练主线** | `swift/pipelines/train/sft.py` | SwiftSft 八步生命周期 |
| **模板** | `swift/template/base.py`、`template_meta.py` | encode/五要素/set_mode |
| 模型注册 | `swift/model/model_meta.py`、`register.py` | ModelMeta、自动匹配、Loader |
| 本模型实现 | `swift/model/models/qwen.py:1412` | Qwen3_5MoeLoader |
| 本模型模板 | `swift/template/templates/qwen.py:593` | Qwen3_5Template |
| 数据集 | `swift/dataset/loader.py`、`utils.py` | 数据流、LazyLLMDataset |

---

## 十一、常见坑与对策（针对本模型）

| 现象 | 原因 | 对策 |
|---|---|---|
| 启动报 `transformers` 版本错误 | 该模型 `requires` transformers≥5.2.0 | 按第一章装依赖；这是 `check_requires` 在提醒你 |
| `--experts_impl grouped_mm` 报错 | grouped_mm 需要 transformers≥5.0 | 升级 transformers，或暂用 `eager` |
| ZeRO-3 下 MoE 训练结果异常 | 专家模块未设为 z3_leaf | ms-swift 已内置处理（`register.py`），确认 model_type 正确识别即可 |
| 多模态 OOM | 图片/视频 token 太多 | 调小 `IMAGE_MAX_TOKEN_NUM` / `VIDEO_MAX_TOKEN_NUM` / `FPS_MAX_FRAMES` |
| 思考模型回答总带 `<think>` | 没关思考 | 推理加 `--enable_thinking false`；训练加 `--add_non_thinking_prefix true` |
| 想确认编码对不对 | 训推模板是否一致 | 用 5.4 的 debug 片段，或对比 `template_backend='jinja'` 的官方渲染 |

---

## 结语

Qwen3.6-35B-A3B 把 ms-swift 几个最有代表性的机制集于一身：**MoE 加载、多模态编码、混合思考、loss_scale、双训练后端**。
顺着"训练一个真实模型"的需求把这些点一个个啃下来，你掌握的就不只是一个模型的用法，
而是 ms-swift "Meta + 注册 + Mapping" 这套可复用的工程范式——
之后无论支持新模型、新数据集，还是读 RLHF/GRPO 代码，路径都是相通的。

下一步建议：完成第十章的练习 1→5，然后尝试给框架提第一个 PR（注册一个新模型或新数据集，套路完全一致）。
