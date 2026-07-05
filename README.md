# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)

version: V4.3.0

[guide](./guide.md)

## set up

```
# 1. install uenv
curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. create
uv venv .venv --python 3.12
# 3. install
uv pip install \
torch torchvision torchaudio \
--index-url https://download.pytorch.org/whl/cu124

cd ms-swift
pip install -e .
# 该模型对依赖版本有要求（来自上面 ModelMeta 的 requires 字段）
pip install 'transformers>=5.2.0' 'qwen_vl_utils>=0.0.14' decord -U
# MoE 高效训练还需要（grouped_mm 专家算子、flash-attn）：
pip install flash-attn --no-build-isolation
```
