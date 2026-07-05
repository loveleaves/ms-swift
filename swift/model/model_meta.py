# Copyright (c) ModelScope Contributors. All rights reserved.

import os
import platform
import re
import torch
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils.versions import require_version
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from swift.utils import HfConfigFactory, get_logger, safe_snapshot_download
from .utils import get_default_torch_dtype

logger = get_logger()


@dataclass
class Model:
    ms_model_id: Optional[str] = None
    hf_model_id: Optional[str] = None
    model_path: Optional[str] = None

    ms_revision: Optional[str] = None
    hf_revision: Optional[str] = None


@dataclass
class ModelGroup:
    models: List[Model]

    # Higher priority. If set to None, the attributes of the ModelMeta will be used.
    template: Optional[str] = None
    ignore_patterns: Optional[List[str]] = None
    requires: Optional[List[str]] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        assert not isinstance(self.template, (list, tuple))  # check ms-swift4.0
        assert isinstance(self.models, (tuple, list)), f'self.models: {self.models}'


class BaseModelLoader(ABC):

    @abstractmethod
    def __init__(self, model_info, model_meta, *args, **kwargs):
        pass

    @abstractmethod
    def load(self) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
        pass


@dataclass
class ModelMeta:
    # [新手导读] 一类模型的"注册信息卡"。注册新模型支持时，主要就是填写这个 dataclass：
    #   model_type:   模型结构的唯一 ID（同一 model_type 共享模板、加载方式等）
    #   model_groups: 该类型下的具体模型列表（魔搭/HF 双 id），用于按模型名自动匹配 model_type
    #   template:     默认对话模板（对应 TEMPLATE_MAPPING 中的 key）
    #   model_arch:   多模态模型中 llm/vit/aligner 各部分的参数前缀映射，
    #                 决定 --freeze_vit 等开关冻结哪些参数（见 model_arch.py）
    #   architectures: config.json 中的 architectures 字段，是自动匹配 model_type 的第二依据
    model_type: Optional[str]
    # Used to list the model_ids from modelscope/huggingface,
    # which participate in the automatic inference of the model_type.
    model_groups: List[ModelGroup]
    loader: Optional[Type[BaseModelLoader]] = None

    template: Optional[str] = None
    model_arch: Optional[str] = None
    mcore_model_type: Optional[str] = None
    architectures: List[str] = field(default_factory=list)
    # Additional files that need to be saved for full parameter training/merge-lora.
    additional_saved_files: List[str] = field(default_factory=list)
    torch_dtype: Optional[torch.dtype] = None

    is_multimodal: bool = False
    is_reward: bool = False
    task_type: Optional[str] = None

    # File patterns to ignore when downloading the model.
    ignore_patterns: Optional[List[str]] = None
    # Usually specifies the version limits of transformers.
    requires: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        from .constant import MLLMModelType, RMModelType
        from .register import ModelLoader
        assert not isinstance(self.loader, str)  # check ms-swift4.0
        if self.loader is None:
            self.loader = ModelLoader
        if not isinstance(self.model_groups, (list, tuple)):
            self.model_groups = [self.model_groups]
        self.candidate_templates = list(
            dict.fromkeys(t for t in [self.template] + [mg.template for mg in self.model_groups] if t is not None))
        if self.model_type in MLLMModelType.__dict__:
            self.is_multimodal = True
        if self.model_type in RMModelType.__dict__:
            self.is_reward = True

    def get_matched_model_group(self, model_name: str) -> Optional[ModelGroup]:
        for model_group in self.model_groups:
            for model in model_group.models:
                for key in ['ms_model_id', 'hf_model_id', 'model_path']:
                    value = getattr(model, key)

                    if isinstance(value, str) and model_name == value.rsplit('/', 1)[-1].lower():
                        return model_group

    def check_requires(self, model_info=None):
        extra_requires = []
        if model_info and model_info.quant_method:
            mapping = {'bnb': ['bitsandbytes'], 'awq': ['autoawq'], 'gptq': ['auto_gptq'], 'aqlm': ['aqlm']}
            extra_requires += mapping.get(model_info.quant_method, [])
        requires = []
        for require in self.requires + extra_requires:
            try:
                require_version(require)
            except ImportError:
                requires.append(f'"{require}"')
        if requires:
            requires = ' '.join(requires)
            logger.warning(f'Please install the package: `pip install {requires} -U`.')


# 全局模型注册表：model_type -> ModelMeta。内置模型在 models/ 目录下通过
# register_model() 填充；--model_type 等参数最终都是来这里查表。
MODEL_MAPPING: Dict[str, ModelMeta] = {}


@dataclass
class ModelInfo:
    # 与 ModelMeta 的区别：ModelMeta 是"注册时"的静态元信息（一类模型一份），
    # ModelInfo 是"运行时"针对具体某个模型目录解析出的信息（dtype、量化方式、最大长度等）。
    model_type: str
    model_dir: str
    torch_dtype: torch.dtype
    max_model_len: int
    quant_method: Literal['gptq', 'awq', 'bnb', 'aqlm', 'hqq', None]
    quant_bits: int

    # extra
    rope_scaling: Optional[Dict[str, Any]] = None
    is_moe_model: bool = False
    is_multimodal: bool = False
    config: Optional[PretrainedConfig] = None
    task_type: Optional[str] = None
    num_labels: Optional[int] = None

    def __post_init__(self):
        self.model_name = get_model_name(self.model_dir)


def get_model_name(model_id_or_path: str) -> Optional[str]:
    assert isinstance(model_id_or_path, str), f'model_id_or_path: {model_id_or_path}'
    # compat hf hub
    model_id_or_path = model_id_or_path.rstrip('/')
    match_ = re.search('/models--.+?--(.+?)/snapshots/', model_id_or_path)
    if match_ is not None:
        return match_.group(1)

    model_name = model_id_or_path.rsplit('/', 1)[-1]
    if platform.system().lower() == 'windows':
        model_name = model_name.rsplit('\\', 1)[-1]
    # compat modelscope snapshot_download
    model_name = model_name.replace('___', '.')
    return model_name


def get_matched_model_meta(model_id_or_path: str) -> Optional[ModelMeta]:
    # 自动匹配 model_type 的第一依据：用 --model 的最后一段名称（如 Qwen3-8B）
    # 与注册表中所有 ModelGroup 的模型 id 后缀逐一比对。
    model_name = get_model_name(model_id_or_path).lower()
    for model_type, model_meta in MODEL_MAPPING.items():
        model_group = ModelMeta.get_matched_model_group(model_meta, model_name)
        if model_group is not None:
            model_meta = deepcopy(model_meta)
            for k, v in asdict(model_group).items():
                if v is not None and k in model_meta.__dict__:
                    setattr(model_meta, k, v)
            return model_meta


def _get_arch_mapping():
    res = {}
    for model_type, model_meta in MODEL_MAPPING.items():
        architectures = model_meta.architectures
        if not architectures:
            architectures.append('null')
        for arch in architectures:
            if arch not in res:
                res[arch] = []
            res[arch].append(model_type)
    return res


def get_matched_model_types(architectures: Optional[List[str]]) -> List[str]:
    """Get possible model_type."""
    architectures = architectures or ['null']
    if architectures:
        architectures = architectures[0]
    arch_mapping = _get_arch_mapping()
    return arch_mapping.get(architectures) or []


def _read_args_json_model_type(model_dir):
    if not os.path.exists(os.path.join(model_dir, 'args.json')):
        return
    from swift.arguments import BaseArguments
    args = BaseArguments.from_pretrained(model_dir)
    return args.model_type


def _get_model_info(model_dir: str, model_type: Optional[str], quantization_config) -> ModelInfo:
    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        config = PretrainedConfig.get_config_dict(model_dir)[0]
    if quantization_config is not None:
        HfConfigFactory.set_config_attr(config, 'quantization_config', quantization_config)
    quant_info = HfConfigFactory.get_quant_info(config) or {}
    torch_dtype = HfConfigFactory.get_torch_dtype(config, quant_info)
    max_model_len = HfConfigFactory.get_max_model_len(config)
    rope_scaling = HfConfigFactory.get_config_attr(config, 'rope_scaling')
    is_moe_model = HfConfigFactory.is_moe_model(config)
    is_multimodal = HfConfigFactory.is_multimodal(config)

    if model_type is None:
        model_type = _read_args_json_model_type(model_dir)
    if model_type is None:
        architectures = HfConfigFactory.get_config_attr(config, 'architectures')
        model_types = get_matched_model_types(architectures)
        if len(model_types) > 1:
            raise ValueError(f'Failed to automatically match `model_type` for `{model_dir}`. '
                             f'Multiple possible types found: {model_types}. '
                             'Please specify `model_type` manually. See documentation: '
                             'https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html')
        elif len(model_types) == 1:
            model_type = model_types[0]
    elif model_type not in MODEL_MAPPING:
        raise ValueError(f"model_type: '{model_type}' not in {list(MODEL_MAPPING.keys())}")

    res = ModelInfo(
        model_type,
        model_dir,
        torch_dtype,
        max_model_len,
        quant_info.get('quant_method'),
        quant_info.get('quant_bits'),
        rope_scaling=rope_scaling,
        is_moe_model=is_moe_model,
        is_multimodal=is_multimodal,
    )
    return res


def get_model_info_meta(
        model_id_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        # hub
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
        download_model: bool = False,
        # model kwargs
        model_type: Optional[str] = None,
        quantization_config=None,
        task_type=None,
        num_labels=None,
        problem_type=None,
        **kwargs) -> Tuple[ModelInfo, ModelMeta]:
    # [新手导读] 模型加载前的"识别"阶段，回答两个问题：这是哪类模型（model_meta）、
    # 这个模型目录的实际配置是什么（model_info）。匹配顺序：
    #   1. 按模型名后缀匹配注册表（get_matched_model_meta）；
    #   2. 下载/定位模型目录（默认 ModelScope，--use_hf 切 HuggingFace）；
    #   3. 仍未确定时读 checkpoint 的 args.json 或按 config.json 的 architectures 匹配；
    #   4. 纯文本模型实在匹配不到则降级为 'dummy' 模板的临时 meta（多模态则直接报错）。
    from .register import ModelLoader
    model_meta = get_matched_model_meta(model_id_or_path)
    model_dir = safe_snapshot_download(
        model_id_or_path,
        revision=revision,
        download_model=download_model,
        use_hf=use_hf,
        ignore_patterns=getattr(model_meta, 'ignore_patterns', None),
        hub_token=hub_token)

    model_type = model_type or getattr(model_meta, 'model_type', None)
    model_info = _get_model_info(model_dir, model_type, quantization_config=quantization_config)
    if model_type is None and model_info.model_type is not None:
        model_type = model_info.model_type
        logger.info(f'Setting model_type: {model_type}')
    if model_type is not None and (model_meta is None or model_meta.model_type != model_type):
        model_meta = MODEL_MAPPING[model_type]
    if model_meta is None:  # not found
        if model_info.is_multimodal:
            raise ValueError(f'Model "{model_id_or_path}" is not supported because no suitable `model_type` was found. '
                             'Please refer to the documentation and specify an appropriate `model_type` manually: '
                             'https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html')
        else:
            model_meta = ModelMeta(None, [], ModelLoader, template='dummy', model_arch=None)
            logger.info(f'Temporarily create model_meta: {model_meta}')
    if torch_dtype is None:
        torch_dtype = model_meta.torch_dtype or get_default_torch_dtype(model_info.torch_dtype)
        logger.info(f'Setting torch_dtype: {torch_dtype}')
    model_info.torch_dtype = torch_dtype
    if task_type is None:
        if model_meta.is_reward:
            num_labels = 1
        if num_labels is None:
            task_type = 'causal_lm'
        else:
            task_type = 'seq_cls'
        if model_meta.task_type is not None:
            task_type = model_meta.task_type

    # Handle reranker task type
    if task_type == 'reranker':
        if num_labels is None:
            num_labels = 1  # Default to 1 for reranker tasks
        logger.info(f'Setting reranker task with num_labels={num_labels}')
    elif task_type == 'generative_reranker':
        # Generative reranker doesn't need num_labels as it uses CausalLM structure
        num_labels = None
        logger.info('Setting generative_reranker task (no num_labels needed)')
    elif task_type == 'seq_cls':
        assert num_labels is not None, 'Please pass the parameter `num_labels`.'
        if problem_type is None:
            if num_labels == 1:
                problem_type = 'regression'
            else:
                problem_type = 'single_label_classification'

    model_info.task_type = task_type
    model_info.num_labels = num_labels
    model_info.problem_type = problem_type
    if model_meta.is_multimodal:
        model_info.is_multimodal = True
    model_meta.check_requires(model_info)
    return model_info, model_meta
