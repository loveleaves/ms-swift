# Copyright (c) ModelScope Contributors. All rights reserved.

from copy import deepcopy
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizerBase
from typing import List, Optional, Type, Union

from .base import Template
from .utils import Prompt, Word


@dataclass
class TemplateMeta:
    """
    Examples:
        chatml (with bos):
            prefix: <s>
            prompt: <|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n
            chat_sep: <|im_end|>\n
            suffix: <|im_end|>
            system_prefix: <s><|im_start|>system\n{{SYSTEM}}<|im_end|>\n

        <s><|im_start|>system  # prefix or system_prefix
        {{SYSTEM}}<|im_end|>
        <|im_start|>user  # prompt
        {{QUERY}}<|im_end|>
        <|im_start|>assistant
        {{RESPONSE}}<|im_end|>  # chat_sep
        <|im_start|>user  # prompt
        {{QUERY}}<|im_end|>
        <|im_start|>assistant
        {{RESPONSE}}<|im_end|>  # suffix
    """
    # [新手导读] 模板的"配方"：注册一个新模板就是填好下面的五要素，
    # 拼接逻辑由 Template._swift_encode 统一实现。
    #   prefix:        整段对话的开头（如 bos），无 system 时使用
    #   prompt:        每轮 user 消息的包装格式，{{QUERY}} 为占位符
    #   chat_sep:      多轮对话中相邻轮次之间的分隔符；为 None 表示不支持多轮
    #   suffix:        最后一轮回复的结尾（通常是 eos），训练时标记生成结束
    #   system_prefix: 含 {{SYSTEM}} 占位符的开头，有 system 时替代 prefix
    # Prompt 类型是 List[str | List[int]]：字符串会被 tokenize，
    # 列表（如 ['eos_token_id']）会在 init() 中被解析为 token id。
    template_type: str
    prefix: Prompt
    prompt: Prompt
    chat_sep: Optional[Prompt]
    suffix: Prompt = field(default_factory=lambda: [['eos_token_id']])
    template_cls: Type[Template] = Template  # 多模态等复杂模板可指定自定义 Template 子类
    system_prefix: Optional[Prompt] = None
    default_system: Optional[str] = None

    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=list)
    agent_template: Optional[str] = None
    # thinking
    is_thinking: bool = False  # Automatically remove think content
    thinking_prefix: str = ''
    non_thinking_prefix: str = ''  # Automatically add non_thinking_prefix for hybrid thinking models
    # During encoding, historical thinking content will be removed.
    # This parameter represents the prefix for the historical part.
    history_thinking_prefix: str = ''

    def to_generate_template_meta(self) -> 'TemplateMeta':
        self = deepcopy(self)
        return TemplateMeta(
            self.template_type,
            prefix=[],
            prompt=['{{QUERY}}'],
            chat_sep=None,
            template_cls=self.template_cls,
            auto_add_bos=True,
            stop_words=self.stop_words,
        )

    @staticmethod
    def _has_system(prefix_or_prompt: Prompt) -> bool:
        return any(['{{SYSTEM}}' in p for p in prefix_or_prompt])

    @staticmethod
    def _replace_system(prefix: Prompt) -> Prompt:
        return [p.replace('{{SYSTEM}}', '') for p in prefix if isinstance(p, str)]

    def _check_template_meta(self):
        # check
        for x in [self.prefix, self.prompt, self.suffix]:
            assert isinstance(x, list)
        for x in [self.chat_sep, self.system_prefix]:
            assert x is None or isinstance(x, list)

    def __post_init__(self):
        # 规整化：允许注册者把 {{SYSTEM}} 直接写在 prefix 里（自动拆分出 system_prefix），
        # 或写在 prompt 里（system 跟在 user 消息中的"后置 system"，如 mistral_nemo）。
        # 并据此推导 support_system / support_multi_round 两个能力标记。
        # system
        if self._has_system(self.prefix):
            assert self.system_prefix is None, 'The prefix already contains {{SYSTEM}}.'
            self.system_prefix = self.prefix
            self.prefix = self._replace_system(self.prefix)

        self.is_post_system = self._has_system(self.prompt)  # mistral_nemo
        if self.is_post_system:
            self.system_prompt = self.prompt
            self.prompt = [context for context in self.prompt if '{{SYSTEM}}' not in context]

        if self.system_prefix is None and not self.is_post_system:
            self.support_system = False
        else:
            self.support_system = True
        self.check_system(self.default_system)

        self.support_multi_round = self.chat_sep is not None

    @staticmethod
    def _token_attr_to_id(tokenizer: PreTrainedTokenizerBase, value: Optional[Prompt]) -> Optional[Prompt]:
        """Turn `eos_token_id` to token id

        e.g. [['eos_token_id']] -> [[2]]
        """
        if value is None:
            return None
        res_value = []
        for v in value:
            if isinstance(v, list):
                v = [getattr(tokenizer, sub_v) if isinstance(sub_v, str) else sub_v for sub_v in v]
            res_value.append(v)
        return res_value

    def init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # 绑定具体 tokenizer：把 'eos_token_id' 这类符号名解析为真实 token id，
        # 并从 suffix 推导停止词（stop_words）与停止 token，供推理时判断生成结束。
        for key in ['prefix', 'prompt', 'chat_sep', 'suffix', 'system_prefix']:
            value = getattr(self, key)
            value = self._token_attr_to_id(tokenizer, value)
            setattr(self, key, value)

        suffix_stop = self.suffix[-1] if self.suffix else None
        if isinstance(suffix_stop, str):
            suffix_stop = suffix_stop.strip()
        self.suffix_stop = suffix_stop
        if suffix_stop and suffix_stop not in self.stop_words:
            self.stop_words.append(suffix_stop)
        if tokenizer.eos_token not in self.stop_words:
            self.stop_words.append(tokenizer.eos_token)

        self.stop_token_id = tokenizer.eos_token_id
        if suffix_stop:
            if isinstance(suffix_stop, str):
                stop_token_id = tokenizer.convert_tokens_to_ids(suffix_stop)
            elif isinstance(suffix_stop, list) and len(suffix_stop) == 1:
                stop_token_id = suffix_stop[0]
            else:
                stop_token_id = None
            if stop_token_id is not None:
                self.stop_token_id = stop_token_id

    def check_system(self, system: Optional[str]) -> None:
        if system is not None:
            assert self.support_system, (
                f'The template does not support `system`, template_type: {self.template_type}, system: {system}')
