#!/usr/bin/env python3
"""Minimal but complete Qwen3-1.7B LoRA SFT with Transformers + PEFT.

Pipeline:
  local JSON/JSONL -> messages -> ms-swift-compatible Qwen3 ChatML
  -> assistant-only labels -> dynamic padding -> base model + LoRA injection
  -> Trainer/Accelerate (single GPU or DDP) -> eval -> resumable checkpoints
  -> adapter-only export -> deterministic generation -> optional merged export.

The implementation deliberately does not import ms-swift.  Comments named
``SFT stage`` explain (1) what the code does, (2) where it sits in the SFT
pipeline, and (3) the corresponding source location in the current ms-swift
repository.  It is therefore both an executable minimum and a reading map.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import peft
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


IGNORE_INDEX = -100
IM_START = '<|im_start|>'
IM_END = '<|im_end|>'
NON_THINKING_PREFIX = '<think>\n\n</think>\n\n'
FORBIDDEN_CONTENT_MARKERS = (IM_START, IM_END)


@dataclass
class DatasetStats:
    source_rows: int = 0
    kept_rows: int = 0
    dropped_overlength: int = 0
    total_tokens: int = 0
    target_tokens: int = 0


class TokenizedSFTDataset(Dataset):
    """Eagerly encode conversations and construct assistant-only labels.

    SFT stage 1-3 -- data preprocessing, template encoding and target masks.
    Function: normalize every row, render the Qwen3 ChatML segments used by
    ms-swift, reject unusable long rows, and keep input/label lengths aligned.
    ms-swift: ``swift/pipelines/train/sft.py::SwiftSft._prepare_dataset`` and
    ``_encode_dataset``; generalized encoding lives in
    ``swift/template/base.py::Template.encode/_encode/_encode_context_list``.
    Qwen3's thinking/non-thinking prefix is registered in
    ``swift/template/templates/qwen.py::Qwen3Template``.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        max_length: int,
        overlength_strategy: str,
        enable_thinking: bool,
        name: str,
    ) -> None:
        self.features: list[dict[str, list[int]]] = []
        self.stats = DatasetStats(source_rows=len(records))

        for row_index, record in enumerate(records, start=1):
            messages = normalize_record(record, row_index=row_index, source_name=name)
            input_ids, labels = encode_qwen3_messages(
                messages,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
            )

            if len(input_ids) > max_length:
                if overlength_strategy == 'drop':
                    self.stats.dropped_overlength += 1
                    continue
                # Left truncation preserves the most recent user/assistant turn,
                # which is usually more useful than cutting off the answer tail.
                input_ids = input_ids[-max_length:]
                labels = labels[-max_length:]

            target_tokens = sum(label != IGNORE_INDEX for label in labels)
            if target_tokens == 0:
                raise ValueError(f'{name} row {row_index}: no assistant target tokens remain after encoding.')
            if len(input_ids) != len(labels):
                raise AssertionError('input_ids and labels must have identical lengths.')

            self.features.append({
                'input_ids': input_ids,
                'attention_mask': [1] * len(input_ids),
                'labels': labels,
            })
            self.stats.total_tokens += len(input_ids)
            self.stats.target_tokens += target_tokens

        self.stats.kept_rows = len(self.features)
        if not self.features:
            raise ValueError(
                f'{name}: no usable records. Increase --max_length, select '
                '--overlength_strategy left, or repair the dataset.')

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


class QwenSFTTrainer(Trainer):
    """Trainer with Qwen3's optional selective-logit optimization.

    SFT stage 6 -- forward/loss memory optimization.  Qwen3 can calculate LM
    logits only at positions needed by the labels.  This does not alter the
    causal cross-entropy objective; it avoids materializing a
    ``batch x sequence x vocabulary`` tensor for masked prompt positions.
    ms-swift: capability detection is in
    ``swift/trainers/mixin.py::get_use_logits_to_keep``; mask construction is
    ``prepare_logits_to_keep``; injection occurs from
    ``swift/trainers/seq2seq_trainer.py::_prepare_inputs``.
    """

    def __init__(self, *args, use_logits_to_keep: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        base_model = self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model
        supports_selective_logits = 'logits_to_keep' in inspect.signature(base_model.forward).parameters
        self.use_logits_to_keep = use_logits_to_keep and supports_selective_logits
        if self.is_world_process_zero():
            print(
                '[trainer] use_logits_to_keep='
                f'{self.use_logits_to_keep} (model_support={supports_selective_logits})')

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        inputs = super()._prepare_inputs(inputs)
        if not self.use_logits_to_keep or 'labels' not in inputs:
            return inputs

        labels = inputs['labels']
        if labels.shape[0] == 1:
            # A boolean selector is maximally efficient for batch size one: it
            # retains every prediction immediately before a supervised token.
            loss_mask = (labels != IGNORE_INDEX)[0]
            inputs['labels'] = F.pad(labels[:, loss_mask], (1, 0), value=IGNORE_INDEX)
            inputs['logits_to_keep'] = F.pad(loss_mask[1:], (0, 1), value=True)
        else:
            # A common suffix is required for a rectangular batch.  Keep from
            # the earliest first-label position (plus the preceding causal logit).
            first_target = (labels != IGNORE_INDEX).int().argmax(dim=-1).min().item()
            logits_to_keep = labels.shape[-1] - first_target + 1
            if logits_to_keep <= 0:
                raise RuntimeError('Every SFT batch must contain supervised tokens.')
            inputs['labels'] = labels[:, -logits_to_keep:]
            inputs['logits_to_keep'] = logits_to_keep
        return inputs


def parse_args() -> argparse.Namespace:
    """Expose the complete, intentionally small experiment contract.

    SFT stage 0 -- configuration.  ms-swift's counterparts are
    ``swift/arguments/sft_args.py::SftArguments`` and
    ``swift/arguments/tuner_args.py::TunerArguments``.  Defaults below mirror
    its mainstream PEFT LoRA settings: all-linear, r=8, alpha=32,
    dropout=0.05, bias=none, learning rate 1e-4 and cosine scheduling.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_name_or_path', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--train_file', required=True, help='Local JSONL or JSON array.')
    parser.add_argument('--eval_file', default=None, help='Optional independent JSONL/JSON validation set.')
    parser.add_argument('--output_dir', default='output/qwen3-1.7b-transformers-lora-sft')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--overlength_strategy', choices=['drop', 'left'], default='drop')
    parser.add_argument('--enable_thinking', action=argparse.BooleanOptionalAction, default=False)

    # SFT stage 4 -- PEFT adapter topology.  ``all-linear`` is expanded before
    # LoraConfig so the exact Qwen3 targets are visible and auditable.
    # ms-swift: defaults are in swift/arguments/tuner_args.py; expansion is
    # swift/pipelines/train/tuner.py::get_target_modules and
    # swift/utils/transformers_utils.py::find_all_linears.
    parser.add_argument(
        '--target_modules',
        default='all-linear',
        help='all-linear or comma-separated module suffixes, e.g. q_proj,v_proj.',
    )
    parser.add_argument(
        '--modules_to_save',
        default='',
        help='Optional comma-separated full modules trained and saved with LoRA, e.g. lm_head.',
    )
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', choices=['none', 'all'], default='none')
    parser.add_argument('--use_rslora', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_dora', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--init_lora_weights',
        choices=['true', 'false', 'gaussian'],
        default='true',
        help='PEFT initialization; true is the standard zero-delta LoRA initialization.',
    )
    parser.add_argument(
        '--lora_dtype',
        choices=['auto', 'fp32', 'bf16'],
        default='auto',
        help='Optional LoRA A/B dtype. PEFT auto normally keeps trainable LoRA weights in FP32.',
    )

    parser.add_argument('--num_train_epochs', type=float, default=1.0)
    parser.add_argument('--max_steps', type=int, default=-1, help='Positive value overrides epochs.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.95)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--lr_scheduler_type', default='cosine')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--optim', default='adamw_torch')
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_logits_to_keep', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--precision', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    parser.add_argument(
        '--attn_implementation',
        choices=['auto', 'eager', 'sdpa', 'flash_attention_2'],
        default='auto',
    )
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--full_determinism', action=argparse.BooleanOptionalAction, default=False)

    # There are deliberately two recovery modes.  A Trainer checkpoint resumes
    # global step + optimizer + scheduler + RNG; adapter_name_or_path only uses
    # existing LoRA weights as a trainable initialization for a new run.
    parser.add_argument(
        '--resume_from_checkpoint',
        nargs='?',
        const='auto',
        default=None,
        help='Full checkpoint path, or flag without value for latest checkpoint-* in output_dir.',
    )
    parser.add_argument(
        '--adapter_name_or_path',
        default=None,
        help='Adapter-only warm start; optimizer, scheduler and global step start over.',
    )

    parser.add_argument('--merge_lora', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--safe_merge', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--trust_remote_code', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--local_files_only', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test_prompt', default='请用两句话解释什么是有监督微调。')
    parser.add_argument('--system_prompt', default='你是一个准确、简洁的中文助手。')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--dry_run', action='store_true', help='Validate tokenizer/data/labels without model loading.')
    return parser.parse_args()


def read_records(path: str) -> list[dict[str, Any]]:
    """Read a JSON array or JSONL stream.

    SFT stage 1 -- ingestion.  ms-swift's production loader is
    ``swift/dataset/loader.py::load_dataset`` and additionally supports hub
    datasets, streaming, sampling and registered preprocessors.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    if file_path.suffix.lower() == '.json':
        payload = json.loads(file_path.read_text(encoding='utf-8'))
        if isinstance(payload, dict) and isinstance(payload.get('data'), list):
            payload = payload['data']
        if not isinstance(payload, list):
            raise TypeError(f'{file_path}: JSON must be an array or an object with a data array.')
        records = payload
    else:
        records = []
        with file_path.open('r', encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f'{file_path}:{line_number}: invalid JSON: {error}') from error

    if not records or not all(isinstance(row, dict) for row in records):
        raise ValueError(f'{file_path}: expected at least one JSON object.')
    return records


def normalize_record(record: dict[str, Any], row_index: int, source_name: str) -> list[dict[str, str]]:
    """Normalize messages and instruction schemas; validate role grammar.

    SFT stage 1 -- schema normalization and quality gate.  ms-swift performs the
    generalized conversion in
    ``swift/dataset/preprocessor/core.py::RowPreprocessor`` and
    ``MessagesPreprocessor``.  Strict marker/role checks make the compact ChatML
    encoder below unambiguous.
    """
    if isinstance(record.get('messages'), list):
        messages = record['messages']
    elif 'instruction' in record and ('output' in record or 'response' in record):
        user_content = str(record['instruction'])
        if record.get('input'):
            user_content += f"\n{record['input']}"
        messages = []
        if record.get('system'):
            messages.append({'role': 'system', 'content': str(record['system'])})
        messages.extend([
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': str(record.get('output', record.get('response')))},
        ])
    else:
        raise ValueError(
            f'{source_name} row {row_index}: expected messages or instruction/(output|response).')

    normalized: list[dict[str, str]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f'{source_name} row {row_index}: message {message_index} is not an object.')
        role = message.get('role')
        content = message.get('content')
        if role not in {'system', 'user', 'assistant'}:
            raise ValueError(
                f'{source_name} row {row_index}: unsupported role {role!r}; '
                'this text-only minimum supports system/user/assistant.')
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f'{source_name} row {row_index}: empty content at message {message_index}.')
        if any(marker in content for marker in FORBIDDEN_CONTENT_MARKERS):
            raise ValueError(f'{source_name} row {row_index}: content may not contain Qwen control markers.')
        normalized.append({'role': role, 'content': content})

    cursor = 1 if normalized and normalized[0]['role'] == 'system' else 0
    expected_role = 'user'
    for message in normalized[cursor:]:
        if message['role'] != expected_role:
            raise ValueError(
                f'{source_name} row {row_index}: expected {expected_role!r}, got {message["role"]!r}.')
        expected_role = 'assistant' if expected_role == 'user' else 'user'
    if not normalized or normalized[-1]['role'] != 'assistant':
        raise ValueError(f'{source_name} row {row_index}: a training conversation must end with assistant.')
    return normalized


def encode_qwen3_messages(
    messages: list[dict[str, str]],
    tokenizer,
    enable_thinking: bool,
) -> tuple[list[int], list[int]]:
    """Encode the current ms-swift Qwen3 text-template subset exactly.

    SFT stage 2-3 -- ChatML rendering and label construction.  Headers and
    system/user content receive ``-100``.  Every assistant response, its
    non-thinking prefix, ``<|im_end|>`` and trailing newline are supervised.
    Encoding segment-by-segment is intentional: ms-swift constructs a context
    list with per-segment loss weights in
    ``swift/template/base.py::_encode_context_list`` and applies dynamic EOS in
    ``_add_dynamic_eos``.  ChatML fragments are declared in
    ``swift/template/templates/utils.py::CHATML_TEMPLATE_META``; Qwen3's empty
    think prefix is in ``swift/template/templates/qwen.py``.

    The official Qwen tokenizer Jinja only inserts an empty think block in the
    final assistant turn.  ms-swift's SFT template inserts it for each ordinary
    assistant answer, so explicitly rendering here matters for multi-turn data.
    """
    input_ids: list[int] = []
    labels: list[int] = []

    def append(text: str, supervise: bool) -> None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(token_ids)
        labels.extend(token_ids if supervise else [IGNORE_INDEX] * len(token_ids))

    for message in messages:
        role = message['role']
        content = message['content']
        append(f'{IM_START}{role}\n', supervise=False)
        if role == 'assistant':
            if not enable_thinking and not content.startswith(('<think>', NON_THINKING_PREFIX)):
                content = NON_THINKING_PREFIX + content
            append(content, supervise=True)
            append(f'{IM_END}\n', supervise=True)
        else:
            append(content, supervise=False)
            append(f'{IM_END}\n', supervise=False)

    return input_ids, labels


def split_records(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic holdout if no eval file is supplied.

    SFT stage 1 -- split.  ms-swift routes this through
    ``swift/pipelines/train/sft.py::SwiftSft._get_dataset`` and the loader's
    ``split_dataset_ratio`` setting.
    """
    if len(records) < 2:
        raise ValueError('At least two records are required when --eval_file is absent.')
    if not 0.0 < val_ratio < 1.0:
        raise ValueError('--val_ratio must be between 0 and 1.')
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    eval_size = max(1, min(len(records) - 1, round(len(records) * val_ratio)))
    eval_indices = set(indices[:eval_size])
    return (
        [row for index, row in enumerate(records) if index not in eval_indices],
        [row for index, row in enumerate(records) if index in eval_indices],
    )


def choose_precision(name: str) -> tuple[torch.dtype, bool, bool]:
    """Resolve base-weight load dtype and Trainer mixed-precision flags.

    SFT stage 4-5 -- numerical policy.  ms-swift resolves ``dtype`` in its model
    arguments/loader and passes bf16/fp16 to TrainingArguments from
    ``swift/trainers/trainer_factory.py::get_training_args``.
    """
    if name == 'auto':
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            name = 'bf16'
        elif torch.cuda.is_available():
            name = 'fp16'
        else:
            name = 'fp32'
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[name]
    return dtype, name == 'bf16', name == 'fp16'


def parse_module_list(value: str) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in value.split(',') if item.strip()))


def resolve_target_modules(model, specification: str) -> list[str]:
    """Expand all-linear and validate every requested module suffix.

    SFT stage 4 -- choose the matrices receiving low-rank deltas.  This mirrors
    ``swift/pipelines/train/tuner.py::get_target_modules`` and
    ``swift/utils/transformers_utils.py::find_all_linears``.  Like ms-swift, it
    excludes output heads.  Qwen3-1.7B therefore resolves to q/k/v/o projections
    plus gate/up/down MLP projections, while ``lm_head`` remains frozen.
    """
    linear_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    if specification.strip().lower() == 'all-linear':
        excluded_suffixes = {'lm_head', 'score', 'v_head', 'classifier'}
        targets = sorted({name.rsplit('.', maxsplit=1)[-1] for name in linear_names
                          if name.rsplit('.', maxsplit=1)[-1] not in excluded_suffixes})
    else:
        targets = parse_module_list(specification)
    if not targets:
        raise ValueError('--target_modules resolved to an empty list.')
    missing = [target for target in targets if not any(
        name == target or name.endswith(f'.{target}') for name in linear_names)]
    if missing:
        raise ValueError(f'LoRA target module suffixes not found in the base model: {missing}')
    return targets


def validate_modules_to_save(model, modules_to_save: list[str]) -> None:
    module_names = [name for name, _ in model.named_modules()]
    missing = [target for target in modules_to_save if not any(
        name == target or name.endswith(f'.{target}') for name in module_names)]
    if missing:
        raise ValueError(f'--modules_to_save entries not found in the base model: {missing}')


def convert_init_lora_weights(value: str) -> bool | str:
    if value == 'true':
        return True
    if value == 'false':
        return False
    return value


def create_or_load_lora_model(
    base_model,
    args: argparse.Namespace,
    target_modules: list[str],
    modules_to_save: list[str],
    resume_checkpoint: str | None,
) -> PeftModel:
    """Freeze the base and either inject or restore a trainable PEFT adapter.

    SFT stage 4 -- parameter-efficient model construction.  Fresh injection
    corresponds to ``swift/pipelines/train/tuner.py::prepare_adapter`` followed
    by ``Swift.prepare_model``.  Loading a checkpoint or adapter corresponds to
    ``swift/pipelines/train/tuner.py::TunerMixin.prepare_model``, which invokes
    ``from_pretrained(..., is_trainable=True)``.  PEFT then owns the actual
    low-rank A/B module injection in both projects.
    """
    adapter_source = resume_checkpoint or args.adapter_name_or_path
    base_model.requires_grad_(False)
    if adapter_source:
        model = PeftModel.from_pretrained(base_model, adapter_source, is_trainable=True)
    else:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save or None,
            bias=args.lora_bias,
            use_rslora=args.use_rslora,
            use_dora=args.use_dora,
            init_lora_weights=convert_init_lora_weights(args.init_lora_weights),
        )
        model = get_peft_model(base_model, config, autocast_adapter_dtype=True)

    # Match ms-swift's FP16-gradient safety fix in TunerMixin.prepare_model:
    # trainable FP16 tensors are promoted because GradScaler cannot unscale them.
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype == torch.float16:
            parameter.data = parameter.data.float()

    if args.lora_dtype != 'auto':
        requested_dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[args.lora_dtype]
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and 'lora_' in name:
                parameter.data = parameter.data.to(dtype=requested_dtype)
    return model


def audit_lora_model(
    model: PeftModel,
    target_modules: list[str],
    modules_to_save: list[str],
) -> dict[str, Any]:
    """Prove that only the intended adapter-side parameters are trainable.

    SFT stage 4 -- trainability audit.  ms-swift freezes the base in
    ``swift/pipelines/train/tuner.py::TunerMixin.prepare_model`` and reports
    parameter statistics through its model-info utilities.  This hard check
    catches a surprisingly costly failure mode: accidentally running full SFT.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_items = [(name, parameter) for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    trainable = sum(parameter.numel() for _, parameter in trainable_items)
    if trainable == 0 or trainable >= total:
        raise RuntimeError(f'Invalid LoRA trainability: trainable={trainable:,}, total={total:,}.')

    unexpected = []
    for name, _ in trainable_items:
        allowed = 'lora_' in name or 'modules_to_save' in name
        if model.peft_config['default'].bias == 'all' and name.endswith('.bias'):
            allowed = True
        if not allowed:
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(f'Unexpected trainable base parameters: {unexpected[:20]}')

    adapted_suffixes = sorted({
        name.rsplit('.', maxsplit=1)[-1]
        for name, module in model.named_modules()
        if hasattr(module, 'lora_A') and 'default' in module.lora_A
    })
    missing_targets = [target for target in target_modules if target not in adapted_suffixes]
    if missing_targets:
        raise RuntimeError(f'PEFT did not inject adapters into requested targets: {missing_targets}')

    return {
        'total_parameters': total,
        'trainable_parameters': trainable,
        'trainable_ratio': trainable / total,
        'target_modules': target_modules,
        'adapted_module_suffixes': adapted_suffixes,
        'modules_to_save': modules_to_save,
        'trainable_parameter_names': [name for name, _ in trainable_items],
    }


def print_dataset_audit(name: str, dataset: TokenizedSFTDataset, tokenizer) -> None:
    """Print the rendered input and exact loss-bearing tokens.

    SFT stage 3 -- label preflight.  ms-swift exposes the same debugging idea in
    ``swift/pipelines/train/sft.py::SwiftSft._show_dataset``.  Inspecting this
    output prevents prompt leakage or an all--100 label tensor from hiding
    behind a plausible training loss.
    """
    stats = dataset.stats
    target_ratio = stats.target_tokens / max(1, stats.total_tokens)
    print(
        f'[{name}] source={stats.source_rows}, kept={stats.kept_rows}, '
        f'dropped_overlength={stats.dropped_overlength}, total_tokens={stats.total_tokens}, '
        f'target_tokens={stats.target_tokens}, target_ratio={target_ratio:.2%}')
    feature = dataset[0]
    target_ids = [token_id for token_id, label in zip(feature['input_ids'], feature['labels'])
                  if label != IGNORE_INDEX]
    print(f'[{name}] first rendered sample:\n{tokenizer.decode(feature["input_ids"], skip_special_tokens=False)}')
    print(f'[{name}] first supervised target:\n{tokenizer.decode(target_ids, skip_special_tokens=False)}')


def resolve_resume_checkpoint(value: str | None, output_dir: str) -> str | None:
    """Resolve a full-state Trainer recovery point.

    SFT stage 8 -- resume.  ms-swift resolves it in
    ``swift/pipelines/train/sft.py::SwiftSft._get_resume_checkpoint``.  Its
    TunerMixin first restores the trainable adapter; Trainer.train then restores
    optimizer, scheduler, RNG and global step.  This script follows that split.
    """
    if value != 'auto':
        if value is not None and not Path(value).is_dir():
            raise FileNotFoundError(f'Checkpoint directory does not exist: {value}')
        return value
    checkpoint = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
    if checkpoint is None:
        raise FileNotFoundError(f'No checkpoint-* directory found under {output_dir}.')
    return checkpoint


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def is_rank_zero() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


def generate_smoke_test(trainer: Trainer, tokenizer, args: argparse.Namespace) -> dict[str, str]:
    """Generate once with the active adapter after training.

    SFT stage 9 -- behavioral verification.  ms-swift configures generation in
    ``swift/pipelines/train/sft.py::SwiftSft._prepare_generation_config`` and
    generated evaluation in ``swift/trainers/seq2seq_trainer.py``.  DDP ranks
    hold complete replicas, so only rank zero needs to run this smoke test.
    """
    trainer.accelerator.wait_for_everyone()
    if not trainer.is_world_process_zero():
        trainer.accelerator.wait_for_everyone()
        return {}

    model = trainer.accelerator.unwrap_model(trainer.model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True
    messages = [
        {'role': 'system', 'content': args.system_prompt},
        {'role': 'user', 'content': args.test_prompt},
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
        return_dict=True,
        return_tensors='pt',
    )
    device = next(model.parameters()).device
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    prompt_length = model_inputs['input_ids'].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
    print(f'[generation] prompt: {args.test_prompt}\n[generation] response: {response}')
    result = {'prompt': args.test_prompt, 'response': response}
    trainer.accelerator.wait_for_everyone()
    return result


def save_final_adapter(trainer: Trainer, tokenizer, final_dir: Path) -> None:
    """Save a small adapter, not a duplicate of Qwen3 base weights.

    SFT stage 8 -- deployable adapter export.  Transformers recognizes
    ``PeftModel`` and calls its ``save_pretrained``; ms-swift makes the same
    adapter-only choice in ``swift/trainers/mixin.py::_save_model/_save`` and
    also persists tokenizer/training metadata.  Trainer checkpoints additionally
    contain optimizer/scheduler/RNG state needed for exact continuation.
    """
    trainer.save_model(str(final_dir))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        required = [final_dir / 'adapter_config.json', final_dir / 'adapter_model.safetensors']
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f'Final PEFT adapter export is incomplete: {missing}')


def merge_and_save_lora(trainer: Trainer, tokenizer, merged_dir: Path, safe_merge: bool) -> None:
    """Optionally fold LoRA deltas into one ordinary Hugging Face model.

    SFT stage 8/9 -- full inference export.  This corresponds to
    ``swift/pipelines/export/merge_lora.py::MergeLoRA.merge_lora`` and
    ``swift/tuners/base.py::SwiftModel.merge_and_unload``.  The operation is
    optional because it writes another full copy of Qwen3-1.7B and the merged
    artifact cannot continue adapter training without reinjection.
    """
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        model = trainer.accelerator.unwrap_model(trainer.model)
        merged_model = model.merge_and_unload(safe_merge=safe_merge)
        merged_model.config.use_cache = True
        merged_model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size='5GB')
        tokenizer.save_pretrained(merged_dir)
        weight_files = list(merged_dir.glob('*.safetensors'))
        if not (merged_dir / 'config.json').is_file() or not weight_files:
            raise RuntimeError(f'Merged model export is incomplete: {merged_dir}')
    trainer.accelerator.wait_for_everyone()


def main() -> None:
    """Execute all LoRA SFT stages from data validation through export."""
    # SFT stage 0 -- parse/validate the run contract and set every RNG.
    # ms-swift: SftArguments/TunerArguments __post_init__ and SwiftSft entry.
    args = parse_args()
    if args.resume_from_checkpoint and args.adapter_name_or_path:
        raise ValueError('Use either --resume_from_checkpoint or --adapter_name_or_path, not both.')
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        raise ValueError('--lora_rank and --lora_alpha must be positive.')
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError('--lora_dropout must be in [0, 1).')
    set_seed(args.seed)

    # SFT stage 2 -- tokenizer/template contract.  ms-swift prepares these in
    # SwiftSft._prepare_model_tokenizer/_prepare_template and registers qwen3 in
    # swift/template/templates/qwen.py.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.chat_template is None:
        raise ValueError('The tokenizer has no chat_template; use the official Qwen3 tokenizer assets.')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # SFT stage 1-3 -- local load, deterministic split, ChatML and loss masks.
    all_train_records = read_records(args.train_file)
    if args.eval_file:
        train_records = all_train_records
        eval_records = read_records(args.eval_file)
    else:
        train_records, eval_records = split_records(all_train_records, args.val_ratio, args.seed)
    train_dataset = TokenizedSFTDataset(
        train_records, tokenizer, args.max_length, args.overlength_strategy,
        args.enable_thinking, 'train')
    eval_dataset = TokenizedSFTDataset(
        eval_records, tokenizer, args.max_length, args.overlength_strategy,
        args.enable_thinking, 'eval')
    if is_rank_zero():
        print_dataset_audit('train', train_dataset, tokenizer)
        print_dataset_audit('eval', eval_dataset, tokenizer)

    # SFT stage 4 -- dynamic right-padding.  label padding is -100, so neither
    # prompt nor batch padding contributes to cross entropy.
    # ms-swift: swift/template/base.py::Template.data_collator/_data_collator.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        return_tensors='pt',
    )
    audit_batch = data_collator([train_dataset[0], train_dataset[min(1, len(train_dataset) - 1)]])
    if not (audit_batch['input_ids'].shape == audit_batch['attention_mask'].shape
            == audit_batch['labels'].shape):
        raise AssertionError('Collated input, mask and label shapes differ.')
    if is_rank_zero():
        print(f'[collator] batch shape: {tuple(audit_batch["input_ids"].shape)}')
    if args.dry_run:
        if is_rank_zero():
            print('[dry-run] tokenizer, schema, ChatML, labels and collation: OK')
        return

    # SFT stage 5 -- TrainingArguments gives device placement and distributed
    # wrapping to Accelerate.  Launching this same file with torchrun enables
    # DDP; do not use device_map for training.  ms-swift builds its enhanced
    # Trainer arguments in swift/trainers/trainer_factory.py.
    dtype, use_bf16, use_fp16 = choose_precision(args.precision)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    global_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    tf32 = bool(
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
        and not args.full_determinism)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        optim=args.optim,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False} if args.gradient_checkpointing else None,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        prediction_loss_only=True,
        remove_unused_columns=True,
        label_names=['labels'],
        report_to='none',
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=args.full_determinism,
        ddp_find_unused_parameters=False,
        include_num_input_tokens_seen=True,
    )
    if is_rank_zero():
        recovery_mode = 'full-resume' if resume_checkpoint else (
            'adapter-warm-start' if args.adapter_name_or_path else 'fresh-adapter')
        print(
            f'[train] world_size={world_size}, global_batch_size={global_batch_size}, '
            f'mode={recovery_mode}, source={resume_checkpoint or args.adapter_name_or_path}')

    # SFT stage 4 -- load immutable Qwen3 base weights.  No device_map is set:
    # Trainer/Accelerate owns placement and DDP replication.  ms-swift enters
    # through swift/model/register.py::get_model_processor, called from
    # SwiftSft._prepare_model_tokenizer.
    model_kwargs: dict[str, Any] = {
        'dtype': dtype,
        'low_cpu_mem_usage': True,
        'trust_remote_code': args.trust_remote_code,
        'local_files_only': args.local_files_only,
    }
    if args.attn_implementation != 'auto':
        model_kwargs['attn_implementation'] = args.attn_implementation
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # SFT stage 4 -- resolve module suffixes and inject/load the only trainable
    # weights.  On Qwen3 all-linear normally means q/k/v/o + gate/up/down.
    target_modules = resolve_target_modules(base_model, args.target_modules)
    modules_to_save = parse_module_list(args.modules_to_save)
    validate_modules_to_save(base_model, modules_to_save)
    model = create_or_load_lora_model(
        base_model, args, target_modules, modules_to_save, resume_checkpoint)
    # A restored adapter owns its topology.  Command-line r/alpha/targets are
    # creation settings and must not pretend to override adapter_config.json.
    # This is also how ms-swift treats --adapters/--resume_from_checkpoint.
    restored_config = model.peft_config['default']
    target_modules = sorted(restored_config.target_modules or [])
    modules_to_save = sorted(restored_config.modules_to_save or [])
    if args.gradient_checkpointing:
        model.config.use_cache = False
        # Required by the reentrant checkpoint path and retained for parity with
        # ms-swift's TrainerMixin._prepare_gradient_checkpointing.  It is safe
        # with the non-reentrant path selected above.
        model.enable_input_require_grads()
    lora_audit = audit_lora_model(model, target_modules, modules_to_save)
    if is_rank_zero():
        print(f'[lora] target_modules={target_modules}, modules_to_save={modules_to_save}')
        print(
            '[lora] total/trainable parameters: '
            f'{lora_audit["total_parameters"]:,}/{lora_audit["trainable_parameters"]:,} '
            f'({lora_audit["trainable_ratio"]:.4%})')

    # SFT stage 6 -- Trainer supplies forward/loss/backward, accumulation,
    # clipping, AdamW, cosine scheduler, AMP and Accelerate DDP.  ms-swift creates
    # its Seq2SeqTrainer in SwiftSft.run; QwenSFTTrainer above adds its principal
    # Qwen3 selective-logit optimization without hiding the standard engine.
    trainer = QwenSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        use_logits_to_keep=args.use_logits_to_keep,
    )

    # SFT stage 6/8 -- optimize LoRA and write checkpoint-*.  Because the model
    # is a PeftModel, each checkpoint stores adapter weights/config rather than
    # the full base; Trainer additionally stores optimizer/scheduler/RNG/state.
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.log_metrics('train', train_result.metrics)
    trainer.save_metrics('train', train_result.metrics)
    trainer.save_state()

    # SFT stage 7 -- teacher-forced validation over assistant-only targets.
    # Perplexity is exp(token NLL), useful as a regression signal but not a
    # complete instruction-following quality metric.
    eval_metrics = trainer.evaluate()
    if 'eval_loss' in eval_metrics:
        eval_metrics['perplexity'] = (
            math.exp(eval_metrics['eval_loss']) if eval_metrics['eval_loss'] < 20 else float('inf'))
    trainer.log_metrics('eval', eval_metrics)
    trainer.save_metrics('eval', eval_metrics)

    # SFT stage 8-9 -- adapter delivery and actual post-train inference check.
    final_adapter_dir = output_dir / 'final_adapter'
    save_final_adapter(trainer, tokenizer, final_adapter_dir)
    generation_result = generate_smoke_test(trainer, tokenizer, args)
    if args.merge_lora:
        merge_and_save_lora(trainer, tokenizer, output_dir / 'merged_model', args.safe_merge)

    if trainer.is_world_process_zero():
        peft_config = trainer.accelerator.unwrap_model(trainer.model).peft_config['default']
        manifest = {
            'script': Path(__file__).name,
            'torch_version': torch.__version__,
            'transformers_version': transformers.__version__,
            'peft_version': peft.__version__,
            'model_name_or_path': args.model_name_or_path,
            'train_file': str(Path(args.train_file).resolve()),
            'train_file_sha256': sha256_file(args.train_file),
            'eval_file': str(Path(args.eval_file).resolve()) if args.eval_file else None,
            'eval_file_sha256': sha256_file(args.eval_file) if args.eval_file else None,
            'arguments': vars(args),
            'effective_peft_config': peft_config.to_dict(),
            'lora_audit': lora_audit,
            'train_stats': asdict(train_dataset.stats),
            'eval_stats': asdict(eval_dataset.stats),
        }
        (output_dir / 'run_manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        (output_dir / 'generation_smoke_test.json').write_text(
            json.dumps(generation_result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[done] resumable adapter checkpoints: {output_dir}/checkpoint-*')
        print(f'[done] inference adapter: {final_adapter_dir}')
        if args.merge_lora:
            print(f'[done] merged inference model: {output_dir / "merged_model"}')


if __name__ == '__main__':
    main()
