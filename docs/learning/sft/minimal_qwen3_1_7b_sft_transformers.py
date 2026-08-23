#!/usr/bin/env python3
"""Minimal, complete Qwen3-1.7B full-parameter SFT with Transformers.

Scope:
  JSON/JSONL -> messages -> Qwen chat template -> assistant-only labels
  -> dynamic padding -> optional FSDP2 full sharding -> Trainer
  -> evaluation -> resumable sharded checkpoints -> full final export
  -> deterministic generation smoke test.

This intentionally performs full-parameter fine-tuning.  It does not depend on
ms-swift, TRL or PEFT, so every essential SFT step remains visible.  Comments
named ``SFT stage`` state both the role of the code and the corresponding
implementation location in the current ms-swift repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from packaging.version import Version
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
QWEN_ASSISTANT_HEADER = '<|im_start|>assistant\n'
QWEN_MESSAGE_END = '<|im_end|>'
FORBIDDEN_CONTENT_MARKERS = ('<|im_start|>', '<|im_end|>')


@dataclass
class DatasetStats:
    source_rows: int = 0
    kept_rows: int = 0
    dropped_overlength: int = 0
    total_tokens: int = 0
    target_tokens: int = 0


class TokenizedSFTDataset(Dataset):
    """Eagerly turn normalized conversations into model inputs and SFT labels.

    SFT stage 2-3 -- template rendering and target construction.
    Function: produce equal-length input_ids/attention_mask/labels and remove
    overlength or target-free examples before a DataLoader can see them.
    ms-swift: ``swift/pipelines/train/sft.py::_encode_dataset`` (lines 341-383)
    dispatches encoding; ``swift/template/base.py::Template.encode/_encode``
    (lines 591-670 and 1462-1502) constructs token ids and labels.  The current
    minimal implementation is deliberately eager, while ms-swift can wrap rows
    with ``swift/dataset/utils.py::LazyLLMDataset`` (lines 57-122).
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

        assistant_header_ids = tokenizer.encode(QWEN_ASSISTANT_HEADER, add_special_tokens=False)
        message_end_ids = tokenizer.encode(QWEN_MESSAGE_END, add_special_tokens=False)
        if not assistant_header_ids or not message_end_ids:
            raise RuntimeError('Failed to tokenize Qwen assistant boundary markers.')

        for row_index, record in enumerate(records, start=1):
            messages = normalize_record(record, row_index=row_index, source_name=name)
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            if not isinstance(input_ids, list) or (input_ids and isinstance(input_ids[0], list)):
                raise TypeError('apply_chat_template should return one flat list of token ids per record.')

            expected_assistant_turns = sum(message['role'] == 'assistant' for message in messages)
            labels, assistant_turns = build_assistant_only_labels(
                input_ids,
                assistant_header_ids=assistant_header_ids,
                message_end_ids=message_end_ids,
            )
            if assistant_turns != expected_assistant_turns:
                raise ValueError(
                    f'{name} row {row_index}: found {assistant_turns} assistant token spans, '
                    f'but messages contain {expected_assistant_turns} assistant turns.')

            if len(input_ids) > max_length:
                if overlength_strategy == 'drop':
                    self.stats.dropped_overlength += 1
                    continue
                input_ids = input_ids[-max_length:]
                labels = labels[-max_length:]

            target_tokens = sum(label != IGNORE_INDEX for label in labels)
            if target_tokens == 0:
                raise ValueError(f'{name} row {row_index}: no assistant target tokens remain after processing.')

            feature = {
                'input_ids': input_ids,
                'attention_mask': [1] * len(input_ids),
                'labels': labels,
            }
            self.features.append(feature)
            self.stats.total_tokens += len(input_ids)
            self.stats.target_tokens += target_tokens

        self.stats.kept_rows = len(self.features)
        if not self.features:
            raise ValueError(
                f'{name}: no usable records. Increase --max_length, use --overlength_strategy left, '
                'or fix the dataset.')

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


def parse_args() -> argparse.Namespace:
    """Define the complete experiment contract exposed by this one-file trainer.

    SFT stage 0 -- configuration.  ms-swift collects the equivalent model,
    template, data, optimizer and distributed settings in
    ``swift/arguments/sft_args.py::SftArguments``; its FSDP2 preset is resolved
    by ``SftArguments._init_fsdp`` (lines 281-326).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_name_or_path', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--train_file', required=True, help='JSONL or JSON array containing messages records.')
    parser.add_argument('--eval_file', default=None, help='Optional independent validation JSONL/JSON file.')
    parser.add_argument('--output_dir', default='output/qwen3-1.7b-transformers-sft')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--overlength_strategy', choices=['drop', 'left'], default='drop')

    parser.add_argument('--num_train_epochs', type=float, default=1.0)
    parser.add_argument('--max_steps', type=int, default=-1, help='Positive value overrides num_train_epochs.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--lr_scheduler_type', default='cosine')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--optim', default='adamw_torch', help='For example: adamw_torch or adafactor.')
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=True)

    # SFT stage 5 -- distributed memory strategy.
    # Function: FSDP2 FULL_SHARD distributes parameters, gradients and AdamW
    # states across ranks; unlike DDP, every GPU need not hold a complete copy.
    # ms-swift: swift/arguments/sft_args.py::_init_fsdp (281-326) maps
    # ``--fsdp fsdp2`` to swift/config/fsdp2.json.  These defaults intentionally
    # mirror that preset.  FSDP2 must be launched with torchrun, not plain python.
    parser.add_argument(
        '--fsdp2',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable PyTorch FSDP2 FULL_SHARD; launch with torchrun and at least two CUDA processes.',
    )
    parser.add_argument(
        '--fsdp2_cpu_ram_efficient_loading',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Only local rank 0 reads full weights on each node; other ranks materialize shards during prepare.',
    )
    parser.add_argument(
        '--fsdp2_activation_checkpointing',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use FSDP-native non-reentrant activation checkpointing for wrapped decoder layers.',
    )
    parser.add_argument(
        '--fsdp2_final_state_dict_type',
        choices=['full', 'sharded'],
        default='full',
        help=(
            'full gathers an inference-ready final model on rank 0 (needs host RAM); sharded keeps a '
            'low-memory distributed checkpoint that must be merged before from_pretrained().'),
    )

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

    parser.add_argument(
        '--resume_from_checkpoint',
        nargs='?',
        const='auto',
        default=None,
        help='Checkpoint path, or pass the flag without a value to use the latest checkpoint.',
    )
    parser.add_argument('--enable_thinking', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--trust_remote_code', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--local_files_only', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test_prompt', default='请用两句话解释什么是有监督微调。')
    parser.add_argument('--system_prompt', default='你是一个准确、简洁的中文助手。')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--dry_run', action='store_true', help='Validate data/masks/collation without loading the model.')
    return parser.parse_args()


def read_records(path: str) -> list[dict[str, Any]]:
    """Read a JSON array or JSONL file into raw training records.

    SFT stage 1 -- dataset ingestion.  The general ms-swift implementation is
    ``swift/dataset/loader.py::load_dataset`` (from line 220), including hubs,
    streaming and registered preprocessors; this file keeps only local JSON.
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
    """Normalize two accepted schemas and validate the conversation grammar.

    SFT stage 1 -- schema normalization/data quality gate.  ms-swift's generic
    counterpart starts at ``swift/dataset/preprocessor/core.py::RowPreprocessor``
    (line 27), with messages-specific conversion in ``MessagesPreprocessor``
    (line 433).  Strict role checks here make later ChatML boundary scanning safe.
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
            f'{source_name} row {row_index}: expected messages or instruction/(output|response) fields.')

    normalized: list[dict[str, str]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f'{source_name} row {row_index}: message {message_index} is not an object.')
        role = message.get('role')
        content = message.get('content')
        if role not in {'system', 'user', 'assistant'}:
            raise ValueError(
                f'{source_name} row {row_index}: unsupported role {role!r}; '
                'this minimal text implementation supports system/user/assistant only.')
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f'{source_name} row {row_index}: empty content at message {message_index}.')
        if any(marker in content for marker in FORBIDDEN_CONTENT_MARKERS):
            raise ValueError(
                f'{source_name} row {row_index}: raw content may not contain Qwen control markers.')
        normalized.append({'role': role, 'content': content})

    cursor = 1 if normalized and normalized[0]['role'] == 'system' else 0
    expected_role = 'user'
    for message in normalized[cursor:]:
        if message['role'] != expected_role:
            raise ValueError(
                f'{source_name} row {row_index}: expected role {expected_role!r}, '
                f"got {message['role']!r}; messages must alternate user/assistant.")
        expected_role = 'assistant' if expected_role == 'user' else 'user'
    if not normalized or normalized[-1]['role'] != 'assistant':
        raise ValueError(f'{source_name} row {row_index}: a training conversation must end with assistant.')
    return normalized


def find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    last_start = len(sequence) - len(pattern)
    for index in range(start, last_start + 1):
        if sequence[index:index + len(pattern)] == pattern:
            return index
    return -1


def build_assistant_only_labels(
    input_ids: list[int],
    assistant_header_ids: list[int],
    message_end_ids: list[int],
) -> tuple[list[int], int]:
    """Mask prompt tokens and supervise assistant content plus ``<|im_end|>``.

    SFT stage 3 -- supervision mask.  ``-100`` removes system/user/header tokens
    from causal cross entropy.  ms-swift performs the generalized equivalent in
    ``swift/template/base.py::_encode_context_list`` (lines 1073-1098), then
    handles answer suffixes in ``_add_dynamic_eos`` (lines 1101-1113).  Qwen3's
    template metadata, including its non-thinking prefix, is registered at
    ``swift/template/templates/qwen.py`` lines 55-61.
    """
    labels = [IGNORE_INDEX] * len(input_ids)
    cursor = 0
    assistant_turns = 0

    while True:
        header_start = find_subsequence(input_ids, assistant_header_ids, cursor)
        if header_start < 0:
            break
        content_start = header_start + len(assistant_header_ids)
        message_end_start = find_subsequence(input_ids, message_end_ids, content_start)
        if message_end_start < 0:
            raise ValueError('Assistant message is missing the Qwen <|im_end|> token.')
        supervised_end = message_end_start + len(message_end_ids)
        labels[content_start:supervised_end] = input_ids[content_start:supervised_end]
        assistant_turns += 1
        cursor = supervised_end

    if assistant_turns == 0:
        raise ValueError('No Qwen assistant span was found after applying the chat template.')
    return labels, assistant_turns


def split_records(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic holdout when no independent eval file is given.

    SFT stage 1 -- train/eval split.  ms-swift routes the same responsibility
    through ``SwiftSft._get_dataset`` in ``swift/pipelines/train/sft.py``
    (lines 91-113) and the dataset loader's ``split_dataset_ratio``.
    """
    if len(records) < 2:
        raise ValueError('At least two records are required when --eval_file is not provided.')
    if not 0.0 < val_ratio < 1.0:
        raise ValueError('--val_ratio must be between 0 and 1.')

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    eval_size = max(1, min(len(records) - 1, round(len(records) * val_ratio)))
    eval_indices = set(indices[:eval_size])
    train_records = [row for index, row in enumerate(records) if index not in eval_indices]
    eval_records = [row for index, row in enumerate(records) if index in eval_indices]
    return train_records, eval_records


def choose_precision(name: str) -> tuple[torch.dtype, bool, bool]:
    """Resolve storage/load dtype and Trainer autocast flags.

    SFT stage 4-5 -- model numerical policy.  ms-swift resolves model dtype in
    its model arguments/loader and forwards bf16/fp16 through
    ``swift/trainers/trainer_factory.py::get_training_args`` (lines 61-73).
    FSDP2 later keeps FP32 master parameters while its mixed-precision policy
    casts forward/backward computation, which is handled by Accelerate.
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


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def is_rank_zero() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


def configure_fsdp2(args: argparse.Namespace) -> dict[str, Any] | None:
    """Build the ms-swift-compatible FSDP2 FULL_SHARD preset and prime env vars.

    SFT stage 0/5 -- distributed initialization.  These values deliberately
    follow ``swift/config/fsdp2.json``: each decoder block is an FSDP unit,
    parameters are re-sharded after forward, only local rank 0 on each node
    initially reads all weights, native checkpointing reduces activations, and
    ordinary training checkpoints remain sharded.  ms-swift loads this preset
    and sets FSDP/NCCL environment in
    ``swift/arguments/sft_args.py::_init_fsdp`` (lines 281-326).

    This must run before ``TrainingArguments`` initializes the process group and
    before ``from_pretrained``.  Otherwise CPU-RAM-efficient loading cannot put
    non-zero ranks on meta tensors and every process reads a full model copy.
    """
    if not args.fsdp2:
        return None

    fsdp_config: dict[str, Any] = {
        'fsdp_version': 2,
        'reshard_after_forward': True,
        'auto_wrap_policy': 'TRANSFORMER_BASED_WRAP',
        'cpu_ram_efficient_loading': args.fsdp2_cpu_ram_efficient_loading,
        'state_dict_type': 'SHARDED_STATE_DICT',
        'activation_checkpointing': args.fsdp2_activation_checkpointing,
    }

    # TrainingArguments forwards most FSDP options through environment variables
    # to Accelerate.  FSDP_VERSION is set explicitly just as ms-swift does,
    # because it selects DTensor-based FSDP2 rather than legacy FSDP1.
    os.environ['ACCELERATE_USE_FSDP'] = 'true'
    os.environ['FSDP_VERSION'] = '2'
    os.environ['FSDP_RESHARD_AFTER_FORWARD'] = 'true'
    os.environ['FSDP_STATE_DICT_TYPE'] = 'SHARDED_STATE_DICT'
    os.environ['FSDP_CPU_RAM_EFFICIENT_LOADING'] = str(args.fsdp2_cpu_ram_efficient_loading).lower()
    os.environ['FSDP_ACTIVATION_CHECKPOINTING'] = str(args.fsdp2_activation_checkpointing).lower()
    os.environ.setdefault('TORCH_NCCL_AVOID_RECORD_STREAMS', '1')
    return fsdp_config


def validate_fsdp2_launch(args: argparse.Namespace) -> None:
    """Fail early when FULL_SHARD was requested without a real CUDA job.

    SFT stage 5 -- launch safety gate.  ``swift/arguments/sft_args.py::_init_fsdp``
    (lines 286-291) similarly rejects incompatible device-map/DeepSpeed modes.
    This minimal script additionally requires two ranks so ``--fsdp2`` cannot
    silently degrade into an unsharded single-process run.
    """
    if not args.fsdp2:
        return
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size < 2 or 'LOCAL_RANK' not in os.environ:
        raise RuntimeError(
            '--fsdp2 requires a distributed launcher. Use torchrun --nproc_per_node=<GPU_COUNT> ...')
    if not torch.distributed.is_available():
        raise RuntimeError('This PyTorch build has no torch.distributed support required by FSDP2.')
    from accelerate.utils.constants import FSDP2_PYTORCH_VERSION

    installed_torch_version = Version(torch.__version__.split('+', maxsplit=1)[0])
    if installed_torch_version < Version(FSDP2_PYTORCH_VERSION):
        raise RuntimeError(
            f'Installed Accelerate requires torch>={FSDP2_PYTORCH_VERSION} for FSDP2; '
            f'found torch=={torch.__version__}.')
    if not torch.cuda.is_available():
        raise RuntimeError('This Qwen3 FSDP2 training path requires CUDA GPUs.')
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f'LOCAL_RANK={local_rank}, but this node exposes only {torch.cuda.device_count()} CUDA devices.')


def audit_fsdp2_trainer(trainer: Trainer, expect_wrapped: bool = False) -> None:
    """Assert that Accelerate selected FSDP2 and, after prepare, DTensor sharding.

    SFT stage 5-6 -- distributed backend verification.  ms-swift supplies the
    configuration, while its Trainer ultimately delegates wrapping to the same
    Transformers/Accelerate stack after construction at
    ``swift/pipelines/train/sft.py`` lines 215-226.
    """
    if not getattr(trainer.args, 'fsdp', None):
        return
    if not trainer.is_fsdp_enabled:
        raise RuntimeError('FSDP was configured, but Trainer/Accelerate did not enable it.')
    plugin = trainer.accelerator.state.fsdp_plugin
    if plugin.fsdp_version != 2 or plugin.reshard_after_forward is not True:
        raise RuntimeError(
            f'Expected FSDP2 FULL_SHARD, got version={plugin.fsdp_version}, '
            f'reshard_after_forward={plugin.reshard_after_forward}.')
    if expect_wrapped:
        from torch.distributed.fsdp import FSDPModule

        if not isinstance(trainer.model_wrapped, FSDPModule):
            raise RuntimeError('Training started, but the model is not an FSDP2 FSDPModule.')
    if trainer.is_world_process_zero():
        print(
            '[fsdp2] verified: '
            f'version={plugin.fsdp_version}, full_shard={plugin.reshard_after_forward}, '
            f'activation_checkpointing={plugin.activation_checkpointing}, '
            f'state_dict_type={plugin.state_dict_type}, wrapped={expect_wrapped}')


def print_dataset_audit(name: str, dataset: TokenizedSFTDataset, tokenizer) -> None:
    """Expose one rendered example and its exact supervised token subsequence.

    SFT stage 3 -- preflight audit.  ms-swift prints encoded training inputs in
    ``swift/pipelines/train/sft.py::SwiftSft._show_dataset`` (lines 323-339).
    Mask inspection is intentionally prominent here because a zero or leaked
    assistant mask can produce plausible-looking but invalid training loss.
    """
    stats = dataset.stats
    ratio = stats.target_tokens / max(1, stats.total_tokens)
    print(
        f'[{name}] source={stats.source_rows}, kept={stats.kept_rows}, '
        f'dropped_overlength={stats.dropped_overlength}, total_tokens={stats.total_tokens}, '
        f'target_tokens={stats.target_tokens}, target_ratio={ratio:.2%}')
    feature = dataset[0]
    target_ids = [token_id for token_id, label in zip(feature['input_ids'], feature['labels'])
                  if label != IGNORE_INDEX]
    print(f'[{name}] first rendered sample:\n{tokenizer.decode(feature["input_ids"], skip_special_tokens=False)}')
    print(f'[{name}] first supervised target:\n{tokenizer.decode(target_ids, skip_special_tokens=False)}')


def resolve_resume_checkpoint(value: str | None, output_dir: str) -> str | None:
    """Resolve an explicit checkpoint or the numerically latest checkpoint.

    SFT stage 8 -- recovery.  ms-swift selects the resume point in
    ``swift/pipelines/train/sft.py::SwiftSft._get_resume_checkpoint``
    (lines 274-293); Transformers then restores model/optimizer/scheduler/RNG,
    including distributed FSDP shards.
    """
    if value != 'auto':
        if value is not None and not Path(value).is_dir():
            raise FileNotFoundError(f'Checkpoint directory does not exist: {value}')
        return value
    checkpoint = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
    if checkpoint is None:
        raise FileNotFoundError(f'No checkpoint-* directory found under {output_dir}.')
    return checkpoint


def generate_smoke_test(trainer: Trainer, tokenizer, args: argparse.Namespace) -> dict[str, str]:
    """Run the post-training prompt through the same Qwen chat contract.

    SFT stage 9 -- behavioral smoke test.  ms-swift configures generation in
    ``SwiftSft._prepare_generation_config`` (sft.py lines 50-55) and its custom
    ``Seq2SeqTrainer.prediction_step`` implements generated evaluation in
    ``swift/trainers/seq2seq_trainer.py`` lines 63-97.

    Every FSDP rank must enter generation because parameter all-gathers are
    collectives; only rank 0 prints and persists the identical decoded result.
    """
    model = trainer.accelerator.unwrap_model(trainer.model)
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
    if trainer.is_world_process_zero():
        print(f'[generation] prompt: {args.test_prompt}\n[generation] response: {response}')
    return {'prompt': args.test_prompt, 'response': response}


def save_final_model(
    trainer: Trainer,
    tokenizer,
    final_dir: Path,
    fsdp2_final_state_dict_type: str,
) -> None:
    """Save either one loadable HF model or a mergeable final FSDP shard set.

    SFT stage 8 -- artifact export.  During FSDP2 training, SHARDED_STATE_DICT
    checkpoints avoid a full-weight gather and retain optimizer shards for
    recovery.  A normal ``from_pretrained`` delivery instead needs a full state
    dict, so this helper switches only the final export to FULL_STATE_DICT.
    ms-swift's asset saving extension is ``swift/trainers/mixin.py::_save``
    (lines 380-409); its FSDP2 checkpoint policy originates from
    ``swift/config/fsdp2.json``.  Actual distributed state-dict I/O in both
    implementations is delegated to Transformers/Accelerate.
    """
    final_dir.mkdir(parents=True, exist_ok=True)
    if not trainer.is_fsdp_enabled:
        trainer.save_model(str(final_dir))
    else:
        plugin = trainer.accelerator.state.fsdp_plugin
        if fsdp2_final_state_dict_type == 'full':
            # set_state_dict_type only creates matching configs when they are
            # None; clear the sharded configs before selecting rank-0 CPU gather.
            plugin.state_dict_config = None
            plugin.optim_state_dict_config = None
            plugin.set_state_dict_type('FULL_STATE_DICT')
            # All ranks participate in get_state_dict collectives; rank 0 alone
            # writes Hugging Face safetensors/config through Trainer._save.
            trainer.save_model(str(final_dir))
        else:
            from accelerate.utils import save_fsdp_model

            save_fsdp_model(plugin, trainer.accelerator, trainer.model, str(final_dir))
        trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        # Explicitly persist processor/config metadata for the sharded branch;
        # it is harmlessly redundant after ordinary Trainer.save_model.
        tokenizer.save_pretrained(final_dir)
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        unwrapped_model.config.save_pretrained(final_dir)
        if getattr(unwrapped_model, 'generation_config', None) is not None:
            unwrapped_model.generation_config.save_pretrained(final_dir)


def main() -> None:
    """Execute the SFT state machine from arguments through deployable output."""
    # SFT stage 0 -- parse and freeze the run contract.
    # ms-swift: SwiftSft.args_class plus SftArguments.__post_init__ in
    # swift/arguments/sft_args.py.  FSDP env must be present before distributed
    # initialization/model loading, so configure it immediately.
    args = parse_args()
    fsdp2_config = configure_fsdp2(args)
    set_seed(args.seed)

    # SFT stage 2 -- load the exact tokenizer/template contract used by Qwen3.
    # ms-swift: SwiftSft._prepare_model_tokenizer/_prepare_template at
    # swift/pipelines/train/sft.py lines 58-89; the qwen3 TemplateMeta is in
    # swift/template/templates/qwen.py lines 55-61.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.chat_template is None:
        raise ValueError('The tokenizer has no chat_template; use the official Qwen3 tokenizer files.')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # SFT stage 1-3 -- ingest, split, render ChatML and build assistant labels.
    # The dataset class above keeps these operations visible.  ms-swift's main
    # orchestration is SwiftSft._prepare_dataset (sft.py lines 125-190).
    all_train_records = read_records(args.train_file)
    if args.eval_file:
        train_records = all_train_records
        eval_records = read_records(args.eval_file)
    else:
        train_records, eval_records = split_records(all_train_records, args.val_ratio, args.seed)

    train_dataset = TokenizedSFTDataset(
        train_records,
        tokenizer,
        max_length=args.max_length,
        overlength_strategy=args.overlength_strategy,
        enable_thinking=args.enable_thinking,
        name='train',
    )
    eval_dataset = TokenizedSFTDataset(
        eval_records,
        tokenizer,
        max_length=args.max_length,
        overlength_strategy=args.overlength_strategy,
        enable_thinking=args.enable_thinking,
        name='eval',
    )
    if is_rank_zero():
        print_dataset_audit('train', train_dataset, tokenizer)
        print_dataset_audit('eval', eval_dataset, tokenizer)

    # SFT stage 4 -- form variable-length examples into one padded tensor batch.
    # labels use -100 rather than pad_token_id so padding contributes no loss.
    # ms-swift: Template.data_collator/_data_collator in
    # swift/template/base.py lines 1652-1685 and 1856-1915.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        return_tensors='pt',
    )
    audit_batch = data_collator([train_dataset[0], train_dataset[min(1, len(train_dataset) - 1)]])
    assert audit_batch['input_ids'].shape == audit_batch['labels'].shape
    assert audit_batch['input_ids'].shape == audit_batch['attention_mask'].shape
    if is_rank_zero():
        print(f'[collator] batch shape: {tuple(audit_batch["input_ids"].shape)}')

    if args.dry_run:
        if args.fsdp2 and is_rank_zero():
            print(f'[dry-run] requested FSDP2 config: {fsdp2_config}')
        if is_rank_zero():
            print('[dry-run] tokenizer, data validation, assistant-only labels and collation: OK')
        return

    # SFT stage 5 -- validate launch and create TrainingArguments *before* model
    # loading.  Constructing TrainingArguments initializes distributed state;
    # Transformers can then skip full checkpoint reads on non-zero FSDP ranks.
    # ms-swift does this through SftArguments._init_fsdp/_init_device before
    # SwiftSft.__init__ calls _prepare_model_tokenizer.
    validate_fsdp2_launch(args)
    dtype, use_bf16, use_fp16 = choose_precision(args.precision)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    global_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    )
    if is_rank_zero():
        print(f'[train] world_size={world_size}, global_batch_size={global_batch_size}, resume={resume_checkpoint}')

    tf32 = bool(
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
        and not args.full_determinism
    )
    # Native FSDP activation checkpointing wraps each auto-wrapped decoder
    # layer.  Enabling Trainer gradient checkpointing at the same time is an
    # error and also causes redundant backward all-gathers.  This matches
    # ms-swift's compatibility handling in sft_args.py lines 328-356.
    trainer_gradient_checkpointing = args.gradient_checkpointing
    if args.fsdp2 and args.fsdp2_activation_checkpointing:
        trainer_gradient_checkpointing = False
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
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        optim=args.optim,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=tf32,
        gradient_checkpointing=trainer_gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {'use_reentrant': False} if trainer_gradient_checkpointing else None),
        # ``full_shard auto_wrap`` gives ZeRO-3-like parameter/gradient/optimizer
        # sharding. Qwen3Model._no_split_modules=["Qwen3DecoderLayer"] tells the
        # transformer-based policy where units may be independently gathered.
        fsdp='full_shard auto_wrap' if args.fsdp2 else None,
        # TrainingArguments normalizes this dictionary in place; pass a copy so
        # run_manifest.json retains the exact user-facing preset above.
        fsdp_config=dict(fsdp2_config) if fsdp2_config is not None else None,
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

    # SFT stage 4 -- load all pretrained causal-LM weights.  No device_map is
    # supplied: Trainer/Accelerate owns placement.  With CPU-efficient FSDP2,
    # local rank 0 loads the checkpoint and peer ranks create meta parameters
    # before Accelerate distributes DTensor shards.  ms-swift's entry is
    # swift/model/register.py::get_model_processor (lines 526 onward), called by
    # SwiftSft._prepare_model_tokenizer.
    model_kwargs: dict[str, Any] = {
        'torch_dtype': dtype,
        'low_cpu_mem_usage': True,
        'trust_remote_code': args.trust_remote_code,
        'local_files_only': args.local_files_only,
    }
    if args.attn_implementation != 'auto':
        model_kwargs['attn_implementation'] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if trainer_gradient_checkpointing or (args.fsdp2 and args.fsdp2_activation_checkpointing):
        model.config.use_cache = False

    # SFT stage 4 -- Full means every base-model parameter receives gradients;
    # FSDP later changes storage placement, not trainability.  ms-swift sets the
    # same invariant in swift/pipelines/train/tuner.py::prepare_model lines
    # 372-378, with optional user-requested freezing that this minimal code omits.
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_parameters != total_parameters:
        raise RuntimeError(
            f'This script is full-parameter SFT, but only {trainable_parameters:,}/{total_parameters:,} '
            'parameters are trainable.')
    if args.fsdp2 and not getattr(model, '_no_split_modules', None):
        raise RuntimeError('FSDP2 transformer auto-wrap needs model._no_split_modules, but it is empty.')
    if is_rank_zero():
        print(f'[model] total/trainable parameters: {total_parameters:,}/{trainable_parameters:,}')

    # SFT stage 6 -- bind model/data/collator to the training engine.  Trainer
    # supplies causal forward/loss/backward, accumulation, clipping, AdamW,
    # scheduler and distributed collectives.  ms-swift selects its enhanced
    # Seq2SeqTrainer at swift/trainers/trainer_factory.py lines 12-73 and creates
    # it in SwiftSft.run (sft.py lines 215-226).
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    if args.fsdp2:
        audit_fsdp2_trainer(trainer, expect_wrapped=False)

    # SFT stage 6/8 -- optimize all parameters and write resumable checkpoint-*.
    # Under FSDP2, Transformers/Accelerate save distributed model and optimizer
    # state using SHARDED_STATE_DICT; the same layout is consumed on resume.
    # ms-swift calls Trainer.train from SwiftSft.train (sft.py lines 295-308).
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    if args.fsdp2:
        audit_fsdp2_trainer(trainer, expect_wrapped=True)
    trainer.log_metrics('train', train_result.metrics)
    trainer.save_metrics('train', train_result.metrics)
    trainer.save_state()

    # SFT stage 7 -- teacher-forced validation over the same assistant-only
    # labels.  eval_loss is token NLL; perplexity is exp(NLL), not a business
    # quality metric.  ms-swift extends this in Seq2SeqTrainer.evaluate.
    eval_metrics = trainer.evaluate()
    if 'eval_loss' in eval_metrics:
        eval_metrics['perplexity'] = (
            math.exp(eval_metrics['eval_loss']) if eval_metrics['eval_loss'] < 20 else float('inf')
        )
    trainer.log_metrics('eval', eval_metrics)
    trainer.save_metrics('eval', eval_metrics)

    # SFT stage 8-9 -- export model assets, then execute one deterministic
    # behavior check.  Every FSDP rank participates in both collective paths.
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped_model.config.use_cache = True
    final_dir = output_dir / 'final'
    save_final_model(trainer, tokenizer, final_dir, args.fsdp2_final_state_dict_type)
    generation_result = generate_smoke_test(trainer, tokenizer, args)
    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        manifest = {
            'script': Path(__file__).name,
            'transformers_version': transformers.__version__,
            'torch_version': torch.__version__,
            'model_name_or_path': args.model_name_or_path,
            'train_file': str(Path(args.train_file).resolve()),
            'train_file_sha256': sha256_file(args.train_file),
            'eval_file': str(Path(args.eval_file).resolve()) if args.eval_file else None,
            'eval_file_sha256': sha256_file(args.eval_file) if args.eval_file else None,
            'arguments': vars(args),
            'effective_fsdp2_config': fsdp2_config,
            'train_stats': vars(train_dataset.stats),
            'eval_stats': vars(eval_dataset.stats),
        }
        (output_dir / 'run_manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        (output_dir / 'generation_smoke_test.json').write_text(
            json.dumps(generation_result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[done] resumable checkpoints: {output_dir}/checkpoint-*')
        if args.fsdp2 and args.fsdp2_final_state_dict_type == 'sharded':
            print(f'[done] mergeable final FSDP shards: {final_dir}/pytorch_model_fsdp_0')
            print(
                '[done] merge before from_pretrained: '
                f'accelerate merge-weights {final_dir}/pytorch_model_fsdp_0 {final_dir}')
        else:
            print(f'[done] inference-ready model: {final_dir}')


if __name__ == '__main__':
    main()
