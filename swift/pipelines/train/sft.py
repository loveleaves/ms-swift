# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from datasets import Dataset as HfDataset
from typing import List, Optional, Union

from swift.arguments import SftArguments
from swift.dataset import (AddLengthPreprocessor, DatasetLoader, EncodePreprocessor, IterablePackingDataset,
                           LazyLLMDataset, PackingDataset, load_dataset)
from swift.infer_engine import prepare_generation_config
from swift.ray_utils import RayHelper
from swift.sequence_parallel import sequence_parallel
from swift.trainers import TrainerFactory
from swift.utils import append_to_jsonl, get_logger, get_model_parameter_info, is_master, plot_images, stat_array
from ..base import SwiftPipeline
from ..utils import get_cached_dataset
from .tuner import TunerMixin

logger = get_logger()


@RayHelper.worker(group=['default'])
class SwiftSft(SwiftPipeline, TunerMixin):
    """[新手导读] `swift sft` 的训练编排类，是理解整个框架的主线。

    完整生命周期（按调用顺序）：
      __init__: 解析参数(基类) -> _prepare_model_tokenizer(查 MODEL_MAPPING 加载模型)
                -> _prepare_template(查 TEMPLATE_MAPPING 取对话模板, set_mode('train'))
      run():    _prepare_dataset(加载+编码数据) -> prepare_model(TunerMixin 注入 LoRA 等)
                -> TrainerFactory 构建 Trainer -> train() -> _save_trainer_state
    RLHF 管线（rlhf.py）继承本类并复用绝大部分步骤，只替换 Trainer 与数据格式。
    """
    args_class = SftArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], SftArguments]] = None) -> None:
        super().__init__(args)
        self.train_msg = {}  # 收集训练过程信息（参数量、checkpoint 路径、指标等），最终写入 logging.jsonl
        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_flash_ckpt()

    @RayHelper.function(group='default')
    def _prepare_flash_ckpt(self):
        if self.args.use_flash_ckpt:
            try:
                import dlrover.trainer.torch.flash_checkpoint.hf_trainer
            except ImportError:
                raise ValueError('Please install dlrover to use flash ckpt `pip install dlrover[k8s,torch]')

    def _prepare_generation_config(self):
        args = self.args
        self.model.origin_generation_config = self.model.generation_config
        self.model.generation_config = prepare_generation_config(self.model.generation_config,
                                                                 args.get_request_config(), self.tokenizer)
        logger.info(f'model.generation_config: {self.model.generation_config}')

    @RayHelper.function(group='default')
    def _prepare_model_tokenizer(self, **kwargs):
        args = self.args
        # 根据 --model 自动匹配 model_type，加载模型与 processor（纯文本模型即 tokenizer，
        # 多模态模型为含图像处理器的 Processor）。实现见 swift/model/register.py: get_model_processor。
        self.model, self.processor = args.get_model_processor(**kwargs)
        if args.sequence_parallel_size > 1:
            sequence_parallel.prepare(
                args.sequence_parallel_size, model=self.model, tokenizer=self.processor, padding_free=args.padding_free)
        if self.model is None:
            return
        if hasattr(self.model, 'hf_device_map'):
            logger.info(f'model.hf_device_map: {self.model.hf_device_map}')

        logger.info(f'model_info: {self.model.model_info}')

        self._prepare_generation_config()

    @RayHelper.function(group='default')
    def _prepare_template(self) -> None:
        args = self.args
        # 模板负责把 messages 编码成 input_ids/labels，是连接数据与模型的枢纽。
        # set_mode('train') 决定编码行为：训练模式会生成 labels（推理模式则不会）。
        template = args.get_template(self.processor)
        template.set_mode('train')
        if template.use_model:
            template.model = self.model
        support_padding_free = template.support_padding_free
        if support_padding_free is None:
            support_padding_free = not args.model_meta.is_multimodal
        if (args.padding_free or args.packing) and not support_padding_free:
            raise ValueError(f'Template `{args.template}` does not support padding free or packing.')
        self.template = template

    def _get_dataset(self):
        # 按 --dataset 表达式加载原始数据集并切分 train/val。
        # 此时样本仍是 messages 等"标准格式"的字典，尚未编码成 token。
        # The random shuffling of the training set occurs in the dataloader of the trainer.
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        train_dataset, val_dataset = None, None
        if args.dataset:
            train_dataset, val_dataset = load_dataset(
                args.dataset,
                split_dataset_ratio=args.split_dataset_ratio,
                shuffle=args.dataset_shuffle,
                **dataset_kwargs)
        if len(args.val_dataset) > 0:
            # Loading val dataset
            dataset_kwargs.pop('interleave_prob', None)
            _, val_dataset = load_dataset(
                args.val_dataset, split_dataset_ratio=1.0, shuffle=args.val_dataset_shuffle, **dataset_kwargs)
            assert args.split_dataset_ratio == 0.
        if args.truncation_strategy == 'split':
            logger.info(f'train_dataset: {train_dataset}')
            logger.info(f'val_dataset: {val_dataset}')
        return train_dataset, val_dataset

    def _save_val_dataset(self, val_dataset):
        args = self.args
        output_dir = getattr(args, 'output_dir', None) or getattr(args, 'save')
        if is_master() and isinstance(val_dataset, HfDataset) and not args.val_dataset:
            os.makedirs(output_dir, exist_ok=True)
            val_dataset_path = os.path.join(output_dir, 'val_dataset.jsonl')
            append_to_jsonl(val_dataset_path, val_dataset.to_list())
            logger.info(f'The split dataset from the training set will be saved at: `{val_dataset_path}`.')

    @RayHelper.function(group='default')
    def _prepare_dataset(self):
        # 数据准备总调度：加载(_get_dataset) -> 编码(_encode_dataset)
        # -> 后处理(_post_process_datasets: lazy 包装/packing)。
        args = self.args
        # Defer encoding to the training phase
        # GRPO/GKD 在训练时需要先 rollout 采样再编码，因此这里跳过预编码。
        pre_process = not (hasattr(args, 'rlhf_type') and args.rlhf_type in ['grpo', 'gkd'])
        if args.cached_dataset or args.cached_val_dataset:
            assert not args.streaming, 'Cached dataset does not support streaming.'
            train_datasets, val_datasets = get_cached_dataset(self.args)
        else:
            train_datasets, val_datasets = [], []
        if args.dataset or args.val_dataset:
            train_dataset, val_dataset = self._get_dataset()
            train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)
            if train_dataset is not None:
                train_datasets.append(train_dataset)
            if val_dataset is not None:
                val_datasets.append(val_dataset)
        train_dataset = DatasetLoader.concat_datasets(train_datasets)
        val_dataset = DatasetLoader.concat_datasets(val_datasets)
        if args.truncation_strategy != 'split':
            logger.info(f'train_dataset: {train_dataset}')
            logger.info(f'val_dataset: {val_dataset}')
        datasets = [train_dataset, val_dataset]
        if not pre_process:
            return datasets
        datasets = self._post_process_datasets(datasets)
        self._show_dataset(*datasets)
        return datasets

    def _post_process_datasets(self, datasets: List) -> List:
        # 把数据集包装成训练时实际使用的形态，三种互斥策略：
        #   1. 默认: LazyLLMDataset —— __getitem__ 时才调用 template.encode，坏样本自动跳过；
        #   2. --packing: 把多条短样本装箱进同一序列，提升吞吐；
        #   3. --streaming: 流式数据边读边编码。
        args = self.args
        predict_with_generate = getattr(args, 'predict_with_generate', False)

        template = self.template
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            if i == 1 and predict_with_generate:
                # val_dataset
                continue
            if not args.streaming and args.truncation_strategy != 'split':
                dataset = LazyLLMDataset(dataset, template.encode, strict=args.strict, random_state=args.data_seed)
            if args.packing:
                packing_dataset_cls = IterablePackingDataset if args.streaming else PackingDataset
                dataset = packing_dataset_cls(
                    template,
                    dataset,
                    num_proc=args.dataset_num_proc,
                    packing_length=args.packing_length,
                    packing_num_proc=args.packing_num_proc,
                    strict=args.strict,
                    load_from_cache_file=args.load_from_cache_file)
            elif args.streaming:
                preprocessor = EncodePreprocessor(template=template)
                dataset = preprocessor(
                    dataset,
                    num_proc=args.dataset_num_proc,
                    load_from_cache_file=args.load_from_cache_file,
                    strict=args.strict)
            datasets[i] = dataset
        return datasets

    @RayHelper.function(group='default')
    def run(self):
        # 训练主流程：数据 -> 保存参数快照 -> 注入 tuner -> 构建 Trainer -> 训练。
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()

        if args.task_type == 'seq_cls':
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)
            logger.info(f'args.problem_type: {args.problem_type}')
        # 把全部参数写入 output_dir/args.json，后续 swift infer/export 可自动读取续传配置。
        args.save_args()

        # prepare_model 来自 TunerMixin：按 --tuner_type 给模型注入 LoRA 等可训练结构，
        # 或全参训练时设置参数冻结。见 pipelines/train/tuner.py。
        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        logger.info(f'model: {self.model}')
        # 打印可训练参数量（LoRA 时通常只有百分之零点几），训练前自检的重要信息。
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        # 按任务类型选择 Trainer 子类（SFT/序列分类/Embedding/Reranker...），
        # 均基于 transformers Trainer 扩展，见 swift/trainers/。
        trainer_cls = TrainerFactory.get_trainer_cls(args)
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            template=self.template,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)

    def _get_trainer_kwargs(self):
        return {}

    def _handle_trainer_state(self, trainer, is_write_rank: bool):
        state = trainer.state
        if hasattr(state, 'last_model_checkpoint'):
            if self.args.create_checkpoint_symlink:
                last_checkpoint = os.path.join(self.args.output_dir, 'last')
                best_checkpoint = os.path.join(self.args.output_dir, 'best')
                if is_write_rank:
                    os.symlink(state.last_model_checkpoint, last_checkpoint)
                    os.symlink(state.best_model_checkpoint, best_checkpoint)
                state.last_model_checkpoint = last_checkpoint
                state.best_model_checkpoint = best_checkpoint
        else:
            state.last_model_checkpoint = None
        logger.info_if(f'last_model_checkpoint: {state.last_model_checkpoint}', cond=is_write_rank)
        logger.info_if(f'best_model_checkpoint: {state.best_model_checkpoint}', cond=is_write_rank)

    def _save_trainer_state(self, trainer):
        training_args = trainer.args
        state = trainer.state
        self._handle_trainer_state(trainer, is_master())

        if is_master():
            # Visualization
            if 'tensorboard' in training_args.report_to:
                images_dir = os.path.join(training_args.output_dir, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)
            if training_args.push_to_hub:
                trainer.push_to_hub()

        self.train_msg.update({
            'last_model_checkpoint': state.last_model_checkpoint,
            'best_model_checkpoint': state.best_model_checkpoint,
            'best_metric': state.best_metric,
            'global_step': state.global_step,
            'log_history': state.log_history,
            'memory': getattr(state, 'max_memory', None),
        })
        if is_master():
            jsonl_path = os.path.join(training_args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, self.train_msg, strict=False)
        return self.train_msg

    def _get_resume_checkpoint(self, trainer):
        args = trainer.args
        if args.resume_from_checkpoint:
            return args.resume_from_checkpoint
        resume_checkpoint = None
        # If flash checkpoint is enabled, try to resume from the last complete checkpoint.
        # If the previous training finished, resume_checkpoint stays None.
        if args.use_flash_ckpt:
            # resume_checkpoint = <resume_dir>/checkpoint-<step>
            resume_checkpoint = trainer.get_resume_checkpoint()

        # Elastic runs require a universal checkpoint; fall back when missing or incomplete.
        callbacks = set(getattr(args, 'callbacks', []))
        elastic_enabled = 'deepspeed_elastic' in callbacks
        if elastic_enabled and (resume_checkpoint is None
                                or not os.path.exists(os.path.join(resume_checkpoint, 'latest_universal'))):
            # get_resume_checkpoint_until_find_ucp returns <resume_dir>/checkpoint-<step> with latest_universal,
            # or None; when None, no universal checkpoint exists and training starts from scratch.
            resume_checkpoint = trainer.get_resume_checkpoint_until_find_ucp()
        return resume_checkpoint

    def train(self, trainer):
        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        resume_checkpoint = self._get_resume_checkpoint(trainer)
        try:
            # 真正的训练循环在 transformers Trainer.train() 内部（前向/反向/优化器步进），
            # swift 通过自定义 Trainer 子类与 data_collator 注入自己的逻辑。
            trainer.train(resume_checkpoint)
        finally:
            res = self._save_trainer_state(trainer)
            if self.args.use_flash_ckpt and hasattr(trainer, 'flash_checkpointer'):
                trainer.wait_latest_checkpoint(trainer.FLASH_CKPT_WAIT_TIMEOUT, trainer.state.global_step)

        return res

    @staticmethod
    def _stat_dataset(dataset: Union[HfDataset, PackingDataset, LazyLLMDataset]):
        if isinstance(dataset, LazyLLMDataset):
            dataset = dataset.dataset
        if isinstance(dataset, HfDataset):
            lengths = dataset['lengths']
            lengths = [max(length) if isinstance(length, list) else length for length in lengths]
        else:
            lengths = dataset.packed_length
        _, stat_str = stat_array(lengths)
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str

    def _show_dataset(self, train_dataset, val_dataset):
        args = self.args
        predict_with_generate = getattr(args, 'predict_with_generate', False)
        if is_master():
            inputs = train_dataset[0] if hasattr(train_dataset, '__len__') else next(iter(train_dataset))
            if isinstance(inputs, list):
                inputs = inputs[0]
            self.template.print_inputs(inputs)
        elif hasattr(train_dataset, '__len__'):
            # Avoid the random mismatch issue in LazyLLMDataset.
            inputs = train_dataset[0]
        if val_dataset is not None and hasattr(val_dataset, '__len__') and len(val_dataset) == 0:
            val_dataset = None
        if not args.lazy_tokenize and not args.streaming:
            self.train_msg['train_dataset'] = self._stat_dataset(train_dataset)
            if val_dataset is not None and not predict_with_generate:
                self.train_msg['val_dataset'] = self._stat_dataset(val_dataset)

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        # 预编码阶段：非 lazy/非流式时，训练前就用多进程 map 把全量数据
        # 过一遍 template.encode（messages -> input_ids/labels），训练期零编码开销。
        # lazy_tokenize（多模态默认）则跳过这里，推迟到 LazyLLMDataset.__getitem__。
        template = self.template
        args = self.args
        self._save_val_dataset(val_dataset)

        predict_with_generate = getattr(args, 'predict_with_generate', False)
        datasets = [train_dataset, val_dataset]
        if not pre_process:
            return datasets

        origin_template_model = template.model
        template.model = None  # Avoid serializing the model.
        if args.truncation_strategy == 'split':
            if args.task_type != 'causal_lm' or template.mode != 'train' or args.use_chat_template:
                raise ValueError('`--truncation_strategy split` is currently only supported for pre-training.')
            assert not args.lazy_tokenize, '`--truncation_strategy split` does not support lazy_tokenize'

        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            if i == 1 and predict_with_generate:
                # val_dataset
                continue
            if not args.lazy_tokenize and not args.streaming:
                # Compatible with cached_dataset, only additionally write length here.
                preprocessor_cls = EncodePreprocessor if args.truncation_strategy == 'split' else AddLengthPreprocessor
                preprocessor = preprocessor_cls(template=template)
                batch_size = 100 if args.model_meta.is_multimodal else 1000
                dataset = preprocessor(
                    dataset,
                    num_proc=args.dataset_num_proc,
                    load_from_cache_file=args.load_from_cache_file,
                    strict=args.strict,
                    batch_size=batch_size)
                if len(dataset) == 0:
                    dataset = None
            datasets[i] = dataset
        template.model = origin_template_model

        return datasets


def sft_main(args: Optional[Union[List[str], SftArguments]] = None):
    # `swift sft` 与 Python API `from swift import sft_main` 的共同入口。
    return SwiftSft(args).main()
