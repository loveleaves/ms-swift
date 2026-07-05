# Copyright (c) ModelScope Contributors. All rights reserved.
import datetime as dt
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import swift
from swift.arguments import AppArguments, BaseArguments, WebUIArguments
from swift.utils import ProcessorMixin, get_logger, parse_args, seed_everything

logger = get_logger()


class SwiftPipeline(ABC, ProcessorMixin):
    """[新手导读] 所有 `swift xxx` 子命令的公共基类（模板方法模式）。

    每个子命令（sft/rlhf/infer/export/eval/...）都是它的子类，只需要做两件事：
      1. 声明 `args_class`：本命令使用哪个参数 dataclass（如 SftArguments）；
      2. 实现 `run()`：本命令的实际业务逻辑。
    基类负责统一的骨架：解析参数 -> 设随机种子 -> 记录起止时间 -> 调用 run()。
    读懂这 60 行，所有 swift 子命令的执行框架就都清楚了。
    """
    args_class = BaseArguments

    def __init__(self, args: Optional[Union[List[str], args_class]] = None):
        # args 可以是 None（从 sys.argv 解析，CLI 场景）、字符串列表、
        # 或已构造好的参数对象（Python API 场景，如 sft_main(SftArguments(...))）。
        self.args = self._parse_args(args)
        args = self.args
        logger.info(f'args: {args}')
        self._set_seed()
        self._compat_dsw_gradio(args)

    def _set_seed(self):
        args = self.args
        if hasattr(args, 'seed'):
            # 种子 = 用户指定 seed + 当前进程 rank：保证整体可复现，
            # 同时各 rank 种子不同（避免分布式下各卡产生完全相同的随机行为）。
            seed = args.seed + max(getattr(args, 'rank', -1), 0)
            seed_everything(seed)
            logger.info(f'Global seed set to {seed}')

    def _parse_args(self, args: Optional[Union[List[str], args_class]] = None) -> args_class:
        if isinstance(args, self.args_class):
            return args
        assert self.args_class is not None
        args, remaining_argv = parse_args(self.args_class, args)
        # 默认对无法识别的参数直接报错（而非静默忽略），避免拼错参数名却悄悄不生效；
        # 可用 --ignore_args_error true 降级为告警。
        if len(remaining_argv) > 0:
            if getattr(args, 'ignore_args_error', False):
                logger.warning(f'remaining_argv: {remaining_argv}')
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')
        return args

    @staticmethod
    def _compat_dsw_gradio(args) -> None:
        if (isinstance(args, (WebUIArguments, AppArguments)) and 'JUPYTER_NAME' in os.environ
                and 'dsw-' in os.environ['JUPYTER_NAME'] and 'GRADIO_ROOT_PATH' not in os.environ):
            os.environ['GRADIO_ROOT_PATH'] = f"/{os.environ['JUPYTER_NAME']}/proxy/{args.server_port}"

    def main(self):
        # 对外的统一入口：cli/sft.py 等调用的就是 SwiftXxx(args).main()。
        logger.info(f'Start time of running main: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
        logger.info(f'swift.__version__: {swift.__version__}')
        result = self.run()
        logger.info(f'End time of running main: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
        return result

    @abstractmethod
    def run(self):
        pass
