# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import json
import os
import subprocess
import sys
import yaml
from typing import Dict, List, Optional

# [新手导读] swift 命令的总入口。`swift sft ...` 实际执行的就是本文件的 cli_main()。
# 子命令 -> 对应入口模块的路由表。例如 `swift sft` 会路由到 swift/cli/sft.py，
# 而 cli/sft.py 内部只有一行：调用 pipelines/train 下的 sft_main()。
ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'swift.cli.pt',
    'sft': 'swift.cli.sft',
    'infer': 'swift.cli.infer',
    'merge-lora': 'swift.cli.merge_lora',
    'web-ui': 'swift.cli.web_ui',
    'deploy': 'swift.cli.deploy',
    'rollout': 'swift.cli.rollout',
    'rlhf': 'swift.cli.rlhf',
    'sample': 'swift.cli.sample',
    'export': 'swift.cli.export',
    'eval': 'swift.cli.eval',
    'app': 'swift.cli.app',
}


def use_torchrun() -> bool:
    # 只要用户设置了 NPROC_PER_NODE 或 NNODES 环境变量，就认为需要多卡/多机分布式，
    # 后续会改用 `python -m torch.distributed.run`（即 torchrun）来拉起训练。
    nproc_per_node = os.getenv('NPROC_PER_NODE')
    nnodes = os.getenv('NNODES')
    if nproc_per_node is None and nnodes is None:
        return False
    return True


def parse_yaml_args(argv):
    # 支持 `swift sft config.yaml/config.json` 的配置文件写法：
    # 把文件里的 k: v 全部展开为 `--k v` 形式的命令行参数（原地替换 argv[0]），
    # 其中 ENV 字段会被注入到环境变量。这样配置文件和命令行参数走同一条解析路径。
    if not argv:
        return
    config = None
    if argv[0].endswith('.json'):
        with open(argv[0], 'r', encoding='utf-8') as f:
            config = json.load(f)
    elif argv[0].endswith('.yaml') or argv[0].endswith('.yml'):
        with open(argv[0], 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    if config is None:
        return
    # Used for saving configurations
    os.environ['SWIFT_CONFIG_FILE'] = argv[0]

    env = config.pop('ENV', None)
    if env:
        for k, v in env.items():
            if k not in os.environ:
                os.environ[k] = str(v)
            elif str(v) != os.environ[k]:
                logger.warning(f'{k} is already set in environment, using `{os.environ[k]}` instead of `{v}`')
    config_argv = []
    for k, v in config.items():
        config_argv.append(f'--{k}')
        if isinstance(v, list):
            config_argv += v
        else:
            if isinstance(v, dict):
                v = json.dumps(v, ensure_ascii=False)
            else:
                v = str(v)
            config_argv.append(v)
    argv[0:1] = config_argv


def get_torchrun_args() -> Optional[List[str]]:
    if not use_torchrun():
        return
    torchrun_args = []
    for env_key in ['NPROC_PER_NODE', 'MASTER_PORT', 'NNODES', 'NODE_RANK', 'MASTER_ADDR']:
        env_val = os.getenv(env_key)
        if env_val is None:
            continue
        torchrun_args += [f'--{env_key.lower()}', env_val]
    return torchrun_args


def cli_main(route_mapping: Optional[Dict[str, str]] = None, is_megatron: bool = False) -> None:
    # 核心流程：取出子命令名 -> 查路由表得到入口 .py 文件 -> 用子进程方式执行它。
    # 注意：这里不是 import 后直接调用，而是起一个新的 python 子进程，
    # 这样才能在分布式场景下无缝替换为 torchrun 启动。
    route_mapping = route_mapping or ROUTE_MAPPING
    argv = sys.argv[1:]
    method_name = argv[0].replace('_', '-')  # 兼容 `swift web_ui` 与 `swift web-ui` 两种写法
    argv = argv[1:]
    file_path = importlib.util.find_spec(route_mapping[method_name]).origin
    parse_yaml_args(argv)
    torchrun_args = get_torchrun_args()
    python_cmd = sys.executable
    # 只有训练/推理类子命令（pt/sft/rlhf/infer）才需要 torchrun 多进程启动；
    # 其余子命令（export/eval/app 等）即使设置了分布式环境变量也走单进程。
    if torchrun_args is None or (not is_megatron and method_name not in {'pt', 'sft', 'rlhf', 'infer'}):
        args = [python_cmd, file_path, *argv]
    else:
        args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == '__main__':
    cli_main()
