import datetime
import importlib.metadata
import math
import os
import re
import shutil
from pathlib import Path

import torch


# =========================
# 概率与数值计算工具
# =========================

# 计算对角高斯分布在给定采样点 t 处的对数概率。
# 这个函数是早期 SAC 实现里常见的辅助函数，当前仓库主流程里不一定直接用到，
# 但保留它对后续扩展其他策略分布实现仍然有帮助。
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


# 以数值更稳定的方式计算 log(sum(exp(x)))。
# 直接先 exp 再求和在数值上容易上溢，这里先减去最大值来稳定计算。
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


# =========================
# 网络参数同步工具
# =========================

# SAC 中的软更新：目标网络参数缓慢向当前网络靠拢。
# tau 越小，目标网络更新越慢，训练通常越稳定。
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# 硬更新：直接把 source 的参数完整复制给 target。
# 主要用于初始化 target 网络。
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# =========================
# 训练运行与日志工具
# =========================

# 根据模型 checkpoint 路径推导对应 replay buffer 的保存路径。
# 例如 xxx.pt -> xxx_buffer.pkl，便于断点恢复时同时加载经验池。
def buffer_path_from_checkpoint(ckpt_path):
    return os.path.splitext(ckpt_path)[0] + "_buffer.pkl"


def prune_old_episode_checkpoints(checkpoints_dir, keep_last_checkpoints):
    # 只轮转普通的 episode checkpoint：
    # - 保留 best.pt / latest.pt
    # - 仅清理 episode_XXXX.pt 及其对应 buffer
    # - 始终保留最近 keep_last_checkpoints 份
    if keep_last_checkpoints is None:
        return

    keep_last_checkpoints = int(keep_last_checkpoints)
    if keep_last_checkpoints <= 0:
        return

    pattern = re.compile(r"^episode_(\d+)\.pt$")
    episode_checkpoints = []

    for filename in os.listdir(checkpoints_dir):
        match = pattern.match(filename)
        if not match:
            continue

        episode_number = int(match.group(1))
        ckpt_path = os.path.join(checkpoints_dir, filename)
        episode_checkpoints.append((episode_number, ckpt_path))

    if len(episode_checkpoints) <= keep_last_checkpoints:
        return

    episode_checkpoints.sort(key=lambda item: item[0])
    to_remove = episode_checkpoints[:-keep_last_checkpoints]

    for episode_number, ckpt_path in to_remove:
        buffer_path = buffer_path_from_checkpoint(ckpt_path)

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"Pruned old checkpoint: episode_{episode_number:04d}.pt")

        if os.path.exists(buffer_path):
            os.remove(buffer_path)
            print(f"Pruned old replay buffer: {os.path.basename(buffer_path)}")


# 一个简单的双写输出工具：
# 让 print 既输出到终端，也同步写入日志文件。
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _safe_package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-found"


def _tensor_preview(tensor, max_elements=8):
    tensor_cpu = tensor.detach().cpu()
    flat = tensor_cpu.reshape(-1)
    values = ", ".join(f"{value:.6g}" for value in flat[:max_elements].tolist())
    if flat.numel() > max_elements:
        values += ", ..."
    return (
        f"shape={tuple(tensor_cpu.shape)}, "
        f"dtype={tensor_cpu.dtype}, "
        f"device={tensor.device}, "
        f"values=[{values}]"
    )


def _append_module_summary(lines, title, module):
    lines.append(title)
    lines.append("-" * 80)
    lines.append(f"class: {module.__class__.__name__}")
    lines.append("architecture:")
    lines.append(str(module))
    lines.append("")

    named_parameters = list(module.named_parameters())
    total_parameters = sum(parameter.numel() for _, parameter in named_parameters)
    trainable_parameters = sum(
        parameter.numel() for _, parameter in named_parameters if parameter.requires_grad
    )

    lines.append("parameter_count:")
    lines.append(f"  total: {total_parameters}")
    lines.append(f"  trainable: {trainable_parameters}")
    lines.append("")

    lines.append("named_parameters:")
    if named_parameters:
        for name, parameter in named_parameters:
            lines.append(
                "  - {}: shape={}, numel={}, requires_grad={}".format(
                    name,
                    tuple(parameter.shape),
                    parameter.numel(),
                    parameter.requires_grad,
                )
            )
    else:
        lines.append("  - none")
    lines.append("")

    named_buffers = list(module.named_buffers())
    lines.append("named_buffers:")
    if named_buffers:
        for name, buffer in named_buffers:
            lines.append(f"  - {name}: {_tensor_preview(buffer)}")
    else:
        lines.append("  - none")
    lines.append("")

    extra_tensor_attributes = []
    for attribute_name in ("action_scale", "action_bias", "noise"):
        if hasattr(module, attribute_name):
            attribute_value = getattr(module, attribute_name)
            if torch.is_tensor(attribute_value):
                extra_tensor_attributes.append((attribute_name, attribute_value))

    lines.append("extra_tensor_attributes:")
    if extra_tensor_attributes:
        for name, tensor in extra_tensor_attributes:
            lines.append(f"  - {name}: {_tensor_preview(tensor)}")
    else:
        lines.append("  - none")
    lines.append("")


def snapshot_run_source(run_dir, args, config_path=None):
    # 只为新 run 复制“最小但够用”的源码快照：
    # - 顶层训练/测试/工具入口
    # - 配置加载器
    # - 当前算法目录
    #
    # 这样未来即便仓库主线继续演化，单看某个 run 目录，
    # 也能知道当时究竟用的是哪套代码。
    repo_root = os.path.dirname(os.path.abspath(__file__))
    snapshot_root = os.path.join(run_dir, "source_snapshot")

    os.makedirs(snapshot_root, exist_ok=True)

    top_level_files = [
        "train.py",
        "test.py",
        "utils.py",
        "requirements.txt",
    ]
    for filename in top_level_files:
        source_path = os.path.join(repo_root, filename)
        if os.path.exists(source_path):
            shutil.copy2(source_path, os.path.join(snapshot_root, filename))

    configs_snapshot_dir = os.path.join(snapshot_root, "configs")
    os.makedirs(configs_snapshot_dir, exist_ok=True)
    for filename in ["__init__.py", "config_loader.py"]:
        source_path = os.path.join(repo_root, "configs", filename)
        if os.path.exists(source_path):
            shutil.copy2(source_path, os.path.join(configs_snapshot_dir, filename))

    # 额外把本次实验实际使用的 YAML 也放进源码快照目录，
    # 这样以后只看 source_snapshot/ 也能直接拿到当时那份配置。
    if config_path is not None:
        config_source_path = os.path.abspath(config_path)
        if os.path.exists(config_source_path):
            shutil.copy2(
                config_source_path,
                os.path.join(configs_snapshot_dir, os.path.basename(config_source_path)),
            )

    algorithms_snapshot_dir = os.path.join(snapshot_root, "algorithms")
    os.makedirs(algorithms_snapshot_dir, exist_ok=True)

    algorithms_init = os.path.join(repo_root, "algorithms", "__init__.py")
    if os.path.exists(algorithms_init):
        shutil.copy2(algorithms_init, os.path.join(algorithms_snapshot_dir, "__init__.py"))

    algorithm_source_dir = os.path.join(repo_root, "algorithms", args.algorithm_name)
    algorithm_target_dir = os.path.join(algorithms_snapshot_dir, args.algorithm_name)
    if os.path.exists(algorithm_source_dir):
        shutil.copytree(
            algorithm_source_dir,
            algorithm_target_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def write_run_summary(run_dir, args, env, agent, config_path):
    # 生成一份人类可读的运行摘要：
    # - 训练参数
    # - 算法名/算法参数
    # - 依赖版本
    # - policy / critic 结构
    summary_path = os.path.join(run_dir, "run_summary.txt")

    lines = []
    lines.append("Run Summary")
    lines.append("=" * 80)
    lines.append(f"created_at: {datetime.datetime.now().isoformat()}")
    lines.append(f"config_source: {config_path}")
    lines.append(f"config_snapshot: {os.path.join(run_dir, 'config.yaml')}")
    lines.append(f"algorithm_name: {args.algorithm_name}")
    lines.append(f"algorithm_variant: {args.algorithm_variant!r}")
    lines.append(f"experiment_name: {args.experiment_name}")
    lines.append(f"env_name: {args.env_name}")
    lines.append(f"observation_space: {env.observation_space}")
    lines.append(f"action_space: {env.action_space}")
    lines.append("")

    lines.append("Versions")
    lines.append("-" * 80)
    lines.append(f"python: {os.sys.version.split()[0]}")
    lines.append(f"torch: {torch.__version__}")
    lines.append(f"gymnasium: {_safe_package_version('gymnasium')}")
    lines.append(f"highway-env: {_safe_package_version('highway-env')}")
    lines.append(f"tensorboard: {_safe_package_version('tensorboard')}")
    lines.append("")

    lines.append("Args")
    lines.append("-" * 80)
    for key, value in sorted(vars(args).items()):
        lines.append(f"{key}: {value!r}")
    lines.append("")

    lines.append("Agent State")
    lines.append("-" * 80)
    for attribute_name in (
        "policy_type",
        "gamma",
        "tau",
        "alpha",
        "automatic_entropy_tuning",
        "target_update_interval",
        "device",
        "target_entropy",
    ):
        if hasattr(agent, attribute_name):
            lines.append(f"{attribute_name}: {getattr(agent, attribute_name)!r}")
    lines.append("")

    if hasattr(agent, "policy"):
        _append_module_summary(lines, "Policy", agent.policy)

    if hasattr(agent, "critic"):
        _append_module_summary(lines, "Critic", agent.critic)

    if hasattr(agent, "critic_target"):
        _append_module_summary(lines, "Critic Target", agent.critic_target)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def prepare_test_artifact_dir(args, config_path):
    # 为 test.py 预留一份可选的测试快照目录。
    # 默认不开启；一旦以后需要，只在对应训练 run 目录下生成：
    # runs/.../test_artifacts/<timestamp>__<config_stem>/
    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if os.path.basename(checkpoint_dir) == "checkpoints":
        run_dir = os.path.dirname(checkpoint_dir)
    else:
        run_dir = checkpoint_dir

    test_root = os.path.join(run_dir, "test_artifacts")
    artifact_name = "{}__{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        Path(config_path).stem,
    )
    artifact_dir = os.path.join(test_root, artifact_name)

    os.makedirs(artifact_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(artifact_dir, "config.yaml"))
    return artifact_dir


def write_test_summary(artifact_dir, args, env, agent, config_path):
    # 记录测试时的关键上下文：
    # - 使用了哪个 checkpoint
    # - 使用了哪份配置
    # - 当前 agent / 环境 / 依赖版本
    #
    # 这样如果以后某次评估结果需要回溯，单看 test_artifacts 目录也能定位上下文。
    summary_path = os.path.join(artifact_dir, "test_summary.txt")

    lines = []
    lines.append("Test Summary")
    lines.append("=" * 80)
    lines.append(f"created_at: {datetime.datetime.now().isoformat()}")
    lines.append(f"checkpoint: {args.checkpoint}")
    lines.append(f"config_source: {config_path}")
    lines.append(f"config_snapshot: {os.path.join(artifact_dir, 'config.yaml')}")
    lines.append(f"algorithm_name: {args.algorithm_name}")
    lines.append(f"algorithm_variant: {args.algorithm_variant!r}")
    lines.append(f"env_name: {args.env_name}")
    lines.append(f"observation_space: {env.observation_space}")
    lines.append(f"action_space: {env.action_space}")
    lines.append("")

    lines.append("Versions")
    lines.append("-" * 80)
    lines.append(f"python: {os.sys.version.split()[0]}")
    lines.append(f"torch: {torch.__version__}")
    lines.append(f"gymnasium: {_safe_package_version('gymnasium')}")
    lines.append(f"highway-env: {_safe_package_version('highway-env')}")
    lines.append(f"tensorboard: {_safe_package_version('tensorboard')}")
    lines.append("")

    lines.append("Args")
    lines.append("-" * 80)
    for key, value in sorted(vars(args).items()):
        lines.append(f"{key}: {value!r}")
    lines.append("")

    if hasattr(agent, "policy"):
        lines.append("Policy")
        lines.append("-" * 80)
        lines.append(str(agent.policy))
        lines.append("")

    if hasattr(agent, "critic"):
        lines.append("Critic")
        lines.append("-" * 80)
        lines.append(str(agent.critic))
        lines.append("")

    if hasattr(agent, "critic_target"):
        lines.append("Critic Target")
        lines.append("-" * 80)
        lines.append(str(agent.critic_target))
        lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# 准备当前训练运行的目录结构。
# 新训练时会创建 runs/<timestamp>__<algorithm_name>__<config_stem>/checkpoints，
# 并把本次实验使用的 YAML 配置复制到 run_dir/config.yaml；
# 从 checkpoint 恢复时则复用已有 run 目录，并继续追加日志。
def prepare_run_paths(args, config_path):
    if args.resume_checkpoint is not None:
        checkpoint_path = os.path.abspath(args.resume_checkpoint)
        checkpoints_dir = os.path.dirname(checkpoint_path)
        run_dir = os.path.dirname(checkpoints_dir)
        log_mode = "a"
    else:
        run_name_parts = [
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.algorithm_name,
        ]
        if getattr(args, "algorithm_variant", None):
            run_name_parts.append(args.algorithm_variant)
        run_name_parts.append(args.experiment_name)
        run_name = "__".join(run_name_parts)
        run_dir = os.path.join("runs", run_name)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        log_mode = "w"

        # 备份当前实验使用的配置文件，方便后续复现实验。
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy2(config_path, os.path.join(run_dir, "config.yaml"))

    os.makedirs(checkpoints_dir, exist_ok=True)
    return run_dir, checkpoints_dir, log_mode
